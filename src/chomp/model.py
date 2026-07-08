# SPDX-License-Identifier: Apache-2.0

"""Model integration.

This file is intentionally the *only* place that knows about the model backend.
The rest of the codebase talks in terms of:
- params pytree (arrays)
- static pytree (non-arrays)
- `training_loss(params, static, batch, ...) -> scalar`

Design intent (senior-engineer hat on):
- You do NOT want random parts of the codebase reaching into Megalodon internals.
- You *will* change model code over time. The training system should barely notice.

Backends:
- `dummy`: a tiny embedding+linear LM for ultra-fast smoke tests
- `megalodon`: your real `megalodon_jax` engine
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from chomp.config import Config, dtype_from_str

if TYPE_CHECKING:
    from chomp.types import Batch

# ------------------------------ Dummy backend ------------------------------


class DummyLM(eqx.Module):
    """A tiny LM used for smoke tests.

    Contract:
        __call__(input_ids: [B, T], attention_mask: [B, T] bool | None) -> logits [B, T, V]

    We include compute_loss so the training code can treat DummyLM and Megalodon
    identically.
    """

    embed: eqx.nn.Embedding
    proj: eqx.nn.Linear
    dropout: eqx.nn.Dropout
    vocab_size: int = eqx.field(static=True)

    # DummyLM carries no recurrent or cross-token state, so segment isolation
    # holds trivially; this keeps strict-multipack smoke tests on the dummy
    # backend passing the capability gate.
    supports_segment_reset: ClassVar[bool] = True

    def __init__(self, *, vocab_size: int, d_model: int, dropout: float, key: jax.Array):
        """Initialize the dummy language model.

        :param int vocab_size: Vocabulary size.
        :param int d_model: Embedding dimension.
        :param float dropout: Dropout rate.
        :param jax.Array key: PRNG key for initialization.
        """
        k1, k2 = jax.random.split(key)
        self.vocab_size = vocab_size
        self.embed = eqx.nn.Embedding(num_embeddings=vocab_size, embedding_size=d_model, key=k1)
        self.proj = eqx.nn.Linear(d_model, vocab_size, use_bias=False, key=k2)
        self.dropout = eqx.nn.Dropout(dropout)

    def __call__(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array | None = None,
        *,
        deterministic: bool = True,
        key: jax.Array | None = None,
    ) -> jax.Array:
        x = self.embed.weight[input_ids]  # [B, T, D]
        if not deterministic:
            if key is None:
                raise ValueError("DummyLM requires a PRNG key when deterministic=False")
            x = self.dropout(x, key=key)
        logits = jnp.einsum("btd,vd->btv", x, self.proj.weight)
        return logits

    def compute_loss(
        self,
        input_ids: jax.Array,
        labels: jax.Array,
        attention_mask: jax.Array | None = None,
        segment_ids: jax.Array | None = None,
        position_ids: jax.Array | None = None,
        *,
        ignore_index: int = -100,
        deterministic: bool = True,
        key: jax.Array | None = None,
    ) -> jax.Array:
        """Compute cross-entropy loss with causal shift.

        :param jax.Array input_ids: Input token IDs of shape [B, T].
        :param jax.Array labels: Label token IDs of shape [B, T].
        :param attention_mask: Optional mask of shape [B, T].
        :param segment_ids: Optional segment IDs of shape [B, T].
        :param position_ids: Optional position IDs of shape [B, T].
        :param int ignore_index: Label value to ignore in loss.
        :param bool deterministic: If False, apply dropout.
        :param key: PRNG key required when deterministic=False.
        :return jax.Array: Scalar mean cross-entropy loss.
        """
        del segment_ids, position_ids
        logits = self(input_ids, attention_mask, deterministic=deterministic, key=key)

        # Shift for causal LM
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]

        if shift_labels.shape[1] == 0:
            return jnp.zeros((), dtype=jnp.float32)

        # Build mask for valid positions
        valid = shift_labels != ignore_index
        if attention_mask is not None:
            # Apply attention mask (shifted)
            valid = valid & attention_mask[:, 1:].astype(bool)

        # Compute cross-entropy
        per_pos = optax.softmax_cross_entropy_with_integer_labels(shift_logits, shift_labels)
        per_pos = per_pos.astype(jnp.float32)

        denom = jnp.maximum(jnp.sum(valid), 1)
        return jnp.sum(jnp.where(valid, per_pos, 0.0)) / denom


# ------------------------------ Builders -----------------------------------

# Repo-wide floor, enforced for every megalodon model build (train and
# generate) regardless of packing mode. Older versions lack full segment
# isolation (supports_segment_reset), and the pyproject git URL cannot carry
# a version specifier — a stale environment is the only way to end up below
# this, so refuse it outright rather than degrade per-feature.
_MIN_MEGALODON_JAX = "0.1.2"


def _require_megalodon_jax_version() -> None:
    """Fail fast when the installed megalodon-jax predates the required floor.

    :raises RuntimeError: If megalodon-jax is older than _MIN_MEGALODON_JAX
        or its version metadata cannot be read.
    """
    from importlib import metadata

    from packaging.version import Version

    try:
        found = metadata.version("megalodon-jax")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "Cannot verify the installed megalodon-jax version (no package "
            f"metadata); chomp requires megalodon-jax >= {_MIN_MEGALODON_JAX}. "
            "Reinstall it: pip install -U "
            "'megalodon-jax @ git+https://github.com/pszemraj/megalodon-jax.git'"
        ) from exc
    if Version(found) < Version(_MIN_MEGALODON_JAX):
        raise RuntimeError(
            f"chomp requires megalodon-jax >= {_MIN_MEGALODON_JAX}, found {found}. "
            "Older versions only isolate attention across packed documents, "
            "leaking ComplexEMA/TimestepNorm state. Upgrade: pip install -U "
            "'megalodon-jax @ git+https://github.com/pszemraj/megalodon-jax.git'"
        )


def build_model(cfg: Config, *, key: jax.Array) -> tuple[Any, Any]:
    """Build model and return (params, static).

    We always partition immediately:
      params, static = eqx.partition(model, eqx.is_array)

    Why?
    - We never want to stash full Modules in TrainState
    - It keeps checkpointing straightforward
    - It forces us to be explicit about what is 'learned' vs 'static'

    :param Config cfg: Model configuration.
    :param jax.Array key: PRNG key for model initialization.
    :raises ImportError: If megalodon backend requested but not installed.
    :raises RuntimeError: If the installed megalodon-jax is older than the repo-wide floor.
    :raises ValueError: If model.backend is unknown.
    :return tuple: (params, static) pytrees from eqx.partition.
    """

    if cfg.model.backend == "dummy":
        model = DummyLM(
            vocab_size=cfg.model.vocab_size,
            d_model=cfg.model.d_model,
            dropout=cfg.model.dropout,
            key=key,
        )
    elif cfg.model.backend == "megalodon":
        try:
            from megalodon_jax.config import MegalodonConfig
            from megalodon_jax.model import MegalodonForCausalLM
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "model.backend='megalodon' requires the `megalodon_jax` package. "
                "Install it (e.g., pip install -e /path/to/megalodon-jax)."
            ) from e
        _require_megalodon_jax_version()

        mcfg = MegalodonConfig(
            vocab_size=cfg.model.vocab_size,
            model_dim=cfg.model.model_dim,
            num_layers=cfg.model.num_layers,
            num_heads=cfg.model.num_heads,
            z_dim=cfg.model.z_dim,
            value_dim=cfg.model.value_dim,
            ffn_hidden_dim=cfg.model.ffn_hidden_dim,
            cema_ndim=cfg.model.cema_ndim,
            chunk_size=cfg.model.chunk_size,
            max_cache_len=cfg.model.max_cache_len,
            cache_unbounded=cfg.model.cache_unbounded,
            norm_num_groups=cfg.model.norm_num_groups,
            norm_eps=cfg.model.norm_eps,
            rope_base=cfg.model.rope_base,
            swiglu=cfg.model.swiglu,
            rescale_nffn=cfg.model.rescale_nffn,
            scale_emb=cfg.model.scale_emb,
            norm_affine=cfg.model.norm_affine,
            dropout=cfg.model.dropout,
            attention_dropout=cfg.model.attention_dropout,
            hidden_dropout=cfg.model.hidden_dropout,
            pad_token_id=cfg.model.pad_token_id,
            bos_token_id=cfg.model.bos_token_id,
            eos_token_id=cfg.model.eos_token_id,
            max_positions=cfg.model.max_positions,
            init_mode=cfg.model.init_mode,
            use_checkpoint=cfg.model.use_checkpoint,
            output_size=cfg.model.output_size,
            use_associative_segment_scan=cfg.model.use_associative_segment_scan,
            param_dtype=dtype_from_str(cfg.model.param_dtype),
            compute_dtype=dtype_from_str(cfg.model.compute_dtype),
            accum_dtype=dtype_from_str(cfg.model.accum_dtype),
            softmax_dtype=dtype_from_str(cfg.model.softmax_dtype),
            gemm_backend=cfg.model.gemm_backend,
        )

        model = MegalodonForCausalLM(mcfg, key=key)
    else:  # pragma: no cover
        raise ValueError(f"Unknown model.backend: {cfg.model.backend!r}")

    params, static = eqx.partition(model, eqx.is_array)
    return params, static


# ------------------------------ Forward/loss wrappers ----------------------


def training_loss(
    params: Any,
    static: Any,
    *,
    batch: Batch,
    deterministic: bool,
    key: jax.Array | None,
    use_packed_attention: bool = False,
) -> jax.Array:
    """Compute training loss.

    Guardrail: this function **does not accept** cache arguments.
    Training should never enable cache; it is an inference concern.

    For Megalodon, `compute_loss` internally enforces `cache=None`.

    :param Any params: Model parameters from eqx.partition.
    :param Any static: Static model components from eqx.partition.
    :param Batch batch: Batch with input_ids, labels, attention_mask, and packed metadata.
    :param bool deterministic: If False, apply dropout.
    :param key: PRNG key required when deterministic=False.
    :param bool use_packed_attention: Whether to pass segment_ids/position_ids to backend loss.
    :return jax.Array: Scalar loss value.
    """

    model = eqx.combine(params, static)

    # Batch tensors come in as [A, B, T]. We compute loss per microbatch and average.
    # The compiled train_step calls this on each microbatch slice (shape [B, T]).
    kwargs: dict[str, Any] = {
        "attention_mask": batch.attention_mask,
        "deterministic": deterministic,
        "key": key,
    }
    if use_packed_attention:
        kwargs["segment_ids"] = batch.segment_ids
        kwargs["position_ids"] = batch.position_ids
    return model.compute_loss(  # type: ignore[attr-defined]
        batch.input_ids,
        batch.labels,
        **kwargs,
    )


def supports_packed_attention(params: Any, static: Any) -> bool:
    """Return True if the model backend supports full packed-segment isolation.

    Checks the ``supports_segment_reset`` capability flag introduced in
    megalodon-jax 0.1.2. Signature introspection of ``compute_loss`` is not
    sufficient: older versions accepted segment_ids/position_ids but only
    isolated attention, leaking ComplexEMA and TimestepNorm state across
    packed document boundaries.

    :param Any params: Model parameters.
    :param Any static: Static model components.
    :return bool: True when the backend resets all recurrent state
        (attention, CEMA, TimestepNorm) at segment boundaries.
    """
    model = eqx.combine(params, static)
    return bool(getattr(model, "supports_segment_reset", False))
