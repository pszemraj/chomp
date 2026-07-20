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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from chomp.config import Config, dtype_from_str
from chomp.utils.tree import path_to_str

if TYPE_CHECKING:
    from chomp.types import Batch

from chomp.types import IGNORE_INDEX

# ------------------------------ Dummy backend ------------------------------


def causal_loss_mask(
    labels: jax.Array,
    attention_mask: jax.Array | None,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> jax.Array:
    """Return the valid causal-label mask after the one-token shift.

    :param jax.Array labels: Label IDs of shape [B, T].
    :param jax.Array | None attention_mask: Optional attention mask [B, T].
    :param int ignore_index: Label value excluded from loss.
    :return jax.Array: Boolean validity mask of shape [B, T - 1].
    """
    valid = labels[:, 1:] != ignore_index
    if attention_mask is not None:
        valid = valid & attention_mask[:, 1:].astype(bool)
    return valid


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
        *,
        ignore_index: int = IGNORE_INDEX,
        deterministic: bool = True,
        key: jax.Array | None = None,
    ) -> jax.Array:
        """Compute cross-entropy loss with causal shift.

        :param jax.Array input_ids: Input token IDs of shape [B, T].
        :param jax.Array labels: Label token IDs of shape [B, T].
        :param attention_mask: Optional mask of shape [B, T].
        :param segment_ids: Optional segment IDs of shape [B, T].
        :param int ignore_index: Label value to ignore in loss.
        :param bool deterministic: If False, apply dropout.
        :param key: PRNG key required when deterministic=False.
        :return jax.Array: Scalar mean cross-entropy loss.
        """
        del segment_ids
        logits = self(input_ids, attention_mask, deterministic=deterministic, key=key)

        # Shift for causal LM
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]

        if shift_labels.shape[1] == 0:
            return jnp.zeros((), dtype=jnp.float32)

        # Build mask for valid positions
        valid = causal_loss_mask(labels, attention_mask, ignore_index=ignore_index)

        # Compute cross-entropy
        per_pos = optax.softmax_cross_entropy_with_integer_labels(shift_logits, shift_labels)
        per_pos = per_pos.astype(jnp.float32)

        denom = jnp.maximum(jnp.sum(valid), 1)
        return jnp.sum(jnp.where(valid, per_pos, 0.0)) / denom


# ------------------------------ Builders -----------------------------------


@dataclass(frozen=True)
class ModelArrayClassification:
    """Optimizer role assigned to one trainable array leaf."""

    family: str
    decay: bool


def _optimizer_group(
    cfg: Config,
    leaf: Any,
    classification: ModelArrayClassification,
) -> str:
    """Return the optimizer group for one classified model array.

    :param Config cfg: Model and optimizer configuration.
    :param Any leaf: Candidate parameter leaf.
    :param ModelArrayClassification classification: Explicit model-family assignment.
    :return str: ``muon`` or ``adam``.
    """
    muon = cfg.optim.muon
    if cfg.optim.name != "muon" or not hasattr(leaf, "ndim") or leaf.ndim != 2:
        return "adam"
    if (
        muon.allow_all_2d
        or classification.family == "projection"
        or (muon.allow_tied_embed and cfg.model.share_emb and classification.family == "embedding")
    ):
        return "muon"
    return "adam"


def classify_model_array(cfg: Config, path: str) -> ModelArrayClassification:
    """Classify one backend trainable array for optimizer assignment.

    Known embeddings and projection weights receive weight decay, and known
    projection matrices may use Muon. Other arrays stay on Adam without decay.

    :param Config cfg: Model/optimizer configuration.
    :param str path: Stable dotted array path.
    :return ModelArrayClassification: Training family and decay policy.
    """
    if cfg.model.backend == "dummy":
        if path.endswith("embed.weight"):
            return ModelArrayClassification("embedding", True)
        if path.endswith("proj.weight"):
            return ModelArrayClassification("projection", True)
    elif cfg.model.backend == "megalodon":
        if path.endswith("embed.weight"):
            return ModelArrayClassification("embedding", True)
        if path.endswith("lm_head.weight"):
            return ModelArrayClassification("projection", True)
        if path.endswith("lm_head.bias"):
            return ModelArrayClassification("bias", False)

        linear_modules = (
            ".attn.wz.",
            ".attn.wv.",
            ".attn.wr.",
            ".attn.wh1.",
            ".attn.wh2.",
            ".ffn.fc1.",
            ".ffn.fc2.",
            ".ffn.fc3.",
        )
        if any(module in path for module in linear_modules):
            if path.endswith(".weight"):
                return ModelArrayClassification("projection", True)
            if path.endswith(".bias"):
                return ModelArrayClassification("bias", False)

        cema_names = ("alpha", "delta", "theta", "gamma_real", "gamma_imag", "omega")
        if ".attn.cema." in path and path.rsplit(".", 1)[-1] in cema_names:
            return ModelArrayClassification("cema", False)

        norm_paths = (
            ".attn.timenorm.weight",
            ".attn.timenorm.bias",
            ".attn.rmsnorm.gamma",
            ".ffn.norm.weight",
            ".ffn.norm.bias",
            "model.norm.weight",
            "model.norm.bias",
        )
        if any(path.endswith(suffix) for suffix in norm_paths):
            return ModelArrayClassification("norm", False)
        if path.endswith(".attn.gamma") or path.endswith(".attn.beta"):
            return ModelArrayClassification("attention_affine", False)
        if path.endswith(".ffn.alpha"):
            return ModelArrayClassification("ffn_residual_scale", False)

    return ModelArrayClassification("other", False)


def parameter_decay_mask(cfg: Config, params: Any) -> Any:
    """Return the explicit model-aware AdamW decay mask.

    :param Config cfg: Model/optimizer configuration.
    :param Any params: Trainable parameter pytree.
    :return Any: Boolean pytree; true only for embeddings and projection weights.
    """
    flat, treedef = jax.tree_util.tree_flatten_with_path(params)
    mask = [
        classify_model_array(cfg, path_to_str(path)).decay if eqx.is_array(leaf) else False
        for path, leaf in flat
    ]
    return treedef.unflatten(mask)


def parameter_optimizer_groups(cfg: Config, params: Any) -> Any:
    """Return the optimizer group for every trainable parameter leaf.

    :param Config cfg: Model and optimizer configuration.
    :param Any params: Trainable parameter pytree.
    :return Any: Pytree containing ``muon`` or ``adam`` labels.
    """

    def label(path: tuple[Any, ...], leaf: Any) -> str:
        """Classify one optimizer leaf from its model family.

        :param tuple[Any, ...] path: JAX pytree path.
        :param Any leaf: Parameter leaf.
        :return str: Optimizer group label.
        """
        if not eqx.is_array(leaf):
            return "adam"
        classification = classify_model_array(cfg, path_to_str(path))
        return _optimizer_group(cfg, leaf, classification)

    return jax.tree_util.tree_map_with_path(label, params)


def build_model(cfg: Config, *, key: jax.Array) -> tuple[Any, Any]:
    """Build model and return (params, static).

    Why?
    - We never want to stash full Modules in TrainState
    - It keeps checkpointing straightforward

    :param Config cfg: Model configuration.
    :param jax.Array key: PRNG key for model initialization.
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
        from megalodon_jax.config import MegalodonConfig
        from megalodon_jax.model import MegalodonForCausalLM

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
            attention_window=cfg.model.attention_window,
            norm_num_groups=cfg.model.norm_num_groups,
            norm_eps=cfg.model.norm_eps,
            rope_base=cfg.model.rope_base,
            swiglu=cfg.model.swiglu,
            rescale_nffn=cfg.model.rescale_nffn,
            scale_emb=cfg.model.scale_emb,
            share_emb=cfg.model.share_emb,
            norm_affine=cfg.model.norm_affine,
            dropout=cfg.model.dropout,
            attention_dropout=cfg.model.attention_dropout,
            attention_dropout_mode=cfg.model.attention_dropout_mode,
            hidden_dropout=cfg.model.hidden_dropout,
            pad_token_id=cfg.model.pad_token_id,
            bos_token_id=cfg.model.bos_token_id,
            eos_token_id=cfg.model.eos_token_id,
            init_mode=cfg.model.init_mode,
            use_checkpoint=cfg.model.use_checkpoint,
            output_size=cfg.model.output_size,
            use_associative_segment_scan=cfg.model.use_associative_segment_scan,
            param_dtype=dtype_from_str(cfg.model.param_dtype),
            compute_dtype=dtype_from_str(cfg.model.compute_dtype),
            accum_dtype=dtype_from_str(cfg.model.accum_dtype),
            attention_softmax_dtype=dtype_from_str(cfg.model.attention_softmax_dtype),
            loss_softmax_dtype=dtype_from_str(cfg.model.loss_softmax_dtype),
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
    use_packed_segments: bool = False,
    loss_chunk_size: int | None = None,
) -> jax.Array:
    """Compute training loss.

    Guardrail: this function **does not accept** cache arguments.
    Training should never enable cache; it is an inference concern.

    For Megalodon, `compute_loss` internally enforces `cache=None`.

    :param Any params: Model parameters from eqx.partition.
    :param Any static: Static model components from eqx.partition.
    :param Batch batch: Batch with input IDs, labels, and segment IDs.
    :param bool deterministic: If False, apply dropout.
    :param key: PRNG key required when deterministic=False.
    :param bool use_packed_segments: Whether to pass segment IDs to backend loss.
    :param int | None loss_chunk_size: Maximum token states projected per loss-head chunk.
    :return jax.Array: Scalar loss value.
    """

    model = eqx.combine(params, static)

    # Batch tensors come in as [A, B, T]. We compute loss per microbatch and average.
    # The compiled train_step calls this on each microbatch slice (shape [B, T]).
    kwargs: dict[str, Any] = {
        "attention_mask": batch.segment_ids > 0,
        "deterministic": deterministic,
        "key": key,
    }
    if use_packed_segments:
        kwargs["segment_ids"] = batch.segment_ids
    if loss_chunk_size is not None:
        kwargs["loss_chunk_size"] = loss_chunk_size
    return model.compute_loss(  # type: ignore[attr-defined]
        batch.input_ids,
        batch.labels,
        **kwargs,
    )


def generate_tokens(
    params: Any,
    static: Any,
    *,
    prompt_tokens: list[int],
    max_new_tokens: int,
    bos_token_id: int,
    eos_token_id: int,
    temperature: float | None,
    top_k: int | None,
    top_p: float | None,
    key: jax.Array | None,
) -> tuple[list[int], jax.Array | None]:
    """Generate continuation token IDs with the Megalodon backend.

    :param Any params: Model parameters from ``eqx.partition``.
    :param Any static: Static model components from ``eqx.partition``.
    :param list[int] prompt_tokens: Non-empty tokenized prompt.
    :param int max_new_tokens: Maximum continuation length.
    :param int bos_token_id: Beginning-of-sequence token ID.
    :param int eos_token_id: End-of-sequence token ID.
    :param float | None temperature: Sampling temperature; 0 selects greedy decoding.
    :param int | None top_k: Optional top-k cutoff.
    :param float | None top_p: Optional nucleus-sampling threshold.
    :param jax.Array | None key: Sampling key; ignored for greedy decoding.
    :return tuple[list[int], jax.Array | None]: Continuation tokens and next sampling key.
    """
    from megalodon_jax import generate as mega_generate

    prompt_ids = jnp.asarray(prompt_tokens, dtype=jnp.int32)[None, :]
    generation_kwargs: dict[str, Any] = {
        "bos_token_id": int(bos_token_id),
        "eos_token_id": int(eos_token_id),
    }
    if temperature is not None:
        generation_kwargs["temperature"] = float(temperature)
    if top_k is not None:
        generation_kwargs["top_k"] = int(top_k)
    if top_p is not None:
        generation_kwargs["top_p"] = float(top_p)

    sampling_key = key if temperature is None or temperature > 0 else None
    output_ids, _cache, next_key = mega_generate(
        eqx.combine(params, static),
        prompt_ids,
        int(max_new_tokens),
        key=sampling_key,
        **generation_kwargs,
    )
    output_tokens = [int(token) for token in jax.device_get(output_ids)[0].tolist()]
    return output_tokens[len(prompt_tokens) :], next_key


def supports_packed_segments(params: Any, static: Any) -> bool:
    """Return True if the model backend supports full packed-segment isolation.

    Checks the ``supports_segment_reset`` capability flag. Signature
    introspection of ``compute_loss`` is insufficient because older versions
    accepted packed metadata without isolating every recurrent state path.

    :param Any params: Model parameters.
    :param Any static: Static model components.
    :return bool: True when the backend resets all recurrent state
        (attention, CEMA, TimestepNorm) at segment boundaries.
    """
    model = eqx.combine(params, static)
    return bool(getattr(model, "supports_segment_reset", False))
