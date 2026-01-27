# SPDX-License-Identifier: Apache-2.0

"""Configuration for chomp.

Rule #1: **One config system.**
If a knob doesn't live in these dataclasses, it doesn't exist.

We use:
- YAML files for readability
- dot-path overrides for quick experiment changes

The loader is intentionally strict: mis-typed keys or invalid values should fail
fast with error messages that tell you exactly what to fix.

Design stance (hard-earned):
- Training must run on *real data* (HF streaming). Debug sources can exist,
  but the default path should never silently fall back to synthetic batches.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml

if TYPE_CHECKING:
    import jax.numpy as jnp

Backend = Literal["dummy", "megalodon"]
DatasetBackend = Literal["hf", "local_text"]
TokenizerKind = Literal["byte", "hf"]
PackingMode = Literal["sequential", "bin"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


@dataclass(frozen=True)
class ModelConfig:
    """Model configuration.

    Minimal set for Phases 0–3:
    - DummyLM backend for fast smoke tests
    - Megalodon backend via `megalodon_jax`

    The Megalodon fields mirror `megalodon_jax.config.MegalodonConfig` where it
    matters. We keep them here so experiments are fully reproducible from YAML.
    """

    backend: Backend = "megalodon"

    # Shared
    vocab_size: int = 256

    # DummyLM fields
    d_model: int = 128
    dropout: float = 0.0

    # Megalodon fields (subset; add as needed)
    model_dim: int = 128
    num_layers: int = 2
    num_heads: int = 1
    z_dim: int = 64
    value_dim: int = 128
    ffn_hidden_dim: int = 256
    cema_ndim: int = 16
    chunk_size: int = 128
    max_cache_len: int | None = None
    cache_unbounded: bool = False
    norm_num_groups: int = 32
    norm_eps: float = 1e-5
    rope_base: float | None = None
    swiglu: bool = False
    rescale_nffn: bool = False
    scale_emb: bool = False
    norm_affine: bool = True
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    max_positions: int = 1_000_000
    init_mode: Literal["gaussian", "xavier", "he", "bert", "none"] = "he"
    use_checkpoint: bool = False
    output_size: int = -1

    # Dtype policy (strings so YAML is clean; converted at runtime)
    param_dtype: Literal["float32", "bfloat16"] = "float32"
    compute_dtype: Literal["float32", "bfloat16"] = "bfloat16"
    accum_dtype: Literal["float32", "bfloat16"] = "float32"
    softmax_dtype: Literal["float32", "bfloat16"] = "float32"

    # Megalodon-jax currently only supports "default"
    gemm_backend: Literal["default"] = "default"


@dataclass(frozen=True)
class TokenizerConfig:
    """Tokenizer configuration.

    We support:
    - kind='byte': cheap, no external files, good for infrastructure bring-up.
    - kind='hf': Hugging Face tokenizer via transformers.AutoTokenizer.

    NOTE: In real pretraining you almost certainly want kind='hf'.
    Byte mode exists so you can bootstrap data/packing without downloading
    a tokenizer first.

    We also support vocab alignment via `vocab_size_multiple` to round model
    embeddings up to a GPU-friendly multiple (default: 128).
    """

    kind: TokenizerKind = "byte"

    # HF tokenizer
    hf_name_or_path: str | None = "pszemraj/bytebpe-tokenizer-32k-mlm"
    hf_use_fast: bool = True
    hf_trust_remote_code: bool = False

    # Round vocab size up to a multiple for aligned embeddings (e.g. 128).
    vocab_size_multiple: int = 128

    # If True, use tokenizer-provided special token IDs to update model config.
    auto_set_special_tokens: bool = True

    # Byte tokenizer
    # If byte_offset>0, we reserve IDs [0..byte_offset-1] for specials and map
    # raw bytes 0..255 to [byte_offset..byte_offset+255].
    byte_offset: int = 0

    # Special token insertion
    add_bos: bool = False
    add_eos: bool = False

    # Optional truncation (in tokens) per document before packing
    max_doc_tokens: int | None = None


@dataclass(frozen=True)
class DataConfig:
    """Data configuration.

    Default dataset is Zyphra/Zyda-2 sample-100BT streaming split, as requested.

    The data pipeline's job is to yield fixed-shape `Batch` objects:
      input_ids:      [A, B, T]
      labels:         [A, B, T]
      attention_mask: [A, B, T]

    where A=grad_accum, B=batch_size, T=seq_len.

    Validation behavior:
    - If an HF validation split exists, we take its first max_eval_samples examples.
    - Otherwise we take the first max_eval_samples examples from the (shuffled) train split.
    """

    backend: DatasetBackend = "hf"

    # HF streaming dataset spec
    # TODO: v0 is single-source only; add multi-source mixing when needed.
    hf_dataset: str = "Zyphra/Zyda-2"
    hf_name: str = "sample-100BT"
    hf_split: str = "train"
    # Preferred eval split; fallback to train if missing.
    hf_eval_split: str = "validation"
    text_key: str = "text"

    shuffle: bool = True
    shuffle_buffer_size: int = 10_000
    seed: int = 0
    repeat: bool = True

    # Network resilience (best-effort; still expect rare failures)
    max_retries: int = 3
    retry_delay_sec: float = 1.0

    # For retry: cache a last-known-good HF state dict every N examples.
    # Smaller => more robust, but more overhead.
    state_update_interval: int = 2_000

    # Debug-only local text source (exercises tokenize+pack path, not synthetic ids)
    local_text: str = "Hello from chomp.\n"

    # Packing mode: sequential stream packer or bin-packing (FFD).
    packing_mode: PackingMode = "sequential"
    # Bin-packing buffer size (documents). Must be >= batch_size * grad_accum.
    packing_buffer_docs: int = 128
    # Optional cap on how many documents may be packed into a single bin.
    packing_max_docs_per_bin: int | None = None

    # Packed-doc loss behavior
    mask_boundary_loss: bool = True
    train_on_eos: bool = True

    # Grain pipeline settings.
    grain_prefetch: int = 0

    # Validation set creation (first N examples from validation or train fallback)
    max_eval_samples: int = 1000

    # Tokenizer
    tokenizer: TokenizerConfig = TokenizerConfig()

    # Simple performance toggle: if True, device_put batches in the iterator.
    # For now, leave False and device_put in the training loop.
    device_put: bool = False


@dataclass(frozen=True)
class TrainConfig:
    """Training loop configuration including batch sizes, steps, and profiling."""

    seed: int = 0
    steps: int = 100
    batch_size: int = 2
    seq_len: int = 128
    grad_accum: int = 1

    jit: bool = True
    deterministic: bool | None = None  # None => derive from dropout

    allow_cpu: bool = False
    log_every: int = 25
    eval_every: int = 2500
    generate_every: int = 5000
    generate_input_len: int | None = None
    generate_max_tokens: int | None = None
    generate_temperature: float | None = None
    generate_top_k: int | None = None
    generate_top_p: float | None = None

    # Simple profiler support (Phase 0): if enabled, write a trace directory.
    profile: bool = False
    profile_dir: str | None = None


@dataclass(frozen=True)
class OptimConfig:
    """Optimizer configuration for AdamW or Muon with warmup+cosine decay."""

    name: Literal["adamw", "muon"] = "adamw"
    lr: float = 3e-4
    muon_lr_scale: float = 100.0
    weight_decay: float = 0.01
    muon_weight_decay_mult: float = 1.0
    grad_clip_norm: float = 1.0
    warmup_steps: int = 10
    decay_steps: int | None = None
    min_lr_ratio: float = 0.0
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    muon_nesterov: bool = True
    muon_consistent_rms: float | None = None
    muon_allow_all_2d: bool = False
    muon_allow_tied_embed: bool = False
    adam_b1: float = 0.9
    adam_b2: float = 0.999
    adam_eps: float = 1e-8
    adam_nesterov: bool = False


@dataclass(frozen=True)
class CheckpointConfig:
    """Orbax checkpointing configuration (Phase 3).

    chomp treats resume correctness as a contract.
    """

    enabled: bool = True
    # If None, checkpoints live under <run_dir>/checkpoints
    root_dir: str | None = None

    save_every: int = 5000
    max_to_keep: int = 3
    async_save: bool = True


@dataclass(frozen=True)
class WandbConfig:
    """Weights & Biases configuration."""

    enabled: bool = False
    project: str | None = None
    entity: str | None = None
    run_name: str | None = None
    mode: Literal["online", "offline", "disabled"] = "online"
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class LoggingConfig:
    """Logging configuration for run directory and metrics output."""

    project: str = "chomp"
    run_dir: str | None = None
    metrics_file: str = "metrics.jsonl"
    level: LogLevel = "INFO"
    console_use_rich: bool = True
    log_file: str | None = "train.log"
    wandb: WandbConfig = WandbConfig()


@dataclass(frozen=True)
class DebugConfig:
    """Debug configuration for NaN checks and device assertions."""

    nan_check: bool = True
    check_device_every: int = 100


@dataclass(frozen=True)
class Config:
    """Top-level configuration combining all sub-configs for a training run."""

    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    train: TrainConfig = TrainConfig()
    optim: OptimConfig = OptimConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    logging: LoggingConfig = LoggingConfig()
    debug: DebugConfig = DebugConfig()

    def to_dict(self) -> dict[str, Any]:
        """Convert the entire config tree to a nested dictionary.

        :return dict[str, Any]: Nested dict representation of all config fields.
        """
        return asdict(self)


# ------------------------------ Loading ---------------------------------


def _set_by_dotted_path(obj: Any, path: str, raw_value: str) -> Any:
    """Set a dataclass field by dotted path, returning a new object.

    Example: path="train.batch_size", raw_value="4"

    We do simple type casting based on the current value type.

    :param Any obj: Root dataclass to modify.
    :param str path: Dot-separated path to the field (e.g., "train.batch_size").
    :param str raw_value: String value to set, will be cast to the field's type.
    :raises ValueError: If the path is invalid or contains unknown keys.
    :return Any: New dataclass with the field updated.
    """

    parts = path.split(".")
    if len(parts) < 1:
        raise ValueError(f"Invalid override path: {path!r}")

    # Walk to the parent
    cur = obj
    parents: list[tuple[Any, str]] = []
    for p in parts[:-1]:
        if not hasattr(cur, p):
            raise ValueError(f"Unknown config key: {path!r} (missing {p!r})")
        parents.append((cur, p))
        cur = getattr(cur, p)

    leaf = parts[-1]
    if not hasattr(cur, leaf):
        raise ValueError(f"Unknown config key: {path!r} (missing {leaf!r})")

    old = getattr(cur, leaf)
    new = _cast_like(old, raw_value)

    # Rebuild dataclasses from the bottom up (frozen dataclasses)
    cur_new = _replace_dataclass(cur, **{leaf: new})
    for parent, field in reversed(parents):
        cur_new = _replace_dataclass(parent, **{field: cur_new})
    return cur_new


def _replace_dataclass(obj: Any, **kwargs: Any) -> Any:
    """Create a new dataclass instance with specified fields replaced.

    :param Any obj: Frozen dataclass to copy.
    :param kwargs: Field names and their new values.
    :return Any: New dataclass instance with updated fields.
    """
    from dataclasses import replace

    return replace(obj, **kwargs)


def _cast_like(old: Any, raw: str) -> Any:
    """Cast a string override to the type of `old`.

    This is intentionally conservative. If we can't cast cleanly, error.

    :param Any old: Reference value whose type determines the cast.
    :param str raw: String value to cast.
    :raises ValueError: If cast fails (e.g., invalid boolean string).
    :return Any: Value cast to the type of `old`.
    """

    if isinstance(old, bool):
        if raw.lower() in {"true", "1", "yes", "y"}:
            return True
        if raw.lower() in {"false", "0", "no", "n"}:
            return False
        raise ValueError(f"Expected boolean, got {raw!r}")
    if isinstance(old, int):
        return int(raw)
    if isinstance(old, float):
        return float(raw)
    if old is None:
        if raw.lower() in {"null", "none"}:
            return None
        # When the default is None, parse YAML scalars to recover numeric/bool types.
        try:
            parsed = yaml.safe_load(raw)
        except yaml.YAMLError:
            return raw
        return raw if parsed is None else parsed
    if isinstance(old, str):
        return raw
    # For Literal or other types, keep string; validation should catch invalid
    return raw


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> Config:
    """Load YAML config file + apply dot-path overrides.

    Overrides format: "train.steps=2000".

    :param path: Path to the YAML config file.
    :param overrides: Optional list of dot-path overrides (e.g., ["train.steps=2000"]).
    :raises ValueError: If override format is invalid or path does not exist.
    :return Config: Validated configuration object.
    """

    path = Path(path)
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}

    data = _resolve_variables(data)

    cfg = _from_nested_dict(data)

    if overrides:
        for o in overrides:
            if "=" not in o:
                raise ValueError(f"Invalid override {o!r}. Expected format like train.steps=123")
            k, v = o.split("=", 1)
            cfg = _set_by_dotted_path(cfg, k.strip(), v.strip())

    validate_config(cfg)
    return cfg


_VAR_INLINE_RE = re.compile(r"\{\$variables\.([A-Za-z0-9_.-]+)\}")
_VAR_BRACE_RE = re.compile(r"\$\{variables\.([A-Za-z0-9_.-]+)\}")
_VAR_FULL_RE = re.compile(r"\$variables\.([A-Za-z0-9_.-]+)$")
_VAR_SUSPICIOUS_RE = re.compile(r"\$variables\.[A-Za-z0-9_.-]+")


def _resolve_variables(data: dict[str, Any]) -> dict[str, Any]:
    """Resolve variable references in a config dict before dataclass parsing.

    Supported forms:
    - Exact value: "$variables.foo" -> replaced with the referenced value (type preserved).
    - Inline string: "seq{$variables.seq_len}" or "${variables.seq_len}" -> interpolated.

    Variable definitions live under top-level key "variables" and may be nested.

    :param dict[str, Any] data: Raw YAML-loaded data.
    :raises ValueError: If a variable reference is missing or circular.
    :return dict[str, Any]: Data with variables resolved (variables removed).
    """
    raw_vars = data.get("variables") or {}
    if not isinstance(raw_vars, dict):
        raise ValueError("variables must be a mapping if provided")

    resolved: dict[str, Any] = {}
    resolving: set[str] = set()

    def _lookup_var(path: str) -> Any:
        """Resolve a variable path from the variables mapping.

        :param str path: Dot-separated variable path (e.g., "foo.bar").
        :raises ValueError: If the reference is missing or circular.
        :return Any: Resolved variable value.
        """
        if path in resolved:
            return resolved[path]
        if path in resolving:
            cycle = " -> ".join(list(resolving) + [path])
            raise ValueError(f"Circular variable reference: {cycle}")

        parts = path.split(".")
        cur: Any = raw_vars
        for part in parts:
            if not isinstance(cur, dict) or part not in cur:
                raise ValueError(f"Unknown variable reference: variables.{path}")
            cur = cur[part]

        resolving.add(path)
        value = _resolve_value(cur)
        resolving.remove(path)
        resolved[path] = value
        return value

    def _sub_var(match: re.Match[str]) -> str:
        """Replace a regex match with the referenced variable value.

        :param re.Match[str] match: Regex match containing the variable path.
        :return str: Stringified variable value.
        """
        return str(_lookup_var(match.group(1)))

    def _resolve_value(value: Any) -> Any:
        """Resolve variables within a nested config value.

        :param Any value: Arbitrary nested value (dict/list/str/etc.).
        :return Any: Resolved value with variables expanded.
        """
        if isinstance(value, dict):
            return {k: _resolve_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_resolve_value(v) for v in value]
        if isinstance(value, str):
            full = _VAR_FULL_RE.fullmatch(value)
            if full:
                return _lookup_var(full.group(1))
            out = _VAR_INLINE_RE.sub(_sub_var, value)
            out = _VAR_BRACE_RE.sub(_sub_var, out)
            remaining = _VAR_SUSPICIOUS_RE.findall(out)
            if remaining:
                warnings.warn(
                    f"String contains unresolved variable-like patterns: {remaining}. "
                    "Use {$variables.name} or ${variables.name} for inline substitution.",
                    stacklevel=2,
                )
            return out
        return value

    resolved_data = {k: _resolve_value(v) for k, v in data.items() if k != "variables"}
    return resolved_data


def _from_nested_dict(data: dict[str, Any]) -> Config:
    """Convert nested dict into Config dataclasses.

    :param dict[str, Any] data: Nested dictionary from YAML parsing.
    :return Config: Fully constructed Config with all sub-configs.
    """

    model = ModelConfig(**(data.get("model") or {}))
    train = TrainConfig(**(data.get("train") or {}))
    optim = OptimConfig(**(data.get("optim") or {}))
    logging_d = data.get("logging") or {}
    wandb_d = logging_d.get("wandb") or {}
    wandb = WandbConfig(**wandb_d)
    logging_d = {k: v for k, v in logging_d.items() if k != "wandb"}
    logging = LoggingConfig(wandb=wandb, **logging_d)
    debug = DebugConfig(**(data.get("debug") or {}))
    checkpoint = CheckpointConfig(**(data.get("checkpoint") or {}))

    # Data + nested tokenizer
    data_d = data.get("data") or {}
    tok_d = data_d.get("tokenizer") or {}
    tok = TokenizerConfig(**tok_d)
    # remove tokenizer from dict before constructing
    data_d = {k: v for k, v in data_d.items() if k != "tokenizer"}
    data_cfg = DataConfig(tokenizer=tok, **data_d)

    return Config(
        model=model,
        data=data_cfg,
        train=train,
        optim=optim,
        checkpoint=checkpoint,
        logging=logging,
        debug=debug,
    )


# ------------------------------ Validation ---------------------------------


def _vfail(msg: str) -> None:
    """Raise ValueError with a standardized config validation prefix.

    :param str msg: Validation failure message.
    :raises ValueError: Always raised with formatted message.
    """
    raise ValueError(f"Config validation failed: {msg}")


def _validate_train(cfg: Config) -> None:
    """Validate training-related config fields."""
    if cfg.train.steps <= 0:
        _vfail(f"train.steps must be positive, got {cfg.train.steps}")
    if cfg.train.batch_size <= 0:
        _vfail(f"train.batch_size must be positive, got {cfg.train.batch_size}")
    if cfg.train.seq_len < 8:
        _vfail(f"train.seq_len must be >= 8, got {cfg.train.seq_len}")
    if cfg.train.grad_accum <= 0:
        _vfail(f"train.grad_accum must be positive, got {cfg.train.grad_accum}")
    if cfg.train.log_every <= 0:
        _vfail(f"train.log_every must be positive, got {cfg.train.log_every}")
    if cfg.train.eval_every < 0:
        _vfail(f"train.eval_every must be >= 0, got {cfg.train.eval_every}")
    if cfg.train.generate_every < 0:
        _vfail(f"train.generate_every must be >= 0, got {cfg.train.generate_every}")
    if cfg.train.generate_input_len is not None:
        if cfg.train.generate_input_len <= 0:
            _vfail(
                "train.generate_input_len must be positive when set, "
                f"got {cfg.train.generate_input_len}"
            )
        if cfg.train.generate_input_len > cfg.train.seq_len:
            _vfail(
                "train.generate_input_len must be <= train.seq_len "
                f"({cfg.train.seq_len}), got {cfg.train.generate_input_len}"
            )
    if cfg.train.generate_max_tokens is not None and cfg.train.generate_max_tokens <= 0:
        _vfail(
            "train.generate_max_tokens must be positive when set, "
            f"got {cfg.train.generate_max_tokens}"
        )
    if cfg.train.generate_temperature is not None and cfg.train.generate_temperature < 0:
        _vfail(
            "train.generate_temperature must be >= 0 when set, "
            f"got {cfg.train.generate_temperature}"
        )
    if cfg.train.generate_top_k is not None and cfg.train.generate_top_k <= 0:
        _vfail(f"train.generate_top_k must be positive when set, got {cfg.train.generate_top_k}")
    if cfg.train.generate_top_p is not None and (
        cfg.train.generate_top_p <= 0 or cfg.train.generate_top_p > 1
    ):
        _vfail(f"train.generate_top_p must be in (0, 1] when set, got {cfg.train.generate_top_p}")


def _validate_optim(cfg: Config) -> None:
    """Validate optimizer-related config fields."""
    if cfg.optim.name not in ("adamw", "muon"):
        _vfail(f"optim.name must be 'adamw' or 'muon', got {cfg.optim.name!r}")
    if cfg.optim.lr <= 0:
        _vfail(f"optim.lr must be positive, got {cfg.optim.lr}")
    if cfg.optim.grad_clip_norm < 0:
        _vfail(f"optim.grad_clip_norm must be >= 0, got {cfg.optim.grad_clip_norm}")
    if cfg.optim.warmup_steps < 0:
        _vfail(f"optim.warmup_steps must be >= 0, got {cfg.optim.warmup_steps}")
    if cfg.optim.decay_steps is not None and cfg.optim.decay_steps <= 0:
        _vfail(f"optim.decay_steps must be positive when set, got {cfg.optim.decay_steps}")
    if cfg.optim.min_lr_ratio < 0 or cfg.optim.min_lr_ratio > 1:
        _vfail(f"optim.min_lr_ratio must be in [0, 1], got {cfg.optim.min_lr_ratio}")
    if cfg.optim.muon_lr_scale <= 0:
        _vfail(f"optim.muon_lr_scale must be positive, got {cfg.optim.muon_lr_scale}")
    if cfg.optim.muon_weight_decay_mult < 0:
        _vfail(f"optim.muon_weight_decay_mult must be >= 0, got {cfg.optim.muon_weight_decay_mult}")
    if cfg.optim.muon_momentum <= 0 or cfg.optim.muon_momentum >= 1:
        _vfail(f"optim.muon_momentum must be in (0, 1), got {cfg.optim.muon_momentum}")
    if cfg.optim.muon_ns_steps <= 0:
        _vfail(f"optim.muon_ns_steps must be positive, got {cfg.optim.muon_ns_steps}")
    if cfg.optim.muon_consistent_rms is not None and cfg.optim.muon_consistent_rms < 0:
        _vfail(
            f"optim.muon_consistent_rms must be >= 0 or None, got {cfg.optim.muon_consistent_rms}"
        )
    if cfg.optim.adam_b1 <= 0 or cfg.optim.adam_b1 >= 1:
        _vfail(f"optim.adam_b1 must be in (0, 1), got {cfg.optim.adam_b1}")
    if cfg.optim.adam_b2 <= 0 or cfg.optim.adam_b2 >= 1:
        _vfail(f"optim.adam_b2 must be in (0, 1), got {cfg.optim.adam_b2}")
    if cfg.optim.adam_eps <= 0:
        _vfail(f"optim.adam_eps must be positive, got {cfg.optim.adam_eps}")
    if cfg.optim.warmup_steps >= cfg.train.steps:
        _vfail(
            f"optim.warmup_steps ({cfg.optim.warmup_steps}) must be < train.steps "
            f"({cfg.train.steps})"
        )


def _validate_checkpoint(cfg: Config) -> None:
    """Validate checkpoint-related config fields."""
    if cfg.checkpoint.enabled:
        if cfg.checkpoint.save_every <= 0:
            _vfail(f"checkpoint.save_every must be positive, got {cfg.checkpoint.save_every}")
        if cfg.checkpoint.max_to_keep <= 0:
            _vfail(f"checkpoint.max_to_keep must be positive, got {cfg.checkpoint.max_to_keep}")


def _validate_model(cfg: Config) -> None:
    """Validate model-related config fields."""
    if cfg.model.vocab_size <= 0:
        _vfail(f"model.vocab_size must be positive, got {cfg.model.vocab_size}")
    if cfg.model.backend == "dummy":
        if cfg.model.d_model <= 0:
            _vfail(f"model.d_model must be positive, got {cfg.model.d_model}")
    elif cfg.model.backend == "megalodon":
        if cfg.model.model_dim <= 0:
            _vfail(f"model.model_dim must be positive, got {cfg.model.model_dim}")
        if cfg.model.num_layers <= 0:
            _vfail(f"model.num_layers must be positive, got {cfg.model.num_layers}")
        if cfg.model.num_heads <= 0:
            _vfail(f"model.num_heads must be positive, got {cfg.model.num_heads}")
        if cfg.model.model_dim % cfg.model.num_heads != 0:
            _vfail(
                f"model.model_dim ({cfg.model.model_dim}) must be divisible by "
                f"model.num_heads ({cfg.model.num_heads})"
            )
        if cfg.model.chunk_size <= 0:
            _vfail(f"model.chunk_size must be positive, got {cfg.model.chunk_size}")
        if cfg.model.chunk_size > cfg.train.seq_len:
            _vfail(
                f"model.chunk_size ({cfg.model.chunk_size}) must be <= train.seq_len ({cfg.train.seq_len})"
            )
        if cfg.train.seq_len % cfg.model.chunk_size != 0:
            _vfail(
                f"train.seq_len ({cfg.train.seq_len}) must be divisible by "
                f"model.chunk_size ({cfg.model.chunk_size})"
            )
    else:
        _vfail(f"model.backend must be 'dummy' or 'megalodon', got {cfg.model.backend!r}")

    if cfg.model.pad_token_id == cfg.model.eos_token_id:
        warnings.warn(
            "model.pad_token_id equals model.eos_token_id. This is allowed, but EOS tokens may "
            "be treated as padding in some models. Prefer a distinct pad token when possible.",
            stacklevel=2,
        )


def _validate_data(cfg: Config) -> None:
    """Validate data pipeline-related config fields."""
    if cfg.data.backend == "hf":
        if not cfg.data.hf_dataset:
            _vfail("data.hf_dataset must be non-empty when data.backend='hf'")
        if not cfg.data.hf_name:
            _vfail("data.hf_name must be non-empty when data.backend='hf' (use named configs)")
        if not cfg.data.hf_split:
            _vfail("data.hf_split must be non-empty when data.backend='hf'")
        if not cfg.data.hf_eval_split:
            _vfail("data.hf_eval_split must be non-empty when data.backend='hf'")
        if not cfg.data.text_key:
            _vfail("data.text_key must be non-empty")
        if cfg.data.shuffle and cfg.data.shuffle_buffer_size <= 0:
            _vfail(
                "data.shuffle_buffer_size must be positive when data.shuffle=true, "
                f"got {cfg.data.shuffle_buffer_size}"
            )
    elif cfg.data.backend == "local_text":
        if not cfg.data.local_text:
            _vfail("data.local_text must be non-empty when data.backend='local_text'")
    else:
        _vfail(f"data.backend must be 'hf' or 'local_text', got {cfg.data.backend!r}")

    if cfg.data.packing_mode not in ("sequential", "bin"):
        _vfail(f"data.packing_mode must be 'sequential' or 'bin', got {cfg.data.packing_mode!r}")
    if cfg.data.packing_mode == "bin":
        if cfg.data.packing_buffer_docs <= 0:
            _vfail(
                "data.packing_buffer_docs must be positive when packing_mode='bin', "
                f"got {cfg.data.packing_buffer_docs}"
            )
        min_docs = cfg.train.batch_size * cfg.train.grad_accum
        if cfg.data.packing_buffer_docs < min_docs:
            _vfail(
                "data.packing_buffer_docs must be >= train.batch_size * train.grad_accum "
                f"({min_docs}), got {cfg.data.packing_buffer_docs}"
            )
        if cfg.data.packing_max_docs_per_bin is not None and cfg.data.packing_max_docs_per_bin <= 0:
            _vfail(
                "data.packing_max_docs_per_bin must be positive when set, "
                f"got {cfg.data.packing_max_docs_per_bin}"
            )

    if cfg.data.grain_prefetch < 0:
        _vfail(f"data.grain_prefetch must be >=0, got {cfg.data.grain_prefetch}")
    if cfg.data.max_eval_samples < 0:
        _vfail(f"data.max_eval_samples must be >=0, got {cfg.data.max_eval_samples}")

    # HF streaming robustness knobs
    if cfg.data.max_retries < 0:
        _vfail(f"data.max_retries must be >=0, got {cfg.data.max_retries}")
    if cfg.data.retry_delay_sec < 0:
        _vfail(f"data.retry_delay_sec must be >=0, got {cfg.data.retry_delay_sec}")
    if cfg.data.state_update_interval <= 0:
        _vfail(f"data.state_update_interval must be >0, got {cfg.data.state_update_interval}")


def _validate_logging(cfg: Config) -> None:
    """Validate logging-related config fields."""
    if cfg.logging.log_file is not None and not str(cfg.logging.log_file).strip():
        _vfail("logging.log_file must be a non-empty string or null")
    if cfg.logging.wandb.mode not in ("online", "offline", "disabled"):
        _vfail(
            "logging.wandb.mode must be 'online', 'offline', or 'disabled', "
            f"got {cfg.logging.wandb.mode!r}"
        )


def _validate_tokenizer(cfg: Config) -> None:
    """Validate tokenizer-related config fields."""
    tok = cfg.data.tokenizer
    if tok.kind == "hf":
        if not tok.hf_name_or_path:
            _vfail("data.tokenizer.hf_name_or_path must be set when tokenizer.kind='hf'")
    elif tok.kind == "byte":
        if tok.byte_offset < 0:
            _vfail(f"data.tokenizer.byte_offset must be >=0, got {tok.byte_offset}")
        min_vocab = tok.byte_offset + 256
        if cfg.model.vocab_size < min_vocab:
            _vfail(
                f"model.vocab_size ({cfg.model.vocab_size}) must be >= byte_offset+256 ({min_vocab}) "
                "when using byte tokenizer"
            )
        if tok.add_bos or tok.add_eos:
            if tok.byte_offset <= 0:
                _vfail(
                    "byte tokenizer with add_bos/add_eos requires data.tokenizer.byte_offset > 0 "
                    "to reserve special-token IDs"
                )
            if tok.add_bos and not (0 <= cfg.model.bos_token_id < tok.byte_offset):
                _vfail(
                    "model.bos_token_id must be within [0, byte_offset) when using byte tokenizer "
                    "with add_bos=true"
                )
            if tok.add_eos and not (0 <= cfg.model.eos_token_id < tok.byte_offset):
                _vfail(
                    "model.eos_token_id must be within [0, byte_offset) when using byte tokenizer "
                    "with add_eos=true"
                )
    else:
        _vfail(f"data.tokenizer.kind must be 'byte' or 'hf', got {tok.kind!r}")

    if tok.vocab_size_multiple <= 0:
        _vfail(
            f"data.tokenizer.vocab_size_multiple must be positive, got {tok.vocab_size_multiple}"
        )

    if tok.max_doc_tokens is not None and tok.max_doc_tokens == 0:
        warnings.warn(
            "data.tokenizer.max_doc_tokens=0 disables truncation; the tokenizer will be "
            "resolved with no max_doc_tokens cap.",
            stacklevel=2,
        )
    if tok.max_doc_tokens is not None and tok.max_doc_tokens < 0:
        warnings.warn(
            "data.tokenizer.max_doc_tokens < 0 disables truncation; the tokenizer will be "
            "resolved with no max_doc_tokens cap.",
            stacklevel=2,
        )

    # Special token ids must be in range if enabled
    if tok.add_bos and not (0 <= cfg.model.bos_token_id < cfg.model.vocab_size):
        _vfail("model.bos_token_id must be within [0, vocab_size) when add_bos=true")
    if tok.add_eos and not (0 <= cfg.model.eos_token_id < cfg.model.vocab_size):
        _vfail("model.eos_token_id must be within [0, vocab_size) when add_eos=true")


def validate_config(cfg: Config) -> None:
    """Validate config with actionable error messages."""
    _validate_train(cfg)
    _validate_optim(cfg)
    _validate_checkpoint(cfg)
    _validate_model(cfg)
    _validate_data(cfg)
    _validate_logging(cfg)
    _validate_tokenizer(cfg)


def derived_deterministic(cfg: Config) -> bool:
    """Compute training determinism.

    If cfg.train.deterministic is set, it wins.
    Otherwise: deterministic iff all dropout rates are zero.

    :param Config cfg: Configuration to check.
    :return bool: True if training should be deterministic.
    """

    if cfg.train.deterministic is not None:
        return bool(cfg.train.deterministic)

    if cfg.model.backend == "dummy":
        return cfg.model.dropout == 0.0
    return (
        cfg.model.dropout == 0.0
        and cfg.model.attention_dropout == 0.0
        and cfg.model.hidden_dropout == 0.0
    )


def resolve_decay_duration(cfg: Config) -> int:
    """Resolve cosine decay duration (post-warmup) in steps.

    If `optim.decay_steps` is unset, we default to `train.steps - optim.warmup_steps`
    so the schedule ends at `train.steps`.

    :param Config cfg: Training configuration.
    :return int: Decay duration in steps.
    """
    if cfg.optim.decay_steps is None:
        return int(cfg.train.steps) - int(cfg.optim.warmup_steps)
    return int(cfg.optim.decay_steps)


def resolve_decay_horizon(cfg: Config) -> int:
    """Resolve the total schedule horizon (warmup + decay) in steps.

    :param Config cfg: Training configuration.
    :return int: Total schedule steps (warmup + decay duration).
    """
    return int(cfg.optim.warmup_steps) + resolve_decay_duration(cfg)


def dtype_from_str(name: str) -> jnp.dtype:
    """Map a dtype string to a JAX dtype.

    :param str name: Dtype name ("float32" or "bfloat16").
    :raises ValueError: If name is not a supported dtype.
    :return jnp.dtype: Corresponding JAX dtype.
    """
    import jax.numpy as jnp

    table = {
        "float32": jnp.float32,
        "bfloat16": jnp.bfloat16,
    }
    if name not in table:
        raise ValueError(f"Unsupported dtype {name!r}. Expected one of {sorted(table)}")
    return table[name]
