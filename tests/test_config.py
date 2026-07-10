"""Config tests consolidated by module."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from chomp.config import (
    Config,
    DataConfig,
    ModelConfig,
    OptimConfig,
    TokenizerConfig,
    TrainConfig,
    build_config,
    load_config,
    validate_config,
)
from chomp.data.pipeline import build_tokenizer, resolve_tokenizer_config
from chomp.utils.ckpt_paths import load_config_for_checkpoint


def _base_cfg() -> Config:
    """Create a base config for validation tests."""
    return Config(
        model=ModelConfig(backend="megalodon", model_dim=128, num_heads=8, chunk_size=16),
        data=DataConfig(
            backend="local_text",
            local_text="config validation text\n",
            tokenizer=TokenizerConfig(kind="byte", byte_offset=0, add_bos=False, add_eos=False),
        ),
        train=TrainConfig(steps=1, batch_size=2, seq_len=16, grad_accum=1, allow_cpu=True),
        optim=OptimConfig(warmup_steps=0),
    )


def _hf_data(*, hf_eval_split: object | None = None) -> DataConfig:
    """Create HF data config for validation tests."""
    return DataConfig(
        backend="hf",
        hf_dataset="dummy",
        hf_name="dummy",
        hf_split="train",
        hf_eval_split=hf_eval_split,  # type: ignore[arg-type]
        text_key="text",
        shuffle=False,
        repeat=True,
        tokenizer=TokenizerConfig(kind="byte", byte_offset=0, add_bos=False, add_eos=False),
    )


def _write_tiny_yaml_config(
    path: Path,
    *,
    variables: str = "",
    vocab_size: int = 256,
    train_steps: int = 10,
    train_seq_len: int | str = 32,
    tokenizer_extra: str = "",
) -> None:
    """Write a tiny local-text YAML config for config-loader tests."""
    tokenizer_extra = f"\n{tokenizer_extra.rstrip()}" if tokenizer_extra else ""
    path.write_text(
        f"""
{variables}model:
  backend: dummy
  vocab_size: {vocab_size}
  d_model: 32
  dropout: 0.0

data:
  backend: local_text
  repeat: true
  local_text: "hello"
  packing_mode: sequential
  tokenizer:
    kind: byte
    byte_offset: 0
    add_bos: false
    add_eos: false{tokenizer_extra}

train:
  steps: {train_steps}
  batch_size: 1
  seq_len: {train_seq_len}
  grad_accum: 1
  jit: false
  deterministic: true
  allow_cpu: true
  log_every: 1
  eval_every: 0

optim:
  lr: 3.0e-4
  warmup_steps: 0

checkpoint:
  enabled: true
  save_every: 5
  max_to_keep: 1
  async_save: false

logging:
  project: chomp
  run_dir: null
  metrics_file: metrics.jsonl
  level: INFO
  console_use_rich: false
  log_file: null
  wandb:
    enabled: false

debug:
  nan_check: true
  check_device_every: 1
""".lstrip()
    )


def test_model_and_train_validation_rejects_invalid_values() -> None:
    """Model/train validation should fail with actionable errors."""
    cases: list[tuple[Callable[[Config], Config], str]] = [
        (lambda cfg: replace(cfg, model=replace(cfg.model, chunk_size=32)), "chunk_size"),
        (lambda cfg: replace(cfg, model=replace(cfg.model, chunk_size=10)), "divisible"),
        # bf16 params without an fp32 master-param path silently give bf16
        # optimizer moments; rejected until that path exists (docs/dev.md).
        (
            lambda cfg: replace(cfg, model=replace(cfg.model, param_dtype="bfloat16")),
            "param_dtype",
        ),
        (
            lambda cfg: replace(cfg, model=replace(cfg.model, model_dim=130, num_heads=8)),
            "model_dim",
        ),
        (lambda cfg: replace(cfg, model=replace(cfg.model, dropout=1.1)), "model.dropout"),
        (lambda cfg: replace(cfg, model=replace(cfg.model, z_dim=65)), "model.z_dim"),
        (lambda cfg: replace(cfg, model=replace(cfg.model, value_dim=130)), "model.value_dim"),
        (
            lambda cfg: replace(cfg, model=replace(cfg.model, norm_num_groups=30)),
            "norm_num_groups",
        ),
        (lambda cfg: replace(cfg, model=replace(cfg.model, norm_eps=0.0)), "norm_eps"),
        (lambda cfg: replace(cfg, model=replace(cfg.model, max_cache_len=8)), "max_cache_len"),
        (lambda cfg: replace(cfg, model=replace(cfg.model, rope_base=0.0)), "rope_base"),
        (
            lambda cfg: replace(cfg, model=replace(cfg.model, attention_dropout=-0.1)),
            "attention_dropout",
        ),
        (
            lambda cfg: replace(cfg, model=replace(cfg.model, hidden_dropout=1.1)),
            "hidden_dropout",
        ),
        (lambda cfg: replace(cfg, model=replace(cfg.model, init_mode="invalid")), "init_mode"),
        (
            lambda cfg: replace(cfg, model=replace(cfg.model, compute_dtype="float16")),
            "compute_dtype",
        ),
        (
            lambda cfg: replace(
                cfg,
                model=replace(
                    cfg.model,
                    compute_dtype="float32",
                    accum_dtype="bfloat16",
                ),
            ),
            "accum_dtype",
        ),
        (
            lambda cfg: replace(cfg, model=replace(cfg.model, softmax_dtype="float16")),
            "softmax_dtype",
        ),
        (
            lambda cfg: replace(cfg, model=replace(cfg.model, gemm_backend="triton")),
            "gemm_backend",
        ),
        (lambda cfg: replace(cfg, model=replace(cfg.model, output_size=0)), "output_size"),
        (lambda cfg: replace(cfg, train=replace(cfg.train, eval_every=-1)), "eval_every"),
        (lambda cfg: replace(cfg, train=replace(cfg.train, generate_every=-1)), "generate_every"),
        (
            lambda cfg: replace(cfg, train=replace(cfg.train, generate_input_len=0)),
            "generate_input_len",
        ),
        (
            lambda cfg: replace(cfg, train=replace(cfg.train, generate_input_len=1024)),
            "generate_input_len",
        ),
        (
            lambda cfg: replace(cfg, train=replace(cfg.train, generate_top_p=1.5)),
            "generate_top_p",
        ),
        (
            lambda cfg: replace(
                cfg,
                optim=replace(cfg.optim, warmup_steps=10),
                train=replace(cfg.train, steps=5),
            ),
            "warmup_steps",
        ),
    ]

    for mutate, match in cases:
        with pytest.raises(ValueError, match=match):
            validate_config(mutate(_base_cfg()))


def test_build_config_rejects_unknown_top_level_sections() -> None:
    """Top-level config typos should fail instead of being silently ignored."""
    data = _base_cfg().to_dict()
    data["trian"] = {"steps": 10}

    with pytest.raises(ValueError, match="unknown top-level"):
        build_config(data)


def test_build_config_allows_resolved_derived_section() -> None:
    """config_resolved.json includes derived metadata that is not schema input."""
    data = _base_cfg().to_dict()
    data["derived"] = {"optim": {"decay_steps_effective": 1}}

    cfg = build_config(data)

    assert cfg == _base_cfg()


def test_override_rejects_empty_dotted_path_component() -> None:
    """Malformed dot-path overrides should fail before attribute lookup."""
    with pytest.raises(ValueError, match="Invalid override path"):
        build_config(_base_cfg().to_dict(), overrides=["train..steps=2"])


def test_optim_validation_rejects_invalid_values() -> None:
    """Optimizer validation should fail for out-of-range settings."""
    cases: list[tuple[Callable[[Config], Config], str]] = [
        (lambda cfg: replace(cfg, optim=replace(cfg.optim, min_lr_ratio=1.5)), "min_lr_ratio"),
        (lambda cfg: replace(cfg, optim=replace(cfg.optim, weight_decay=-0.1)), "weight_decay"),
        (
            lambda cfg: replace(
                cfg, optim=replace(cfg.optim, muon=replace(cfg.optim.muon, lr_scale=0.0))
            ),
            "optim.muon.lr_scale",
        ),
        (
            lambda cfg: replace(
                cfg,
                optim=replace(cfg.optim, muon=replace(cfg.optim.muon, weight_decay_mult=-1.0)),
            ),
            "optim.muon.weight_decay_mult",
        ),
        (lambda cfg: replace(cfg, optim=replace(cfg.optim, name="sgd")), "optim.name"),
        (
            lambda cfg: replace(
                cfg,
                optim=replace(cfg.optim, muon=replace(cfg.optim.muon, momentum=1.5)),
            ),
            "optim.muon.momentum",
        ),
        (
            lambda cfg: replace(
                cfg,
                optim=replace(cfg.optim, muon=replace(cfg.optim.muon, ns_steps=0)),
            ),
            "optim.muon.ns_steps",
        ),
        (
            lambda cfg: replace(
                cfg,
                optim=replace(cfg.optim, muon=replace(cfg.optim.muon, consistent_rms=-0.1)),
            ),
            "optim.muon.consistent_rms",
        ),
        (
            lambda cfg: replace(
                cfg, optim=replace(cfg.optim, adam=replace(cfg.optim.adam, b1=1.1))
            ),
            "optim.adam.b1",
        ),
        (
            lambda cfg: replace(
                cfg, optim=replace(cfg.optim, adam=replace(cfg.optim.adam, b2=0.0))
            ),
            "optim.adam.b2",
        ),
        (
            lambda cfg: replace(
                cfg, optim=replace(cfg.optim, adam=replace(cfg.optim.adam, eps=0.0))
            ),
            "optim.adam.eps",
        ),
    ]

    for mutate, match in cases:
        with pytest.raises(ValueError, match=match):
            validate_config(mutate(_base_cfg()))


@pytest.mark.parametrize(
    ("mode", "knob"),
    [
        ("bin", "packing_buffer_docs"),
        ("multipack", "packing_group_docs"),
    ],
)
def test_max_eval_samples_below_pack_threshold_is_valid(mode: str, knob: str) -> None:
    """max_eval_samples below the packer's pack threshold must validate: the
    FFD packers flush partially filled buffers at end of stream, so a small
    eval doc set still emits windows."""
    cfg = _base_cfg()
    cfg = replace(
        cfg,
        data=replace(cfg.data, packing_mode=mode, max_eval_samples=4, **{knob: 8}),
        train=replace(cfg.train, eval_every=1),
    )
    validate_config(cfg)


def test_data_and_logging_validation_rejects_invalid_values() -> None:
    """Data/logging validation should reject invalid configuration values."""
    cases: list[tuple[Callable[[Config], Config], str]] = [
        (
            lambda cfg: replace(
                cfg,
                data=replace(cfg.data, packing_mode="bin", packing_buffer_docs=1),
            ),
            "packing_buffer_docs",
        ),
        (
            lambda cfg: replace(
                cfg,
                data=replace(cfg.data, packing_mode="multipack", packing_group_docs=0),
            ),
            "packing_group_docs",
        ),
        (
            lambda cfg: replace(
                cfg,
                data=replace(cfg.data, packing_mode="multipack", packing_max_docs_per_bin=0),
            ),
            "packing_max_docs_per_bin",
        ),
        (lambda cfg: replace(cfg, data=replace(cfg.data, packing_mode="unknown")), "packing_mode"),
        (
            lambda cfg: replace(cfg, data=replace(cfg.data, max_eval_samples=-1)),
            "max_eval_samples",
        ),
        (lambda cfg: replace(cfg, data=replace(cfg.data, seed=-1)), "data.seed"),
        (
            lambda cfg: replace(
                cfg,
                logging=replace(cfg.logging, wandb=replace(cfg.logging.wandb, mode="bogus")),
            ),
            "wandb.mode",
        ),
        (lambda cfg: replace(cfg, logging=replace(cfg.logging, level="TRACE")), "level"),
        (lambda cfg: replace(cfg, logging=replace(cfg.logging, project=" ")), "project"),
        (
            lambda cfg: replace(cfg, logging=replace(cfg.logging, metrics_file=" ")),
            "metrics_file",
        ),
        (
            lambda cfg: replace(
                cfg,
                logging=replace(
                    cfg.logging,
                    wandb=replace(cfg.logging.wandb, tags=("valid", "")),
                ),
            ),
            "wandb.tags",
        ),
        (
            lambda cfg: replace(
                cfg,
                logging=replace(
                    cfg.logging,
                    wandb=replace(cfg.logging.wandb, tags="scalar"),
                ),
            ),
            "wandb.tags",
        ),
        (lambda cfg: replace(cfg, logging=replace(cfg.logging, log_file=" ")), "log_file"),
        (
            lambda cfg: replace(
                cfg,
                logging=replace(
                    cfg.logging,
                    wandb=replace(cfg.logging.wandb, project=" "),
                ),
            ),
            "wandb.project",
        ),
        (
            lambda cfg: replace(cfg, checkpoint=replace(cfg.checkpoint, root_dir=" ")),
            "checkpoint.root_dir",
        ),
        (
            lambda cfg: replace(cfg, train=replace(cfg.train, profile_dir=" ")),
            "train.profile_dir",
        ),
        (
            lambda cfg: replace(cfg, debug=replace(cfg.debug, check_device_every=-1)),
            "check_device_every",
        ),
    ]

    for mutate, match in cases:
        with pytest.raises(ValueError, match=match):
            validate_config(mutate(_base_cfg()))


def test_wandb_tags_require_and_normalize_a_string_list() -> None:
    """YAML/JSON tag lists should normalize to the immutable config type."""
    data = _base_cfg().to_dict()
    data["logging"]["wandb"]["tags"] = ["baseline", "smoke"]

    cfg = build_config(data)

    assert cfg.logging.wandb.tags == ("baseline", "smoke")

    data["logging"]["wandb"]["tags"] = "baseline,smoke"
    with pytest.raises(ValueError, match="YAML/JSON list"):
        build_config(data)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("data.shuffle", "false"),
        ("data.device_put", "false"),
        ("model.scale_emb", "false"),
        ("model.model_dim", 128.0),
        ("optim.lr", "0.0003"),
        ("logging.project", 1),
    ],
)
def test_config_rejects_values_with_the_wrong_type(path: str, value: object) -> None:
    """YAML-looking strings and cross-category scalars must not change semantics silently."""
    data = _base_cfg().to_dict()
    target: dict[str, object] = data
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[part]  # type: ignore[assignment]
    target[parts[-1]] = value

    with pytest.raises(ValueError, match=re.escape(path)):
        build_config(data)


@pytest.mark.parametrize("mode", ["bin", "multipack"])
def test_strict_packed_rejects_disabled_boundary_masking(mode: str) -> None:
    """Strict packing with mask_boundary_loss=false must fail validation.

    The backend excludes cross-segment label pairs whenever segment_ids are
    passed (megalodon-jax >= 0.1.2), so disabling chomp's pre-masking desyncs
    host-side token counts from the model's loss denominator — a silent change
    to gradient normalization, not a preference.
    """
    cfg = _base_cfg()
    cfg = replace(
        cfg,
        data=replace(
            cfg.data,
            packing_mode=mode,
            packing_group_docs=2,
            packing_strict_segments=True,
            mask_boundary_loss=False,
            max_eval_samples=0,
        ),
    )
    with pytest.raises(ValueError, match="mask_boundary_loss"):
        validate_config(cfg)

    non_strict = replace(
        cfg, data=replace(cfg.data, packing_strict_segments=False, packing_buffer_docs=2)
    )
    validate_config(non_strict)  # deliberate bleed stays allowed


def test_hf_eval_split_allows_null() -> None:
    """hf_eval_split=None should validate and imply train-split eval fallback."""
    cfg = _base_cfg()
    validate_config(replace(cfg, data=_hf_data()))


def test_hf_eval_split_default_is_null() -> None:
    """DataConfig should default hf_eval_split to None for train-only corpora."""
    assert DataConfig().hf_eval_split is None


@pytest.mark.parametrize("bad_split", [False, 0, 1.5, [], {}])
def test_hf_eval_split_rejects_non_string_types(bad_split: object) -> None:
    """hf_eval_split must be either None or a non-empty string."""
    cfg = _base_cfg()
    with pytest.raises(ValueError, match="hf_eval_split"):
        validate_config(replace(cfg, data=_hf_data(hf_eval_split=bad_split)))


def test_muon_defaults_reflect_sweep_results() -> None:
    """Muon defaults should match the best 1k-step sweep settings."""
    cfg = _base_cfg()
    assert cfg.optim.muon.lr_scale == pytest.approx(100.0)
    assert cfg.optim.muon.consistent_rms is None


def test_default_init_mode_is_he() -> None:
    """Default init_mode should be 'he'."""
    cfg = Config()
    assert cfg.model.init_mode == "he"


def test_pad_token_id_equal_to_eos_warns() -> None:
    """pad_token_id equal to eos_token_id should warn but still validate."""
    cfg = _base_cfg()
    bad_model = replace(cfg.model, pad_token_id=2, eos_token_id=2)
    bad = replace(cfg, model=bad_model)
    with pytest.warns(UserWarning, match="pad_token_id equals model.eos_token_id"):
        validate_config(bad)


class _DummyTokenizer:
    """Mock tokenizer with configurable special tokens."""

    def __init__(self, size: int, *, bos: int | None, eos: int | None, pad: int | None) -> None:
        self._size = int(size)
        self._bos = bos
        self._eos = eos
        self._pad = pad

    def __len__(self) -> int:
        return self._size

    @property
    def bos_token_id(self) -> int | None:
        return self._bos

    @property
    def eos_token_id(self) -> int | None:
        return self._eos

    @property
    def pad_token_id(self) -> int | None:
        return self._pad


def _tokenizer_resolution_cfg(
    *,
    model: ModelConfig | None = None,
    tokenizer: TokenizerConfig | None = None,
    train_steps: int = 1,
    train_seq_len: int = 8,
) -> Config:
    """Create a local-text config for tokenizer-resolution tests."""
    return Config(
        model=model or ModelConfig(backend="dummy", vocab_size=512, d_model=32),
        data=DataConfig(
            backend="local_text",
            local_text="tokenizer config text\n",
            tokenizer=tokenizer
            or TokenizerConfig(kind="byte", vocab_size_multiple=128, max_doc_tokens=None),
        ),
        train=TrainConfig(
            steps=train_steps,
            batch_size=1,
            seq_len=train_seq_len,
            grad_accum=1,
            allow_cpu=True,
        ),
        optim=OptimConfig(warmup_steps=0),
    )


def test_vocab_size_rounds_up_to_multiple() -> None:
    """Vocab size should round up to configured multiple."""
    cfg = _tokenizer_resolution_cfg(
        model=ModelConfig(backend="dummy", vocab_size=300, d_model=32),
        tokenizer=TokenizerConfig(kind="byte", vocab_size_multiple=128),
    )
    tok = _DummyTokenizer(size=256, bos=None, eos=None, pad=None)
    updated = resolve_tokenizer_config(cfg, tok)
    assert updated.model.vocab_size == 384


def test_auto_sets_special_token_ids() -> None:
    """auto_set_special_tokens should copy IDs from tokenizer to config."""
    cfg = _tokenizer_resolution_cfg(
        model=ModelConfig(
            backend="dummy",
            vocab_size=512,
            d_model=32,
            bos_token_id=0,
            eos_token_id=1,
            pad_token_id=2,
        ),
        tokenizer=TokenizerConfig(
            kind="hf",
            hf_name_or_path="dummy",
            auto_set_special_tokens=True,
            add_bos=False,
            add_eos=False,
        ),
    )
    tok = _DummyTokenizer(size=512, bos=10, eos=11, pad=12)
    updated = resolve_tokenizer_config(cfg, tok)
    assert updated.model.bos_token_id == 10
    assert updated.model.eos_token_id == 11
    assert updated.model.pad_token_id == 12


def test_tokenizer_pad_equals_eos_warns() -> None:
    """Tokenizer with pad==eos should warn but still resolve."""
    cfg = _tokenizer_resolution_cfg(
        model=ModelConfig(
            backend="dummy",
            vocab_size=512,
            d_model=32,
            bos_token_id=0,
            eos_token_id=1,
            pad_token_id=2,
        ),
        tokenizer=TokenizerConfig(
            kind="hf",
            hf_name_or_path="dummy",
            auto_set_special_tokens=True,
            add_bos=False,
            add_eos=False,
        ),
    )
    tok = _DummyTokenizer(size=512, bos=0, eos=0, pad=0)
    with pytest.warns(UserWarning, match="pad_token_id equals model.eos_token_id"):
        updated = resolve_tokenizer_config(cfg, tok)
    assert updated.model.pad_token_id == 0
    assert updated.model.eos_token_id == 0


def test_default_max_doc_tokens_inferred() -> None:
    """max_doc_tokens should default to 4 * seq_len when unset."""
    cfg = _tokenizer_resolution_cfg(train_steps=10, train_seq_len=16)
    tok = _DummyTokenizer(size=256, bos=None, eos=None, pad=None)
    updated = resolve_tokenizer_config(cfg, tok)
    assert updated.data.tokenizer.max_doc_tokens == 64


def test_zero_max_doc_tokens_disables_truncation() -> None:
    """max_doc_tokens=0 should resolve to None (no truncation)."""
    cfg = _tokenizer_resolution_cfg(
        tokenizer=TokenizerConfig(kind="byte", vocab_size_multiple=128, max_doc_tokens=0),
        train_steps=10,
        train_seq_len=16,
    )
    tok = _DummyTokenizer(size=256, bos=None, eos=None, pad=None)
    updated = resolve_tokenizer_config(cfg, tok)
    assert updated.data.tokenizer.max_doc_tokens is None


def test_load_config_for_checkpoint_resolves_variables(tmp_path: Path) -> None:
    """Variable placeholders in override configs should resolve before validation."""
    config_path = tmp_path / "config.yaml"
    _write_tiny_yaml_config(
        config_path,
        variables="variables:\n  seq_len: 64\n\n",
        train_seq_len="$variables.seq_len",
    )

    cfg = load_config_for_checkpoint(
        step_dir=tmp_path, run_dir=None, config_override=str(config_path)
    )

    assert cfg.train.seq_len == 64


def test_generate_config_applies_tokenizer_derived_fields(tmp_path: Path) -> None:
    """Generate config loading should apply tokenizer-derived fields."""
    config_path = tmp_path / "config.yaml"
    _write_tiny_yaml_config(
        config_path,
        vocab_size=300,
        tokenizer_extra="    vocab_size_multiple: 128",
    )

    cfg = load_config_for_checkpoint(
        step_dir=tmp_path, run_dir=None, config_override=str(config_path)
    )

    assert cfg.model.vocab_size == 300

    tokenizer = build_tokenizer(cfg)
    cfg_resolved = resolve_tokenizer_config(cfg, tokenizer)
    assert cfg_resolved.model.vocab_size == 384


def test_override_casts_float_when_default_none(tmp_path: Path) -> None:
    """Optional float overrides should parse into float values."""
    config_path = tmp_path / "config.yaml"
    _write_tiny_yaml_config(config_path, train_steps=20, train_seq_len=8)

    cfg = load_config(config_path, overrides=["optim.muon.consistent_rms=0.2"])

    assert isinstance(cfg.optim.muon.consistent_rms, float)
    assert cfg.optim.muon.consistent_rms == pytest.approx(0.2)
