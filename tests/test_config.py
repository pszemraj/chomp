"""Config tests consolidated by module."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields, replace
from pathlib import Path

import jax
import pytest

from chomp.config import (
    Config,
    DataConfig,
    ModelConfig,
    OptimConfig,
    TokenizerConfig,
    TrainConfig,
    build_config,
    derived_deterministic,
    load_config,
    read_config_mapping,
    resolve_window_shuffle_rows,
    strict_packed_segments,
    validate_config,
)
from chomp.data.pipeline import build_tokenizer, resolve_tokenizer_config
from chomp.model import build_model
from chomp.utils.ckpt_paths import load_config_for_checkpoint
from chomp.utils.tree import param_count


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
        hf_revision="0" * 40,
        hf_eval_split=hf_eval_split,  # type: ignore[arg-type]
        text_key="text",
        shuffle=False,
        repeat=True,
        tokenizer=TokenizerConfig(kind="byte", byte_offset=0, add_bos=False, add_eos=False),
    )


def test_model_and_train_validation_rejects_invalid_values() -> None:
    """Model/train validation should fail with actionable errors."""
    cases: list[tuple[Callable[[Config], Config], str]] = [
        (lambda cfg: replace(cfg, model=replace(cfg.model, chunk_size=32)), "chunk_size"),
        (lambda cfg: replace(cfg, model=replace(cfg.model, chunk_size=10)), "divisible"),
        # bf16 params without an fp32 master-param path silently give bf16
        # optimizer moments; rejected until such a path exists.
        (
            lambda cfg: replace(cfg, model=replace(cfg.model, param_dtype="bfloat16")),
            "param_dtype",
        ),
        (
            lambda cfg: replace(cfg, model=replace(cfg.model, model_dim=130, num_heads=8)),
            "model_dim",
        ),
        (lambda cfg: replace(cfg, model=replace(cfg.model, dropout=1.0)), "model.dropout"),
        (lambda cfg: replace(cfg, model=replace(cfg.model, z_dim=65)), "model.z_dim"),
        (
            lambda cfg: replace(cfg, model=replace(cfg.model, z_dim=72, num_heads=8)),
            "even for RoPE",
        ),
        (lambda cfg: replace(cfg, model=replace(cfg.model, value_dim=130)), "model.value_dim"),
        (
            lambda cfg: replace(cfg, model=replace(cfg.model, norm_num_groups=30)),
            "norm_num_groups",
        ),
        (
            lambda cfg: replace(cfg, model=replace(cfg.model, norm_num_groups=128)),
            "smaller than model.model_dim",
        ),
        (lambda cfg: replace(cfg, model=replace(cfg.model, norm_eps=0.0)), "norm_eps"),
        (
            lambda cfg: replace(cfg, model=replace(cfg.model, attention_window=0)),
            "attention_window",
        ),
        (lambda cfg: replace(cfg, model=replace(cfg.model, rope_base=0.0)), "rope_base"),
        (
            lambda cfg: replace(cfg, model=replace(cfg.model, attention_dropout=-0.1)),
            "attention_dropout",
        ),
        (
            lambda cfg: replace(cfg, model=replace(cfg.model, hidden_dropout=1.0)),
            "hidden_dropout",
        ),
        (
            lambda cfg: replace(cfg, model=replace(cfg.model, attention_dropout_mode="other")),
            "attention_dropout_mode",
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
            lambda cfg: replace(cfg, model=replace(cfg.model, attention_softmax_dtype="float16")),
            "attention_softmax_dtype",
        ),
        (lambda cfg: replace(cfg, model=replace(cfg.model, output_size=0)), "output_size"),
        (
            lambda cfg: replace(cfg, model=replace(cfg.model, share_emb=True, output_size=128)),
            "share_emb",
        ),
        (
            lambda cfg: replace(
                cfg,
                model=replace(
                    cfg.model,
                    share_emb=False,
                    output_size=cfg.model.vocab_size - 1,
                ),
            ),
            "output_size.*vocab_size",
        ),
        (lambda cfg: replace(cfg, model=replace(cfg.model, pad_token_id=-1)), "pad_token_id"),
        (
            lambda cfg: replace(cfg, model=replace(cfg.model, eos_token_id=cfg.model.vocab_size)),
            "eos_token_id",
        ),
        (lambda cfg: replace(cfg, train=replace(cfg.train, seed=-1)), "train.seed"),
        (lambda cfg: replace(cfg, train=replace(cfg.train, eval_every=-1)), "eval_every"),
        (
            lambda cfg: replace(cfg, train=replace(cfg.train, eval_failure_policy="ignore")),
            "eval_failure_policy",
        ),
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
        (
            lambda cfg: replace(
                cfg,
                checkpoint=replace(cfg.checkpoint, resume_compat="invalid"),
            ),
            "resume_compat",
        ),
    ]

    for mutate, match in cases:
        with pytest.raises(ValueError, match=match):
            validate_config(mutate(_base_cfg()))


def test_export_dir_name_must_stay_inside_the_run_directory() -> None:
    """dir_name is joined onto the run directory, so only a plain name is safe.

    Anything with a separator could put the export outside the run, or on top of
    the run's own checkpoints and tokenizer snapshot.
    """
    for name in ("../escape", "nested/export", "/absolute", "", ".", ".."):
        cfg = replace(_base_cfg(), export=replace(_base_cfg().export, dir_name=name))
        with pytest.raises(ValueError, match="export.dir_name"):
            validate_config(cfg)

    validate_config(replace(_base_cfg(), export=replace(_base_cfg().export, dir_name="model")))


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


def test_warmup_ratio_resolves_to_steps() -> None:
    """A warmup ratio should derive warmup_steps from the step budget."""
    data = _base_cfg().to_dict()
    data["train"]["steps"] = 62000
    data["optim"]["warmup_ratio"] = 0.032
    del data["optim"]["warmup_steps"]

    cfg = build_config(data)

    assert cfg.optim.warmup_steps == 1984
    assert cfg.optim.warmup_ratio == 0.032


def test_warmup_ratio_rescales_when_overrides_shorten_the_run() -> None:
    """Shortening train.steps must rescale warmup instead of failing validation."""
    data = _base_cfg().to_dict()
    data["train"]["steps"] = 62000
    data["optim"]["warmup_ratio"] = 0.05
    del data["optim"]["warmup_steps"]

    cfg = build_config(data, overrides=["train.steps=200"])

    assert cfg.optim.warmup_steps == 10


def test_warmup_ratio_round_trips_through_resolved_dict() -> None:
    """config_resolved.json stores ratio and resolved steps together."""
    data = _base_cfg().to_dict()
    data["train"]["steps"] = 1000
    data["optim"]["warmup_ratio"] = 0.1
    del data["optim"]["warmup_steps"]

    cfg = build_config(data)

    assert build_config(cfg.to_dict()) == cfg


def test_warmup_ratio_rejects_contradictory_step_count() -> None:
    """An explicit warmup_steps that disagrees with the ratio must fail loudly."""
    data = _base_cfg().to_dict()
    data["train"]["steps"] = 1000
    data["optim"]["warmup_ratio"] = 0.1
    data["optim"]["warmup_steps"] = 25

    with pytest.raises(ValueError, match="contradicts optim.warmup_ratio"):
        build_config(data)


def test_warmup_steps_override_conflicts_with_active_ratio() -> None:
    """Overriding the derived value would be silently discarded, so reject it."""
    data = _base_cfg().to_dict()
    data["train"]["steps"] = 1000
    data["optim"]["warmup_ratio"] = 0.1
    del data["optim"]["warmup_steps"]

    with pytest.raises(ValueError, match="conflicts with optim.warmup_ratio"):
        build_config(data, overrides=["optim.warmup_steps=25"])

    cfg = build_config(data, overrides=["optim.warmup_ratio=null", "optim.warmup_steps=25"])
    assert cfg.optim.warmup_steps == 25
    assert cfg.optim.warmup_ratio is None


@pytest.mark.parametrize("ratio", [-0.1, 1.0, 2.0])
def test_warmup_ratio_rejects_out_of_range(ratio: float) -> None:
    """The ratio is a fraction of the run that must leave room to decay."""
    data = _base_cfg().to_dict()
    data["optim"]["warmup_ratio"] = ratio
    del data["optim"]["warmup_steps"]

    with pytest.raises(ValueError, match=r"optim.warmup_ratio must be in \[0, 1\)"):
        build_config(data)


def test_unresolved_warmup_ratio_fails_validation() -> None:
    """A hand-built Config must not silently ignore its warmup ratio."""
    base = _base_cfg()
    cfg = replace(
        base,
        train=replace(base.train, steps=1000),
        optim=replace(base.optim, warmup_ratio=0.1, warmup_steps=0),
    )

    with pytest.raises(ValueError, match="implies optim.warmup_steps=100"):
        validate_config(cfg)


@pytest.mark.parametrize(
    ("data", "path"),
    [
        ({"derived": []}, "derived"),
        ({"model": False}, "model"),
        ({"train": []}, "train"),
        ({"optim": "adamw"}, "optim"),
        ({"optim": {"muon": False}}, "optim.muon"),
        ({"optim": {"adam": []}}, "optim.adam"),
        ({"checkpoint": 0}, "checkpoint"),
        ({"logging": "INFO"}, "logging"),
        ({"logging": {"wandb": False}}, "logging.wandb"),
        ({"debug": []}, "debug"),
        ({"data": False}, "data"),
        ({"data": {"tokenizer": []}}, "data.tokenizer"),
    ],
)
def test_build_config_rejects_non_mapping_sections(data: dict[str, object], path: str) -> None:
    """Falsy and truthy section scalars must not silently select defaults."""
    with pytest.raises(ValueError, match=rf"{path} must be a mapping or null"):
        build_config(data)


def test_config_reference_matches_config_fields() -> None:
    """The config reference should contain every config field and no stale fields."""
    reference = read_config_mapping(Path(__file__).parents[1] / "docs/config-reference.yaml")
    assert set(reference) == {field.name for field in fields(Config)} | {"variables", "derived"}

    reference = {
        key: value for key, value in reference.items() if key not in {"variables", "derived"}
    }

    def _assert_matching_keys(
        documented: dict[str, object], defaults: dict[str, object], path: str = ""
    ) -> None:
        """Assert that one documented config mapping matches its dataclass mapping.

        :param dict[str, object] documented: Config-reference mapping to inspect.
        :param dict[str, object] defaults: Runtime default mapping to compare.
        :param str path: Nested field path for assertion messages, defaults to "".
        """
        assert set(documented) == set(defaults), path or "config"
        for key, default in defaults.items():
            if isinstance(default, dict):
                child = documented[key]
                assert isinstance(child, dict), f"{path}{key}"
                _assert_matching_keys(child, default, f"{path}{key}.")

    _assert_matching_keys(reference, Config().to_dict())


@pytest.mark.parametrize(
    ("name", "data_backend", "allow_cpu"),
    [
        ("offline_cpu_smoke.yaml", "local_text", True),
        ("hf_streaming_smoke.yaml", "hf", False),
    ],
)
def test_dev_smoke_configs_load(name: str, data_backend: str, allow_cpu: bool) -> None:
    """Checked-in smoke configs should select their intended I/O and device paths."""
    config_path = Path(__file__).parents[1] / "configs/dev" / name
    cfg = load_config(config_path)
    assert cfg.model.backend == "dummy"
    assert cfg.data.backend == data_backend
    assert cfg.train.allow_cpu is allow_cpu


@pytest.mark.parametrize(
    ("name", "expected_parameters", "batch_size", "grad_accum"),
    [
        ("megalodon_100m_2048.yaml", 113_854_464, 2, 8),
        ("megalodon_200m_2048.yaml", 188_777_472, 2, 8),
        ("megalodon_500m_2048.yaml", 513_672_192, 1, 16),
        ("megalodon_1b_2048.yaml", 976_978_944, 1, 16),
    ],
)
def test_maintained_pretrain_recipe_contract(
    name: str,
    expected_parameters: int,
    batch_size: int,
    grad_accum: int,
) -> None:
    """Maintained recipes should preserve their labeled scale and correctness policy."""
    config_path = Path(__file__).parents[1] / "configs/pretrain" / name
    cfg = load_config(config_path)
    abstract_params = jax.eval_shape(
        lambda key: build_model(cfg, key=key)[0],
        jax.random.key(0),
    )

    assert param_count(abstract_params) == expected_parameters
    assert cfg.model.backend == "megalodon"
    # Gated FFN at a param-matched width. swiglu adds a third model_dim x
    # ffn_hidden_dim matrix rather than rescaling it, so flipping this without
    # also rescaling ffn_hidden_dim silently inflates the recipe by ~50% of its
    # feed-forward mass -- which the parameter assertion above would catch.
    assert cfg.model.swiglu is True
    assert cfg.model.share_emb is True
    assert cfg.model.chunk_size == 512
    assert cfg.model.attention_window is None
    assert cfg.model.use_checkpoint is True
    assert derived_deterministic(cfg) is False
    assert cfg.model.param_dtype == "float32"
    assert cfg.model.compute_dtype == "bfloat16"
    assert cfg.model.accum_dtype == "float32"
    assert cfg.model.attention_softmax_dtype == "float32"
    assert cfg.data.packing_strict_segments is True
    assert cfg.data.mask_boundary_loss is True
    assert strict_packed_segments(cfg) is True
    assert cfg.optim.name == "muon"
    assert cfg.optim.muon.lr_scale == 100.0
    assert cfg.optim.muon.consistent_rms is None
    assert cfg.checkpoint.resume_compat == "strict"
    assert cfg.train.eval_failure_policy == "fatal"
    assert cfg.train.batch_size == batch_size
    assert cfg.train.grad_accum == grad_accum
    assert cfg.optim.warmup_steps * 100 == cfg.train.steps

    max_target_positions = (
        cfg.train.steps * cfg.train.grad_accum * cfg.train.batch_size * (cfg.train.seq_len - 1)
    )
    assert max_target_positions >= 20 * expected_parameters


def test_yaml_loader_rejects_duplicate_explicit_keys(tmp_path: Path) -> None:
    """Repeated explicit YAML keys must fail instead of silently taking the last value."""
    config_path = tmp_path / "duplicate.yaml"
    config_path.write_text("train:\n  steps: 10\n  steps: 20\n")

    with pytest.raises(ValueError, match="duplicate key 'steps'"):
        read_config_mapping(config_path)


def test_yaml_loader_allows_explicit_merge_overrides(tmp_path: Path) -> None:
    """Standard YAML merge defaults may still be overridden explicitly."""
    config_path = tmp_path / "merge.yaml"
    config_path.write_text(
        "defaults: &defaults\n  steps: 10\ntrain:\n  <<: *defaults\n  steps: 20\n"
    )

    assert read_config_mapping(config_path)["train"]["steps"] == 20


def test_defaults_select_the_real_training_data_path() -> None:
    """Bare defaults should use the maintained packed HF training path."""
    cfg = Config()

    assert cfg.data.packing_mode == "bin"
    assert cfg.data.packing_buffer_docs == 256
    assert cfg.data.grain_prefetch == 2
    assert cfg.data.tokenizer.kind == "hf"
    assert cfg.data.tokenizer.add_eos is True
    validate_config(cfg)


def test_hf_dataset_name_may_be_omitted() -> None:
    """Single-config HF repositories should accept the Datasets default config."""
    cfg = replace(_base_cfg(), data=replace(_hf_data(), hf_name=None))

    validate_config(cfg)


def test_config_variables_must_be_a_mapping() -> None:
    """Variable resolution requires a mapping."""
    with pytest.raises(ValueError, match="variables"):
        build_config({"variables": []})


def test_override_rejects_empty_dotted_path_component() -> None:
    """Malformed dot-path overrides should fail before attribute lookup."""
    with pytest.raises(ValueError, match="Invalid override path"):
        build_config(_base_cfg().to_dict(), overrides=["train..steps=2"])


def test_override_cast_failure_names_dotted_path() -> None:
    """Invalid scalar overrides should identify the field that failed."""
    with pytest.raises(ValueError, match="Invalid override 'train.steps'"):
        build_config(_base_cfg().to_dict(), overrides=["train.steps=abc"])


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
                data=replace(cfg.data, packing_mode="bin", packing_buffer_docs=0),
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
            lambda cfg: replace(cfg, data=replace(cfg.data, eval_packing="bin")),
            "eval_packing",
        ),
        (
            lambda cfg: replace(cfg, data=replace(cfg.data, max_eval_samples=-1)),
            "max_eval_samples",
        ),
        (
            lambda cfg: replace(cfg, data=replace(cfg.data, window_shuffle_tokens=-1)),
            "window_shuffle_tokens",
        ),
        (
            lambda cfg: replace(cfg, data=replace(cfg.data, window_shuffle_tokens=31)),
            "window_shuffle_tokens",
        ),
        (
            lambda cfg: replace(cfg, data=replace(cfg.data, window_shuffle_max_rows=0)),
            "window_shuffle_max_rows",
        ),
        (
            lambda cfg: replace(cfg, data=replace(cfg.data, window_shuffle_max_rows=1)),
            "window_shuffle_max_rows",
        ),
        (
            lambda cfg: replace(
                cfg,
                data=replace(_hf_data(), shuffle=True, shuffle_buffer_bytes=0),
            ),
            "shuffle_buffer_bytes",
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
        (lambda cfg: replace(cfg, logging=replace(cfg.logging, log_file=" ")), "log_file"),
        (lambda cfg: replace(cfg, logging=replace(cfg.logging, run_dir=" ")), "run_dir"),
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
            lambda cfg: replace(cfg, train=replace(cfg.train, profile_dir=" ")),
            "train.profile_dir",
        ),
    ]

    for mutate, match in cases:
        with pytest.raises(ValueError, match=match):
            validate_config(mutate(_base_cfg()))


@pytest.mark.parametrize(
    (
        "token_budget",
        "max_rows",
        "seq_len",
        "batch_size",
        "grad_accum",
        "expected_rows",
    ),
    [
        (0, None, 16, 2, 1, 0),
        (8_388_608, None, 8, 1, 1, 4_096),
        (8_388_608, None, 16, 1, 1, 4_096),
        (8_388_608, None, 128, 1, 1, 4_096),
        (8_388_608, None, 2_048, 1, 1, 4_096),
        (8_388_608, None, 32_768, 1, 1, 256),
        (1_000, None, 16, 3, 2, 60),
        (1_000, 50, 16, 3, 2, 48),
    ],
)
def test_window_shuffle_budget_resolves_to_batch_aligned_rows(
    token_budget: int,
    max_rows: int | None,
    seq_len: int,
    batch_size: int,
    grad_accum: int,
    expected_rows: int,
) -> None:
    """Packed-row shuffle should honor both bounds and preserve batch geometry."""
    cfg = _base_cfg()
    data = replace(cfg.data, window_shuffle_tokens=token_budget)
    if max_rows is not None:
        data = replace(data, window_shuffle_max_rows=max_rows)
    cfg = replace(
        cfg,
        data=data,
        model=replace(cfg.model, chunk_size=min(cfg.model.chunk_size, seq_len)),
        train=replace(
            cfg.train,
            seq_len=seq_len,
            batch_size=batch_size,
            grad_accum=grad_accum,
        ),
    )

    validate_config(cfg)
    assert resolve_window_shuffle_rows(cfg) == expected_rows


def test_wandb_tags_require_and_normalize_a_string_list() -> None:
    """YAML/JSON tag lists should normalize to the immutable config type."""
    data = _base_cfg().to_dict()
    data["logging"]["wandb"]["tags"] = ["baseline", "smoke"]

    cfg = build_config(data)

    assert cfg.logging.wandb.tags == ("baseline", "smoke")

    data["logging"]["wandb"]["tags"] = "baseline,smoke"
    with pytest.raises(ValueError, match="YAML/JSON list"):
        build_config(data)

    data["logging"]["wandb"]["tags"] = ["baseline", 1]
    with pytest.raises(ValueError, match="YAML/JSON list"):
        build_config(data)

    for raw in ("null", "baseline,smoke"):
        with pytest.raises(ValueError, match="logging.wandb.tags.*list-valued"):
            build_config(
                _base_cfg().to_dict(),
                overrides=[f"logging.wandb.tags={raw}"],
            )


@pytest.mark.parametrize("mode", ["bin", "multipack"])
def test_strict_packed_rejects_disabled_boundary_masking(mode: str) -> None:
    """Strict packing with mask_boundary_loss=false must fail validation.

    The backend excludes cross-segment label pairs whenever segment_ids are
    passed (megalodon-jax >= 0.2.2), so disabling chomp's pre-masking desyncs
    asynchronous host token accounting from the backend's authoritative count.
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


@pytest.mark.parametrize(
    ("section", "field", "value", "match"),
    [
        ("data", "shuffle", "false", "data.shuffle must be a boolean"),
        ("data", "packing_strict_segments", "true", "must be a boolean"),
        ("optim", "lr", "0.0003", "optim.lr must be a number"),
        ("optim", "lr", float("nan"), "optim.lr must be finite"),
        ("optim", "weight_decay", float("inf"), "must be finite"),
        ("train", "steps", 1.5, "train.steps must be an integer"),
    ],
)
def test_config_rejects_mistyped_scalars(
    section: str, field: str, value: object, match: str
) -> None:
    """Quoted YAML scalars and non-finite numbers fail at config time."""
    cfg = _base_cfg()
    cfg = replace(cfg, **{section: replace(getattr(cfg, section), **{field: value})})
    with pytest.raises(ValueError, match=match):
        validate_config(cfg)


def test_override_null_rejected_for_non_optional_field() -> None:
    """--override key=null cannot silently clear a required scalar."""
    cfg = _base_cfg()
    with pytest.raises(ValueError, match="optim.lr must be a number"):
        build_config(cfg.to_dict(), overrides=["optim.lr=null"])


def test_hf_eval_split_allows_null() -> None:
    """hf_eval_split=None should validate as a disjoint content holdout."""
    cfg = _base_cfg()
    validate_config(replace(cfg, data=_hf_data()))


@pytest.mark.parametrize("revision", [None, "main", "abc123"])
def test_hf_source_allows_any_revision(revision: str | None) -> None:
    """Refs and null are valid at config time; run() resolves them to a commit."""
    cfg = _base_cfg()
    validate_config(replace(cfg, data=replace(_hf_data(), hf_revision=revision)))


def test_hf_source_allows_null_revision_override() -> None:
    """A CLI override may clear the default HF revision."""
    cfg = _base_cfg()
    cfg = build_config(
        replace(cfg, data=_hf_data()).to_dict(),
        overrides=["data.hf_revision=null"],
    )

    assert cfg.data.hf_revision is None


@pytest.mark.parametrize("bad_split", [False, 0, 1.5, [], {}])
def test_hf_eval_split_rejects_non_string_types(bad_split: object) -> None:
    """hf_eval_split must be either None or a non-empty string."""
    cfg = _base_cfg()
    with pytest.raises(ValueError, match="hf_eval_split"):
        validate_config(replace(cfg, data=_hf_data(hf_eval_split=bad_split)))


def test_hf_eval_split_rejects_enabled_training_split() -> None:
    """An eval_loss split cannot be the same split consumed by training."""
    cfg = _base_cfg()
    with pytest.raises(ValueError, match="must differ"):
        validate_config(replace(cfg, data=_hf_data(hf_eval_split="train")))


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.1])
def test_hf_eval_holdout_fraction_is_open_unit_interval(fraction: float) -> None:
    """Content holdout fractions must leave nonempty probability mass on both sides."""
    cfg = _base_cfg()
    data = replace(_hf_data(), hf_eval_holdout_fraction=fraction)
    with pytest.raises(ValueError, match="hf_eval_holdout_fraction"):
        validate_config(replace(cfg, data=data))


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
            or TokenizerConfig(
                kind="byte",
                vocab_size_multiple=128,
                add_eos=False,
                max_doc_tokens=None,
            ),
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
        tokenizer=TokenizerConfig(kind="byte", vocab_size_multiple=128, add_eos=False),
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


def test_tokenizer_resolution_rejects_out_of_range_special_ids() -> None:
    """Resolved special IDs must fit the final aligned model vocabulary."""
    cfg = _tokenizer_resolution_cfg(
        model=ModelConfig(
            backend="dummy",
            vocab_size=512,
            d_model=32,
            pad_token_id=512,
        )
    )
    tok = _DummyTokenizer(size=256, bos=None, eos=None, pad=None)

    with pytest.raises(ValueError, match="pad_token_id.*resolved vocab_size"):
        resolve_tokenizer_config(cfg, tok)


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


def test_default_max_doc_tokens_disables_truncation() -> None:
    """An unset document cap must remain None rather than silently truncating."""
    cfg = _tokenizer_resolution_cfg(train_steps=10, train_seq_len=16)
    tok = _DummyTokenizer(size=256, bos=None, eos=None, pad=None)
    updated = resolve_tokenizer_config(cfg, tok)
    assert updated.data.tokenizer.max_doc_tokens is None


@pytest.mark.parametrize("value", [0, -1])
def test_nonpositive_max_doc_tokens_is_rejected(value: int) -> None:
    """Only null or a positive explicit truncation cap is valid."""
    cfg = _tokenizer_resolution_cfg(
        tokenizer=TokenizerConfig(
            kind="byte",
            vocab_size_multiple=128,
            add_eos=False,
            max_doc_tokens=value,
        ),
        train_steps=10,
        train_seq_len=16,
    )
    tok = _DummyTokenizer(size=256, bos=None, eos=None, pad=None)
    with pytest.raises(ValueError, match="max_doc_tokens must be null"):
        resolve_tokenizer_config(cfg, tok)


def test_load_config_for_checkpoint_resolves_variables(tmp_path: Path) -> None:
    """Variable placeholders in override configs should resolve before validation."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """variables:
  seq_len: 64
model:
  backend: dummy
  d_model: 32
train:
  seq_len: $variables.seq_len
"""
    )

    cfg = load_config_for_checkpoint(step_dir=tmp_path, config_override=str(config_path))

    assert cfg.train.seq_len == 64


def test_generate_config_applies_tokenizer_derived_fields(tmp_path: Path) -> None:
    """Generate config loading should apply tokenizer-derived fields."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """model:
  backend: dummy
  vocab_size: 300
  d_model: 32
data:
  backend: local_text
  local_text: hello
  tokenizer:
    kind: byte
    add_eos: false
    vocab_size_multiple: 128
"""
    )

    cfg = load_config_for_checkpoint(step_dir=tmp_path, config_override=str(config_path))

    assert cfg.model.vocab_size == 300

    tokenizer = build_tokenizer(cfg)
    cfg_resolved = resolve_tokenizer_config(cfg, tokenizer)
    assert cfg_resolved.model.vocab_size == 384


def test_override_casts_float_when_default_none(tmp_path: Path) -> None:
    """Optional float overrides should parse into float values."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """model:
  backend: dummy
  d_model: 32
train:
  steps: 20
  seq_len: 8
"""
    )

    cfg = load_config(config_path, overrides=["optim.muon.consistent_rms=0.2"])

    assert isinstance(cfg.optim.muon.consistent_rms, float)
    assert cfg.optim.muon.consistent_rms == pytest.approx(0.2)
