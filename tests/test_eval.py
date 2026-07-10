"""Evaluation tests consolidated by module."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from chomp.config import (
    CheckpointConfig,
    Config,
    DataConfig,
    DebugConfig,
    LoggingConfig,
    ModelConfig,
    OptimConfig,
    TokenizerConfig,
    TrainConfig,
)
from chomp.data import (
    build_eval_iterator,
    build_generation_text_stream,
    build_tokenizer,
    load_or_create_eval_tokens,
)
from chomp.train import run
from tests.helpers.hf_fakes import FakeHFIterable
from tests.helpers.io import read_jsonl


def _eval_run_cfg(
    run_dir: Path, *, steps: int = 2, checkpoint: CheckpointConfig | None = None
) -> Config:
    """Build a local-text run config that exercises periodic eval."""
    return Config(
        model=ModelConfig(backend="dummy", vocab_size=256, d_model=32, dropout=0.0),
        data=DataConfig(
            backend="local_text",
            repeat=True,
            local_text="abcdefghijklmnopqrstuvwxyz" * 4,
            max_eval_samples=3,
            tokenizer=TokenizerConfig(kind="byte", byte_offset=0, add_bos=False, add_eos=False),
        ),
        train=TrainConfig(
            seed=0,
            steps=steps,
            batch_size=1,
            seq_len=8,
            grad_accum=1,
            jit=False,
            deterministic=True,
            allow_cpu=True,
            log_every=1000,
            eval_every=1,
        ),
        optim=OptimConfig(lr=1e-3, weight_decay=0.0, grad_clip_norm=0.0, warmup_steps=0),
        checkpoint=checkpoint or CheckpointConfig(enabled=False),
        debug=DebugConfig(nan_check=True, check_device_every=0),
        logging=LoggingConfig(project="chomp", run_dir=str(run_dir)),
    )


def test_eval_batches_assembled_once_and_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Eval batches are deterministic; the iterator is built once, not per eval."""
    from chomp.data import build_eval_iterator as real_build

    calls = {"n": 0}

    def _counting_build(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        return real_build(*args, **kwargs)

    monkeypatch.setattr("chomp.train.build_eval_iterator", _counting_build)

    run_dir = tmp_path / "run_cache"
    cfg = _eval_run_cfg(run_dir)

    run(cfg, config_path=None, resume="none")

    # The training entrypoint pins the eval set to the run directory.
    assert (run_dir / "eval_tokens.json.gz").exists()

    assert calls["n"] == 1, f"eval iterator rebuilt {calls['n']} times for 2 evals"
    metrics_path = run_dir / cfg.logging.metrics_file
    rows = read_jsonl(metrics_path)
    eval_rows = [row for row in rows if row.get("eval_loss") not in (None, "")]
    assert len(eval_rows) == 2  # both evals produced a loss from the cached batches
    assert any("step" in row for row in rows)
    for row in rows:
        for key in (
            "eval_tokens",
            "wall_time_s",
            "packing_tokens",
            "packing_capacity",
            "device_memory_gb",
        ):
            assert key not in row


def _packed_eval_cfg(run_dir: Path, *, grad_accum: int = 1, **data_overrides: Any) -> Config:
    """Build a tiny packed-mode config for eval flush/zero-batch tests.

    :param Path run_dir: Run directory for logging.
    :param int grad_accum: Gradient accumulation steps (rows_per_batch knob).
    :param data_overrides: DataConfig field overrides (packing mode/knobs).
    :return Config: Ready-to-run configuration.
    """
    return Config(
        model=ModelConfig(backend="dummy", vocab_size=256, d_model=32, dropout=0.0),
        data=DataConfig(
            backend="hf",
            hf_dataset="dummy",
            hf_name="dummy",
            hf_split="train",
            hf_eval_split="validation",
            shuffle=False,
            repeat=True,
            tokenizer=TokenizerConfig(kind="byte", byte_offset=0, add_bos=False, add_eos=False),
            **data_overrides,
        ),
        train=TrainConfig(
            seed=0,
            steps=1,
            batch_size=1,
            seq_len=8,
            grad_accum=grad_accum,
            jit=False,
            deterministic=True,
            allow_cpu=True,
            log_every=1000,
            eval_every=1,
        ),
        optim=OptimConfig(lr=1e-3, weight_decay=0.0, grad_clip_norm=0.0, warmup_steps=0),
        checkpoint=CheckpointConfig(enabled=False),
        debug=DebugConfig(nan_check=True, check_device_every=0),
        logging=LoggingConfig(project="chomp", run_dir=str(run_dir)),
    )


@pytest.mark.parametrize(
    ("mode", "knobs"),
    [
        pytest.param("bin", {"packing_buffer_docs": 4}, id="bin"),
        pytest.param("multipack", {"packing_group_docs": 8}, id="multipack"),
    ],
)
def test_eval_flushes_partial_buffer_at_stream_end(
    tmp_path: Path,
    patch_hf_load_dataset: Callable[..., dict[str, int]],
    mode: str,
    knobs: dict[str, int],
) -> None:
    """Eval doc sets below the pack threshold must still produce eval batches.

    The eval split yields 2 docs, below the packers' thresholds; the
    end-of-stream flush packs them into a padded window instead of starving
    eval with zero batches.
    """
    patch_hf_load_dataset(
        {
            "train": [{"text": "xxxx"} for _ in range(64)],
            "validation": [{"text": "yy"}, {"text": "zz"}],
        }
    )

    run_dir = tmp_path / f"run_flush_{mode}"
    cfg = _packed_eval_cfg(run_dir, packing_mode=mode, max_eval_samples=8, **knobs)

    run(cfg, config_path=None, resume="none")

    metrics_path = run_dir / cfg.logging.metrics_file
    rows = read_jsonl(metrics_path)
    eval_rows = [row for row in rows if row.get("eval_loss") not in (None, "")]
    assert eval_rows, "flushed eval windows should have produced an eval loss"


def test_eval_pads_missing_rows_and_ignores_train_grad_accum(
    tmp_path: Path, patch_hf_load_dataset: Callable[..., dict[str, int]]
) -> None:
    """Finite eval uses A=1 and retains a row below the train batch geometry."""
    patch_hf_load_dataset(
        {
            "train": [{"text": "xxxx"} for _ in range(64)],
            # 4 tokens flush into a single [T=8] window < rows_per_batch=2.
            "validation": [{"text": "yy"}, {"text": "zz"}],
        }
    )

    run_dir = tmp_path / "run_partial_batch"
    cfg = _packed_eval_cfg(
        run_dir,
        grad_accum=2,
        packing_mode="bin",
        packing_buffer_docs=4,
        max_eval_samples=8,
    )

    run(cfg, config_path=None, resume="none")

    rows = read_jsonl(run_dir / cfg.logging.metrics_file)
    eval_rows = [row for row in rows if row.get("eval_loss") not in (None, "")]
    assert eval_rows
    eval_batches = list(build_eval_iterator(cfg, tokens=[[1, 2], [3, 4]]))
    assert len(eval_batches) == 1
    assert eval_batches[0].input_ids.shape == (1, 1, 8)


def test_eval_fails_fast_on_zero_valid_tokens(
    tmp_path: Path, patch_hf_load_dataset: Callable[..., dict[str, int]]
) -> None:
    """Eval batches whose labels are entirely masked must raise, not log
    eval_loss=None.

    Single-token eval docs make every adjacent window position a segment
    transition, so boundary masking wipes all shifted labels — batches emit
    but token_sum stays 0, which would otherwise silently null the eval curve
    for the whole run.
    """
    patch_hf_load_dataset(
        {
            "train": [{"text": "abcdefghijklmnop"} for _ in range(64)],
            "validation": [{"text": "a"} for _ in range(64)],
        }
    )

    run_dir = tmp_path / "run_zero_tokens"
    cfg = _packed_eval_cfg(run_dir, packing_mode="sequential", max_eval_samples=64)

    with pytest.raises(RuntimeError, match="zero valid loss tokens"):
        run(cfg, config_path=None, resume="none")


def _base_cfg() -> Config:
    """Create a base config for eval text selection tests."""
    return Config(
        model=ModelConfig(backend="dummy", vocab_size=256, d_model=16, dropout=0.0),
        data=DataConfig(
            backend="hf",
            hf_dataset="dummy",
            hf_name="dummy",
            hf_split="train",
            hf_eval_split="validation",
            text_key="text",
            shuffle=False,
            repeat=False,
            max_eval_samples=2,
            tokenizer=TokenizerConfig(kind="byte", byte_offset=0, add_bos=False, add_eos=False),
        ),
        train=TrainConfig(
            steps=1,
            batch_size=1,
            seq_len=8,
            grad_accum=1,
            jit=False,
            deterministic=True,
            allow_cpu=True,
            log_every=1000,
            eval_every=0,
        ),
    )


def test_eval_split_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit eval split is authoritative and never opens train."""
    requested_splits: list[str] = []

    def _load_dataset(
        dataset: str, *, name: str, split: str, streaming: bool, revision: str | None
    ) -> FakeHFIterable:
        _ = (dataset, name, streaming, revision)
        requested_splits.append(split)
        return FakeHFIterable(items=[{"text": "val-a"}, {"text": "val-b"}])

    import datasets as hf_datasets

    monkeypatch.setattr(hf_datasets, "load_dataset", _load_dataset)
    cfg = _base_cfg()
    tok = build_tokenizer(cfg)
    tokens = load_or_create_eval_tokens(cfg, tokenizer=tok)

    assert tokens == [tok.encode("val-a"), tok.encode("val-b")]
    assert requested_splits == ["validation"]


def test_eval_collection_closes_hf_stream(
    patch_hf_load_dataset: Callable[..., dict[str, int]],
) -> None:
    """Bounded eval collection releases its source iterator immediately."""
    record: dict[str, Any] = {}
    patch_hf_load_dataset(
        {"validation": [{"text": "val-a"}, {"text": "val-b"}]},
        record=record,
    )
    cfg = _base_cfg()

    load_or_create_eval_tokens(cfg, tokenizer=build_tokenizer(cfg))

    assert record["close_calls"] == 1


def test_failed_eval_collection_closes_hf_stream(
    patch_hf_load_dataset: Callable[..., dict[str, int]],
) -> None:
    """Eval collection errors do not leak the partially consumed source."""
    record: dict[str, Any] = {"fail_consumed": False}
    patch_hf_load_dataset(
        {"validation": [{"text": "val-a"}, {"text": "val-b"}]},
        fail_at=1,
        record=record,
    )
    base = _base_cfg()
    cfg = replace(base, data=replace(base.data, max_retries=0))

    with pytest.raises(RuntimeError, match="Failed to collect evaluation documents"):
        load_or_create_eval_tokens(cfg, tokenizer=build_tokenizer(cfg))

    assert record["close_calls"] == 1


@pytest.mark.parametrize(
    "failure",
    [FileNotFoundError("missing validation split"), PermissionError("authentication failed")],
)
def test_explicit_eval_split_failure_never_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    """Missing or inaccessible explicit eval data must not become train-set eval."""
    requested_splits: list[str] = []

    def _load_dataset(
        dataset: str,
        *,
        name: str,
        split: str,
        streaming: bool,
        revision: str | None,
    ) -> FakeHFIterable:
        _ = (dataset, name, streaming, revision)
        requested_splits.append(split)
        if split == "validation":
            raise failure
        return FakeHFIterable(items=[{"text": "training data must not be used"}])

    import datasets as hf_datasets

    monkeypatch.setattr(hf_datasets, "load_dataset", _load_dataset)
    cfg = _base_cfg()

    with pytest.raises(RuntimeError, match="never falls back"):
        load_or_create_eval_tokens(cfg, tokenizer=build_tokenizer(cfg), run_dir=tmp_path)

    assert requested_splits == ["validation"]
    assert not (tmp_path / "eval_tokens.json.gz").exists()


def test_positive_eval_sample_count_rejects_empty_source(
    tmp_path: Path, patch_hf_load_dataset: Callable[..., dict[str, int]]
) -> None:
    """An empty configured source must fail instead of silently disabling eval."""
    patch_hf_load_dataset({"validation": [], "train": [{"text": "unused"}]})
    cfg = _base_cfg()

    with pytest.raises(RuntimeError, match="collected zero documents"):
        load_or_create_eval_tokens(cfg, tokenizer=build_tokenizer(cfg), run_dir=tmp_path)

    assert not (tmp_path / "eval_tokens.json.gz").exists()


def test_eval_cache_manifest_records_actual_source_split(
    tmp_path: Path, patch_hf_load_dataset: Callable[..., dict[str, int]]
) -> None:
    """Pinned eval identity must record the split that actually supplied documents."""
    import gzip

    patch_hf_load_dataset(
        {
            "validation": [{"text": "val-a"}, {"text": "val-b"}],
            "train": [{"text": "train-a"}],
        }
    )
    cfg = _base_cfg()
    load_or_create_eval_tokens(cfg, tokenizer=build_tokenizer(cfg), run_dir=tmp_path)

    with gzip.open(tmp_path / "eval_tokens.json.gz", "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["manifest"]["source_split"] == "validation"
    assert payload["manifest"]["hf_revision"] == cfg.data.hf_revision


def test_eval_empty_when_disabled() -> None:
    """Eval should return empty list when max_eval_samples=0."""
    cfg = _base_cfg()
    cfg = replace(cfg, data=replace(cfg.data, max_eval_samples=0))
    tok = build_tokenizer(cfg)
    assert load_or_create_eval_tokens(cfg, tokenizer=tok) == []


def _local_eval_cfg(local_text: str = "eval persistence corpus\n") -> Config:
    """Local-text config for eval token cache tests.

    :param str local_text: Source text for the local backend.
    :return Config: Minimal configuration with 3 eval samples.
    """
    return Config(
        model=ModelConfig(backend="dummy", vocab_size=256, d_model=32, dropout=0.0),
        data=DataConfig(
            backend="local_text",
            repeat=True,
            local_text=local_text,
            max_eval_samples=3,
            tokenizer=TokenizerConfig(kind="byte", byte_offset=0, add_bos=False, add_eos=False),
        ),
    )


def test_eval_tokens_pinned_to_run_dir(tmp_path: Path) -> None:
    """The eval set is created once and reloaded from the run directory, so
    upstream source drift cannot silently change what a resumed run
    evaluates on (config-visible drift is resume-compat's job)."""
    cfg = _local_eval_cfg(local_text="first corpus")
    tok = build_tokenizer(cfg)
    tokens_first = load_or_create_eval_tokens(cfg, tokenizer=tok, run_dir=tmp_path)
    assert tokens_first
    assert (tmp_path / "eval_tokens.json.gz").exists()

    drifted = replace(cfg, data=replace(cfg.data, local_text="different corpus"))
    assert load_or_create_eval_tokens(drifted, tokenizer=tok, run_dir=tmp_path) == tokens_first
    # Sanity: without the cache the drifted source yields a different set.
    assert load_or_create_eval_tokens(drifted, tokenizer=tok) != tokens_first


def test_eval_tokens_cache_rejects_eval_knob_drift(tmp_path: Path) -> None:
    """Changing an eval-identity knob against an existing cache fails loudly."""
    cfg = _local_eval_cfg()
    tok = build_tokenizer(cfg)
    load_or_create_eval_tokens(cfg, tokenizer=tok, run_dir=tmp_path)

    drifted = replace(cfg, data=replace(cfg.data, max_eval_samples=5))
    with pytest.raises(RuntimeError, match="different eval settings"):
        load_or_create_eval_tokens(drifted, tokenizer=tok, run_dir=tmp_path)


def test_eval_cache_missing_on_resume_fails(tmp_path: Path) -> None:
    """A resume whose pinned eval set vanished must fail hard: recollecting
    silently would compare post-resume eval losses against a different token
    set. data.recreate_eval_cache is the explicit one-shot override."""
    cfg = _local_eval_cfg()
    tok = build_tokenizer(cfg)
    tokens = load_or_create_eval_tokens(cfg, tokenizer=tok, run_dir=tmp_path, resume=False)
    assert tokens
    (tmp_path / "eval_tokens.json.gz").unlink()

    with pytest.raises(RuntimeError, match="pinned eval set is missing"):
        load_or_create_eval_tokens(cfg, tokenizer=tok, run_dir=tmp_path, resume=True)

    override = replace(cfg, data=replace(cfg.data, recreate_eval_cache=True))
    recreated = load_or_create_eval_tokens(override, tokenizer=tok, run_dir=tmp_path, resume=True)
    assert recreated == tokens
    assert (tmp_path / "eval_tokens.json.gz").exists()


def test_run_resume_requires_eval_cache(tmp_path: Path) -> None:
    """The training entrypoint treats resume + missing eval cache as fatal."""
    run_dir = tmp_path / "run"
    cfg = _eval_run_cfg(
        run_dir,
        steps=1,
        checkpoint=CheckpointConfig(enabled=True, save_every=1, max_to_keep=2, async_save=False),
    )

    run(cfg, config_path=None, resume="none")
    cache = run_dir / "eval_tokens.json.gz"
    assert cache.exists()
    cache.unlink()

    with pytest.raises(RuntimeError, match="pinned eval set is missing"):
        run(cfg, config_path=None, resume="latest")


def test_failed_resume_does_not_persist_recreated_eval_cache(tmp_path: Path) -> None:
    """A rejected resume must not poison a missing eval cache.

    Recollection may use a changed source, tokenizer, or eval identity. The
    replacement therefore cannot enter the run directory until checkpoint
    compatibility accepts the current configuration.
    """
    run_dir = tmp_path / "run"
    cfg = _eval_run_cfg(
        run_dir,
        steps=1,
        checkpoint=CheckpointConfig(enabled=True, save_every=1, max_to_keep=2, async_save=False),
    )
    run(cfg, config_path=None, resume="none")

    cache = run_dir / "eval_tokens.json.gz"
    tok = build_tokenizer(cfg)
    expected_tokens = load_or_create_eval_tokens(cfg, tokenizer=tok, run_dir=run_dir)
    cache.unlink()

    incompatible = replace(
        cfg,
        data=replace(
            cfg.data,
            local_text="incompatible replacement corpus",
            recreate_eval_cache=True,
        ),
    )
    with pytest.raises(RuntimeError, match="local_text_hash"):
        run(incompatible, config_path=None, resume="latest")
    assert not cache.exists()

    correct = replace(cfg, data=replace(cfg.data, recreate_eval_cache=True))
    run(correct, config_path=None, resume="latest")
    assert cache.exists()
    assert load_or_create_eval_tokens(cfg, tokenizer=tok, run_dir=run_dir) == expected_tokens


def test_eval_tokens_cache_rejects_corruption(tmp_path: Path) -> None:
    """A cache whose content no longer matches its hash is refused."""
    import gzip

    cfg = _local_eval_cfg()
    tok = build_tokenizer(cfg)
    load_or_create_eval_tokens(cfg, tokenizer=tok, run_dir=tmp_path)

    path = tmp_path / "eval_tokens.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as f:
        payload = json.load(f)
    payload["tokens"][0][0] = int(payload["tokens"][0][0]) + 1
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f)

    with pytest.raises(RuntimeError, match="corrupt"):
        load_or_create_eval_tokens(cfg, tokenizer=tok, run_dir=tmp_path)


def test_null_eval_split_wires_complementary_content_partitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Null eval split selects held-out content and excludes it from train prompts."""
    train_items = [{"text": f"train-{index}"} for index in range(9)]
    seen_specs: list[Any] = []

    def _capture_stream(spec: Any) -> Any:
        """Capture the resolved stream spec without shadowing shuffle behavior."""
        seen_specs.append(spec)
        return iter(item["text"] for item in train_items)

    monkeypatch.setattr("chomp.data.pipeline.HFStreamingTextStream", _capture_stream)

    cfg = _base_cfg()
    cfg = replace(cfg, data=replace(cfg.data, hf_eval_split=None, shuffle=True, seed=0))
    tok = build_tokenizer(cfg)
    tokens = load_or_create_eval_tokens(cfg, tokenizer=tok)
    _ = build_generation_text_stream(cfg)

    assert tokens == [tok.encode("train-0"), tok.encode("train-1")]
    assert len(seen_specs) == 2
    assert seen_specs[0].content_partition == "eval"
    assert seen_specs[1].content_partition == "train"
    assert seen_specs[0].seed == cfg.data.seed
    assert seen_specs[0].shuffle_buffer_size == cfg.data.shuffle_buffer_size
    assert seen_specs[0].shuffle_buffer_bytes == cfg.data.shuffle_buffer_bytes
