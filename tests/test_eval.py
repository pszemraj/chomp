"""Evaluation tests consolidated by module."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from _pytest.logging import LogCaptureFixture

from chomp.config import (
    CheckpointConfig,
    Config,
    TokenizerConfig,
)
from chomp.data import (
    build_eval_iterator,
    build_tokenizer,
    load_generation_prompt_tokens,
    load_or_create_eval_tokens,
)
from chomp.train import run
from tests.helpers.config_factories import make_pipeline_cfg
from tests.helpers.hf_fakes import FakeHFIterable
from tests.helpers.io import read_jsonl


def _eval_cfg(
    run_dir: Path | None = None,
    *,
    backend: str = "local_text",
    steps: int = 1,
    grad_accum: int = 1,
    checkpoint: CheckpointConfig | None = None,
    **data_overrides: Any,
) -> Config:
    """Build the shared minimal configuration for evaluation tests.

    :param Path | None run_dir: Optional run directory.
    :param str backend: ``local_text`` or ``hf`` source.
    :param int steps: Training steps for run-level tests.
    :param int grad_accum: Training accumulation geometry.
    :param CheckpointConfig | None checkpoint: Optional checkpoint policy.
    :param data_overrides: Data fields specific to the behavior under test.
    :return Config: Minimal evaluation configuration.
    """
    data: dict[str, Any] = {
        "tokenizer": TokenizerConfig(kind="byte", byte_offset=0, add_bos=False, add_eos=False)
    }
    if backend == "hf":
        data.update(
            backend="hf",
            hf_dataset="dummy",
            hf_name="dummy",
            hf_split="train",
            hf_eval_split="validation",
            hf_revision=None,
            text_key="text",
            shuffle=False,
            repeat=False,
            max_eval_samples=2,
        )
    else:
        data.update(
            backend="local_text",
            repeat=True,
            local_text="abcdefghijklmnopqrstuvwxyz" * 4,
            max_eval_samples=3,
        )
    data.update(data_overrides)
    cfg = make_pipeline_cfg(seq_len=8, vocab_size=256, **data)
    return replace(
        cfg,
        train=replace(
            cfg.train,
            steps=steps,
            grad_accum=grad_accum,
            log_every=1000,
            eval_every=1,
        ),
        optim=replace(cfg.optim, lr=1e-3, weight_decay=0.0, grad_clip_norm=0.0),
        checkpoint=checkpoint or CheckpointConfig(enabled=False),
        debug=replace(cfg.debug, nan_check=True),
        logging=replace(cfg.logging, run_dir=None if run_dir is None else str(run_dir)),
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
    cfg = _eval_cfg(run_dir, steps=2)

    run(cfg, config_path=None, resume="none")

    assert calls["n"] == 1, f"eval iterator rebuilt {calls['n']} times for 2 evals"
    metrics_path = run_dir / cfg.logging.metrics_file
    rows = read_jsonl(metrics_path)
    eval_rows = [row for row in rows if row.get("eval_loss") not in (None, "")]
    assert len(eval_rows) == 2  # both evals produced a loss from the cached batches


def test_nonfinite_eval_disables_eval_before_metric_logging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    """A non-finite eval reduction disables eval without logging NaN."""
    run_dir = tmp_path / "run_nonfinite_eval"
    cfg = _eval_cfg(run_dir)

    def _nonfinite_eval_step(_params: Any, _batch: Any) -> tuple[float, int]:
        """Return a poisoned loss with a nonzero denominator.

        :param _params: Ignored model parameters.
        :param _batch: Ignored eval batch.
        :return tuple[float, int]: NaN loss sum and one valid token.
        """
        return float("nan"), 1

    monkeypatch.setattr(
        "chomp.train.make_eval_step",
        lambda *_args, **_kwargs: _nonfinite_eval_step,
    )

    with caplog.at_level(logging.WARNING, logger="chomp.train"):
        run(cfg, config_path=None, resume="none", dry_run=False)

    assert "non-finite loss sum" in caplog.text
    rows = read_jsonl(run_dir / cfg.logging.metrics_file)
    assert not any("eval_loss" in row for row in rows)


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
    cfg = _eval_cfg(run_dir, backend="hf", packing_mode=mode, max_eval_samples=8, **knobs)

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
    cfg = _eval_cfg(
        run_dir,
        backend="hf",
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


def test_eval_zero_valid_tokens_disables_eval(
    tmp_path: Path,
    patch_hf_load_dataset: Callable[..., dict[str, int]],
    caplog: LogCaptureFixture,
) -> None:
    """Entirely masked eval labels disable eval without logging a null loss.

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
    cfg = _eval_cfg(run_dir, backend="hf", packing_mode="sequential", max_eval_samples=64)

    with caplog.at_level(logging.WARNING, logger="chomp.train"):
        run(cfg, config_path=None, resume="none")
    assert "zero valid loss tokens" in caplog.text
    rows = read_jsonl(run_dir / cfg.logging.metrics_file)
    assert not any("eval_loss" in row for row in rows)


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
    cfg = _eval_cfg(backend="hf")
    tok = build_tokenizer(cfg)
    tokens = load_or_create_eval_tokens(cfg, tokenizer=tok)

    assert tokens == [tok.encode("val-a"), tok.encode("val-b")]
    assert requested_splits == ["validation"]


@pytest.mark.parametrize("fail_at", [None, 1], ids=["success", "failure"])
def test_eval_collection_always_closes_hf_stream(
    patch_hf_load_dataset: Callable[..., dict[str, int]], fail_at: int | None
) -> None:
    """Eval collection releases its source after success or failure."""
    record: dict[str, Any] = {}
    patch_hf_load_dataset(
        {"validation": [{"text": "val-a"}, {"text": "val-b"}]},
        fail_at=fail_at,
        record=record,
    )
    cfg = _eval_cfg(backend="hf")

    if fail_at is not None:
        with pytest.raises(RuntimeError, match="Failed to collect evaluation documents"):
            load_or_create_eval_tokens(cfg, tokenizer=build_tokenizer(cfg))
    else:
        load_or_create_eval_tokens(cfg, tokenizer=build_tokenizer(cfg))

    assert record["close_calls"] == 1


def test_malformed_eval_row_fails(
    patch_hf_load_dataset: Callable[..., dict[str, int]],
) -> None:
    """A non-string text field must fail evaluation collection."""
    patch_hf_load_dataset({"validation": [{"text": None}]})
    cfg = _eval_cfg(backend="hf")

    with pytest.raises(RuntimeError, match="Failed to collect evaluation documents"):
        load_or_create_eval_tokens(cfg, tokenizer=build_tokenizer(cfg))


@pytest.mark.parametrize(
    "failure",
    [FileNotFoundError("missing validation split"), PermissionError("authentication failed")],
)
def test_explicit_eval_split_failure_never_falls_back(
    monkeypatch: pytest.MonkeyPatch, failure: Exception
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
    cfg = _eval_cfg(backend="hf")

    with pytest.raises(RuntimeError, match="never falls back"):
        load_or_create_eval_tokens(cfg, tokenizer=build_tokenizer(cfg))

    assert requested_splits == ["validation"]


def test_positive_eval_sample_count_rejects_empty_source(
    patch_hf_load_dataset: Callable[..., dict[str, int]],
) -> None:
    """An empty configured source must fail instead of silently disabling eval."""
    patch_hf_load_dataset({"validation": [], "train": [{"text": "unused"}]})
    cfg = _eval_cfg(backend="hf")

    with pytest.raises(RuntimeError, match="collected zero documents"):
        load_or_create_eval_tokens(cfg, tokenizer=build_tokenizer(cfg))


def test_explicit_eval_split_ignores_document_shuffle(
    patch_hf_load_dataset: Callable[..., dict[str, int]],
) -> None:
    """Eval documents keep literal split order even when data.shuffle=true."""
    docs = [{"text": f"doc{i:02d}"} for i in range(16)]
    patch_hf_load_dataset({"train": [{"text": "unused"}], "validation": docs})
    cfg = _eval_cfg(
        backend="hf",
        max_eval_samples=8,
        shuffle=True,
        shuffle_buffer_size=4,
    )
    tok = build_tokenizer(cfg)

    tokens = load_or_create_eval_tokens(cfg, tokenizer=tok)

    assert tokens == [tok.encode(f"doc{i:02d}") for i in range(8)]


def test_eval_empty_when_disabled() -> None:
    """Eval should return empty list when max_eval_samples=0."""
    cfg = _eval_cfg(backend="hf")
    cfg = replace(cfg, data=replace(cfg.data, max_eval_samples=0))
    tok = build_tokenizer(cfg)
    assert load_or_create_eval_tokens(cfg, tokenizer=tok) == []


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

    cfg = _eval_cfg(backend="hf")
    cfg = replace(cfg, data=replace(cfg.data, hf_eval_split=None, shuffle=True, seed=0))
    tok = build_tokenizer(cfg)
    tokens = load_or_create_eval_tokens(cfg, tokenizer=tok)
    prompts = load_generation_prompt_tokens(cfg, tokenizer=tok, max_samples=2)

    assert tokens == [tok.encode("train-0"), tok.encode("train-1")]
    assert len(prompts) == 2
    assert len(seen_specs) == 2
    assert seen_specs[0].content_partition == "eval"
    assert seen_specs[1].content_partition == "train"
    assert seen_specs[1].shuffle is False
    assert seen_specs[0].seed == cfg.data.seed
    assert seen_specs[0].shuffle_buffer_size == cfg.data.shuffle_buffer_size
    assert seen_specs[0].shuffle_buffer_bytes == cfg.data.shuffle_buffer_bytes
