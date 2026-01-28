"""Data pipeline tests consolidated by module."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from chomp.config import (
    CheckpointConfig,
    Config,
    DataConfig,
    LoggingConfig,
    ModelConfig,
    OptimConfig,
    TokenizerConfig,
    TrainConfig,
)
from chomp.data.hf import HFStreamingTextStream, HFStreamSpec
from chomp.data.pipeline import BinPacker, ByteTokenizer, build_train_iterator
from chomp.train import run

if TYPE_CHECKING:
    from chomp.types import Batch


def _doc(token: int, length: int) -> list[int]:
    """Create a document of repeated tokens.

    :param int token: Token value to repeat.
    :param int length: Number of repetitions.
    :return list[int]: Token list of length ``length``.
    """
    return [token] * length


def test_bin_packer_packs_multiple_docs() -> None:
    """Bin packer should combine multiple documents into packed bins."""
    packer = BinPacker(
        seq_len=8,
        add_bos=False,
        add_eos=False,
        bos_id=1,
        eos_id=2,
        max_doc_tokens=None,
        bins_per_pack=2,
        buffer_docs=2,
        max_docs_per_bin=None,
        pad_id=0,
    )

    for tok, length in [(10, 6), (11, 2), (12, 6), (13, 2)]:
        packer.add_document(_doc(tok, length))

    assert packer.can_pop()
    seq1, seg1 = packer.pop_seq_with_segments()
    seq2, seg2 = packer.pop_seq_with_segments()

    assert seq1.shape == (8,)
    assert seq2.shape == (8,)

    for seq, segs in [(seq1, seg1), (seq2, seg2)]:
        pad_mask = seq == 0
        if np.any(pad_mask):
            assert np.all(segs[pad_mask] == 0)

        unique = np.unique(segs[segs > 0])
        assert unique.size >= 2


def test_bin_packer_state_roundtrip() -> None:
    """Bin packer state should roundtrip correctly via get/set_state."""
    packer = BinPacker(
        seq_len=8,
        add_bos=False,
        add_eos=False,
        bos_id=1,
        eos_id=2,
        max_doc_tokens=None,
        bins_per_pack=2,
        buffer_docs=2,
        max_docs_per_bin=None,
        pad_id=0,
    )

    for tok, length in [(21, 6), (22, 2), (23, 6), (24, 2)]:
        packer.add_document(_doc(tok, length))

    _ = packer.pop_seq_with_segments()
    state = packer.get_state()
    seq_b = packer.pop_seq_with_segments()

    restored = BinPacker(
        seq_len=8,
        add_bos=False,
        add_eos=False,
        bos_id=1,
        eos_id=2,
        max_doc_tokens=None,
        bins_per_pack=2,
        buffer_docs=2,
        max_docs_per_bin=None,
        pad_id=0,
    )
    restored.set_state(state)
    seq_b2 = restored.pop_seq_with_segments()

    np.testing.assert_array_equal(seq_b[0], seq_b2[0])
    np.testing.assert_array_equal(seq_b[1], seq_b2[1])


def test_grain_iterator_state_roundtrip() -> None:
    """Grain iterator should produce same batches after state restore."""
    cfg = Config(
        model=ModelConfig(backend="dummy", vocab_size=512, d_model=32, dropout=0.0),
        data=DataConfig(
            backend="local_text",
            repeat=True,
            local_text="hi",
            packing_mode="bin",
            packing_buffer_docs=4,
            packing_max_docs_per_bin=None,
            mask_boundary_loss=True,
            train_on_eos=True,
            grain_prefetch=2,
            tokenizer=TokenizerConfig(kind="byte", byte_offset=4, add_bos=True, add_eos=True),
        ),
        train=TrainConfig(
            steps=1,
            batch_size=1,
            seq_len=8,
            grad_accum=1,
            jit=False,
            deterministic=True,
            allow_cpu=True,
        ),
        optim=OptimConfig(warmup_steps=0),
    )

    it = build_train_iterator(cfg)
    _ = next(it)
    stats = it.get_stats()
    assert stats.get("packing_mode") == cfg.data.packing_mode
    state = it.get_state()

    next_a = next(it)

    it2 = build_train_iterator(cfg)
    it2.set_state(state)
    next_b = next(it2)

    np.testing.assert_array_equal(next_a.input_ids, next_b.input_ids)
    np.testing.assert_array_equal(next_a.labels, next_b.labels)
    np.testing.assert_array_equal(next_a.attention_mask, next_b.attention_mask)
    np.testing.assert_array_equal(next_a.segment_ids, next_b.segment_ids)


def test_grain_iterator_stats_disabled_with_device_put() -> None:
    """Packing stats should be empty when device_put=True."""
    cfg = Config(
        model=ModelConfig(backend="dummy", vocab_size=512, d_model=32, dropout=0.0),
        data=DataConfig(
            backend="local_text",
            repeat=True,
            local_text="hi",
            packing_mode="bin",
            packing_buffer_docs=4,
            packing_max_docs_per_bin=None,
            mask_boundary_loss=True,
            train_on_eos=True,
            grain_prefetch=0,
            device_put=True,
            tokenizer=TokenizerConfig(kind="byte", byte_offset=4, add_bos=True, add_eos=True),
        ),
        train=TrainConfig(
            steps=1,
            batch_size=1,
            seq_len=8,
            grad_accum=1,
            jit=False,
            deterministic=True,
            allow_cpu=True,
        ),
        optim=OptimConfig(warmup_steps=0),
    )

    it = build_train_iterator(cfg)
    _ = next(it)
    assert it.get_stats() == {}


def test_labels_align_with_inputs_except_masked() -> None:
    """Labels should match inputs except where masked with -100."""
    cfg = Config(
        model=ModelConfig(backend="dummy", vocab_size=512, d_model=32, dropout=0.0),
        data=DataConfig(
            backend="local_text",
            repeat=True,
            local_text="hi",
            mask_boundary_loss=True,
            train_on_eos=False,
            tokenizer=TokenizerConfig(kind="byte", byte_offset=4, add_bos=True, add_eos=True),
        ),
        train=TrainConfig(
            steps=1,
            batch_size=1,
            seq_len=8,
            grad_accum=2,
            jit=False,
            deterministic=True,
            allow_cpu=True,
        ),
        optim=OptimConfig(warmup_steps=0),
    )

    it = build_train_iterator(cfg)
    batch = next(it)

    labels = np.asarray(batch.labels)
    inputs = np.asarray(batch.input_ids)
    mask = labels != -100

    assert mask.any()
    assert np.all(labels[mask] == inputs[mask])


def test_pipeline_segment_ids_multiple_docs() -> None:
    """Pipeline should emit multiple segment IDs and mask boundaries."""
    cfg = Config(
        model=ModelConfig(backend="dummy", vocab_size=512, d_model=32, dropout=0.0),
        data=DataConfig(
            backend="local_text",
            repeat=True,
            local_text="hi",
            mask_boundary_loss=True,
            train_on_eos=True,
            tokenizer=TokenizerConfig(kind="byte", byte_offset=4, add_bos=True, add_eos=True),
        ),
        train=TrainConfig(
            steps=1,
            batch_size=1,
            seq_len=8,
            grad_accum=1,
            jit=False,
            deterministic=True,
            allow_cpu=True,
        ),
        optim=OptimConfig(warmup_steps=0),
    )

    it = build_train_iterator(cfg)
    batch = next(it)
    segs = batch.segment_ids[0, 0]
    unique = np.unique(segs)
    assert unique.size >= 2
    assert np.all(unique > 0)

    boundary = segs[1:] != segs[:-1]
    assert boundary.any()
    masked_labels = batch.labels[0, 0][1:][boundary]
    assert np.all(masked_labels == -100)


def test_boundary_loss_mask_toggle() -> None:
    """With mask_boundary_loss=False, boundary labels should not be masked."""
    cfg = Config(
        model=ModelConfig(backend="dummy", vocab_size=512, d_model=32, dropout=0.0),
        data=DataConfig(
            backend="local_text",
            repeat=True,
            local_text="hi",
            mask_boundary_loss=False,
            train_on_eos=True,
            tokenizer=TokenizerConfig(kind="byte", byte_offset=4, add_bos=True, add_eos=True),
        ),
        train=TrainConfig(
            steps=1,
            batch_size=1,
            seq_len=8,
            grad_accum=1,
            jit=False,
            deterministic=True,
            allow_cpu=True,
        ),
        optim=OptimConfig(warmup_steps=0),
    )

    it = build_train_iterator(cfg)
    batch = next(it)
    segs = batch.segment_ids[0, 0]
    boundary = segs[1:] != segs[:-1]
    assert boundary.any()
    labels_at_boundary = batch.labels[0, 0][1:][boundary]
    assert np.all(labels_at_boundary != -100)


def test_pipeline_bin_packing_segment_ids() -> None:
    """Bin packing should produce multiple segments with packing stats."""
    cfg = Config(
        model=ModelConfig(backend="dummy", vocab_size=512, d_model=32, dropout=0.0),
        data=DataConfig(
            backend="local_text",
            repeat=True,
            local_text="hi",
            packing_mode="bin",
            packing_buffer_docs=4,
            packing_max_docs_per_bin=None,
            mask_boundary_loss=True,
            train_on_eos=True,
            tokenizer=TokenizerConfig(kind="byte", byte_offset=4, add_bos=True, add_eos=True),
        ),
        train=TrainConfig(
            steps=1,
            batch_size=1,
            seq_len=8,
            grad_accum=1,
            jit=False,
            deterministic=True,
            allow_cpu=True,
        ),
        optim=OptimConfig(warmup_steps=0),
    )

    it = build_train_iterator(cfg)
    batch = next(it)
    segs = batch.segment_ids[0, 0]
    unique = np.unique(segs[segs > 0])
    assert unique.size >= 2
    assert np.array_equal(batch.attention_mask, batch.segment_ids > 0)

    stats = it.get_stats()
    assert stats["packing_mode"] == "bin"
    assert stats["packing_capacity"] == batch.segment_ids.size
    expected_util = float(np.count_nonzero(batch.attention_mask) / batch.attention_mask.size)
    assert stats["packing_utilization"] == expected_util


@dataclass
class _FakeHFPipelineIterable:
    """Mock HF iterable dataset for testing."""

    items: list[dict[str, Any]]
    index: int = 0

    def select_columns(self, _columns: list[str]) -> _FakeHFPipelineIterable:
        """Return self (columns not used in tests).

        :param list[str] _columns: Column names to select.
        :return _FakeHFPipelineIterable: Self for chaining.
        """
        return self

    def shuffle(self, *, seed: int, buffer_size: int) -> _FakeHFPipelineIterable:
        """Return self (shuffle not used in tests).

        :param int seed: Shuffle seed.
        :param int buffer_size: Shuffle buffer size.
        :return _FakeHFPipelineIterable: Self for chaining.
        """
        _ = (seed, buffer_size)
        return self

    def state_dict(self) -> dict[str, Any]:
        """Return iterator state.

        :return dict[str, Any]: State dictionary.
        """
        return {"index": int(self.index)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore iterator state."""
        self.index = int(state["index"])

    def __iter__(self) -> _FakeHFPipelineIterable:
        return self

    def __next__(self) -> dict[str, Any]:
        if self.index >= len(self.items):
            raise StopIteration
        item = self.items[self.index]
        self.index += 1
        return item


def test_hf_pipeline_segment_ids_and_label_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HF pipeline should emit segment IDs and mask labels at boundaries."""
    items = [
        {"text": "hi"},
        {"text": "ok"},
        {"text": "yo"},
        {"text": "sup"},
    ]

    def _load_dataset(
        dataset: str, *, name: str, split: str, streaming: bool
    ) -> _FakeHFPipelineIterable:
        """Mock load_dataset returning fake iterable.

        :param str dataset: Dataset name.
        :param str name: Config name.
        :param str split: Split name.
        :param bool streaming: Streaming flag.
        :return _FakeHFPipelineIterable: Fake dataset iterable.
        """
        _ = (dataset, name, split, streaming)
        return _FakeHFPipelineIterable(items=items)

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", _load_dataset)

    cfg = Config(
        model=ModelConfig(backend="dummy", vocab_size=256, d_model=32, dropout=0.0),
        data=DataConfig(
            backend="hf",
            hf_dataset="dummy",
            hf_name="dummy",
            hf_split="train",
            text_key="text",
            shuffle=False,
            shuffle_buffer_size=8,
            seed=0,
            repeat=False,
            mask_boundary_loss=True,
            train_on_eos=True,
            tokenizer=TokenizerConfig(kind="byte", byte_offset=4, add_bos=True, add_eos=True),
        ),
        train=TrainConfig(
            steps=1,
            batch_size=1,
            seq_len=8,
            grad_accum=1,
            jit=False,
            deterministic=True,
            allow_cpu=True,
        ),
        optim=OptimConfig(warmup_steps=0),
    )

    it = build_train_iterator(cfg)
    batch = next(it)

    segs = batch.segment_ids[0, 0]
    unique = np.unique(segs)
    assert unique.size >= 2
    assert np.all(unique > 0)

    boundary = segs[1:] != segs[:-1]
    assert boundary.any()
    masked_labels = batch.labels[0, 0][1:][boundary]
    assert np.all(masked_labels == -100)


@dataclass
class _FakeHFStateIterable:
    """Mock HF iterable dataset with optional failure injection."""

    items: list[dict[str, Any]]
    index: int = 0
    fail_at: int | None = None
    record: dict[str, Any] | None = None

    def select_columns(self, _columns: list[str]) -> _FakeHFStateIterable:
        """Return self (columns not used in tests).

        :param list[str] _columns: Column names to select.
        :return _FakeHFStateIterable: Self for chaining.
        """
        return self

    def shuffle(self, *, seed: int, buffer_size: int) -> _FakeHFStateIterable:
        """Return self (shuffle not used in tests).

        :param int seed: Shuffle seed.
        :param int buffer_size: Shuffle buffer size.
        :return _FakeHFStateIterable: Self for chaining.
        """
        _ = (seed, buffer_size)
        return self

    def state_dict(self) -> dict[str, Any]:
        """Return iterator state.

        :return dict[str, Any]: State dictionary.
        """
        return {"index": int(self.index)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore iterator state and record load calls."""
        self.index = int(state["index"])
        if self.record is not None:
            self.record["load_calls"] = self.record.get("load_calls", 0) + 1
            self.record["last_loaded"] = dict(state)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return _FakeHFStateIterator(self)


class _FakeHFStateIterator:
    """Mock HF iterator with optional failure injection."""

    def __init__(self, ds: _FakeHFStateIterable) -> None:
        """Initialize iterator from dataset."""
        self._ds = ds
        self._i = int(ds.index)

    def __iter__(self) -> _FakeHFStateIterator:
        return self

    def __next__(self) -> dict[str, Any]:
        if self._ds.fail_at is not None and self._i == self._ds.fail_at:
            rec = self._ds.record
            if rec is None or not rec.get("fail_consumed", False):
                if rec is not None:
                    rec["fail_consumed"] = True
                raise RuntimeError("transient failure")
        if self._i >= len(self._ds.items):
            raise StopIteration
        item = self._ds.items[self._i]
        self._i += 1
        self._ds.index = self._i
        return item


@pytest.mark.slow
def test_hf_state_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    """HF stream should resume to same position after state roundtrip."""
    items = [{"text": "alpha"}, {"text": "bravo"}, {"text": "charlie"}]

    def _load_dataset(
        dataset: str, *, name: str, split: str, streaming: bool
    ) -> _FakeHFStateIterable:
        """Mock load_dataset returning fake iterable.

        :param str dataset: Dataset name.
        :param str name: Config name.
        :param str split: Split name.
        :param bool streaming: Streaming flag.
        :return _FakeHFStateIterable: Fake dataset iterable.
        """
        _ = (dataset, name, split, streaming)
        return _FakeHFStateIterable(items=items)

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", _load_dataset)

    spec = HFStreamSpec(
        dataset="dummy",
        name="dummy",
        split="train",
        text_key="text",
        shuffle=False,
        shuffle_buffer_size=8,
        seed=0,
        repeat=False,
        max_retries=0,
        retry_delay_sec=0.0,
        state_update_interval=2,
    )

    stream = HFStreamingTextStream(spec)
    _ = next(stream)
    _ = next(stream)
    state = stream.get_state()
    expected = next(stream)

    resumed = HFStreamingTextStream(spec)
    resumed.set_state(state)
    assert next(resumed) == expected


@pytest.mark.slow
def test_hf_retry_rebuild_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    """HF stream should recover from transient failure via state restore."""
    items = [{"text": "alpha"}, {"text": "bravo"}, {"text": "charlie"}]
    record: dict[str, Any] = {"builds": 0, "fail_consumed": False}

    def _load_dataset(
        dataset: str, *, name: str, split: str, streaming: bool
    ) -> _FakeHFStateIterable:
        """Mock load_dataset with failure injection.

        :param str dataset: Dataset name.
        :param str name: Config name.
        :param str split: Split name.
        :param bool streaming: Streaming flag.
        :return _FakeHFStateIterable: Fake dataset iterable.
        """
        _ = (dataset, name, split, streaming)
        record["builds"] += 1
        return _FakeHFStateIterable(items=items, fail_at=1, record=record)

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", _load_dataset)

    spec = HFStreamSpec(
        dataset="dummy",
        name="dummy",
        split="train",
        text_key="text",
        shuffle=False,
        shuffle_buffer_size=8,
        seed=0,
        repeat=False,
        max_retries=1,
        retry_delay_sec=0.0,
        state_update_interval=1,
    )

    stream = HFStreamingTextStream(spec)
    assert next(stream) == "alpha"
    assert next(stream) == "bravo"

    assert record["builds"] >= 2
    assert record.get("load_calls", 0) >= 1
    assert record.get("last_loaded") == {"index": 1}


def _batch_arrays(batch: Batch) -> tuple:
    """Extract arrays from batch for comparison.

    :param Batch batch: Batch to extract from.
    :return tuple: Tuple of (input_ids, labels, attention_mask, segment_ids).
    """
    return batch.input_ids, batch.labels, batch.attention_mask, batch.segment_ids


def test_packer_alignment_after_restore() -> None:
    """Restored iterator should produce same batches as continued iterator."""
    cfg = Config(
        model=ModelConfig(backend="dummy", vocab_size=512, d_model=32, dropout=0.0),
        data=DataConfig(
            backend="local_text",
            repeat=True,
            local_text="abcde",
            tokenizer=TokenizerConfig(kind="byte", byte_offset=4, add_bos=True, add_eos=True),
        ),
        train=TrainConfig(
            steps=1,
            batch_size=1,
            seq_len=8,
            grad_accum=1,
            jit=False,
            deterministic=True,
            allow_cpu=True,
        ),
        optim=OptimConfig(warmup_steps=0),
    )

    it = build_train_iterator(cfg)
    _ = next(it)
    state = it.get_state()

    remaining = state.get("packer", {}).get("remaining_tokens")
    assert remaining, "expected non-empty packer buffer for alignment test"

    cont = [_batch_arrays(next(it)) for _ in range(3)]

    it2 = build_train_iterator(cfg)
    it2.set_state(state)
    resumed = [_batch_arrays(next(it2)) for _ in range(3)]

    for batch_a, batch_b in zip(cont, resumed, strict=True):
        for arr_a, arr_b in zip(batch_a, batch_b, strict=True):
            np.testing.assert_array_equal(arr_a, arr_b)


def test_train_on_eos_false_masks_eos_labels() -> None:
    """With train_on_eos=False, EOS token labels should be masked to -100."""
    cfg = Config(
        model=ModelConfig(backend="dummy", vocab_size=512, d_model=32, dropout=0.0),
        data=DataConfig(
            backend="local_text",
            repeat=True,
            local_text="hi",
            mask_boundary_loss=False,
            train_on_eos=False,
            tokenizer=TokenizerConfig(kind="byte", byte_offset=4, add_bos=True, add_eos=True),
        ),
        train=TrainConfig(
            steps=1,
            batch_size=1,
            seq_len=8,
            grad_accum=1,
            jit=False,
            deterministic=True,
            allow_cpu=True,
        ),
        optim=OptimConfig(warmup_steps=0),
    )

    it = build_train_iterator(cfg)
    batch = next(it)

    input_ids = batch.input_ids[0, 0]
    labels = batch.labels[0, 0]

    eos_id = int(cfg.model.eos_token_id)
    eos_positions = input_ids[1:] == eos_id
    assert eos_positions.any()
    masked = labels[1:][eos_positions]
    assert np.all(masked == -100)


def test_byte_tokenizer_roundtrip() -> None:
    """Byte tokenizer should round-trip ASCII text."""
    tok = ByteTokenizer(byte_offset=0)
    text = "hello world"
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_byte_tokenizer_skips_special_tokens() -> None:
    """Special tokens should be skipped when requested."""
    tok = ByteTokenizer(byte_offset=4)
    ids = [0, 1] + tok.encode("hi")
    assert tok.decode(ids) == "hi"


def test_tokenizer_snapshot_saved(tmp_path: Path) -> None:
    """Training should save tokenizer.json with kind metadata."""
    base = Config(
        model=ModelConfig(backend="dummy", vocab_size=256, d_model=32, dropout=0.0),
        data=DataConfig(
            backend="local_text",
            repeat=True,
            local_text="Tokenizer snapshot test.\n",
            tokenizer=TokenizerConfig(kind="byte", byte_offset=0, add_bos=False, add_eos=False),
        ),
        train=TrainConfig(
            seed=0,
            steps=1,
            batch_size=1,
            seq_len=16,
            grad_accum=1,
            jit=False,
            deterministic=True,
            allow_cpu=True,
            log_every=1000,
        ),
        optim=OptimConfig(warmup_steps=0),
        checkpoint=CheckpointConfig(enabled=False),
        logging=LoggingConfig(
            project="chomp", run_dir=None, metrics_file="metrics.jsonl", level="INFO"
        ),
    )

    run_dir = tmp_path / "run"
    cfg = replace(base, logging=replace(base.logging, run_dir=str(run_dir)))
    run(cfg, config_path=None, resume="none")

    tok_file = run_dir / "tokenizer" / "tokenizer.json"
    assert tok_file.exists()

    data = json.loads(tok_file.read_text(encoding="utf-8"))
    assert data["kind"] == "byte"
