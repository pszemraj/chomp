"""Data pipeline tests consolidated by module."""

from __future__ import annotations

import json
from dataclasses import replace
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
from chomp.data.pack import MultipackPacker, TokenPacker
from chomp.data.pipeline import (
    BinPacker,
    ByteTokenizer,
    build_eval_iterator,
    build_train_iterator,
)
from chomp.train import run
from tests.helpers.hf_fakes import FakeHFIterable, FakeHFStateIterable

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


def test_multipack_packer_emits_segment_local_positions() -> None:
    """MultipackPacker should emit segment IDs and per-segment position IDs."""
    packer = MultipackPacker(
        seq_len=8,
        add_bos=False,
        add_eos=False,
        bos_id=1,
        eos_id=2,
        max_doc_tokens=None,
        bins_per_pack=1,
        group_docs=2,
        max_docs_per_bin=None,
        pad_id=0,
    )

    packer.add_document([10, 11, 12])
    packer.add_document([20, 21])
    assert packer.can_pop()
    toks, segs, pos = packer.pop_seq_with_metadata()

    np.testing.assert_array_equal(toks[:5], np.asarray([10, 11, 12, 20, 21], dtype=np.int32))
    np.testing.assert_array_equal(segs[:5], np.asarray([1, 1, 1, 2, 2], dtype=np.int32))
    np.testing.assert_array_equal(pos[:5], np.asarray([0, 1, 2, 0, 1], dtype=np.int32))
    assert np.all(segs[5:] == 0)
    assert np.all(pos[5:] == 0)


def test_multipack_packer_state_roundtrip() -> None:
    """Multipack packer state should roundtrip via get/set_state."""
    packer = MultipackPacker(
        seq_len=8,
        add_bos=False,
        add_eos=False,
        bos_id=1,
        eos_id=2,
        max_doc_tokens=None,
        bins_per_pack=1,
        group_docs=2,
        max_docs_per_bin=None,
        pad_id=0,
    )
    packer.add_document([31, 32, 33])
    packer.add_document([41, 42])

    state = packer.get_state()
    seq_a = packer.pop_seq_with_metadata()

    restored = MultipackPacker(
        seq_len=8,
        add_bos=False,
        add_eos=False,
        bos_id=1,
        eos_id=2,
        max_doc_tokens=None,
        bins_per_pack=1,
        group_docs=2,
        max_docs_per_bin=None,
        pad_id=0,
    )
    restored.set_state(state)
    seq_b = restored.pop_seq_with_metadata()

    np.testing.assert_array_equal(seq_a[0], seq_b[0])
    np.testing.assert_array_equal(seq_a[1], seq_b[1])
    np.testing.assert_array_equal(seq_a[2], seq_b[2])


def test_token_packer_legacy_state_normalizes_large_segment_ids() -> None:
    """TokenPacker should accept legacy large segment IDs and preserve boundaries."""
    packer = TokenPacker(
        seq_len=8,
        add_bos=False,
        add_eos=False,
        bos_id=1,
        eos_id=2,
        max_doc_tokens=None,
    )

    legacy_state = {
        "remaining_tokens": [10, 11, 12, 20, 21, 30, 31, 32],
        "remaining_segments": [
            2_147_483_600,
            2_147_483_600,
            2_147_483_600,
            2_147_483_601,
            2_147_483_601,
            2_147_483_602,
            2_147_483_602,
            2_147_483_602,
        ],
        "next_segment_id": 2_147_483_603,
    }
    packer.set_state(legacy_state)

    assert packer.can_pop()
    _, segs = packer.pop_seq_with_segments()
    assert segs.dtype == np.int32
    assert np.all(segs > 0)

    boundary = segs[1:] != segs[:-1]
    np.testing.assert_array_equal(
        boundary,
        np.asarray([False, False, True, False, True, False, False], dtype=bool),
    )

    # Adding a new document after restore should remain safe and deterministic.
    packer.add_document([41, 42, 43, 44, 45, 46, 47, 48])
    seq2, segs2 = packer.pop_seq_with_segments()
    assert seq2.shape == (8,)
    np.testing.assert_array_equal(segs2, np.ones((8,), dtype=np.int32))

    state = packer.get_state()
    assert int(state["next_segment_id"]) in (1, 2)


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


def _window_shuffle_cfg(*, window: int, repeat: bool = False) -> Config:
    """Build an HF-backed config for window-shuffle tests.

    Each fake document tokenizes (byte, offset 4, BOS/EOS) to exactly seq_len=8,
    so every packed row is one whole document and row identity is readable from
    its second token.

    :param int window: data.window_shuffle_windows value.
    :param bool repeat: Whether to repeat the stream.
    :return Config: Test configuration.
    """
    return Config(
        model=ModelConfig(backend="dummy", vocab_size=512, d_model=32, dropout=0.0),
        data=DataConfig(
            backend="hf",
            hf_dataset="dummy",
            hf_name="dummy",
            hf_split="train",
            text_key="text",
            shuffle=False,
            shuffle_buffer_size=8,
            seed=0,
            repeat=repeat,
            packing_mode="sequential",
            mask_boundary_loss=True,
            train_on_eos=True,
            window_shuffle_windows=window,
            tokenizer=TokenizerConfig(kind="byte", byte_offset=4, add_bos=True, add_eos=True),
        ),
        train=TrainConfig(
            steps=1,
            batch_size=2,
            seq_len=8,
            grad_accum=1,
            jit=False,
            deterministic=True,
            allow_cpu=True,
        ),
        optim=OptimConfig(warmup_steps=0),
    )


def _distinct_docs(count: int) -> list[dict[str, str]]:
    """Create fake HF items with distinct, identifiable 6-char texts.

    :param int count: Number of documents.
    :return list[dict[str, str]]: Items usable by FakeHFIterable.
    """
    return [{"text": chr(ord("A") + i % 50) * 6} for i in range(count)]


def _row_doc_tokens(batch: Batch) -> list[int]:
    """Extract each row's document-identifying token (position 1, after BOS).

    :param Batch batch: Batch of packed rows.
    :return list[int]: One token value per row.
    """
    rows = np.asarray(batch.input_ids).reshape(-1, batch.input_ids.shape[-1])
    return [int(r[1]) for r in rows]


def test_window_shuffle_disabled_matches_unshuffled(monkeypatch: pytest.MonkeyPatch) -> None:
    """window_shuffle_windows=0 must preserve raw packer output order."""
    items = _distinct_docs(20)

    def _load_dataset(dataset: str, *, name: str, split: str, streaming: bool) -> FakeHFIterable:
        _ = (dataset, name, split, streaming)
        return FakeHFIterable(items=items)

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", _load_dataset)

    cfg = _window_shuffle_cfg(window=0)
    it = build_train_iterator(cfg)
    seen = _row_doc_tokens(next(it)) + _row_doc_tokens(next(it))

    expected = [ord(item["text"][0]) + 4 for item in items[: len(seen)]]
    assert seen == expected


def test_window_shuffle_permutes_within_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """W>0 must permute rows within each window without loss or duplication."""
    items = _distinct_docs(20)

    def _load_dataset(dataset: str, *, name: str, split: str, streaming: bool) -> FakeHFIterable:
        _ = (dataset, name, split, streaming)
        return FakeHFIterable(items=items)

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", _load_dataset)

    window = 8
    cfg = _window_shuffle_cfg(window=window)
    it = build_train_iterator(cfg)
    seen = []
    for _ in range(window // 2):  # batch_size=2, grad_accum=1 -> 2 rows per batch
        seen.extend(_row_doc_tokens(next(it)))

    expected_window = [ord(item["text"][0]) + 4 for item in items[:window]]
    assert sorted(seen) == sorted(expected_window)  # same multiset: nothing lost/duplicated
    assert seen != expected_window  # order actually changed


def test_eval_iterator_never_shuffles() -> None:
    """Eval batches must come out in strict document order regardless of W."""
    cfg = _window_shuffle_cfg(window=4096)
    tokens = [[100 + i] * 6 for i in range(8)]
    tokenizer = ByteTokenizer(byte_offset=4)
    it = build_eval_iterator(cfg, tokens=tokens, tokenizer=tokenizer)

    seen = _row_doc_tokens(next(it)) + _row_doc_tokens(next(it))
    assert seen == [100, 101, 102, 103]


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

    labels = np.asarray(batch.labels)
    attn = np.asarray(batch.attention_mask, dtype=bool)
    valid_loss = labels[..., 1:] != -100
    valid_loss = valid_loss & attn[..., 1:]
    assert stats["loss_tokens_host"] == int(np.count_nonzero(valid_loss))

    segs_all = np.asarray(batch.segment_ids)
    boundary = (
        (segs_all[..., 1:] != segs_all[..., :-1])
        & (segs_all[..., 1:] > 0)
        & (segs_all[..., :-1] > 0)
    )
    assert stats["boundary_transitions"] == int(np.count_nonzero(boundary))

    flat_segs = segs_all.reshape(-1, segs_all.shape[-1])
    has_tokens = np.any(flat_segs > 0, axis=1)
    seq_boundary = (
        (flat_segs[:, 1:] != flat_segs[:, :-1]) & (flat_segs[:, 1:] > 0) & (flat_segs[:, :-1] > 0)
    )
    docs_per_seq = np.where(has_tokens, 1 + seq_boundary.sum(axis=1), 0).astype(np.int32)
    assert stats["docs_per_seq_mean"] == float(np.mean(docs_per_seq))
    assert stats["docs_per_seq_min"] == int(np.min(docs_per_seq))
    assert stats["docs_per_seq_max"] == int(np.max(docs_per_seq))


def test_pipeline_multipack_position_ids_and_stats() -> None:
    """Multipack mode should emit per-segment position IDs and packing stats."""
    cfg = Config(
        model=ModelConfig(backend="dummy", vocab_size=512, d_model=32, dropout=0.0),
        data=DataConfig(
            backend="local_text",
            repeat=True,
            local_text="hi",
            packing_mode="multipack",
            packing_group_docs=4,
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
    segs = np.asarray(batch.segment_ids[0, 0], dtype=np.int32)
    pos = np.asarray(batch.position_ids[0, 0], dtype=np.int32)
    attn = np.asarray(batch.attention_mask[0, 0], dtype=bool)

    assert np.array_equal(batch.attention_mask, batch.segment_ids > 0)
    assert np.all(pos[~attn] == 0)
    for idx in range(int(segs.size)):
        if segs[idx] <= 0:
            continue
        if idx == 0 or segs[idx] != segs[idx - 1]:
            assert pos[idx] == 0
        else:
            assert pos[idx] == pos[idx - 1] + 1

    stats = it.get_stats()
    assert stats["packing_mode"] == "multipack"
    assert stats["packing_capacity"] == batch.segment_ids.size


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

    def _load_dataset(dataset: str, *, name: str, split: str, streaming: bool) -> FakeHFIterable:
        """Mock load_dataset returning fake iterable.

        :param str dataset: Dataset name.
        :param str name: Config name.
        :param str split: Split name.
        :param bool streaming: Streaming flag.
        :return FakeHFIterable: Fake dataset iterable.
        """
        _ = (dataset, name, split, streaming)
        return FakeHFIterable(items=items)

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


@pytest.mark.slow
def test_hf_state_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    """HF stream should resume to same position after state roundtrip."""
    items = [{"text": "alpha"}, {"text": "bravo"}, {"text": "charlie"}]

    def _load_dataset(
        dataset: str, *, name: str, split: str, streaming: bool
    ) -> FakeHFStateIterable:
        """Mock load_dataset returning fake iterable.

        :param str dataset: Dataset name.
        :param str name: Config name.
        :param str split: Split name.
        :param bool streaming: Streaming flag.
        :return FakeHFStateIterable: Fake dataset iterable.
        """
        _ = (dataset, name, split, streaming)
        return FakeHFStateIterable(items=items)

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
    ) -> FakeHFStateIterable:
        """Mock load_dataset with failure injection.

        :param str dataset: Dataset name.
        :param str name: Config name.
        :param str split: Split name.
        :param bool streaming: Streaming flag.
        :return FakeHFStateIterable: Fake dataset iterable.
        """
        _ = (dataset, name, split, streaming)
        record["builds"] += 1
        return FakeHFStateIterable(items=items, fail_at=1, record=record)

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
