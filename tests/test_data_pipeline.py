"""Data pipeline tests consolidated by module."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from chomp.config import (
    Config,
    DataConfig,
    TokenizerConfig,
    resolve_window_shuffle_rows,
)
from chomp.data.grain import (
    _batch_segment_stats,
    _TrainSequenceIterDataset,
)
from chomp.data.hf import HFStreamingTextStream, HFStreamSpec
from chomp.data.pack import FFDPacker, FFDPackerState, TokenPacker
from chomp.data.pipeline import (
    ByteTokenizer,
    ZeroLossTokensError,
    _SequenceProducer,
    build_eval_iterator,
    build_train_iterator,
    effective_window_shuffle_seed,
)
from tests.helpers.config_factories import make_pipeline_cfg

if TYPE_CHECKING:
    from collections.abc import Callable

    from chomp.types import Batch


def _doc(token: int, length: int) -> list[int]:
    """Create a document of repeated tokens.

    :param int token: Token value to repeat.
    :param int length: Number of repetitions.
    :return list[int]: Token list of length ``length``.
    """
    return [token] * length


def test_window_shuffle_seed_normalizes_to_uint32() -> None:
    """Large valid data seeds must remain acceptable to Grain's uint32 shuffler."""
    cfg = Config(data=replace(Config().data, seed=2**32 + 7))

    assert effective_window_shuffle_seed(cfg) == 104_736


def _hf_stream_spec(**overrides: Any) -> HFStreamSpec:
    """Create a deterministic HF stream spec for tests."""
    params: dict[str, Any] = {
        "dataset": "dummy",
        "name": "dummy",
        "split": "train",
        "text_key": "text",
        "revision": None,
        "shuffle": False,
        "shuffle_buffer_size": 8,
        "shuffle_buffer_bytes": 1_000_000,
        "seed": 0,
        "repeat": False,
        "content_partition": "all",
        "eval_holdout_fraction": 0.01,
    }
    params.update(overrides)
    return HFStreamSpec(**params)


def _packer(mode: str, **overrides: Any) -> TokenPacker | FFDPacker:
    """Build a standard tiny packer for the requested mode.

    :param str mode: Packing mode to construct.
    :param overrides: Common packer settings to replace.
    :return TokenPacker | FFDPacker: Configured test packer.
    """
    params: dict[str, Any] = {
        "seq_len": 8,
        "add_bos": False,
        "add_eos": False,
        "bos_id": 1,
        "eos_id": 2,
        "max_doc_tokens": None,
        "pad_id": 0,
    }
    params.update(overrides)
    if mode == "sequential":
        return TokenPacker(**params)
    return FFDPacker(
        mode=mode,
        bins_per_pack=2 if mode == "bin" else 1,
        lookahead_docs=2,
        max_docs_per_bin=None,
        **params,
    )


def _ffd_pending_docs(packer: Any) -> list[list[int]]:
    """Decode pending FFD chunks for queue-policy assertions."""
    state = FFDPackerState.from_dict(packer.get_state())
    return [row.tolist() for row in state.pending_docs]


def test_bin_packer_packs_multiple_docs() -> None:
    """Bin packer should combine multiple documents into packed bins."""
    packer = _packer("bin")
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


@pytest.mark.parametrize("mode", ["bin", "multipack"])
def test_ffd_leftover_requeue_preserves_arrival_order(mode: str) -> None:
    """Leftovers must requeue in arrival order, not FFD descending-size order.

    Sizes 10, 8, 9 with capacity 16 and one bin per cycle: the size-10 doc
    seeds the bin, the other two don't fit and become leftovers. Descending
    FFD order would requeue [9, 8]; arrival order is [8, 9].
    """
    kwargs: dict[str, Any] = {
        "seq_len": 16,
        "add_bos": False,
        "add_eos": False,
        "bos_id": 1,
        "eos_id": 2,
        "max_doc_tokens": None,
        "bins_per_pack": 1,
        "max_docs_per_bin": None,
        "pad_id": 0,
    }
    packer = FFDPacker(mode=mode, lookahead_docs=3, **kwargs)

    packer.add_document(_doc(10, 10))
    packer.add_document(_doc(20, 8))
    packer.add_document(_doc(30, 9))

    assert packer.can_pop()
    seq, _ = packer.pop_seq_with_segments()
    np.testing.assert_array_equal(seq[:10], np.full((10,), 10, dtype=np.int32))

    assert _ffd_pending_docs(packer) == [_doc(20, 8), _doc(30, 9)]


def test_ffd_queue_policies_remain_distinct() -> None:
    """Bin consumes the full pool while multipack consumes one bounded group."""
    common: dict[str, Any] = {
        "seq_len": 8,
        "add_bos": False,
        "add_eos": False,
        "bos_id": 1,
        "eos_id": 2,
        "max_doc_tokens": None,
        "bins_per_pack": 2,
        "max_docs_per_bin": None,
        "pad_id": 0,
    }
    bin_packer = FFDPacker(mode="bin", lookahead_docs=3, **common)
    multipack = FFDPacker(mode="multipack", lookahead_docs=3, **common)
    for packer in (bin_packer, multipack):
        packer.add_document(_doc(10, 2))
        packer.add_document(_doc(20, 2))
        # Adds two chunks at once, taking the queue past the threshold.
        packer.add_document(_doc(30, 12))
        _ = packer.pop_seq_with_segments()
        _ = packer.pop_seq_with_segments()

    assert _ffd_pending_docs(bin_packer) == [_doc(30, 8)]
    assert _ffd_pending_docs(multipack) == [_doc(30, 8), _doc(30, 4)]


@pytest.mark.parametrize("mode", ["bin", "multipack"])
def test_ffd_fifo_seed_prevents_adversarial_starvation(mode: str) -> None:
    """The oldest short candidate must progress despite endless full rows."""
    packer = _packer(mode)
    packer.add_document(_doc(6, 6))

    emitted: list[int] = []
    for token in range(10, 16):
        packer.add_document(_doc(token, 8))
        if packer.can_pop():
            row, _ = packer.pop_seq_with_segments()
            emitted.append(int(row[0]))

    assert emitted == [6, 10, 11, 12, 13, 14]
    assert _ffd_pending_docs(packer) == [_doc(15, 8)]


@pytest.mark.parametrize("mode", ["bin", "multipack"])
def test_ffd_long_document_tail_progress_and_restore(mode: str) -> None:
    """A long-document tail must be the next mandatory seed after restore."""
    packer = _packer(mode)
    packer.add_document(_doc(7, 14))  # chunks [8, 6]
    first, _ = packer.pop_seq_with_segments()
    state = packer.get_state()

    packer.add_document(_doc(9, 8))
    expected, _ = packer.pop_seq_with_segments()

    restored = _packer(mode)
    restored.set_state(state)
    restored.add_document(_doc(9, 8))
    actual, _ = restored.pop_seq_with_segments()

    np.testing.assert_array_equal(first, np.full((8,), 7, dtype=np.int32))
    np.testing.assert_array_equal(expected[:6], np.full((6,), 7, dtype=np.int32))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("mode", ["bin", "multipack"])
def test_ffd_packer_flushes_pending_docs_at_stream_end(mode: str) -> None:
    """finish() must flush sub-threshold pending docs instead of dropping them.

    One doc is below both packers' thresholds (buffer_docs/group_docs = 2),
    so can_pop stays False until finish() marks the stream exhausted.
    """
    packer = _packer(mode)
    packer.add_document(_doc(7, 5))
    assert not packer.can_pop()

    packer.finish()
    assert packer.can_pop()
    seq, segs = packer.pop_seq_with_segments()
    np.testing.assert_array_equal(seq[:5], np.full((5,), 7, dtype=np.int32))
    np.testing.assert_array_equal(segs[:5], np.ones((5,), dtype=np.int32))
    assert np.all(segs[5:] == 0)

    # Drained: nothing pending, nothing ready, no infinite re-flush.
    assert not packer.can_pop()


@pytest.mark.parametrize("mode", ["sequential", "bin", "multipack"])
def test_packer_rejects_add_document_after_finish(mode: str) -> None:
    """add_document after finish() must raise, not silently buffer.

    A silently buffered document would still be emitted by a later flush
    cycle, so every packer shares the same misuse contract.
    """
    packer = _packer(mode)
    packer.finish()
    with pytest.raises(RuntimeError, match="after"):
        packer.add_document(_doc(7, 3))


@pytest.mark.parametrize("mode", ["bin", "multipack"])
def test_ffd_packer_flush_state_roundtrips(mode: str) -> None:
    """The exhausted flag must survive get/set_state so a resumed run still
    flushes the same tail windows as the continuous one."""
    packer = _packer(mode)
    packer.add_document(_doc(9, 3))
    packer.finish()
    state = packer.get_state()
    assert state["exhausted"] is True
    expected = packer.pop_seq_with_segments()

    restored = _packer(mode)
    restored.set_state(state)
    assert restored.can_pop()
    actual = restored.pop_seq_with_segments()
    for arr_a, arr_b in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(arr_a, arr_b)


@pytest.mark.parametrize(
    ("mode", "docs", "pops_before_snapshot"),
    [
        pytest.param("bin", [_doc(21, 6), _doc(22, 2), _doc(23, 6), _doc(24, 2)], 1),
        pytest.param("multipack", [[31, 32, 33], [41, 42]], 0),
    ],
)
def test_ffd_packer_state_roundtrip(
    mode: str,
    docs: list[list[int]],
    pops_before_snapshot: int,
) -> None:
    """FFD packer state must roundtrip via get/set_state (shared base-class state)."""
    packer = _packer(mode)
    for doc in docs:
        packer.add_document(doc)
    for _ in range(pops_before_snapshot):
        _ = packer.pop_seq_with_segments()
    state = packer.get_state()
    assert "pending_docs" not in state
    assert isinstance(state["pending_tokens_i32_b64"], str)
    assert json.loads(json.dumps(state)) == state
    expected_stats = packer.get_stats()
    expected = packer.pop_seq_with_segments()

    restored = _packer(mode)
    restored.set_state(state)
    actual = restored.pop_seq_with_segments()

    for arr_a, arr_b in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(arr_a, arr_b)

    # docs_seen/docs_truncated diagnostics must survive save/load, not reset to 0.
    assert restored.get_stats() == expected_stats
    assert expected_stats["docs_seen"] == len(docs)


def test_token_packer_finite_tail_state_roundtrip() -> None:
    """Sequential exhaustion and its padded tail must survive a restore."""
    packer = _packer("sequential")
    packer.add_document([4, 5, 6])
    packer.finish()
    state = packer.get_state()
    assert state["exhausted"] is True

    restored = _packer("sequential")
    restored.set_state(state)
    tokens, segments = restored.pop_seq_with_segments()
    np.testing.assert_array_equal(tokens, [4, 5, 6, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(segments, [1, 1, 1, 0, 0, 0, 0, 0])
    assert not restored.can_pop()


@pytest.mark.parametrize("mode", ["sequential", "bin", "multipack"])
def test_document_truncation_metrics_are_complete_and_resume_stable(mode: str) -> None:
    """Explicit truncation should report token loss across resume."""
    packer = _packer(mode, max_doc_tokens=4)
    restored = _packer(mode, max_doc_tokens=4)

    for length in (0, 1, 2, 4, 5, 8, 9):
        packer.add_document(list(range(length)))
    expected = packer.get_stats()

    assert expected == {
        "docs_seen": 7,
        "docs_truncated": 3,
        "source_tokens_observed": 29,
        "source_tokens_retained": 19,
        "source_tokens_discarded": 10,
        "source_truncation_fraction": 10 / 29,
    }
    restored.set_state(packer.get_state())
    assert restored.get_stats() == expected


def test_grain_iterator_state_roundtrip() -> None:
    """Grain iterator should produce same batches after state restore."""
    cfg = make_pipeline_cfg(packing_mode="bin", packing_buffer_docs=4, grain_prefetch=2)

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
    np.testing.assert_array_equal(next_a.segment_ids, next_b.segment_ids)


def test_default_small_context_window_caps_pre_first_yield_work(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A small-context default should request no more than the packed-row cap."""
    cfg = make_pipeline_cfg(window_shuffle_tokens=DataConfig().window_shuffle_tokens)
    calls = 0

    def _counted_next_window(_producer: Any) -> tuple[np.ndarray, np.ndarray]:
        """Return one valid row while counting Grain's source requests."""
        nonlocal calls
        calls += 1
        return (
            np.full(cfg.train.seq_len, 7, dtype=np.int32),
            np.ones(cfg.train.seq_len, dtype=np.int32),
        )

    monkeypatch.setattr(_SequenceProducer, "next_window", _counted_next_window)
    with caplog.at_level(logging.INFO, logger="chomp.data.grain"):
        it = build_train_iterator(cfg)

    assert calls == 0
    assert any("4096 rows / 32768 tokens" in record.getMessage() for record in caplog.records)
    try:
        _ = next(it)
    finally:
        it.close()

    assert calls == cfg.data.window_shuffle_max_rows


def _window_shuffle_cfg(*, window: int, repeat: bool = False) -> Config:
    """Build an HF-backed config for window-shuffle tests.

    Each fake document tokenizes (byte, offset 4, BOS/EOS) to exactly seq_len=8,
    so every packed row is one whole document and row identity is readable from
    its second token.

    :param int window: Desired packed-window row count.
    :param bool repeat: Whether to repeat the stream.
    :return Config: Test configuration.
    """
    return make_pipeline_cfg(
        batch_size=2,
        backend="hf",
        hf_dataset="dummy",
        hf_name="dummy",
        hf_split="train",
        shuffle=False,
        shuffle_buffer_size=8,
        repeat=repeat,
        window_shuffle_tokens=window * 8,
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


def test_window_shuffle_disabled_matches_unshuffled(
    patch_hf_load_dataset: Callable[..., dict[str, int]],
) -> None:
    """window_shuffle_tokens=0 must preserve raw packer output order."""
    items = _distinct_docs(20)
    patch_hf_load_dataset(items)

    cfg = _window_shuffle_cfg(window=0)
    it = build_train_iterator(cfg)
    seen = _row_doc_tokens(next(it)) + _row_doc_tokens(next(it))

    expected = [ord(item["text"][0]) + 4 for item in items[: len(seen)]]
    assert seen == expected


def test_window_shuffle_permutes_within_window(
    patch_hf_load_dataset: Callable[..., dict[str, int]],
) -> None:
    """W>0 must permute rows within each window without loss or duplication."""
    items = _distinct_docs(20)
    patch_hf_load_dataset(items)

    window = 8
    cfg = _window_shuffle_cfg(window=window)
    it = build_train_iterator(cfg)
    seen = []
    for _ in range(window // 2):  # batch_size=2, grad_accum=1 -> 2 rows per batch
        seen.extend(_row_doc_tokens(next(it)))

    expected_window = [ord(item["text"][0]) + 4 for item in items[:window]]
    assert sorted(seen) == sorted(expected_window)  # same multiset: nothing lost/duplicated
    assert seen != expected_window  # order actually changed


def test_window_shuffle_state_roundtrip(
    patch_hf_load_dataset: Callable[..., dict[str, int]],
) -> None:
    """Resume with W>0 must reproduce the continuous run exactly.

    Snapshots mid-window (3 batches = 6 of 8 window slots consumed) so the
    restored iterator must replay the parent to the window start, re-apply the
    same permutation, fast-forward, and then cross a window boundary.
    """
    items = _distinct_docs(60)
    patch_hf_load_dataset(items)

    cfg = _window_shuffle_cfg(window=16)
    cfg = replace(
        cfg,
        data=replace(cfg.data, window_shuffle_max_rows=8, grain_prefetch=2),
    )
    assert resolve_window_shuffle_rows(cfg) == 8

    it = build_train_iterator(cfg)
    for _ in range(3):
        _ = next(it)
    state = it.get_state()

    followups_a = [next(it) for _ in range(3)]

    it2 = build_train_iterator(cfg)
    it2.set_state(state)
    followups_b = [next(it2) for _ in range(3)]

    for a, b in zip(followups_a, followups_b, strict=True):
        np.testing.assert_array_equal(a.input_ids, b.input_ids)
        np.testing.assert_array_equal(a.labels, b.labels)
        np.testing.assert_array_equal(a.segment_ids, b.segment_ids)


def test_eval_iterator_never_shuffles() -> None:
    """Eval batches must come out in strict document order regardless of W."""
    cfg = _window_shuffle_cfg(window=4096)
    tokens = [[100 + i] * 6 for i in range(8)]
    it = build_eval_iterator(cfg, tokens=tokens)

    seen = _row_doc_tokens(next(it)) + _row_doc_tokens(next(it))
    assert seen == [100, 101, 102, 103]


def _assert_multi_segment_boundary_masked(batch: Batch) -> None:
    """Assert a row holds >=2 positive segments with boundary labels masked.

    :param Batch batch: Batch whose first row is checked.
    """
    segs = batch.segment_ids[0, 0]
    unique = np.unique(segs)
    assert unique.size >= 2
    assert np.all(unique > 0)

    boundary = segs[1:] != segs[:-1]
    assert boundary.any()
    masked_labels = batch.labels[0, 0][1:][boundary]
    assert np.all(masked_labels == -100)


def test_finite_sequential_tail_is_padded_without_token_loss() -> None:
    """A finite sequential tail fills the final row and preserves alignment."""
    # 10 chars + BOS/EOS = 12 tokens: one full row and one four-token tail.
    cfg = make_pipeline_cfg(local_text="x" * 10, repeat=False, window_shuffle_tokens=0)
    cfg = replace(cfg, train=replace(cfg.train, grad_accum=2))

    it = build_train_iterator(cfg)
    state_before = json.dumps(it.get_state(), sort_keys=True, default=str)
    batch = next(it)
    assert batch.input_ids.shape == (2, 1, 8)
    assert int(np.count_nonzero(batch.segment_ids)) == 12
    np.testing.assert_array_equal(batch.segment_ids[1, 0, 4:], np.zeros(4, dtype=np.int32))
    state_after = json.dumps(it.get_state(), sort_keys=True, default=str)
    assert state_after != state_before
    with pytest.raises(StopIteration):
        next(it)


def test_stopiteration_at_exact_batch_boundary_consumes_zero_windows() -> None:
    """Exact EOF after full batches should report no partial batch consumption."""
    cfg = make_pipeline_cfg(
        local_text="x" * 16,
        repeat=False,
        window_shuffle_tokens=0,
        seq_len=8,
        tokenizer=TokenizerConfig(kind="byte", byte_offset=4, add_bos=False, add_eos=False),
    )

    it = build_train_iterator(cfg)
    next(it)
    next(it)
    with pytest.raises(StopIteration):
        next(it)


def test_batch_assembly_rejects_zero_valid_loss_tokens() -> None:
    """One-token segments must fail before a zero-objective batch reaches training."""
    cfg = make_pipeline_cfg(
        local_text="a",
        seq_len=8,
        repeat=True,
        window_shuffle_tokens=0,
        tokenizer=TokenizerConfig(kind="byte", byte_offset=0, add_bos=False, add_eos=False),
    )

    with pytest.raises(ZeroLossTokensError, match="zero valid loss tokens"):
        next(build_train_iterator(cfg))


def test_pipeline_segment_ids_multiple_docs() -> None:
    """Pipeline should emit multiple segment IDs and mask boundaries."""
    cfg = make_pipeline_cfg()

    it = build_train_iterator(cfg)
    _assert_multi_segment_boundary_masked(next(it))


def test_boundary_loss_mask_toggle() -> None:
    """With mask_boundary_loss=False, boundary labels should not be masked."""
    cfg = make_pipeline_cfg(mask_boundary_loss=False)

    it = build_train_iterator(cfg)
    batch = next(it)
    segs = batch.segment_ids[0, 0]
    boundary = segs[1:] != segs[:-1]
    assert boundary.any()
    labels_at_boundary = batch.labels[0, 0][1:][boundary]
    assert np.all(labels_at_boundary != -100)


@pytest.mark.parametrize(
    ("mode", "mode_config"),
    [
        ("bin", {"packing_buffer_docs": 4}),
        ("multipack", {"packing_group_docs": 4}),
    ],
)
def test_ffd_pipeline_emits_segments_and_stats(mode: str, mode_config: dict[str, int]) -> None:
    """Both FFD policies should emit multiple segments and packing stats."""
    cfg = make_pipeline_cfg(packing_mode=mode, **mode_config)

    it = build_train_iterator(cfg)
    batch = next(it)
    segs = batch.segment_ids[0, 0]
    unique = np.unique(segs[segs > 0])
    assert unique.size >= 2

    stats = it.get_stats()
    assert stats["packing_mode"] == mode
    assert stats["segments_per_seq_max"] >= 2
    assert stats["packing_capacity"] == batch.segment_ids.size


def test_batch_segment_stats_use_literal_segment_geometry() -> None:
    """Packing diagnostics should match a hand-enumerated segment batch."""
    segments = np.asarray([[[1, 1, 2, 0], [3, 3, 3, 3]]], dtype=np.int32)
    assert _batch_segment_stats(segments) == {
        "packing_tokens": 7,
        "packing_capacity": 8,
        "packing_utilization": 0.875,
        "boundary_transitions": 1,
        "segments_per_seq_mean": 1.5,
        "segments_per_seq_min": 1,
        "segments_per_seq_max": 2,
    }


def test_loss_token_count_stays_paired_through_prefetch() -> None:
    """Exact host accounting travels with its batch through prefetch."""
    cfg = make_pipeline_cfg(grain_prefetch=2)
    iterator = build_train_iterator(cfg)
    batch = next(iterator)
    labels = np.asarray(batch.labels)
    attention = np.asarray(batch.segment_ids) > 0
    expected = int(np.count_nonzero((labels[..., 1:] != -100) & attention[..., 1:]))

    assert iterator.get_loss_tokens() == expected


def test_packing_array_diagnostics_can_skip_without_losing_token_count() -> None:
    """Non-log steps should avoid full batch stats while preserving exact accounting."""
    cfg = make_pipeline_cfg(window_shuffle_tokens=0)
    iterator = build_train_iterator(cfg)
    iterator.set_collect_stats(False)

    _ = next(iterator)

    skipped_stats = iterator.get_stats()
    assert "packing_capacity" not in skipped_stats
    assert skipped_stats["docs_seen"] > 0
    assert iterator.get_loss_tokens() > 0

    iterator.set_collect_stats(True)
    _ = next(iterator)
    assert iterator.get_stats()["packing_capacity"] > 0


def test_hf_pipeline_segment_ids_and_label_mask(
    patch_hf_load_dataset: Callable[..., dict[str, int]],
) -> None:
    """HF pipeline should emit segment IDs and mask labels at boundaries."""
    patch_hf_load_dataset([{"text": "hi"}, {"text": "ok"}, {"text": "yo"}, {"text": "sup"}])

    cfg = make_pipeline_cfg(
        vocab_size=256,
        backend="hf",
        hf_dataset="dummy",
        hf_name="dummy",
        hf_split="train",
        shuffle=False,
        shuffle_buffer_size=8,
        repeat=False,
    )

    it = build_train_iterator(cfg)
    _assert_multi_segment_boundary_masked(next(it))


def test_grain_close_reaches_hf_source_with_prefetch(
    patch_hf_load_dataset: Callable[..., dict[str, int]],
) -> None:
    """Explicit close must stop prefetch and release the HF source generator."""
    record: dict[str, Any] = {}
    patch_hf_load_dataset(
        [{"text": f"document-{index}"} for index in range(100)],
        record=record,
    )
    cfg = make_pipeline_cfg(
        vocab_size=256,
        backend="hf",
        hf_dataset="dummy",
        hf_name="dummy",
        hf_split="train",
        shuffle=False,
        repeat=True,
        grain_prefetch=2,
        window_shuffle_tokens=0,
    )

    iterator = build_train_iterator(cfg)
    next(iterator)
    iterator.close()

    assert record["close_calls"] == 1


def test_hf_state_roundtrip(patch_hf_load_dataset: Callable[..., dict[str, int]]) -> None:
    """HF stream should resume to same position after state roundtrip."""
    patch_hf_load_dataset([{"text": "alpha"}, {"text": "bravo"}, {"text": "charlie"}])

    spec = _hf_stream_spec()
    stream = HFStreamingTextStream(spec)
    _ = next(stream)
    _ = next(stream)
    state = stream.get_state()
    expected = next(stream)

    resumed = HFStreamingTextStream(spec)
    resumed.set_state(state)
    assert next(resumed) == expected


@pytest.mark.parametrize(
    ("item", "match"),
    [
        ({"text": None}, "must contain strings"),
        ({"body": "wrong column"}, "without text key 'text'"),
    ],
)
def test_hf_schema_errors_fail(
    patch_hf_load_dataset: Callable[..., dict[str, int]],
    item: dict[str, Any],
    match: str,
) -> None:
    """Rows must contain a string at the configured text key."""
    patch_hf_load_dataset([item])
    stream = HFStreamingTextStream(_hf_stream_spec())

    with pytest.raises(ValueError, match=match):
        next(stream)


def test_hf_close_honors_remote_parquet_shutdown_grace(
    patch_hf_load_dataset: Callable[..., dict[str, int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sources flagged by Datasets must receive its Arrow shutdown grace."""
    from datasets import config as datasets_config

    record: dict[str, Any] = {}
    patch_hf_load_dataset([{"text": "alpha"}], record=record)
    stream = HFStreamingTextStream(_hf_stream_spec())
    next(stream)
    stream._ds._ex_iterable = SimpleNamespace(sleep_on_threads_shutdown=True)
    sleeps: list[float] = []
    monkeypatch.setattr("chomp.data.hf.time.sleep", sleeps.append)

    stream.close()
    stream.close()

    assert record["close_calls"] == 1
    assert sleeps == [datasets_config.SLEEP_TIME_ON_THREADS_SHUTDOWN]
    with pytest.raises(ValueError, match="closed HF streaming iterator"):
        next(stream)


def test_real_hf_iterable_resume_is_exact_across_fresh_processes(tmp_path: Path) -> None:
    """Serialized real-HF state should reproduce continuation in a new interpreter."""
    worker = Path(__file__).parent / "helpers" / "hf_resume_worker.py"
    state_path = tmp_path / "state.pkl"
    expected_path = tmp_path / "expected.json"
    actual_path = tmp_path / "actual.json"

    subprocess.run(
        [sys.executable, str(worker), "prepare", str(state_path), str(expected_path)],
        check=True,
    )
    subprocess.run(
        [sys.executable, str(worker), "resume", str(state_path), str(actual_path)],
        check=True,
    )

    assert json.loads(actual_path.read_text()) == json.loads(expected_path.read_text())


def test_hf_document_shuffle_byte_budget_bounds_window_and_replays_exactly(
    patch_hf_load_dataset: Callable[..., dict[str, int]],
) -> None:
    """UTF-8 bytes should cap a window without weakening exact replay."""
    items = [{"text": f"doc-{index}-" + ("é" * 5)} for index in range(30)]
    patch_hf_load_dataset(items)
    spec = _hf_stream_spec(
        shuffle=True,
        shuffle_buffer_size=100,
        shuffle_buffer_bytes=32,
        seed=11,
    )
    stream = HFStreamingTextStream(spec)

    consumed = [next(stream) for _ in range(3)]
    stats = stream.get_stats()
    assert stats["shuffle_window_docs"] == 2
    assert stats["shuffle_window_bytes"] >= spec.shuffle_buffer_bytes
    assert stats["shuffle_window_bytes"] < spec.shuffle_buffer_bytes + max(
        len(item["text"].encode("utf-8")) for item in items
    )
    state = stream.get_state()
    expected = [next(stream) for _ in range(9)]

    restored = HFStreamingTextStream(spec)
    restored.set_state(state)
    assert [next(restored) for _ in range(9)] == expected
    replay_stats = restored.get_stats()
    assert replay_stats["shuffle_replayed_window_docs"] >= 2
    assert replay_stats["shuffle_replayed_window_bytes"] >= spec.shuffle_buffer_bytes
    assert len(set(consumed + expected)) == len(consumed + expected)


def test_hf_shuffled_state_requires_replay_metadata(
    patch_hf_load_dataset: Callable[..., dict[str, int]],
) -> None:
    """Shuffled restore must reject state that cannot reconstruct its buffer."""
    patch_hf_load_dataset([{"text": "alpha"}, {"text": "bravo"}])
    stream = HFStreamingTextStream(_hf_stream_spec(shuffle=True))

    with pytest.raises(RuntimeError, match="shuffle_state"):
        stream.set_state({"epoch": 0, "hf_state": {"index": 1}})


def test_hf_set_state_raises_on_missing_hf_state(
    patch_hf_load_dataset: Callable[..., dict[str, int]],
) -> None:
    """set_state must fail loud when hf_state is missing, not silently rebuild."""
    patch_hf_load_dataset([{"text": "alpha"}, {"text": "bravo"}])

    spec = _hf_stream_spec()
    stream = HFStreamingTextStream(spec)
    with pytest.raises(RuntimeError, match="hf_state"):
        stream.set_state({"epoch": 0, "hf_state": None})
    with pytest.raises(RuntimeError, match="hf_state"):
        stream.set_state({"epoch": 0})


def test_hf_repeat_rolls_epochs(
    patch_hf_load_dataset: Callable[..., dict[str, int]],
) -> None:
    """A repeating stream rebuilds after ordinary source exhaustion."""
    patch_hf_load_dataset([{"text": "alpha"}])
    stream = HFStreamingTextStream(_hf_stream_spec(repeat=True))

    assert [next(stream) for _ in range(3)] == ["alpha", "alpha", "alpha"]


def test_hf_repeat_rejects_logically_empty_epoch(
    patch_hf_load_dataset: Callable[..., dict[str, int]],
) -> None:
    """A repeated source filtered to no documents fails descriptively."""
    patch_hf_load_dataset([{"text": "alpha"}])
    stream = HFStreamingTextStream(
        _hf_stream_spec(
            repeat=True,
            content_partition="eval",
            eval_holdout_fraction=1e-20,
        )
    )

    with pytest.raises(RuntimeError, match="no documents in a complete epoch"):
        next(stream)


def test_hf_content_holdout_is_disjoint_complete_and_resumable(
    patch_hf_load_dataset: Callable[..., dict[str, int]],
) -> None:
    """Hash partitioning assigns every content identity to exactly one side."""
    texts = [f"document-{index:03d}" for index in range(200)] + ["document-000"]
    patch_hf_load_dataset([{"text": text} for text in texts])
    common = {"shuffle": False, "eval_holdout_fraction": 0.25}

    train = list(HFStreamingTextStream(_hf_stream_spec(content_partition="train", **common)))
    eval_ = list(HFStreamingTextStream(_hf_stream_spec(content_partition="eval", **common)))
    assert train and eval_
    assert set(train).isdisjoint(eval_)
    assert set(train) | set(eval_) == set(texts)
    assert (train.count("document-000"), eval_.count("document-000")) in {(2, 0), (0, 2)}

    stream = HFStreamingTextStream(_hf_stream_spec(content_partition="train", **common))
    _ = [next(stream) for _ in range(7)]
    state = stream.get_state()
    expected = [next(stream) for _ in range(12)]
    restored = HFStreamingTextStream(_hf_stream_spec(content_partition="train", **common))
    restored.set_state(state)
    assert [next(restored) for _ in range(12)] == expected


def _batch_arrays(batch: Batch) -> tuple:
    """Extract arrays from batch for comparison.

    :param Batch batch: Batch to extract from.
    :return tuple: Tuple of (input_ids, labels, segment_ids).
    """
    return batch.input_ids, batch.labels, batch.segment_ids


def test_packer_alignment_after_restore() -> None:
    """Restored iterator should produce same batches as continued iterator."""
    # Raw packer-state layout is only exposed without window shuffling.
    cfg = make_pipeline_cfg(local_text="abcde", window_shuffle_tokens=0)

    it = build_train_iterator(cfg)
    _ = next(it)
    state = it.get_state()

    remaining = state.get("packer", {}).get("remaining_tokens")
    assert remaining, "expected non-empty packer buffer for alignment test"
    docs_seen_at_snapshot = state["packer"]["document_stats"]["docs_seen"]
    assert docs_seen_at_snapshot > 0

    cont = [_batch_arrays(next(it)) for _ in range(3)]

    it2 = build_train_iterator(cfg)
    it2.set_state(state)
    # docs_seen/docs_truncated diagnostics must survive save/load, not reset to 0.
    assert it2.get_state()["packer"]["document_stats"]["docs_seen"] == docs_seen_at_snapshot
    resumed = [_batch_arrays(next(it2)) for _ in range(3)]

    for batch_a, batch_b in zip(cont, resumed, strict=True):
        for arr_a, arr_b in zip(batch_a, batch_b, strict=True):
            np.testing.assert_array_equal(arr_a, arr_b)


def test_grain_shuffle_source_does_not_buffer_position_ids() -> None:
    """Packed shuffle rows should hold only tokens and segment IDs."""
    cfg = make_pipeline_cfg(window_shuffle_tokens=64)
    dataset = _TrainSequenceIterDataset(cfg=cfg, tokenizer=ByteTokenizer(byte_offset=4))

    window = next(iter(dataset))

    assert len(window) == 2
    assert all(array.shape == (cfg.train.seq_len,) for array in window)


def test_train_on_eos_false_masks_eos_labels() -> None:
    """With train_on_eos=False, EOS token labels should be masked to -100."""
    cfg = make_pipeline_cfg(mask_boundary_loss=False, train_on_eos=False)

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
