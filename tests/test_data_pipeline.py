"""Data pipeline tests consolidated by module."""

from __future__ import annotations

import json
import logging
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
from chomp.data.grain import _packer_stats_from_chain, effective_window_shuffle_seed
from chomp.data.hf import HFStreamingTextStream, HFStreamSpec
from chomp.data.pack import MultipackPacker, TokenPacker
from chomp.data.pipeline import (
    BatchAssemblyStopIteration,
    BinPacker,
    ByteTokenizer,
    build_eval_iterator,
    build_train_iterator,
)
from chomp.train import run
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
        "seed": 0,
        "repeat": False,
        "max_retries": 0,
        "retry_delay_sec": 0.0,
        "state_update_interval": 2,
    }
    params.update(overrides)
    return HFStreamSpec(**params)


def _bin_packer() -> BinPacker:
    """Standard tiny BinPacker for packer-level tests."""
    return BinPacker(
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


def _multipack_packer() -> MultipackPacker:
    """Standard tiny MultipackPacker for packer-level tests."""
    return MultipackPacker(
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


def test_bin_packer_packs_multiple_docs() -> None:
    """Bin packer should combine multiple documents into packed bins."""
    packer = _bin_packer()
    for tok, length in [(10, 6), (11, 2), (12, 6), (13, 2)]:
        packer.add_document(_doc(tok, length))

    assert packer.can_pop()
    seq1, seg1, _ = packer.pop_seq_with_metadata()
    seq2, seg2, _ = packer.pop_seq_with_metadata()

    assert seq1.shape == (8,)
    assert seq2.shape == (8,)

    for seq, segs in [(seq1, seg1), (seq2, seg2)]:
        pad_mask = seq == 0
        if np.any(pad_mask):
            assert np.all(segs[pad_mask] == 0)

        unique = np.unique(segs[segs > 0])
        assert unique.size >= 2


def test_multipack_packer_emits_segment_local_positions() -> None:
    """MultipackPacker should emit segment IDs and per-segment position IDs."""
    packer = _multipack_packer()
    packer.add_document([10, 11, 12])
    packer.add_document([20, 21])
    assert packer.can_pop()
    toks, segs, pos = packer.pop_seq_with_metadata()

    np.testing.assert_array_equal(toks[:5], np.asarray([10, 11, 12, 20, 21], dtype=np.int32))
    np.testing.assert_array_equal(segs[:5], np.asarray([1, 1, 1, 2, 2], dtype=np.int32))
    np.testing.assert_array_equal(pos[:5], np.asarray([0, 1, 2, 0, 1], dtype=np.int32))
    assert np.all(segs[5:] == 0)
    assert np.all(pos[5:] == 0)


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
    if mode == "bin":
        packer: Any = BinPacker(buffer_docs=3, **kwargs)
    else:
        packer = MultipackPacker(group_docs=3, **kwargs)

    packer.add_document(_doc(10, 10))
    packer.add_document(_doc(20, 8))
    packer.add_document(_doc(30, 9))

    assert packer.can_pop()
    seq, _, _ = packer.pop_seq_with_metadata()
    np.testing.assert_array_equal(seq[:10], np.full((10,), 10, dtype=np.int32))

    pending = packer.get_state()["pending_docs"]
    assert pending == [_doc(20, 8), _doc(30, 9)]


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
    bin_packer = BinPacker(buffer_docs=3, **common)
    multipack = MultipackPacker(group_docs=3, **common)
    for packer in (bin_packer, multipack):
        packer.add_document(_doc(10, 2))
        packer.add_document(_doc(20, 2))
        # Adds two chunks at once, taking the queue past the threshold.
        packer.add_document(_doc(30, 12))
        _ = packer.pop_seq_with_metadata()
        _ = packer.pop_seq_with_metadata()

    assert bin_packer.get_state()["pending_docs"] == []
    assert multipack.get_state()["pending_docs"] == [_doc(30, 4)]


@pytest.mark.parametrize("make_packer", [_bin_packer, _multipack_packer])
def test_ffd_packer_flushes_pending_docs_at_stream_end(
    make_packer: Callable[[], Any],
) -> None:
    """finish() must flush sub-threshold pending docs instead of dropping them.

    One doc is below both packers' thresholds (buffer_docs/group_docs = 2),
    so can_pop stays False until finish() marks the stream exhausted.
    """
    packer = make_packer()
    packer.add_document(_doc(7, 5))
    assert not packer.can_pop()

    packer.finish()
    assert packer.can_pop()
    seq, segs, _ = packer.pop_seq_with_metadata()
    np.testing.assert_array_equal(seq[:5], np.full((5,), 7, dtype=np.int32))
    np.testing.assert_array_equal(segs[:5], np.ones((5,), dtype=np.int32))
    assert np.all(segs[5:] == 0)

    # Drained: nothing pending, nothing ready, no infinite re-flush.
    assert not packer.can_pop()


@pytest.mark.parametrize("make_packer", [_bin_packer, _multipack_packer])
def test_ffd_packer_flush_state_roundtrips(make_packer: Callable[[], Any]) -> None:
    """The exhausted flag must survive get/set_state so a resumed run still
    flushes the same tail windows as the continuous one."""
    packer = make_packer()
    packer.add_document(_doc(9, 3))
    packer.finish()
    state = packer.get_state()
    assert state["exhausted"] is True
    expected = packer.pop_seq_with_metadata()

    restored = make_packer()
    restored.set_state(state)
    assert restored.can_pop()
    actual = restored.pop_seq_with_metadata()
    for arr_a, arr_b in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(arr_a, arr_b)


@pytest.mark.parametrize(
    ("make_packer", "docs", "pops_before_snapshot"),
    [
        pytest.param(
            _bin_packer, [_doc(21, 6), _doc(22, 2), _doc(23, 6), _doc(24, 2)], 1, id="bin"
        ),
        pytest.param(_multipack_packer, [[31, 32, 33], [41, 42]], 0, id="multipack"),
    ],
)
def test_ffd_packer_state_roundtrip(
    make_packer: Callable[[], Any],
    docs: list[list[int]],
    pops_before_snapshot: int,
) -> None:
    """FFD packer state must roundtrip via get/set_state (shared base-class state)."""
    packer = make_packer()
    for doc in docs:
        packer.add_document(doc)
    for _ in range(pops_before_snapshot):
        _ = packer.pop_seq_with_metadata()
    state = packer.get_state()
    expected_stats = packer.get_stats()
    expected = packer.pop_seq_with_metadata()

    restored = make_packer()
    restored.set_state(state)
    actual = restored.pop_seq_with_metadata()

    for arr_a, arr_b in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(arr_a, arr_b)

    # docs_seen/docs_truncated diagnostics must survive save/load, not reset to 0.
    assert restored.get_stats() == expected_stats
    assert expected_stats["docs_seen"] == len(docs)


@pytest.mark.parametrize(
    "missing_key",
    ["pending_docs", "ready_tokens", "ready_segments", "docs_seen", "docs_truncated", "exhausted"],
)
def test_ffd_packer_state_from_dict_is_strict(missing_key: str) -> None:
    """FFD packer set_state must fail loud on corrupt/foreign state, not default to []/0."""
    packer = _bin_packer()
    full_state = {
        "pending_docs": [],
        "ready_tokens": [],
        "ready_segments": [],
        "docs_seen": 0,
        "docs_truncated": 0,
        "exhausted": False,
    }
    del full_state[missing_key]
    with pytest.raises(KeyError):
        packer.set_state(full_state)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        # ready row pair with mismatched inner lengths
        (
            {"ready_tokens": [[1, 2, 3, 4, 5, 6, 7, 8]], "ready_segments": [[1, 1, 1]]},
            r"ready_tokens\[0\] and ready_segments\[0\]",
        ),
        # ready row shorter than seq_len (rows are padded to fixed length)
        (
            {"ready_tokens": [[1, 2, 3]], "ready_segments": [[1, 1, 1]]},
            "expected exactly seq_len",
        ),
        # negative segment id in a ready row
        (
            {
                "ready_tokens": [[1, 2, 3, 4, 5, 6, 7, 8]],
                "ready_segments": [[1, 1, 1, 1, -1, 0, 0, 0]],
            },
            "negative segment ids",
        ),
        # empty pending chunk (chunks are non-empty by construction)
        ({"pending_docs": [[]]}, r"pending_docs\[0\]"),
        # pending chunk longer than capacity (chunks are pre-split)
        ({"pending_docs": [list(range(9))]}, r"pending_docs\[0\]"),
        # negative counters / truncated > seen
        ({"docs_seen": -1}, "invalid document counters"),
        ({"docs_seen": 1, "docs_truncated": 2}, "invalid document counters"),
    ],
    ids=[
        "row_pair_mismatch",
        "short_ready_row",
        "negative_segment",
        "empty_pending_chunk",
        "oversized_pending_chunk",
        "negative_counter",
        "truncated_exceeds_seen",
    ],
)
@pytest.mark.parametrize("make_packer", [_bin_packer, _multipack_packer])
def test_ffd_packer_state_rejects_corrupt_queues(
    make_packer: Callable[[], Any], mutation: dict[str, Any], match: str
) -> None:
    """Nested queue invariants fail loud at restore: row pairing, fixed
    seq_len ready rows, capacity-bounded pending chunks, sane counters."""
    packer = make_packer()
    state = {
        "pending_docs": [],
        "ready_tokens": [],
        "ready_segments": [],
        "docs_seen": 3,
        "docs_truncated": 0,
        "exhausted": False,
    }
    state.update(mutation)
    with pytest.raises(ValueError, match=match):
        packer.set_state(state)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"remaining_tokens": [10], "remaining_segments": [0]}, "remaining_segments"),
        ({"next_segment_id": 3}, "next_segment_id"),
        ({"docs_seen": -1}, "invalid document counters"),
        ({"docs_seen": 1, "docs_truncated": 2}, "invalid document counters"),
    ],
    ids=["invalid_segment", "invalid_next_segment", "negative_counter", "truncated_gt_seen"],
)
def test_token_packer_state_rejects_invalid_current_state(
    mutation: dict[str, Any], match: str
) -> None:
    """TokenPacker rejects state outside its current compact-ID invariants."""
    packer = TokenPacker(
        seq_len=8,
        add_bos=False,
        add_eos=False,
        bos_id=1,
        eos_id=2,
        max_doc_tokens=None,
    )

    state = {
        "remaining_tokens": [],
        "remaining_segments": [],
        "next_segment_id": 1,
        "docs_seen": 0,
        "docs_truncated": 0,
    }
    state.update(mutation)
    with pytest.raises(ValueError, match=match):
        packer.set_state(state)


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
    return make_pipeline_cfg(
        batch_size=2,
        backend="hf",
        hf_dataset="dummy",
        hf_name="dummy",
        hf_split="train",
        shuffle=False,
        shuffle_buffer_size=8,
        repeat=repeat,
        window_shuffle_windows=window,
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
    """window_shuffle_windows=0 must preserve raw packer output order."""
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

    cfg = _window_shuffle_cfg(window=8)
    cfg = replace(cfg, data=replace(cfg.data, grain_prefetch=2))

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
        np.testing.assert_array_equal(a.attention_mask, b.attention_mask)
        np.testing.assert_array_equal(a.segment_ids, b.segment_ids)
        np.testing.assert_array_equal(a.position_ids, b.position_ids)


def test_docs_added_this_batch_accounts_for_all_pulls(
    patch_hf_load_dataset: Callable[..., dict[str, int]],
) -> None:
    """Sum of docs_added_this_batch must equal the packer's docs_seen counter."""
    patch_hf_load_dataset(_distinct_docs(64))

    cfg = _window_shuffle_cfg(window=8)
    it = build_train_iterator(cfg)

    total = 0
    for _ in range(4):
        _ = next(it)
        stats = it.get_stats()
        added = stats.get("docs_added_this_batch")
        assert added is not None and added >= 0
        total += added
    assert total == it.get_stats().get("docs_seen")


def test_disabled_stats_skip_per_batch_chain_walks(monkeypatch: pytest.MonkeyPatch) -> None:
    """data.device_put=true disables stats; batch assembly must then skip the
    per-batch iterator-chain walks entirely — nothing ever reads the snapshot."""
    import chomp.data.grain as grain_mod

    calls = {"n": 0}
    real = grain_mod._packer_stats_from_chain

    def _counting(it: Any) -> dict[str, Any]:
        """Count chain walks while delegating to the real implementation.

        :param it: Outermost Grain DatasetIterator.
        :return dict[str, Any]: Packer stats from the real walk.
        """
        calls["n"] += 1
        return real(it)

    monkeypatch.setattr(grain_mod, "_packer_stats_from_chain", _counting)

    cfg = make_pipeline_cfg(packing_mode="bin", packing_buffer_docs=4, device_put=True)
    it = build_train_iterator(cfg)
    for _ in range(2):
        _ = next(it)
        assert it.get_stats() == {}
    assert calls["n"] == 0


def test_eval_iterator_never_shuffles() -> None:
    """Eval batches must come out in strict document order regardless of W."""
    cfg = _window_shuffle_cfg(window=4096)
    tokens = [[100 + i] * 6 for i in range(8)]
    it = build_eval_iterator(cfg, tokens=tokens)

    seen = _row_doc_tokens(next(it)) + _row_doc_tokens(next(it))
    assert seen == [100, 101, 102, 103]


class _RaisingStatsNode:
    """Chain node whose get_stats() always raises, to exercise error handling."""

    def get_stats(self) -> dict[str, int]:
        """Raise unconditionally to simulate a real packer bug."""
        raise RuntimeError("boom")


def test_packer_stats_from_chain_swallows_and_logs_get_stats_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A packer's get_stats() raising must yield {} and log once, not spam per call."""
    node = _RaisingStatsNode()

    with caplog.at_level(logging.WARNING, logger="chomp.data.grain"):
        first = _packer_stats_from_chain(node)
        second = _packer_stats_from_chain(node)

    assert first == {}
    assert second == {}

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, "must warn once per node, not once per call"
    assert "RuntimeError" in warnings[0].getMessage()
    assert "boom" in warnings[0].getMessage()


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


def test_stopiteration_mid_assembly_advances_iterator_state() -> None:
    """Exhaustion during batch assembly is not a no-op.

    Windows are popped (and discarded) and the stream advances before
    StopIteration surfaces from a partial batch, so the train loop must
    treat exhaustion as data-state misalignment (no final checkpoint).
    """
    # 10 chars + BOS/EOS = 12 tokens = one seq_len=8 window; grad_accum=2
    # needs two, so assembly pops window 1 and then runs dry.
    cfg = make_pipeline_cfg(local_text="x" * 10, repeat=False, window_shuffle_windows=0)
    cfg = replace(cfg, train=replace(cfg.train, grad_accum=2))

    it = build_train_iterator(cfg)
    state_before = json.dumps(it.get_state(), sort_keys=True, default=str)
    with pytest.raises(BatchAssemblyStopIteration) as exc_info:
        next(it)
    assert exc_info.value.windows_consumed == 1
    state_after = json.dumps(it.get_state(), sort_keys=True, default=str)
    assert state_after != state_before


def test_stopiteration_at_exact_batch_boundary_consumes_zero_windows() -> None:
    """Exact EOF after full batches should report no partial batch consumption."""
    cfg = make_pipeline_cfg(
        local_text="x" * 16,
        repeat=False,
        window_shuffle_windows=0,
        seq_len=8,
        tokenizer=TokenizerConfig(kind="byte", byte_offset=4, add_bos=False, add_eos=False),
    )

    it = build_train_iterator(cfg)
    next(it)
    next(it)
    with pytest.raises(BatchAssemblyStopIteration) as exc_info:
        next(it)
    assert exc_info.value.windows_consumed == 0


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


def test_pipeline_bin_packing_segment_ids() -> None:
    """Bin packing should produce multiple segments with packing stats."""
    cfg = make_pipeline_cfg(packing_mode="bin", packing_buffer_docs=4)

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
    cfg = make_pipeline_cfg(packing_mode="multipack", packing_group_docs=4)

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


def test_real_hf_iterable_shuffled_state_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Owned document shuffle must resume exactly over a real HF iterable."""
    import datasets

    def _items() -> Any:
        """Yield identifiable local documents through the real HF state machinery."""
        for index in range(101):
            yield {"text": f"document-{index:03d}"}

    def _load_dataset(
        dataset: str,
        *,
        name: str,
        split: str,
        streaming: bool,
        revision: str | None,
    ) -> Any:
        _ = (dataset, name, split, streaming, revision)
        return datasets.IterableDataset.from_generator(_items)

    monkeypatch.setattr(datasets, "load_dataset", _load_dataset)
    spec = _hf_stream_spec(shuffle=True, shuffle_buffer_size=17, seed=29)
    stream = HFStreamingTextStream(spec)
    consumed = [next(stream) for _ in range(23)]
    state = stream.get_state()
    expected = [next(stream) for _ in range(41)]

    resumed = HFStreamingTextStream(spec)
    resumed.set_state(state)
    actual = [next(resumed) for _ in range(41)]

    assert actual == expected
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


def test_hf_retry_rebuild_roundtrip(patch_hf_load_dataset: Callable[..., dict[str, int]]) -> None:
    """HF stream should recover from transient failure via state restore."""
    items = [{"text": "alpha"}, {"text": "bravo"}, {"text": "charlie"}]
    record: dict[str, Any] = {"fail_consumed": False}
    calls = patch_hf_load_dataset(items, fail_at=1, record=record)

    spec = _hf_stream_spec(max_retries=1, state_update_interval=1)
    stream = HFStreamingTextStream(spec)
    assert next(stream) == "alpha"
    assert next(stream) == "bravo"

    assert calls["builds"] >= 2
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
    # Raw packer-state layout is only exposed without window shuffling.
    cfg = make_pipeline_cfg(local_text="abcde", window_shuffle_windows=0)

    it = build_train_iterator(cfg)
    _ = next(it)
    state = it.get_state()

    remaining = state.get("packer", {}).get("remaining_tokens")
    assert remaining, "expected non-empty packer buffer for alignment test"
    docs_seen_at_snapshot = state["packer"]["docs_seen"]
    assert docs_seen_at_snapshot > 0

    cont = [_batch_arrays(next(it)) for _ in range(3)]

    it2 = build_train_iterator(cfg)
    it2.set_state(state)
    # docs_seen/docs_truncated diagnostics must survive save/load, not reset to 0.
    assert it2.get_state()["packer"]["docs_seen"] == docs_seen_at_snapshot
    resumed = [_batch_arrays(next(it2)) for _ in range(3)]

    for batch_a, batch_b in zip(cont, resumed, strict=True):
        for arr_a, arr_b in zip(batch_a, batch_b, strict=True):
            np.testing.assert_array_equal(arr_a, arr_b)


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
