"""Grain-backed iterator wrappers for chomp."""

from __future__ import annotations

import logging
from typing import Any, Protocol

import numpy as np

from chomp.config import Config
from chomp.types import IGNORE_INDEX, Batch

logger = logging.getLogger(__name__)

# Seed offset for the packed-window shuffle so it is decoupled from the HF
# document-shuffle seed (which uses data.seed directly).
_WINDOW_SHUFFLE_SEED_OFFSET = 104_729


class _IteratorProtocol(Protocol):
    """Protocol for Grain dataset iterators."""

    def __next__(self) -> Batch: ...

    def get_state(self) -> dict[str, Any]:
        """Return iterator state for checkpointing."""
        ...

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore iterator state from a checkpoint."""
        ...


class GrainTrainBatchIterator:
    """Iterator wrapper that runs the pipeline through Grain."""

    def __init__(self, *, ds: Any, packing_mode: str, enable_stats: bool) -> None:
        """Initialize the Grain-backed iterator.

        :param ds: Grain IterDataset yielding Batch objects.
        :param str packing_mode: Packing mode name for metrics.
        :param bool enable_stats: Whether to compute packing stats on each batch.
        """
        self._ds = ds
        self._it: _IteratorProtocol = iter(ds)
        self._packing_mode = str(packing_mode)
        self._enable_stats = bool(enable_stats)
        self._last_stats: dict[str, float | int | str] = {}

    def __iter__(self) -> GrainTrainBatchIterator:
        return self

    def __next__(self) -> Batch:
        batch = next(self._it)
        if not self._enable_stats:
            self._last_stats = {}
            return batch
        attn = np.asarray(batch.attention_mask, dtype=bool)
        labels = np.asarray(batch.labels)
        segs = np.asarray(batch.segment_ids)

        tokens_used = int(np.count_nonzero(attn))
        capacity = int(attn.size)
        utilization = float(tokens_used / capacity) if capacity > 0 else 0.0
        valid_loss = labels[..., 1:] != int(IGNORE_INDEX)
        valid_loss = valid_loss & attn[..., 1:]
        loss_tokens_host = int(np.count_nonzero(valid_loss))

        # Reshape commutes with the per-last-axis boundary op, so one [rows, T-1]
        # array serves both the global count and the per-sequence doc counts.
        flat_segs = segs.reshape(-1, segs.shape[-1])
        seq_boundary = (
            (flat_segs[:, 1:] != flat_segs[:, :-1])
            & (flat_segs[:, 1:] > 0)
            & (flat_segs[:, :-1] > 0)
        )
        boundary_transitions = int(np.count_nonzero(seq_boundary))

        has_tokens = np.any(flat_segs > 0, axis=1)
        docs_per_seq = np.where(has_tokens, 1 + seq_boundary.sum(axis=1), 0).astype(np.int32)
        self._last_stats = {
            "packing_mode": self._packing_mode,
            "packing_tokens": tokens_used,
            "packing_capacity": capacity,
            "packing_utilization": utilization,
            "loss_tokens_host": loss_tokens_host,
            "boundary_transitions": boundary_transitions,
            "docs_per_seq_mean": float(np.mean(docs_per_seq)),
            "docs_per_seq_min": int(np.min(docs_per_seq)),
            "docs_per_seq_max": int(np.max(docs_per_seq)),
        }
        return batch

    def get_state(self) -> dict[str, Any]:
        """Return iterator state for checkpointing.

        :return dict[str, Any]: Serializable iterator state.
        """
        return self._it.get_state()

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore iterator state from checkpoint."""
        self._it.set_state(state)
        self._last_stats = {}

    def checkpoint_target(self) -> Any:
        """Return the Grain iterator used for Orbax checkpointing.

        :return Any: The underlying Grain DatasetIterator.
        """
        return self._it

    def get_stats(self) -> dict[str, float | int | str]:
        """Return latest packing stats from the iterator.

        :return dict[str, float | int | str]: Utilization stats for the last batch.
        """
        if not self._enable_stats:
            return {}
        stats: dict[str, float | int | str] = dict(self._last_stats)
        extra = _packer_stats_from_chain(self._it)
        if extra:
            stats.update(extra)
        return stats


def _packer_stats_from_chain(it: Any) -> dict[str, Any]:
    """Walk a Grain iterator chain to the first node exposing get_stats().

    Intermediate nodes (window shuffle, prefetch) don't expose packer stats;
    the sequence-producer iterator at the bottom of the chain does.

    Two failure modes are handled deliberately differently: grain's
    `DatasetIterator._parent` is a property that raises AssertionError (not
    AttributeError) on parentless nodes, which is expected and kept narrow to
    the parent walk. A packer's own get_stats() raising is a real bug and
    must not vanish silently, so it is caught separately, logged once per
    node (to avoid per-batch log spam), and only then downgraded to {}.

    :param it: Outermost Grain DatasetIterator.
    :return dict[str, Any]: Packer stats, or an empty dict if unreachable.
    """
    node = it
    while node is not None:
        get_stats = getattr(node, "get_stats", None)
        if callable(get_stats):
            try:
                return dict(get_stats())
            except Exception as exc:
                if not getattr(node, "_chomp_stats_error_warned", False):
                    logger.warning(
                        "packer get_stats() raised %s: %s; packing stats will be "
                        "empty for this iterator until the underlying bug is fixed.",
                        type(exc).__name__,
                        exc,
                    )
                    node._chomp_stats_error_warned = True
                return {}
        # grain's DatasetIterator._parent is a property that raises
        # AssertionError (not AttributeError) on parentless nodes.
        try:
            node = getattr(node, "_parent", None)
        except AssertionError:
            return {}
    return {}


def _make_grain_iter_classes(grain: Any) -> tuple[type[Any], type[Any], type[Any]]:
    """Create Grain dataset classes without importing grain at module import time.

    :param grain: Imported grain module.
    :return tuple[type[Any], type[Any], type[Any]]: Sequence-level, batch-assembly,
        and resume-safe window-shuffle IterDatasets.
    """

    class _TrainSequenceDatasetIterator(grain.DatasetIterator):  # type: ignore[misc]
        """Source DatasetIterator yielding packed [T] windows from _SequenceProducer."""

        def __init__(self, *, cfg: Config, tokenizer: Any) -> None:
            """Initialize the sequence iterator.

            :param Config cfg: Training configuration.
            :param tokenizer: Tokenizer instance for encoding text.
            """
            super().__init__()
            from chomp.data.pipeline import _SequenceProducer

            self._producer = _SequenceProducer(cfg, tokenizer=tokenizer)

        def __next__(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            return self._producer.next_window()

        def get_state(self) -> dict[str, Any]:
            """Return producer state for checkpointing.

            :return dict[str, Any]: Serializable producer state.
            """
            return self._producer.get_state()

        def set_state(self, state: dict[str, Any]) -> None:
            """Restore producer state from checkpoint."""
            self._producer.set_state(state)

        def get_stats(self) -> dict[str, int]:
            """Return packer-level document stats.

            :return dict[str, int]: Packer stats, or an empty dict if unavailable.
            """
            return self._producer.get_stats()

    class _TrainSequenceIterDataset(grain.IterDataset):  # type: ignore[misc]
        """IterDataset yielding packed [T] windows."""

        def __init__(self, *, cfg: Config, tokenizer: Any) -> None:
            """Initialize the dataset.

            :param Config cfg: Training configuration.
            :param tokenizer: Tokenizer instance for encoding text.
            """
            super().__init__()
            self._cfg = cfg
            self._tokenizer = tokenizer

        def __iter__(self) -> grain.DatasetIterator:
            return _TrainSequenceDatasetIterator(cfg=self._cfg, tokenizer=self._tokenizer)

    class _ResumeSafeWindowShuffleIterDataset(  # type: ignore[misc]
        grain.experimental.WindowShuffleIterDataset
    ):
        """WindowShuffleIterDataset with an exact-resume fix.

        Upstream `_WindowShuffleDatasetIterator.set_state` leaves the
        iterator's `_init` flag True, so the first window refill after a
        restore skips the window_index increment and replays the previous
        window's permutation seed, diverging from the continuous run.
        `set_state` itself fills the current window, so every later refill
        must increment: clearing `_init` after restore is always correct.
        """

        def __iter__(self) -> Any:
            it = super().__iter__()
            if not hasattr(it, "_init"):
                # Fail fast: silently skipping the patch would resurrect the
                # upstream resume bug with no signal, breaking exact resume.
                raise RuntimeError(
                    "grain's _WindowShuffleDatasetIterator no longer has an _init "
                    "attribute; the resume-exactness workaround in "
                    "_ResumeSafeWindowShuffleIterDataset must be re-verified against "
                    "this grain version (see chomp docs/packing.md, window shuffling)."
                )
            original_set_state = it.set_state

            def _set_state(state: dict[str, Any]) -> None:
                """Restore state, then clear _init so restore wins over lazy init.

                :param dict[str, Any] state: Serialized iterator state.
                """
                original_set_state(state)
                it._init = False

            it.set_state = _set_state
            return it

    class _BatchAssembleDatasetIterator(grain.DatasetIterator):  # type: ignore[misc]
        """Assembles [A, B, T] Batch objects from a stream of packed windows.

        Holds no checkpoint state of its own: it fully drains exactly A*B
        windows per __next__, and checkpointing only happens between batches,
        so state forwards 1:1 to the parent iterator.

        docs_added_this_batch is measured here, below any prefetch layer, so
        it reflects actual stream pulls for the assembled batch rather than
        prefetch-thread timing. With prefetch enabled the reported value may
        belong to a batch up to prefetch-depth ahead of the one just consumed.

        Thread safety: with prefetch, __next__ runs on the prefetch thread
        while get_stats is called from the consumer thread. All packer/producer
        access happens inside __next__; the finished stats dict is published
        via a single attribute assignment (atomic under the GIL) and get_stats
        only reads that snapshot — it never walks into producer-owned objects.
        """

        def __init__(self, parent: Any, *, cfg: Config) -> None:
            """Initialize the batch assembler.

            :param parent: Parent DatasetIterator yielding packed windows.
            :param Config cfg: Training configuration.
            """
            super().__init__(parent)
            self._A = int(cfg.train.grad_accum)
            self._B = int(cfg.train.batch_size)
            self._T = int(cfg.train.seq_len)
            self._device_put = bool(cfg.data.device_put)
            self._mask_boundary_loss = bool(cfg.data.mask_boundary_loss)
            self._train_on_eos = bool(cfg.data.train_on_eos)
            self._eos_id = int(cfg.model.eos_token_id)
            self._stats_snapshot: dict[str, Any] = {}

        def _docs_seen(self) -> int | None:
            """Read the packer's docs_seen counter from the parent chain.

            :return int | None: docs_seen if reachable, else None.
            """
            value = _packer_stats_from_chain(self._parent).get("docs_seen")
            return int(value) if value is not None else None

        def __next__(self) -> Batch:
            from chomp.data.pipeline import _assemble_batch

            docs_seen_before = self._docs_seen()
            batch = _assemble_batch(
                lambda: next(self._parent),
                grad_accum=self._A,
                batch_size=self._B,
                seq_len=self._T,
                mask_boundary_loss=self._mask_boundary_loss,
                train_on_eos=self._train_on_eos,
                eos_id=self._eos_id,
                device_put=self._device_put,
            )
            stats = _packer_stats_from_chain(self._parent)
            docs_seen_after = stats.get("docs_seen")
            if docs_seen_before is not None and docs_seen_after is not None:
                # Fresh documents pulled from the stream while assembling this
                # batch. Collapses toward 0 while already-buffered content
                # drains; bursty when a shuffle window refills.
                stats["docs_added_this_batch"] = int(docs_seen_after) - docs_seen_before
            # Single-assignment publish; never mutated afterwards.
            self._stats_snapshot = stats
            return batch

        def get_state(self) -> dict[str, Any]:
            """Return parent iterator state for checkpointing.

            :return dict[str, Any]: Serializable iterator state.
            """
            return self._parent.get_state()

        def set_state(self, state: dict[str, Any]) -> None:
            """Restore parent iterator state from checkpoint."""
            self._parent.set_state(state)
            self._stats_snapshot = {}

        def get_stats(self) -> dict[str, Any]:
            """Return the stats snapshot published by the last assembled batch.

            Safe to call from the consumer thread under prefetch: reads a
            reference published atomically by __next__, never producer state.

            :return dict[str, Any]: Packer stats merged with docs_added_this_batch.
            """
            return dict(self._stats_snapshot)

    class _BatchAssembleIterDataset(grain.IterDataset):  # type: ignore[misc]
        """IterDataset that yields chomp Batch objects."""

        def __init__(self, parent: Any, *, cfg: Config) -> None:
            """Initialize the dataset.

            :param parent: Parent IterDataset yielding packed windows.
            :param Config cfg: Training configuration.
            """
            super().__init__(parent)
            self._cfg = cfg

        def __iter__(self) -> grain.DatasetIterator:
            return _BatchAssembleDatasetIterator(self._parent.__iter__(), cfg=self._cfg)

    return _TrainSequenceIterDataset, _BatchAssembleIterDataset, _ResumeSafeWindowShuffleIterDataset


def build_grain_iterator(cfg: Config, *, tokenizer: Any) -> GrainTrainBatchIterator:
    """Build a Grain-backed batch iterator.

    Pipeline: sequence producer -> optional packed-window shuffle -> batch
    assembly -> optional thread prefetch. The window shuffle decorrelates
    batches from raw packer-output order (data.window_shuffle_windows).

    :param Config cfg: Training configuration.
    :param tokenizer: Tokenizer instance for encoding text.
        :raises RuntimeError: If grain is not installed.
        :return GrainTrainBatchIterator: Iterator yielding Batch objects.
    """
    try:
        import grain.python as grain
    except Exception as exc:  # pragma: no cover - missing dependency
        raise RuntimeError("Grain is not installed. Install with `pip install grain`.") from exc

    _TrainSequenceIterDataset, _BatchAssembleIterDataset, _WindowShuffle = _make_grain_iter_classes(
        grain
    )
    ds = _TrainSequenceIterDataset(cfg=cfg, tokenizer=tokenizer)

    if cfg.data.window_shuffle_windows > 0:
        ds = _WindowShuffle(
            ds,
            window_size=int(cfg.data.window_shuffle_windows),
            seed=int(cfg.data.seed) + _WINDOW_SHUFFLE_SEED_OFFSET,
        )

    ds = _BatchAssembleIterDataset(ds, cfg=cfg)

    if cfg.data.device_put and cfg.data.grain_prefetch > 0:
        logger.warning(
            "data.device_put=True with grain_prefetch>0 may place device transfers on "
            "background threads; consider setting data.device_put=false."
        )

    if cfg.data.grain_prefetch > 0:
        ds = grain.experimental.ThreadPrefetchIterDataset(
            ds, prefetch_buffer_size=int(cfg.data.grain_prefetch)
        )

    return GrainTrainBatchIterator(
        ds=ds,
        packing_mode=cfg.data.packing_mode,
        enable_stats=not cfg.data.device_put,
    )
