"""Grain-backed iterator wrappers for chomp."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

import grain.python as grain
import numpy as np

from chomp.config import Config, resolve_window_shuffle_rows
from chomp.data.pipeline import (
    _assemble_batch,
    _BatchAssemblySpec,
    _SequenceProducer,
    effective_window_shuffle_seed,
)
from chomp.types import Batch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _BatchEnvelope:
    """One batch paired with exact host accounting and source statistics."""

    batch: Batch
    loss_tokens_host: int
    pipeline_stats: dict[str, Any]


def _batch_segment_stats(segment_ids: Any) -> dict[str, float | int]:
    """Compute packing diagnostics from one fixed-shape segment-ID batch.

    :param Any segment_ids: Array with positive packed segments and zero padding.
    :return dict[str, float | int]: Utilization, boundary, and segment-density metrics.
    """
    segs = np.asarray(segment_ids)
    tokens_used = int(np.count_nonzero(segs))
    capacity = int(segs.size)
    flat_segs = segs.reshape(-1, segs.shape[-1])
    boundaries = (
        (flat_segs[:, 1:] != flat_segs[:, :-1]) & (flat_segs[:, 1:] > 0) & (flat_segs[:, :-1] > 0)
    )
    has_tokens = np.any(flat_segs > 0, axis=1)
    segments_per_seq = np.where(has_tokens, 1 + boundaries.sum(axis=1), 0)
    return {
        "packing_tokens": tokens_used,
        "packing_capacity": capacity,
        "packing_utilization": float(tokens_used / capacity),
        "boundary_transitions": int(np.count_nonzero(boundaries)),
        "segments_per_seq_mean": float(np.mean(segments_per_seq)),
        "segments_per_seq_min": int(np.min(segments_per_seq)),
        "segments_per_seq_max": int(np.max(segments_per_seq)),
    }


class _IteratorProtocol(Protocol):
    """Protocol for Grain dataset iterators."""

    def __next__(self) -> _BatchEnvelope: ...

    def get_state(self) -> dict[str, Any]:
        """Return iterator state for checkpointing."""
        ...

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore iterator state from a checkpoint."""
        ...


class GrainTrainBatchIterator:
    """Iterator wrapper that runs the pipeline through Grain."""

    def __init__(self, *, ds: Any, packing_mode: str) -> None:
        """Initialize the Grain-backed iterator.

        :param ds: Grain IterDataset yielding Batch objects.
        :param str packing_mode: Packing mode name for metrics.
        """
        self._it: _IteratorProtocol = iter(ds)
        self._packing_mode = str(packing_mode)
        self._last_stats: dict[str, float | int | str] = {}
        self._last_loss_tokens: int | None = None
        self._collect_next_stats = True

    def __iter__(self) -> GrainTrainBatchIterator:
        return self

    def __next__(self) -> Batch:
        envelope = next(self._it)
        batch = envelope.batch
        self._last_loss_tokens = int(envelope.loss_tokens_host)
        self._last_stats = dict(envelope.pipeline_stats)
        if not self._collect_next_stats:
            return batch
        self._last_stats.update(_batch_segment_stats(batch.segment_ids))
        self._last_stats.update(
            {
                "packing_mode": self._packing_mode,
                "loss_tokens_host": self._last_loss_tokens,
            }
        )
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
        self._last_loss_tokens = None
        self._collect_next_stats = True

    def get_loss_tokens(self) -> int:
        """Return the exact valid-target count paired with the last batch.

        :raises RuntimeError: If no batch has been yielded since construction/restore.
        :return int: Host-computed valid causal target count.
        """
        if self._last_loss_tokens is None:
            raise RuntimeError("No batch is available for host loss-token accounting")
        return self._last_loss_tokens

    def set_collect_stats(self, enabled: bool) -> None:
        """Choose whether the next consumed batch computes full packing diagnostics.

        Exact loss-token counting is unaffected; it is carried in the batch
        envelope and remains available on every step.

        :param bool enabled: Whether to scan the next batch for diagnostics.
        """
        self._collect_next_stats = bool(enabled)

    def checkpoint_target(self) -> Any:
        """Return the Grain iterator used for Orbax checkpointing.

        :return Any: The underlying Grain DatasetIterator.
        """
        return self._it

    def close(self) -> None:
        """Stop prefetch and recursively release the source iterator."""
        # Grain's ordinary DatasetIterator.close() recursively closes parents,
        # but ThreadPrefetchDatasetIterator overrides it to stop only its own
        # worker. Close that outer node first, then its parent; the parent's
        # base implementation handles the remainder of the chain.
        try:
            parent = getattr(self._it, "_parent", None)
        except AssertionError:
            parent = None
        self._it.close()
        if parent is not None:
            parent.close()

    def get_stats(self) -> dict[str, float | int | str]:
        """Return latest packing stats from the iterator.

        :return dict[str, float | int | str]: Utilization stats for the last batch.
        """
        return dict(self._last_stats)


def _packer_stats_from_chain(it: Any) -> dict[str, Any]:
    """Walk a Grain iterator chain to the first node exposing get_stats().

    Intermediate nodes (window shuffle, prefetch) don't expose packer stats;
    the sequence-producer iterator at the bottom of the chain does.

    :param it: Outermost Grain DatasetIterator.
    :return dict[str, Any]: Packer stats, or an empty dict if unreachable.
    """
    node = it
    while node is not None:
        get_stats = getattr(node, "get_stats", None)
        if callable(get_stats):
            return dict(get_stats())
        # grain's DatasetIterator._parent is a property that raises
        # AssertionError (not AttributeError) on parentless nodes.
        try:
            node = getattr(node, "_parent", None)
        except AssertionError:
            return {}
    return {}


class _TrainSequenceDatasetIterator(grain.DatasetIterator):
    """Source iterator yielding packed ``[T]`` windows."""

    def __init__(self, *, cfg: Config, tokenizer: Any) -> None:
        """Initialize the sequence iterator.

        :param Config cfg: Training configuration.
        :param tokenizer: Tokenizer instance for encoding text.
        """
        super().__init__()
        self._producer = _SequenceProducer(cfg, tokenizer=tokenizer)

    def __next__(self) -> tuple[np.ndarray, np.ndarray]:
        return self._producer.next_window()

    def get_state(self) -> dict[str, Any]:
        """Return producer state for checkpointing.

        :return dict[str, Any]: Serializable producer state.
        """
        return self._producer.get_state()

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore producer state from checkpoint."""
        self._producer.set_state(state)

    def get_stats(self) -> dict[str, int | float]:
        """Return packer-level document stats.

        :return dict[str, int | float]: Packer stats, or an empty dict if unavailable.
        """
        return self._producer.get_stats()

    def close(self) -> None:
        """Release the producer before closing the Grain source node."""
        try:
            self._producer.close()
        finally:
            super().close()


class _TrainSequenceIterDataset(grain.IterDataset):
    """Dataset yielding packed ``[T]`` windows."""

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


class _ResumeSafeWindowShuffleIterDataset(grain.experimental.WindowShuffleIterDataset):
    """Window shuffle with an exact-resume fix for pinned Grain.

    Upstream ``set_state`` leaves the iterator's ``_init`` flag true, so the
    next refill reuses the previous window's permutation seed. Restoring fills
    the current window, which means every later refill must increment normally.
    """

    def __iter__(self) -> Any:
        it = super().__iter__()
        if not hasattr(it, "_init"):
            raise RuntimeError(
                "grain's _WindowShuffleDatasetIterator no longer has an _init "
                "attribute; the resume-exactness workaround in "
                "_ResumeSafeWindowShuffleIterDataset must be re-verified against "
                "this grain version (see chomp docs/packing.md, window shuffling)."
            )
        original_set_state = it.set_state

        def _set_state(state: dict[str, Any]) -> None:
            """Restore state, then clear ``_init`` so restore wins over lazy init.

            :param dict[str, Any] state: Serialized iterator state.
            """
            original_set_state(state)
            it._init = False

        it.set_state = _set_state
        return it


class _BatchAssembleDatasetIterator(grain.DatasetIterator):
    """Assemble fixed batches and forward parent checkpoint state."""

    def __init__(self, parent: Any, *, cfg: Config) -> None:
        """Initialize the batch assembler.

        :param parent: Parent iterator yielding packed windows.
        :param Config cfg: Training configuration.
        """
        super().__init__(parent)
        self._spec = _BatchAssemblySpec.from_config(cfg)

    def __next__(self) -> _BatchEnvelope:
        batch, loss_tokens_host = _assemble_batch(lambda: next(self._parent), self._spec)
        stats = _packer_stats_from_chain(self._parent)
        return _BatchEnvelope(
            batch=batch,
            loss_tokens_host=loss_tokens_host,
            pipeline_stats=stats,
        )

    def get_state(self) -> dict[str, Any]:
        """Return parent iterator state for checkpointing.

        :return dict[str, Any]: Serializable iterator state.
        """
        return self._parent.get_state()

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore parent iterator state from checkpoint."""
        self._parent.set_state(state)


class _BatchAssembleIterDataset(grain.IterDataset):
    """Dataset yielding fixed chomp batches."""

    def __init__(self, parent: Any, *, cfg: Config) -> None:
        """Initialize the dataset.

        :param parent: Parent dataset yielding packed windows.
        :param Config cfg: Training configuration.
        """
        super().__init__(parent)
        self._cfg = cfg

    def __iter__(self) -> grain.DatasetIterator:
        return _BatchAssembleDatasetIterator(self._parent.__iter__(), cfg=self._cfg)


def build_grain_iterator(cfg: Config, *, tokenizer: Any) -> GrainTrainBatchIterator:
    """Build a Grain-backed batch iterator.

    Pipeline: sequence producer -> optional packed-window shuffle -> batch
    assembly -> optional thread prefetch. The window shuffle decorrelates
    batches from raw packer-output order within token and row bounds.

    :param Config cfg: Training configuration.
    :param tokenizer: Tokenizer instance for encoding text.
    :return GrainTrainBatchIterator: Iterator yielding Batch objects.
    """
    ds = _TrainSequenceIterDataset(cfg=cfg, tokenizer=tokenizer)

    window_rows = resolve_window_shuffle_rows(cfg)
    if window_rows > 0:
        window_tokens = window_rows * int(cfg.train.seq_len)
        logger.info(
            "Packed-window shuffle: %d rows / %d tokens (token budget=%d, max rows=%d)",
            window_rows,
            window_tokens,
            cfg.data.window_shuffle_tokens,
            cfg.data.window_shuffle_max_rows,
        )
        ds = _ResumeSafeWindowShuffleIterDataset(
            ds,
            window_size=window_rows,
            seed=effective_window_shuffle_seed(cfg),
        )

    ds = _BatchAssembleIterDataset(ds, cfg=cfg)

    if cfg.data.grain_prefetch > 0:
        ds = grain.experimental.ThreadPrefetchIterDataset(
            ds, prefetch_buffer_size=int(cfg.data.grain_prefetch)
        )

    return GrainTrainBatchIterator(
        ds=ds,
        packing_mode=cfg.data.packing_mode,
    )
