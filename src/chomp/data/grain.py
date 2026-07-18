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

    def __init__(self, *, ds: Any, packing_mode: str, enable_stats: bool) -> None:
        """Initialize the Grain-backed iterator.

        :param ds: Grain IterDataset yielding Batch objects.
        :param str packing_mode: Packing mode name for metrics.
        :param bool enable_stats: Whether to compute packing stats on each batch.
        """
        self._it: _IteratorProtocol = iter(ds)
        self._packing_mode = str(packing_mode)
        self._enable_stats = bool(enable_stats)
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
        if not self._enable_stats or not self._collect_next_stats:
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
        if not self._enable_stats:
            return {}
        return dict(self._last_stats)


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

    def __init__(self, parent: Any, *, cfg: Config, enable_stats: bool) -> None:
        """Initialize the batch assembler.

        :param parent: Parent iterator yielding packed windows.
        :param Config cfg: Training configuration.
        :param bool enable_stats: Whether to compute per-batch packer stats.
        """
        super().__init__(parent)
        self._spec = _BatchAssemblySpec.from_config(cfg)
        self._device_put = bool(cfg.data.device_put)
        self._enable_stats = bool(enable_stats)

    def __next__(self) -> _BatchEnvelope:
        batch, loss_tokens_host = _assemble_batch(lambda: next(self._parent), self._spec)
        stats = _packer_stats_from_chain(self._parent) if self._enable_stats else {}
        if self._device_put:
            import jax

            batch = jax.device_put(batch)
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

    def __init__(self, parent: Any, *, cfg: Config, enable_stats: bool) -> None:
        """Initialize the dataset.

        :param parent: Parent dataset yielding packed windows.
        :param Config cfg: Training configuration.
        :param bool enable_stats: Whether iterators compute per-batch packer stats.
        """
        super().__init__(parent)
        self._cfg = cfg
        self._enable_stats = bool(enable_stats)

    def __iter__(self) -> grain.DatasetIterator:
        return _BatchAssembleDatasetIterator(
            self._parent.__iter__(), cfg=self._cfg, enable_stats=self._enable_stats
        )


def build_grain_iterator(cfg: Config, *, tokenizer: Any) -> GrainTrainBatchIterator:
    """Build a Grain-backed batch iterator.

    Pipeline: sequence producer -> optional packed-window shuffle -> batch
    assembly -> optional thread prefetch. The window shuffle decorrelates
    batches from raw packer-output order within a token memory budget.

    :param Config cfg: Training configuration.
    :param tokenizer: Tokenizer instance for encoding text.
    :return GrainTrainBatchIterator: Iterator yielding Batch objects.
    """
    ds = _TrainSequenceIterDataset(cfg=cfg, tokenizer=tokenizer)

    window_rows = resolve_window_shuffle_rows(cfg)
    if window_rows > 0:
        ds = _ResumeSafeWindowShuffleIterDataset(
            ds,
            window_size=window_rows,
            seed=effective_window_shuffle_seed(cfg),
        )

    # Single source of the stats-gating rule: device_put moves batches to
    # device inside the iterator, and stats are disabled with it (see
    # GrainTrainBatchIterator.get_stats). The batch assembler receives the
    # same flag so it skips the per-batch chain walks nothing would read.
    enable_stats = not cfg.data.device_put
    ds = _BatchAssembleIterDataset(ds, cfg=cfg, enable_stats=enable_stats)

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
        enable_stats=enable_stats,
    )
