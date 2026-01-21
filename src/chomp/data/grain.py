"""Grain-backed iterator wrappers for chomp."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from chomp.config import Config
from chomp.types import Batch


class _IteratorProtocol(Protocol):
    """Protocol for Grain dataset iterators."""

    def __next__(self) -> Batch: ...

    def get_state(self) -> dict[str, Any]: ...

    def set_state(self, state: dict[str, Any]) -> None: ...


class GrainTrainBatchIterator:
    """Iterator wrapper that runs the pipeline through Grain."""

    def __init__(self, *, ds: Any, packing_mode: str) -> None:
        """Initialize the Grain-backed iterator.

        :param ds: Grain IterDataset yielding Batch objects.
        :param str packing_mode: Packing mode name for metrics.
        """
        self._ds = ds
        self._it: _IteratorProtocol = iter(ds)
        self._packing_mode = str(packing_mode)
        self._last_stats: dict[str, float | int | str] = {}

    def __iter__(self) -> GrainTrainBatchIterator:
        return self

    def __next__(self) -> Batch:
        batch = next(self._it)
        attn = batch.attention_mask
        tokens_used = int(np.count_nonzero(attn))
        capacity = int(attn.size)
        utilization = float(tokens_used / capacity) if capacity > 0 else 0.0
        self._last_stats = {
            "packing_mode": self._packing_mode,
            "packing_tokens": tokens_used,
            "packing_capacity": capacity,
            "packing_utilization": utilization,
        }
        return batch

    def get_state(self) -> dict[str, Any]:
        """Return iterator state for checkpointing."""
        return self._it.get_state()

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore iterator state from checkpoint."""
        self._it.set_state(state)
        self._last_stats = {}

    def get_stats(self) -> dict[str, float | int | str]:
        """Return latest packing stats from the iterator."""
        return dict(self._last_stats)


def build_grain_iterator(cfg: Config, *, tokenizer: Any) -> GrainTrainBatchIterator:
    """Build a Grain-backed batch iterator.

    :param Config cfg: Training configuration.
    :param tokenizer: Tokenizer instance for encoding text.
        :raises RuntimeError: If grain is not installed.
        :return GrainTrainBatchIterator: Iterator yielding Batch objects.
    """
    try:
        import grain.python as grain
    except Exception as exc:  # pragma: no cover - missing dependency
        raise RuntimeError("Grain is not installed. Install with `pip install grain`.") from exc

    class _TrainBatchDatasetIterator(grain.DatasetIterator):  # type: ignore[misc]
        """DatasetIterator that delegates to TrainBatchIterator."""

        def __init__(self, *, cfg: Config, tokenizer: Any) -> None:
            """Initialize the dataset iterator.

            :param Config cfg: Training configuration.
            :param tokenizer: Tokenizer instance for encoding text.
            """
            super().__init__()
            from chomp.data.pipeline import TrainBatchIterator

            self._it = TrainBatchIterator(cfg, tokenizer=tokenizer)

        def __next__(self) -> Batch:
            return next(self._it)

        def get_state(self) -> dict[str, Any]:
            """Return iterator state for checkpointing."""
            return self._it.get_state()

        def set_state(self, state: dict[str, Any]) -> None:
            """Restore iterator state from checkpoint."""
            self._it.set_state(state)

    class _TrainBatchIterDataset(grain.IterDataset):  # type: ignore[misc]
        """IterDataset that yields chomp Batch objects."""

        def __init__(self, *, cfg: Config, tokenizer: Any) -> None:
            """Initialize the dataset.

            :param Config cfg: Training configuration.
            :param tokenizer: Tokenizer instance for encoding text.
            """
            super().__init__()
            self._cfg = cfg
            self._tokenizer = tokenizer

        def __iter__(self) -> grain.DatasetIterator:
            return _TrainBatchDatasetIterator(cfg=self._cfg, tokenizer=self._tokenizer)

    ds = _TrainBatchIterDataset(cfg=cfg, tokenizer=tokenizer)

    if cfg.data.grain_prefetch > 0:
        ds = grain.experimental.ThreadPrefetchIterDataset(
            ds, prefetch_buffer_size=int(cfg.data.grain_prefetch)
        )

    return GrainTrainBatchIterator(ds=ds, packing_mode=cfg.data.packing_mode)
