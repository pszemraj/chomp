"""Grain-backed iterator wrappers for chomp."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Protocol

import numpy as np

from chomp.config import Config, resolve_window_shuffle_rows
from chomp.types import IGNORE_INDEX, Batch

logger = logging.getLogger(__name__)

# Seed offset for the packed-window shuffle so it is decoupled from the HF
# document-shuffle seed (which uses data.seed directly).
_WINDOW_SHUFFLE_SEED_OFFSET = 104_729
_UINT32_MODULUS = 2**32


@dataclass(frozen=True)
class _BatchEnvelope:
    """One batch paired with its exact pre-device loss-token count."""

    batch: Batch
    loss_tokens_host: int


def effective_window_shuffle_seed(cfg: Config) -> int:
    """Return the deterministic seed consumed by packed-window shuffling.

    :param Config cfg: Training configuration.
    :return int: Effective Grain window-shuffle seed.
    """
    return (int(cfg.data.seed) + _WINDOW_SHUFFLE_SEED_OFFSET) % _UINT32_MODULUS


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
        self._ds = ds
        self._it: _IteratorProtocol = iter(ds)
        self._packing_mode = str(packing_mode)
        self._enable_stats = bool(enable_stats)
        self._last_stats: dict[str, float | int | str] = {}
        self._last_loss_tokens: int | None = None

    def __iter__(self) -> GrainTrainBatchIterator:
        return self

    def __next__(self) -> Batch:
        envelope = next(self._it)
        batch = envelope.batch
        self._last_loss_tokens = int(envelope.loss_tokens_host)
        if not self._enable_stats:
            self._last_stats = {}
            return batch
        attn = np.asarray(batch.attention_mask, dtype=bool)
        segs = np.asarray(batch.segment_ids)

        tokens_used = int(np.count_nonzero(attn))
        capacity = int(attn.size)
        utilization = float(tokens_used / capacity) if capacity > 0 else 0.0
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
            "loss_tokens_host": self._last_loss_tokens,
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
        self._last_loss_tokens = None

    def get_loss_tokens(self) -> int:
        """Return the exact valid-target count paired with the last batch.

        :raises RuntimeError: If no batch has been yielded since construction/restore.
        :return int: Host-computed valid causal target count.
        """
        if self._last_loss_tokens is None:
            raise RuntimeError("No batch is available for host loss-token accounting")
        return self._last_loss_tokens

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

        `_init` is a private grain internal, so pyproject.toml upper-bounds
        the grain dependency; re-verify this workaround before raising that
        cap. The `hasattr` tripwire below fires at iterator construction on
        every run (fresh or resumed), so an incompatible grain fails fast.
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

        Holds no checkpoint state of its own: it drains up to A*B windows per
        __next__ and pads missing rows only after exact upstream exhaustion.
        Checkpointing happens between batches, so state forwards 1:1 to the
        parent iterator.

        docs_added_this_batch is measured here, below any prefetch layer, so
        it reflects actual stream pulls for the assembled batch rather than
        prefetch-thread timing. With prefetch enabled the reported value may
        belong to a batch up to prefetch-depth ahead of the one just consumed.

        Thread safety: with prefetch, __next__ runs on the prefetch thread
        while get_stats is called from the consumer thread. All packer/producer
        access happens inside __next__; the finished stats dict is published
        via a single attribute assignment (atomic under the GIL) and get_stats
        only reads that snapshot — it never walks into producer-owned objects.

        With enable_stats=False the per-batch chain walks are skipped entirely
        (nothing would ever read the snapshot).
        """

        def __init__(self, parent: Any, *, cfg: Config, enable_stats: bool) -> None:
            """Initialize the batch assembler.

            :param parent: Parent DatasetIterator yielding packed windows.
            :param Config cfg: Training configuration.
            :param bool enable_stats: Whether to compute per-batch packer stats.
            """
            super().__init__(parent)
            from chomp.data.pipeline import _BatchAssemblySpec

            # Count targets while arrays are still on host, then optionally
            # transfer the batch. The count travels beside the batch through
            # prefetch, so accounting cannot observe an ahead-of-consumer
            # stats snapshot.
            self._spec = replace(_BatchAssemblySpec.from_config(cfg), device_put=False)
            self._device_put = bool(cfg.data.device_put)
            self._enable_stats = bool(enable_stats)
            self._stats_snapshot: dict[str, Any] = {}

        def _docs_seen(self) -> int | None:
            """Read the packer's docs_seen counter from the parent chain.

            :return int | None: docs_seen if reachable, else None.
            """
            value = _packer_stats_from_chain(self._parent).get("docs_seen")
            return int(value) if value is not None else None

        def __next__(self) -> _BatchEnvelope:
            from chomp.data.pipeline import _assemble_batch

            docs_seen_before = self._docs_seen() if self._enable_stats else None
            batch = _assemble_batch(lambda: next(self._parent), self._spec)
            labels = np.asarray(batch.labels)
            attention = np.asarray(batch.attention_mask, dtype=bool)
            valid = (labels[..., 1:] != int(IGNORE_INDEX)) & attention[..., 1:]
            loss_tokens_host = int(np.count_nonzero(valid))
            if self._enable_stats:
                stats = _packer_stats_from_chain(self._parent)
                docs_seen_after = stats.get("docs_seen")
                if docs_seen_before is not None and docs_seen_after is not None:
                    # Fresh documents pulled from the stream while assembling this
                    # batch. Collapses toward 0 while already-buffered content
                    # drains; bursty when a shuffle window refills.
                    stats["docs_added_this_batch"] = int(docs_seen_after) - docs_seen_before
                # Single-assignment publish; never mutated afterwards.
                self._stats_snapshot = stats
            if self._device_put:
                import jax

                batch = jax.device_put(batch)
            return _BatchEnvelope(batch=batch, loss_tokens_host=loss_tokens_host)

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

        def __init__(self, parent: Any, *, cfg: Config, enable_stats: bool) -> None:
            """Initialize the dataset.

            :param parent: Parent IterDataset yielding packed windows.
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

    return _TrainSequenceIterDataset, _BatchAssembleIterDataset, _ResumeSafeWindowShuffleIterDataset


def build_grain_iterator(cfg: Config, *, tokenizer: Any) -> GrainTrainBatchIterator:
    """Build a Grain-backed batch iterator.

    Pipeline: sequence producer -> optional packed-window shuffle -> batch
    assembly -> optional thread prefetch. The window shuffle decorrelates
    batches from raw packer-output order within a token memory budget.

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

    window_rows = resolve_window_shuffle_rows(cfg)
    if window_rows > 0:
        ds = _WindowShuffle(
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
