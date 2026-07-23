"""Fresh-process worker for real Hugging Face stream resume tests."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

import datasets

from chomp.data.hf import HFStreamingTextStream, HFStreamSpec


def _items() -> Any:
    """Yield stable local documents through the real HF iterable machinery."""
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
    """Return a process-local real IterableDataset without network access."""
    _ = (dataset, name, split, streaming, revision)
    return datasets.IterableDataset.from_generator(_items)


def _spec() -> HFStreamSpec:
    """Build the resume contract shared by both worker processes."""
    return HFStreamSpec(
        dataset="local",
        name="default",
        split="train",
        text_key="text",
        revision=None,
        shuffle=True,
        shuffle_buffer_size=17,
        shuffle_buffer_bytes=1_000_000,
        seed=29,
        repeat=False,
        content_partition="all",
        eval_holdout_fraction=0.01,
    )


def main() -> None:
    """Write a checkpoint/continuation or restore it in a fresh process."""
    mode, state_arg, output_arg = sys.argv[1:4]
    state_path = Path(state_arg)
    output_path = Path(output_arg)
    datasets.load_dataset = _load_dataset  # type: ignore[assignment]
    stream = HFStreamingTextStream(_spec())

    if mode == "prepare":
        consumed = [next(stream) for _ in range(23)]
        with state_path.open("wb") as handle:
            pickle.dump(stream.get_state(), handle)
        continuation = [next(stream) for _ in range(41)]
        assert len(set(consumed + continuation)) == len(consumed + continuation)
    elif mode == "resume":
        with state_path.open("rb") as handle:
            state = pickle.load(handle)
        stream.set_state(state)
        continuation = [next(stream) for _ in range(41)]
    else:
        raise ValueError(f"Unknown worker mode: {mode!r}")

    output_path.write_text(json.dumps(continuation), encoding="utf-8")


if __name__ == "__main__":
    main()
