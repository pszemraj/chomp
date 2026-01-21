"""Iterator state alignment test.

We checkpoint the data iterator while the packer buffer is non-empty and ensure
that restoring into a fresh iterator yields the exact same token stream.
"""

from __future__ import annotations

import numpy as np

from chomp.config import Config, DataConfig, ModelConfig, TokenizerConfig, TrainConfig
from chomp.data.pipeline import build_train_iterator


def _batch_arrays(batch):
    return batch.input_ids, batch.labels, batch.attention_mask, batch.segment_ids


def test_packer_alignment_after_restore():
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
