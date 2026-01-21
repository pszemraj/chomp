"""Grain-backed iterator should support state roundtrip."""

from __future__ import annotations

import numpy as np

from chomp.config import Config, DataConfig, ModelConfig, TokenizerConfig, TrainConfig
from chomp.data.pipeline import build_train_iterator


def test_grain_iterator_state_roundtrip():
    cfg = Config(
        model=ModelConfig(
            backend="dummy", vocab_size=512, d_model=32, dropout=0.0, segment_masking=True
        ),
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
    )

    it = build_train_iterator(cfg)
    _ = next(it)
    state = it.get_state()

    next_a = next(it)

    it2 = build_train_iterator(cfg)
    it2.set_state(state)
    next_b = next(it2)

    np.testing.assert_array_equal(next_a.input_ids, next_b.input_ids)
    np.testing.assert_array_equal(next_a.labels, next_b.labels)
    np.testing.assert_array_equal(next_a.attention_mask, next_b.attention_mask)
    np.testing.assert_array_equal(next_a.segment_ids, next_b.segment_ids)
