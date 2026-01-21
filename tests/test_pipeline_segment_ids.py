"""Pipeline should emit segment IDs from real text tokenization."""

from __future__ import annotations

import numpy as np

from chomp.config import Config, DataConfig, ModelConfig, TokenizerConfig, TrainConfig
from chomp.data.pipeline import build_train_iterator


def test_pipeline_segment_ids_multiple_docs():
    cfg = Config(
        model=ModelConfig(
            backend="dummy", vocab_size=512, d_model=32, dropout=0.0, segment_masking=True
        ),
        data=DataConfig(
            backend="local_text",
            repeat=True,
            local_text="hi",
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
    batch = next(it)
    segs = batch.segment_ids[0, 0]
    unique = np.unique(segs)
    assert unique.size >= 2
    assert np.all(unique > 0)

    boundary = segs[1:] != segs[:-1]
    assert boundary.any()
    masked_labels = batch.labels[0, 0][:-1][boundary]
    assert np.all(masked_labels == -100)
