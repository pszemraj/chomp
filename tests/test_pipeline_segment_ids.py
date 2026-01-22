"""Pipeline should emit segment IDs from real text tokenization."""

from __future__ import annotations

import numpy as np

from chomp.config import Config, DataConfig, ModelConfig, TokenizerConfig, TrainConfig
from chomp.data.pipeline import build_train_iterator


def test_pipeline_segment_ids_multiple_docs():
    cfg = Config(
        model=ModelConfig(
            backend="dummy", vocab_size=512, d_model=32, dropout=0.0, segment_masking=False
        ),
        data=DataConfig(
            backend="local_text",
            repeat=True,
            local_text="hi",
            mask_boundary_loss=True,
            train_on_eos=True,
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
    masked_labels = batch.labels[0, 0][1:][boundary]
    assert np.all(masked_labels == -100)


def test_boundary_loss_mask_toggle():
    cfg = Config(
        model=ModelConfig(
            backend="dummy", vocab_size=512, d_model=32, dropout=0.0, segment_masking=False
        ),
        data=DataConfig(
            backend="local_text",
            repeat=True,
            local_text="hi",
            mask_boundary_loss=False,
            train_on_eos=True,
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
    boundary = segs[1:] != segs[:-1]
    assert boundary.any()
    labels_at_boundary = batch.labels[0, 0][1:][boundary]
    assert np.all(labels_at_boundary != -100)


def test_pipeline_bin_packing_segment_ids():
    cfg = Config(
        model=ModelConfig(
            backend="dummy", vocab_size=512, d_model=32, dropout=0.0, segment_masking=False
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
    unique = np.unique(segs[segs > 0])
    assert unique.size >= 2
    assert np.array_equal(batch.attention_mask, batch.segment_ids > 0)

    stats = it.get_stats()
    assert stats["packing_mode"] == "bin"
    assert stats["packing_capacity"] == batch.segment_ids.size
    expected_util = float(np.count_nonzero(batch.attention_mask) / batch.attention_mask.size)
    assert stats["packing_utilization"] == expected_util
