"""Labels should align with input_ids (model shifts internally)."""

from __future__ import annotations

import numpy as np

from chomp.config import Config, DataConfig, ModelConfig, TokenizerConfig, TrainConfig
from chomp.data.pipeline import build_train_iterator


def test_labels_align_with_inputs_except_masked() -> None:
    """Labels should match inputs except where masked with -100."""
    cfg = Config(
        model=ModelConfig(
            backend="dummy", vocab_size=512, d_model=32, dropout=0.0, segment_masking=False
        ),
        data=DataConfig(
            backend="local_text",
            repeat=True,
            local_text="hi",
            mask_boundary_loss=True,
            train_on_eos=False,
            tokenizer=TokenizerConfig(kind="byte", byte_offset=4, add_bos=True, add_eos=True),
        ),
        train=TrainConfig(
            steps=1,
            batch_size=1,
            seq_len=8,
            grad_accum=2,
            jit=False,
            deterministic=True,
            allow_cpu=True,
        ),
    )

    it = build_train_iterator(cfg)
    batch = next(it)

    labels = np.asarray(batch.labels)
    inputs = np.asarray(batch.input_ids)
    mask = labels != -100

    assert mask.any()
    assert np.all(labels[mask] == inputs[mask])
