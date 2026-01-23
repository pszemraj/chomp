"""Train-on-EOS toggle should mask EOS labels when disabled."""

from __future__ import annotations

import numpy as np

from chomp.config import Config, DataConfig, ModelConfig, TokenizerConfig, TrainConfig
from chomp.data.pipeline import build_train_iterator


def test_train_on_eos_false_masks_eos_labels() -> None:
    """With train_on_eos=False, EOS token labels should be masked to -100."""
    cfg = Config(
        model=ModelConfig(backend="dummy", vocab_size=512, d_model=32, dropout=0.0),
        data=DataConfig(
            backend="local_text",
            repeat=True,
            local_text="hi",
            mask_boundary_loss=False,
            train_on_eos=False,
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

    input_ids = batch.input_ids[0, 0]
    labels = batch.labels[0, 0]

    eos_id = int(cfg.model.eos_token_id)
    eos_positions = input_ids[1:] == eos_id
    assert eos_positions.any()
    masked = labels[1:][eos_positions]
    assert np.all(masked == -100)
