"""Tokenizer-derived config updates (vocab rounding + special tokens)."""

from __future__ import annotations

from chomp.config import Config, DataConfig, ModelConfig, TokenizerConfig, TrainConfig
from chomp.data.pipeline import resolve_tokenizer_config


class _DummyTokenizer:
    def __init__(self, size: int, *, bos: int | None, eos: int | None, pad: int | None):
        self._size = int(size)
        self._bos = bos
        self._eos = eos
        self._pad = pad

    def __len__(self) -> int:
        return self._size

    @property
    def bos_token_id(self) -> int | None:
        return self._bos

    @property
    def eos_token_id(self) -> int | None:
        return self._eos

    @property
    def pad_token_id(self) -> int | None:
        return self._pad


def test_vocab_size_rounds_up_to_multiple():
    cfg = Config(
        model=ModelConfig(vocab_size=300),
        data=DataConfig(
            backend="local_text",
            local_text="tokenizer config text\n",
            tokenizer=TokenizerConfig(kind="byte", vocab_size_multiple=128),
        ),
        train=TrainConfig(steps=1, batch_size=1, seq_len=8, grad_accum=1, allow_cpu=True),
    )
    tok = _DummyTokenizer(size=256, bos=None, eos=None, pad=None)
    updated = resolve_tokenizer_config(cfg, tok)
    assert updated.model.vocab_size == 384


def test_auto_sets_special_token_ids():
    cfg = Config(
        model=ModelConfig(vocab_size=512, bos_token_id=0, eos_token_id=1, pad_token_id=2),
        data=DataConfig(
            backend="local_text",
            local_text="tokenizer config text\n",
            tokenizer=TokenizerConfig(
                kind="hf",
                hf_name_or_path="dummy",
                auto_set_special_tokens=True,
                add_bos=False,
                add_eos=False,
            ),
        ),
        train=TrainConfig(steps=1, batch_size=1, seq_len=8, grad_accum=1, allow_cpu=True),
    )
    tok = _DummyTokenizer(size=512, bos=10, eos=11, pad=12)
    updated = resolve_tokenizer_config(cfg, tok)
    assert updated.model.bos_token_id == 10
    assert updated.model.eos_token_id == 11
    assert updated.model.pad_token_id == 12
