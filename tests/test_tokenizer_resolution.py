"""Tokenizer-derived config updates (vocab rounding + special tokens)."""

from __future__ import annotations

import pytest

from chomp.config import Config, DataConfig, ModelConfig, OptimConfig, TokenizerConfig, TrainConfig
from chomp.data.pipeline import resolve_tokenizer_config


class _DummyTokenizer:
    """Mock tokenizer with configurable special tokens."""

    def __init__(self, size: int, *, bos: int | None, eos: int | None, pad: int | None) -> None:
        """Initialize mock tokenizer.

        :param int size: Tokenizer vocab size.
        :param int | None bos: BOS token id.
        :param int | None eos: EOS token id.
        :param int | None pad: PAD token id.
        """
        self._size = int(size)
        self._bos = bos
        self._eos = eos
        self._pad = pad

    def __len__(self) -> int:
        return self._size

    @property
    def bos_token_id(self) -> int | None:
        """Return BOS token ID.

        :return int | None: BOS token id.
        """
        return self._bos

    @property
    def eos_token_id(self) -> int | None:
        """Return EOS token ID.

        :return int | None: EOS token id.
        """
        return self._eos

    @property
    def pad_token_id(self) -> int | None:
        """Return pad token ID.

        :return int | None: PAD token id.
        """
        return self._pad


def test_vocab_size_rounds_up_to_multiple() -> None:
    """Vocab size should round up to configured multiple."""
    cfg = Config(
        model=ModelConfig(backend="dummy", vocab_size=300, d_model=32),
        data=DataConfig(
            backend="local_text",
            local_text="tokenizer config text\n",
            tokenizer=TokenizerConfig(kind="byte", vocab_size_multiple=128),
        ),
        train=TrainConfig(steps=1, batch_size=1, seq_len=8, grad_accum=1, allow_cpu=True),
        optim=OptimConfig(warmup_steps=0),
    )
    tok = _DummyTokenizer(size=256, bos=None, eos=None, pad=None)
    updated = resolve_tokenizer_config(cfg, tok)
    assert updated.model.vocab_size == 384


def test_auto_sets_special_token_ids() -> None:
    """auto_set_special_tokens should copy IDs from tokenizer to config."""
    cfg = Config(
        model=ModelConfig(
            backend="dummy",
            vocab_size=512,
            d_model=32,
            bos_token_id=0,
            eos_token_id=1,
            pad_token_id=2,
        ),
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
        optim=OptimConfig(warmup_steps=0),
    )
    tok = _DummyTokenizer(size=512, bos=10, eos=11, pad=12)
    updated = resolve_tokenizer_config(cfg, tok)
    assert updated.model.bos_token_id == 10
    assert updated.model.eos_token_id == 11
    assert updated.model.pad_token_id == 12


def test_tokenizer_pad_equals_eos_warns() -> None:
    """Tokenizer with pad==eos should warn but still resolve."""
    cfg = Config(
        model=ModelConfig(
            backend="dummy",
            vocab_size=512,
            d_model=32,
            bos_token_id=0,
            eos_token_id=1,
            pad_token_id=2,
        ),
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
        optim=OptimConfig(warmup_steps=0),
    )
    tok = _DummyTokenizer(size=512, bos=0, eos=0, pad=0)
    with pytest.warns(UserWarning, match="pad_token_id equals model.eos_token_id"):
        updated = resolve_tokenizer_config(cfg, tok)
    assert updated.model.pad_token_id == 0
    assert updated.model.eos_token_id == 0


def test_default_max_doc_tokens_inferred() -> None:
    """max_doc_tokens should default to 4 * seq_len when unset."""
    cfg = Config(
        model=ModelConfig(backend="dummy", vocab_size=512, d_model=32),
        data=DataConfig(
            backend="local_text",
            local_text="tokenizer config text\n",
            tokenizer=TokenizerConfig(kind="byte", vocab_size_multiple=128, max_doc_tokens=None),
        ),
        train=TrainConfig(steps=10, batch_size=1, seq_len=16, grad_accum=1, allow_cpu=True),
        optim=OptimConfig(warmup_steps=0),
    )
    tok = _DummyTokenizer(size=256, bos=None, eos=None, pad=None)
    updated = resolve_tokenizer_config(cfg, tok)
    assert updated.data.tokenizer.max_doc_tokens == 64


def test_zero_max_doc_tokens_disables_truncation() -> None:
    """max_doc_tokens=0 should resolve to None (no truncation)."""
    cfg = Config(
        model=ModelConfig(backend="dummy", vocab_size=512, d_model=32),
        data=DataConfig(
            backend="local_text",
            local_text="tokenizer config text\n",
            tokenizer=TokenizerConfig(kind="byte", vocab_size_multiple=128, max_doc_tokens=0),
        ),
        train=TrainConfig(steps=10, batch_size=1, seq_len=16, grad_accum=1, allow_cpu=True),
        optim=OptimConfig(warmup_steps=0),
    )
    tok = _DummyTokenizer(size=256, bos=None, eos=None, pad=None)
    updated = resolve_tokenizer_config(cfg, tok)
    assert updated.data.tokenizer.max_doc_tokens is None
