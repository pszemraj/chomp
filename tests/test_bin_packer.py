"""Bin packer should pack documents into fixed-length bins."""

from __future__ import annotations

import numpy as np

from chomp.data.pack import BinPacker


def _doc(token: int, length: int) -> list[int]:
    return [token] * length


def test_bin_packer_packs_multiple_docs():
    packer = BinPacker(
        seq_len=8,
        add_bos=False,
        add_eos=False,
        bos_id=1,
        eos_id=2,
        max_doc_tokens=None,
        bins_per_pack=2,
        buffer_docs=2,
        max_docs_per_bin=None,
        pad_id=0,
    )

    for tok, length in [(10, 6), (11, 2), (12, 6), (13, 2)]:
        packer.add_document(_doc(tok, length))

    assert packer.can_pop()
    seq1, seg1 = packer.pop_seq_plus_one_with_segments()
    seq2, seg2 = packer.pop_seq_plus_one_with_segments()

    assert seq1.shape == (9,)
    assert seq2.shape == (9,)

    for seq, segs in [(seq1, seg1), (seq2, seg2)]:
        assert np.any(seq == 0)
        pad_mask = seq == 0
        assert np.all(segs[pad_mask] == 0)

        unique = np.unique(segs[segs > 0])
        assert unique.size >= 2


def test_bin_packer_state_roundtrip():
    packer = BinPacker(
        seq_len=8,
        add_bos=False,
        add_eos=False,
        bos_id=1,
        eos_id=2,
        max_doc_tokens=None,
        bins_per_pack=2,
        buffer_docs=2,
        max_docs_per_bin=None,
        pad_id=0,
    )

    for tok, length in [(21, 6), (22, 2), (23, 6), (24, 2)]:
        packer.add_document(_doc(tok, length))

    _ = packer.pop_seq_plus_one_with_segments()
    state = packer.get_state()
    seq_b = packer.pop_seq_plus_one_with_segments()

    restored = BinPacker(
        seq_len=8,
        add_bos=False,
        add_eos=False,
        bos_id=1,
        eos_id=2,
        max_doc_tokens=None,
        bins_per_pack=2,
        buffer_docs=2,
        max_docs_per_bin=None,
        pad_id=0,
    )
    restored.set_state(state)
    seq_b2 = restored.pop_seq_plus_one_with_segments()

    np.testing.assert_array_equal(seq_b[0], seq_b2[0])
    np.testing.assert_array_equal(seq_b[1], seq_b2[1])
