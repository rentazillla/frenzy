from __future__ import annotations

from frenzy.fingerprints import batch_morgan_bits, morgan_bits, morgan_fp, tanimoto


def test_morgan_bits_shape():
    bits = morgan_bits("CCO")
    assert bits.shape == (2048,)
    assert bits.dtype.name == "uint8"


def test_morgan_bits_invalid():
    import pytest

    with pytest.raises(ValueError):
        morgan_bits("not-smiles")


def test_morgan_fp_type():
    fp = morgan_fp("CCO")
    assert fp.GetNumBits() == 2048


def test_tanimoto_identity():
    assert tanimoto("CCO", "CCO") == 1.0


def test_tanimoto_disjoint():
    sim = tanimoto("CCO", "c1ccccc1")
    assert 0.0 <= sim < 1.0


def test_batch_morgan_bits():
    arr = batch_morgan_bits(["CCO", "c1ccccc1", "bad"])
    assert arr.shape == (3, 2048)
    # invalid row is all zeros
    assert arr[2].sum() == 0