from __future__ import annotations

from frenzy.similarity import band_filter, dedupe, has_charges, has_metals


def test_band_filter_returns_sorted():
    candidates = ["CCO", "CC(=O)O", "c1ccccc1", "Cc1ccccc1"]
    scored = band_filter(candidates, "CCO", min_sim=0.0, max_sim=1.0)
    assert len(scored) == len(candidates)
    sims = [s for _, s in scored]
    assert sims == sorted(sims, reverse=True)


def test_band_filter_respects_bounds():
    candidates = ["CCO", "c1ccccc1"]
    scored = band_filter(candidates, "CCO", min_sim=0.5, max_sim=0.95)
    for _, sim in scored:
        assert 0.5 <= sim <= 0.95


def test_band_filter_excludes_self():
    scored = band_filter(["CCO", "CCN"], "CCO", min_sim=0.0, max_sim=0.99)
    smiles = [smi for smi, _ in scored]
    assert "CCO" not in smiles


def test_dedupe_preserves_order():
    assert dedupe(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_has_metals_false_for_organic():
    assert not has_metals("CCO")
    assert not has_metals("c1ccccc1")


def test_has_metals_true():
    assert has_metals("[Na+].[Cl-]")
    assert has_metals("c1ccccc1[Zn]")


def test_has_charges_false_for_neutral():
    assert not has_charges("CCO")
    assert not has_charges("c1ccccc1")


def test_has_charges_true():
    assert has_charges("[O-]c1ccccc1")
    assert has_charges("C[N+](C)(C)C")


def test_has_charges_true_for_zwitterion():
    assert has_charges("[NH3+]CC(=O)[O-]")


def test_has_charges_false_for_invalid():
    assert not has_charges("not-a-smiles")