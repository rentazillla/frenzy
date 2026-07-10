from __future__ import annotations

from frenzy.generators.stoned import generate
from frenzy.io import to_mol


def test_generate_returns_valid_smiles():
    entry = to_mol("CC(=O)Oc1ccccc1C(=O)O")
    candidates = generate(entry, n=10, seed=42)
    assert len(candidates) <= 10
    assert len(candidates) > 0
    # all candidates are valid SMILES (parsable) and distinct from input
    from rdkit import Chem

    for smi in candidates:
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None
        assert smi != entry.canon_smiles


def test_generate_reproducible():
    entry = to_mol("CCO")
    a = generate(entry, n=5, seed=7)
    b = generate(entry, n=5, seed=7)
    assert a == b


def test_generate_different_seeds_differ():
    entry = to_mol("CC(=O)Oc1ccccc1C(=O)O")
    a = generate(entry, n=10, seed=1)
    b = generate(entry, n=10, seed=2)
    assert a != b


def test_generate_no_duplicates():
    entry = to_mol("c1ccccc1")
    candidates = generate(entry, n=15, seed=99)
    assert len(candidates) == len(set(candidates))