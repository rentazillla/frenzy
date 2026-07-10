from __future__ import annotations

from pathlib import Path

import pytest

from frenzy.io import MolEntry, parse_inputs, read_smi_file, to_mol

FIXTURES = Path(__file__).parent / "fixtures"


def test_to_mol_smiles():
    entry = to_mol("CCO")
    assert entry is not None
    assert entry.canon_smiles == "CCO"
    assert entry.source == "CCO"


def test_to_mol_canonicalizes():
    entry = to_mol("OCC")
    assert entry is not None
    assert entry.canon_smiles == "CCO"


def test_to_mol_inchi():
    entry = to_mol("InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3")
    assert entry is not None
    assert entry.canon_smiles == "CCO"


def test_to_mol_invalid():
    assert to_mol("not-a-smiles") is None


def test_parse_inputs_dedupes():
    valid, invalid = parse_inputs(["CCO", "OCC", "bad"])
    assert len(valid) == 1
    assert valid[0].canon_smiles == "CCO"
    assert invalid == ["bad"]


def test_read_smi_file():
    path = FIXTURES / "inputs.smi"
    identifiers = read_smi_file(path)
    assert len(identifiers) == 2
    assert identifiers[0] == "CC(=O)Oc1ccccc1C(=O)O"


@pytest.fixture
def mol_entry() -> MolEntry:
    entry = to_mol("CC(=O)Oc1ccccc1C(=O)O")
    assert entry is not None
    return entry