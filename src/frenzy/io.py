from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem


@dataclass(frozen=True)
class MolEntry:
    smiles: str
    canon_smiles: str
    source: str  # raw identifier the user provided (SMILES or InChI)


def to_mol(identifier: str) -> MolEntry | None:
    """Parse a SMILES or InChI string into a canonical MolEntry.

    Returns None if the identifier cannot be parsed into a valid molecule.
    """
    mol = Chem.MolFromSmiles(identifier)
    if mol is not None:
        canon = Chem.MolToSmiles(mol)
        return MolEntry(smiles=canon, canon_smiles=canon, source=identifier)

    mol = Chem.MolFromInchi(identifier)
    if mol is not None:
        canon = Chem.MolToSmiles(mol)
        return MolEntry(smiles=canon, canon_smiles=canon, source=identifier)

    return None


def parse_inputs(identifiers: list[str]) -> tuple[list[MolEntry], list[str]]:
    """Parse a list of identifier strings, returning (valid, invalid)."""
    valid: list[MolEntry] = []
    invalid: list[str] = []
    seen: set[str] = set()
    for ident in identifiers:
        entry = to_mol(ident.strip())
        if entry is None:
            invalid.append(ident)
        elif entry.canon_smiles in seen:
            continue
        else:
            seen.add(entry.canon_smiles)
            valid.append(entry)
    return valid, invalid


def read_smi_file(path: Path) -> list[str]:
    """Read a .smi file: one identifier per line, whitespace-delimited first column."""
    identifiers: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        identifiers.append(line.split()[0])
    return identifiers