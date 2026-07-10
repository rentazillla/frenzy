from __future__ import annotations

from rdkit import Chem, DataStructs

from .fingerprints import morgan_fp


def has_metals(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    metals = {
        3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        31, 32, 33, 34, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
        50, 51, 52, 53, 54, 55, 56, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
        82, 83, 84,
    }
    return any(atom.GetAtomicNum() in metals for atom in mol.GetAtoms())


def band_filter(
    candidates: list[str],
    ref_smiles: str,
    min_sim: float = 0.55,
    max_sim: float = 0.85,
    radius: int = 2,
    n_bits: int = 2048,
) -> list[tuple[str, float]]:
    """Return [(smiles, tanimoto)] for candidates in [min_sim, max_sim], sorted desc."""
    ref_fp = morgan_fp(ref_smiles, radius, n_bits)
    scored: list[tuple[str, float]] = []
    for smi in candidates:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = morgan_fp(smi, radius, n_bits)
        sim = DataStructs.TanimotoSimilarity(ref_fp, fp)
        if min_sim <= sim <= max_sim:
            scored.append((smi, sim))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def dedupe(candidates: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for smi in candidates:
        if smi not in seen:
            seen.add(smi)
            result.append(smi)
    return result