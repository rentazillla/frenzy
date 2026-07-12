from __future__ import annotations

import gc

import numpy as np
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

_gen_cache: dict[tuple[int, int], object] = {}


def _gen(radius: int, n_bits: int):
    key = (radius, n_bits)
    if key not in _gen_cache:
        _gen_cache[key] = GetMorganGenerator(radius=radius, fpSize=n_bits)
    return _gen_cache[key]


def morgan_bits(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Return a uint8 array of the Morgan fingerprint bits for *smiles*."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    fp = _gen(radius, n_bits).GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048):
    """Return an ExplicitBitVect (for RDKit Tanimoto computation)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    return _gen(radius, n_bits).GetFingerprint(mol)


def tanimoto(a_smiles: str, b_smiles: str, radius: int = 2, n_bits: int = 2048) -> float:
    fp_a = morgan_fp(a_smiles, radius, n_bits)
    fp_b = morgan_fp(b_smiles, radius, n_bits)
    return DataStructs.TanimotoSimilarity(fp_a, fp_b)


def batch_morgan_bits(smiles_list: list[str], radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Stacked [N, n_bits] uint8 array. Invalid SMILES produce an all-zero row.

    Mol/fp objects are released each iteration and the cyclic GC is triggered
    periodically so large corpora don't accumulate unreachable RDKit objects.
    """
    gen = _gen(radius, n_bits)
    arr = np.zeros((len(smiles_list), n_bits), dtype=np.uint8)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = gen.GetFingerprint(mol)
        DataStructs.ConvertToNumpyArray(fp, arr[i])
        del mol, fp
        if i % 8192 == 8191:
            gc.collect()
    return arr
