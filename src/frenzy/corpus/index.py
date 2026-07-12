from __future__ import annotations

import gc
from collections.abc import Iterable, Iterator
from pathlib import Path

import faiss
import numpy as np
from rdkit import Chem
from rdkit.Chem import DataStructs

from ..fingerprints import _gen, batch_morgan_bits

_BATCH = 65536


def build_index(
    smiles: Iterable[str] | list[str],
    out_path: Path,
    radius: int = 2,
    n_bits: int = 2048,
) -> None:
    """Build a faiss binary index over Morgan fingerprints of *smiles*.

    Streams the corpus in batches so peak memory is bounded by the batch size,
    not the corpus size. Persists the index to *out_path* and a sibling
    ``.ids`` file mapping faiss row -> SMILES.
    """
    gen = _gen(radius, n_bits)
    index = faiss.IndexBinaryFlat(n_bits)
    ids_path = out_path.with_suffix(out_path.suffix + ".ids")
    with ids_path.open("w", encoding="utf-8") as ids_f:
        for batch in _batched(smiles, _BATCH):
            packed = _fingerprint_batch(batch, gen, n_bits)
            index.add(packed)
            ids_f.write("\n".join(batch))
            ids_f.write("\n")
            del packed
            gc.collect()
    faiss.write_index_binary(index, str(out_path))


def _fingerprint_batch(
    batch: list[str], gen: object, n_bits: int
) -> np.ndarray:
    """Packed [len(batch), n_bits//8] uint8 array. Invalid SMILES -> zero row."""
    bits = np.zeros((len(batch), n_bits), dtype=np.uint8)
    for i, smi in enumerate(batch):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = gen.GetFingerprint(mol)
        DataStructs.ConvertToNumpyArray(fp, bits[i])
        del mol, fp
    return np.packbits(bits, axis=1)


def _batched(smiles: Iterable[str], size: int) -> Iterator[list[str]]:
    batch: list[str] = []
    for smi in smiles:
        batch.append(smi)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def load_index(index_path: Path) -> tuple[faiss.IndexBinaryFlat, list[str]]:
    """Load a faiss index and its companion ``.ids`` file."""
    index = faiss.read_index_binary(str(index_path))
    ids_path = index_path.with_suffix(index_path.suffix + ".ids")
    smiles_list = ids_path.read_text(encoding="utf-8").splitlines()
    return index, smiles_list


def query_knn(
    index: faiss.IndexBinaryFlat,
    queries: list[str],
    k: int = 1,
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """Return [N, k] array of Tanimoto similarities for each query's k nearest corpus members."""
    bits = batch_morgan_bits(queries, radius=radius, n_bits=n_bits)
    packed = np.packbits(bits, axis=1)
    _distances, _indices = index.search(packed, k)
    return 1.0 - _distances / n_bits


def corpus_neighbors(
    index: faiss.IndexBinaryFlat,
    smiles_list: list[str],
    queries: list[str],
    k: int = 5,
    radius: int = 2,
    n_bits: int = 2048,
) -> list[list[tuple[str, float]]]:
    """Return for each query a list of (corpus_smiles, tanimoto) for its k nearest members."""
    bits = batch_morgan_bits(queries, radius=radius, n_bits=n_bits)
    packed = np.packbits(bits, axis=1)
    _distances, indices = index.search(packed, k)
    sims = 1.0 - _distances / n_bits
    results: list[list[tuple[str, float]]] = []
    for row, sim_row in zip(indices, sims, strict=True):
        neighbors = [
            (smiles_list[idx], float(sim))
            for idx, sim in zip(row, sim_row, strict=True)
            if idx >= 0
        ]
        results.append(neighbors)
    return results