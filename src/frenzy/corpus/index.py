from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

from ..fingerprints import batch_morgan_bits


def build_index(
    smiles_list: list[str],
    out_path: Path,
    radius: int = 2,
    n_bits: int = 2048,
) -> None:
    """Build a faiss binary index over Morgan fingerprints of *smiles_list*.

    Persists the index to *out_path* and a sibling ``.ids`` file mapping
    faiss row -> SMILES.
    """
    bits = batch_morgan_bits(smiles_list, radius=radius, n_bits=n_bits)
    packed = np.packbits(bits, axis=1)
    index = faiss.IndexBinaryFlat(n_bits)
    index.add(packed)
    faiss.write_index_binary(index, str(out_path))
    ids_path = out_path.with_suffix(out_path.suffix + ".ids")
    ids_path.write_text("\n".join(smiles_list), encoding="utf-8")


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