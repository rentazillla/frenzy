from __future__ import annotations

from pathlib import Path

from ..corpus.index import load_index, query_knn


def gate_candidates(
    candidates: list[str],
    index_path: Path,
    gate_threshold: float = 0.35,
    k: int = 1,
    radius: int = 2,
    n_bits: int = 2048,
) -> list[str]:
    """Filter candidates whose nearest-corpus Tanimoto >= *gate_threshold*."""
    if not candidates:
        return []
    index, _smiles_list = load_index(index_path)
    sims = query_knn(index, candidates, k=k, radius=radius, n_bits=n_bits)
    # sims is [N, k]; take the max across k
    best = sims.max(axis=1) if k > 1 else sims[:, 0]
    return [smi for smi, s in zip(candidates, best, strict=True) if s >= gate_threshold]


def corpus_retrieve(
    index_path: Path,
    query_smiles: str,
    n: int,
    min_sim: float = 0.55,
    max_sim: float = 0.85,
    radius: int = 2,
    n_bits: int = 2048,
) -> list[tuple[str, float]]:
    """Retrieve real corpus compounds similar to *query_smiles* within the Tanimoto band."""
    from ..corpus.index import corpus_neighbors

    index, smiles_list = load_index(index_path)
    # oversample k to have enough after band filtering
    k = min(max(n * 10, 50), index.ntotal)
    neighbors = corpus_neighbors(
        index, smiles_list, [query_smiles], k=k, radius=radius, n_bits=n_bits
    )[0]
    banded = [(smi, sim) for smi, sim in neighbors if min_sim <= sim <= max_sim]
    return banded[:n]