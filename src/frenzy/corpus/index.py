from __future__ import annotations

import gc
from collections.abc import Iterable, Iterator
from itertools import islice
from pathlib import Path

import faiss
import numpy as np
from rdkit import Chem
from rdkit.Chem import DataStructs

from ..fingerprints import _gen, batch_morgan_bits

_BATCH = 65536
_IVF_MIN_CORPUS = 1000
DEFAULT_NPROBE = 64


def build_index(
    smiles: Iterable[str],
    out_path: Path,
    corpus_size: int | None = None,
    radius: int = 2,
    n_bits: int = 2048,
) -> None:
    """Build a faiss binary index over Morgan fingerprints of *smiles*.

    Uses IndexBinaryIVF for corpora >= 1000 entries (fast approximate search)
    and IndexBinaryFlat for smaller corpora (exact search, no training needed).

    Streams the corpus in a single pass so peak memory is bounded by the batch
    size, not the corpus size. For large corpora the caller must pass
    *corpus_size* (e.g. from a line count); if None the iterable is buffered
    to count, which is fine for small corpora but defeats streaming on large ones.

    Persists the index to *out_path*, a sibling ``.ids`` file (faiss row ->
    SMILES), and a ``.nprobe`` file for IVF indexes (faiss does not persist
    nprobe).
    """
    gen = _gen(radius, n_bits)
    ids_path = out_path.with_suffix(out_path.suffix + ".ids")
    nprobe_path = out_path.with_suffix(out_path.suffix + ".nprobe")

    if corpus_size is None:
        smiles = list(smiles)
        n = len(smiles)
    else:
        n = corpus_size

    if n < _IVF_MIN_CORPUS:
        index = faiss.IndexBinaryFlat(n_bits)
        with ids_path.open("w", encoding="utf-8") as ids_f:
            for batch in _batched(smiles, _BATCH):
                _append_batch(index, batch, gen, n_bits, ids_f)
        faiss.write_index_binary(index, str(out_path))
        return

    nlist = min(int(np.sqrt(n)), 4096)
    nprobe = min(nlist // 4, DEFAULT_NPROBE)
    quantizer = faiss.IndexBinaryFlat(n_bits)
    index = faiss.IndexBinaryIVF(quantizer, n_bits, faiss.METRIC_L2)
    index.nprobe = nprobe

    train_size = min(max(nlist * 40, _BATCH), n)
    train_batch = list(islice(smiles, train_size))
    train_packed = _fingerprint_batch(train_batch, gen, n_bits)
    index.train(train_packed)
    del train_packed
    gc.collect()

    with ids_path.open("w", encoding="utf-8") as ids_f:
        ids_f.write("\n".join(train_batch))
        ids_f.write("\n")
        for batch in _batched(smiles, _BATCH):
            _append_batch(index, batch, gen, n_bits, ids_f)
    faiss.write_index_binary(index, str(out_path))
    nprobe_path.write_text(str(nprobe), encoding="utf-8")


def _append_batch(index, batch, gen, n_bits, ids_f) -> None:
    packed = _fingerprint_batch(batch, gen, n_bits)
    index.add(packed)
    ids_f.write("\n".join(batch))
    ids_f.write("\n")
    del packed
    gc.collect()


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


def load_index(index_path: Path) -> tuple[faiss.IndexBinary, list[str]]:
    """Load a faiss index and its companion ``.ids`` file.

    For IVF indexes, restores ``nprobe`` from the sibling ``.nprobe`` file
    (faiss does not persist this attribute).
    """
    index = faiss.read_index_binary(str(index_path))
    if hasattr(index, "nprobe"):
        nprobe_path = index_path.with_suffix(index_path.suffix + ".nprobe")
        if nprobe_path.exists():
            index.nprobe = int(nprobe_path.read_text(encoding="utf-8").strip())
        else:
            index.nprobe = DEFAULT_NPROBE
    ids_path = index_path.with_suffix(index_path.suffix + ".ids")
    smiles_list = ids_path.read_text(encoding="utf-8").splitlines()
    return index, smiles_list


def query_knn(
    index: faiss.IndexBinary,
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
    index: faiss.IndexBinary,
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