from __future__ import annotations

from pathlib import Path

import pytest

from frenzy.corpus.index import build_index, corpus_neighbors, load_index, query_knn

FIXTURES = Path(__file__).parent / "fixtures"
CORPUS = FIXTURES / "corpus.smi"


@pytest.fixture(scope="module")
def built_index(tmp_path_factory):
    from frenzy.io import read_smi_file

    out_dir = tmp_path_factory.mktemp("idx")
    idx_path = out_dir / "test.faiss"
    smiles = read_smi_file(CORPUS)
    build_index(smiles, idx_path)
    return idx_path


def test_build_and_load_index(built_index):
    index, smiles_list = load_index(built_index)
    assert index.ntotal > 0
    assert len(smiles_list) == index.ntotal


def test_query_knn_shape(built_index):
    index, _ = load_index(built_index)
    sims = query_knn(index, ["CCO", "c1ccccc1"], k=3)
    assert sims.shape == (2, 3)
    assert (sims >= 0).all() and (sims <= 1).all()


def test_corpus_neighbors(built_index):
    index, smiles_list = load_index(built_index)
    neighbors = corpus_neighbors(index, smiles_list, ["CC(=O)Oc1ccccc1C(=O)O"], k=3)
    assert len(neighbors) == 1
    assert len(neighbors[0]) == 3
    # the top neighbor should be in the aspirin family
    top_smi, top_sim = neighbors[0][0]
    assert top_sim > 0.5