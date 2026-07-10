from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from frenzy.cli import app

runner = CliRunner()
FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def built_index(tmp_path_factory):
    from frenzy.corpus.index import build_index
    from frenzy.io import read_smi_file

    out_dir = tmp_path_factory.mktemp("cli_idx")
    idx_path = out_dir / "index.faiss"
    smiles = read_smi_file(FIXTURES / "corpus.smi")
    build_index(smiles, idx_path)
    return idx_path


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "similar" in result.stdout


def test_similar_stoned_basic():
    result = runner.invoke(
        app,
        [
            "similar",
            "--input", "CC(=O)Oc1ccccc1C(=O)O",
            "--n", "5",
            "--strategy", "stoned",
            "--min-sim", "0.1",
            "--max-sim", "0.99",
            "--seed", "42",
            "--format", "csv",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "smiles" in result.stdout
    assert "tanimoto" in result.stdout


def test_similar_stoned_excludes_ions_by_default():
    result = runner.invoke(
        app,
        [
            "similar",
            "--input", "CC(=O)Oc1ccccc1C(=O)O",
            "--n", "20",
            "--strategy", "stoned",
            "--min-sim", "0.1",
            "--max-sim", "0.99",
            "--seed", "42",
            "--format", "csv",
        ],
    )
    assert result.exit_code == 0, result.stdout
    from rdkit import Chem

    lines = result.stdout.strip().splitlines()[1:]
    assert len(lines) > 0
    for line in lines:
        smi = line.split(",")[1]
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None
        assert all(a.GetFormalCharge() == 0 for a in mol.GetAtoms())


def test_similar_stoned_allow_ions():
    result = runner.invoke(
        app,
        [
            "similar",
            "--input", "CC(=O)Oc1ccccc1C(=O)O",
            "--n", "20",
            "--strategy", "stoned",
            "--min-sim", "0.1",
            "--max-sim", "0.99",
            "--seed", "42",
            "--format", "csv",
            "--allow-ions",
        ],
    )
    assert result.exit_code == 0, result.stdout
    lines = result.stdout.strip().splitlines()[1:]
    assert len(lines) > 0


def test_similar_stoned_smi_format(tmp_path):
    out = tmp_path / "out.smi"
    result = runner.invoke(
        app,
        [
            "similar",
            "--input", "CCO",
            "--n", "3",
            "--strategy", "stoned",
            "--min-sim", "0.0",
            "--max-sim", "1.0",
            "--seed", "1",
            "--format", "smi",
            "--out", str(out),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert out.exists()
    lines = out.read_text().strip().splitlines()
    assert len(lines) > 0
    assert "tanimoto=" in lines[0]


def test_similar_multi_input():
    result = runner.invoke(
        app,
        [
            "similar",
            "--multi", str(FIXTURES / "inputs.smi"),
            "--n", "3",
            "--strategy", "stoned",
            "--min-sim", "0.0",
            "--max-sim", "1.0",
            "--seed", "5",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "smiles" in result.stdout


def test_similar_hybrid(built_index):
    result = runner.invoke(
        app,
        [
            "similar",
            "--input", "CC(=O)Oc1ccccc1C(=O)O",
            "--n", "5",
            "--strategy", "hybrid",
            "--min-sim", "0.3",
            "--max-sim", "0.95",
            "--seed", "42",
            "--corpus-index", str(built_index),
            "--gate-threshold", "0.2",
        ],
    )
    assert result.exit_code == 0, result.stdout


def test_similar_corpus_strategy(built_index):
    result = runner.invoke(
        app,
        [
            "similar",
            "--input", "CC(=O)Oc1ccccc1C(=O)O",
            "--n", "5",
            "--strategy", "corpus",
            "--min-sim", "0.3",
            "--max-sim", "1.0",
            "--corpus-index", str(built_index),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "smiles" in result.stdout


def test_similar_merge(built_index):
    result = runner.invoke(
        app,
        [
            "similar",
            "--multi", str(FIXTURES / "inputs.smi"),
            "--n", "5",
            "--strategy", "stoned",
            "--min-sim", "0.0",
            "--max-sim", "1.0",
            "--seed", "10",
            "--merge",
        ],
    )
    assert result.exit_code == 0, result.stdout


def test_similar_props(tmp_path):
    out = tmp_path / "props.csv"
    result = runner.invoke(
        app,
        [
            "similar",
            "--input", "CCO",
            "--n", "3",
            "--strategy", "stoned",
            "--min-sim", "0.0",
            "--max-sim", "1.0",
            "--seed", "3",
            "--props",
            "--out", str(out),
        ],
    )
    assert result.exit_code == 0, result.stdout
    content = out.read_text()
    assert "MW" in content
    assert "QED" in content


def test_similar_strict_invalid():
    result = runner.invoke(
        app,
        [
            "similar",
            "--input", "not-a-smiles",
            "--n", "3",
            "--strategy", "stoned",
            "--strict",
        ],
    )
    assert result.exit_code != 0


def test_similar_no_input_errors():
    result = runner.invoke(app, ["similar", "--n", "3"])
    assert result.exit_code != 0


def test_similar_hybrid_missing_index():
    result = runner.invoke(
        app,
        [
            "similar",
            "--input", "CCO",
            "--strategy", "hybrid",
            "--corpus-index", "/nonexistent/index.faiss",
        ],
    )
    assert result.exit_code != 0