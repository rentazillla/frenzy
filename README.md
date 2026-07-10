# frenzy

Generate similar compounds in the same chemical space from SMILES input.

frenzy is a command-line tool that, given a SMILES identifier (or a list of
SMILES), produces a desired number of similar compounds using the STONED-SELFIES
generative method gated by a known-compound corpus (ChEMBL / ZINC20 / a
user-supplied `.smi` file). The hybrid strategy returns *novel* structures that
remain plausibly in known chemical space.

## Install

```bash
uv sync                        # creates .venv + installs deps
uv run frenzy --help
```

## Quick start

```bash
# 1. Download a corpus (optional for --strategy stoned; required for hybrid)
uv run frenzy download --source chembl --out data/

# 2. Build a one-time faiss index over the corpus
uv run frenzy index --corpus data/chembl.smi --out data/index.faiss

# 3. Generate 50 similar compounds (Tanimoto 0.55-0.85, corpus-gated)
uv run frenzy similar --input "CC(=O)Oc1ccccc1C(=O)O" --n 50 --out results.csv

# Batch input from a .smi file; pool and re-rank globally
uv run frenzy similar --multi inputs.smi --n 100 --merge --out results.csv
```

## Commands

```
frenzy download --source chembl|zinc|local PATH [--out data/]
frenzy index    --corpus data/<src>.smi [--fp morgan|maccs] [--out data/index.faiss]
frenzy similar  --input "CCO"|--multi in.smi --n 50 \
    --strategy hybrid|stoned|corpus --similarity 0.6 \
    --min-sim 0.55 --max-sim 0.85 \
    [--merge] [--seed 42] [--props] [--out results.csv] [--format csv|smi|sdf]
```

### `similar` options

| Flag | Default | Description |
|---|---|---|
| `--input` / `--multi` | — | Single SMILES/InChI, or path to a `.smi` file (one per line) |
| `--n` | `50` | Number of similar compounds to return |
| `--strategy` | `hybrid` | `hybrid` (STONED + corpus gate), `stoned` (generate only), `corpus` (retrieve only) |
| `--similarity` | `0.6` | Target Tanimoto (informational; the band controls filtering) |
| `--min-sim` / `--max-sim` | `0.55` / `0.85` | Tanimoto band for keeping candidates |
| `--merge` | off | Pool candidates across all inputs, dedupe, re-rank globally |
| `--seed` | `0` | RNG seed for reproducible STONED mutations |
| `--props` | off | Append MW, LogP, TPSA, HBD, HBA, QED columns |
| `--out` | stdout | Output file path |
| `--format` | `csv` | `csv`, `smi`, or `sdf` |
| `--corpus-index` | `data/index.faiss` | Path to a prebuilt faiss index (hybrid/corpus strategies) |
| `--gate-threshold` | `0.35` | Min Tanimoto to nearest corpus member to keep a candidate (hybrid) |
| `--keep-stereo` | off | Preserve input stereochemistry during mutation |
| `--allow-metals` | off | Allow metal-containing candidates |
| `--strict` | off | Exit nonzero if any input SMILES is invalid |

## How it works

1. **Parse & canonicalize** inputs (RDKit).
2. **STONED-SELFIES**: convert each input to SELFIES, apply random mutations
   (atom swaps, additions, deletions), decode back to SMILES. SELFIES guarantees
   syntactically valid molecules.
3. **Corpus gate** (hybrid): fingerprint each candidate, query the faiss index
   for its nearest corpus neighbor; reject candidates whose nearest-corpus
   Tanimoto is below `--gate-threshold`.
4. **Tanimoto band**: keep candidates within `[min-sim, max-sim]` of the input;
   rank descending.
5. **Output**: CSV / SMI / SDF, optionally with computed properties.

## License

MIT