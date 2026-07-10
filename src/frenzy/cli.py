from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import io as mol_io
from .corpus import download as dl
from .corpus.index import build_index
from .generators import corpus as corpus_gen
from .generators import stoned as stoned_gen
from .output import write_output
from .similarity import band_filter, has_charges, has_metals

app = typer.Typer(
    name="frenzy",
    help="Generate similar compounds in the same chemical space from SMILES input.",
    no_args_is_help=True,
)
console = Console()

DEFAULT_INDEX = Path("data/index.faiss")


@app.command()
def download(
    source: str = typer.Option(..., "--source", help="chembl | zinc | local"),
    path: Path | None = typer.Option(None, "--path", help="local corpus file (for --source local)"),
    out: Path = typer.Option(Path("data"), "--out", help="output directory"),
    tranche: str | None = typer.Option(None, "--tranche", help="ZINC tranche id (e.g. EAEDAA)"),
) -> None:
    """Download or register a compound corpus as a .smi file."""
    if source == "chembl":
        smi = dl.download_chembl(out)
    elif source == "zinc":
        smi = dl.download_zinc(out, tranche=tranche)
    elif source == "local":
        if path is None:
            console.print("[red]--path required for --source local[/red]")
            raise typer.Exit(1)
        smi = dl.register_local(path)
    else:
        console.print(f"[red]unknown source: {source}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]Corpus ready:[/green] {smi}")


@app.command()
def index(
    corpus: Path = typer.Option(..., "--corpus", help="path to a .smi corpus file"),
    fp: str = typer.Option("morgan", "--fp", help="fingerprint type (morgan only for now)"),
    out: Path = typer.Option(DEFAULT_INDEX, "--out", help="output index path"),
) -> None:
    """Build a faiss index over a corpus .smi file."""
    if not corpus.exists():
        console.print(f"[red]corpus not found: {corpus}[/red]")
        raise typer.Exit(1)
    if fp != "morgan":
        console.print(f"[red]unsupported fingerprint: {fp} (only 'morgan')[/red]")
        raise typer.Exit(1)
    smiles_list = mol_io.read_smi_file(corpus)
    if not smiles_list:
        console.print("[red]no molecules found in corpus[/red]")
        raise typer.Exit(1)
    console.print(f"[blue]Indexing {len(smiles_list)} molecules...[/blue]")
    build_index(smiles_list, out)
    console.print(f"[green]Index written:[/green] {out}")


@app.command()
def similar(
    input_smiles: str | None = typer.Option(None, "--input", help="single SMILES or InChI"),
    multi: Path | None = typer.Option(None, "--multi", help=".smi file with one identifier per line"),
    n: int = typer.Option(50, "--n", help="number of similar compounds to return"),
    strategy: str = typer.Option("hybrid", "--strategy", help="hybrid | stoned | corpus"),
    min_sim: float = typer.Option(0.55, "--min-sim", help="minimum Tanimoto to keep"),
    max_sim: float = typer.Option(0.85, "--max-sim", help="maximum Tanimoto to keep"),
    merge: bool = typer.Option(False, "--merge", help="pool candidates across inputs and re-rank globally"),
    seed: int = typer.Option(0, "--seed", help="RNG seed for STONED mutations"),
    props: bool = typer.Option(False, "--props", help="append computed property columns"),
    out: Path | None = typer.Option(None, "--out", help="output file (default: stdout)"),
    fmt: str = typer.Option("csv", "--format", help="csv | smi | sdf"),
    corpus_index: Path = typer.Option(DEFAULT_INDEX, "--corpus-index", help="path to faiss index"),
    gate_threshold: float = typer.Option(0.35, "--gate-threshold", help="min corpus Tanimoto to keep a candidate (hybrid)"),
    keep_stereo: bool = typer.Option(False, "--keep-stereo", help="preserve stereochemistry during mutation"),
    allow_metals: bool = typer.Option(False, "--allow-metals", help="allow metal-containing candidates"),
    allow_ions: bool = typer.Option(False, "--allow-ions", help="allow charged/ionic candidates"),
    strict: bool = typer.Option(False, "--strict", help="exit nonzero if any input is invalid"),
) -> None:
    """Generate similar compounds in the same chemical space."""
    if input_smiles is None and multi is None:
        console.print("[red]provide --input or --multi[/red]")
        raise typer.Exit(1)

    identifiers = mol_io.read_smi_file(multi) if multi is not None else [input_smiles]

    valid, invalid = mol_io.parse_inputs(identifiers)
    if invalid:
        for bad in invalid:
            console.print(f"[yellow]invalid identifier:[/yellow] {bad}")
        if strict:
            raise typer.Exit(1)

    if not valid:
        console.print("[red]no valid inputs[/red]")
        raise typer.Exit(1)

    needs_corpus = strategy in ("hybrid", "corpus")
    if needs_corpus and not corpus_index.exists():
        console.print(
            f"[red]corpus index not found: {corpus_index}. "
            "Run 'frenzy download' + 'frenzy index' first, or use --strategy stoned.[/red]"
        )
        raise typer.Exit(1)

    # (input_smiles, candidate_smiles, tanimoto)
    all_scored: list[tuple[str, str, float]] = []

    for entry in valid:
        if strategy == "corpus":
            retrieved = corpus_gen.corpus_retrieve(
                corpus_index, entry.canon_smiles, n, min_sim=min_sim, max_sim=max_sim
            )
            all_scored.extend((entry.canon_smiles, smi, sim) for smi, sim in retrieved)
            continue

        # stoned or hybrid: generate candidates
        raw = stoned_gen.generate(entry, n * 4, seed=seed, keep_stereo=keep_stereo)

        if not allow_metals:
            raw = [smi for smi in raw if not has_metals(smi)]

        if not allow_ions:
            raw = [smi for smi in raw if not has_charges(smi)]

        if strategy == "hybrid":
            raw = corpus_gen.gate_candidates(
                raw, corpus_index, gate_threshold=gate_threshold
            )

        scored = band_filter(raw, entry.canon_smiles, min_sim=min_sim, max_sim=max_sim)
        all_scored.extend((entry.canon_smiles, smi, sim) for smi, sim in scored)

    if merge:
        # dedupe by candidate SMILES, keep the best Tanimoto across inputs
        best: dict[str, tuple[str, str, float]] = {}
        for in_smi, cand_smi, sim in all_scored:
            if cand_smi not in best or sim > best[cand_smi][2]:
                best[cand_smi] = (in_smi, cand_smi, sim)
        merged = list(best.values())
        merged.sort(key=lambda x: x[2], reverse=True)
        all_scored = merged[:n]
    else:
        # per-input top-N
        from collections import defaultdict

        grouped: defaultdict[str, list[tuple[str, str, float]]] = defaultdict(list)
        for row in all_scored:
            grouped[row[0]].append(row)
        all_scored = []
        for _in_smi, rows in grouped.items():
            rows.sort(key=lambda x: x[2], reverse=True)
            all_scored.extend(rows[:n])

    if not all_scored:
        console.print(
            "[yellow]no candidates passed the filters. "
            "Try widening the similarity band or lowering --gate-threshold.[/yellow]"
        )
        raise typer.Exit(0)

    write_output(all_scored, fmt, out, props)

    if out:
        table = Table(title="Summary")
        table.add_column("input")
        table.add_column("returned")
        from collections import Counter

        counts = Counter(in_smi for in_smi, _, _ in all_scored)
        for in_smi, count in counts.items():
            table.add_row(in_smi, str(count))
        console.print(table)


if __name__ == "__main__":
    app()