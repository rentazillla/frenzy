from __future__ import annotations

import csv
import sys
from pathlib import Path

from rdkit import Chem

from .properties import compute_many


def _rows(
    scored: list[tuple[str, str, float]],
    include_props: bool,
) -> list[dict[str, float | int | str]]:
    """Build output rows from (input_smiles, candidate_smiles, tanimoto) tuples."""
    rows: list[dict[str, float | int | str]] = []
    for in_smi, cand_smi, sim in scored:
        row: dict[str, float | int | str] = {
            "input_smiles": in_smi,
            "smiles": cand_smi,
            "tanimoto": round(sim, 4),
        }
        rows.append(row)
    if include_props:
        props = compute_many([cand for _, cand, _ in scored])
        for row, p in zip(rows, props, strict=True):
            row.update(p)
    return rows


def write_csv(
    scored: list[tuple[str, str, float]],
    out_path: Path | None,
    include_props: bool,
) -> None:
    rows = _rows(scored, include_props)
    fieldnames = list(rows[0].keys()) if rows else ["input_smiles", "smiles", "tanimoto"]
    if out_path:
        with out_path.open("w", newline="", encoding="utf-8") as fh:
            _write_csv_rows(fh, rows, fieldnames)
    else:
        _write_csv_rows(sys.stdout, rows, fieldnames)


def _write_csv_rows(fh, rows, fieldnames) -> None:
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def write_smi(
    scored: list[tuple[str, str, float]],
    out_path: Path | None,
    include_props: bool,
) -> None:
    lines = [f"{cand_smi} tanimoto={sim:.4f} input={in_smi}\n"
             for in_smi, cand_smi, sim in scored]
    if out_path:
        out_path.write_text("".join(lines), encoding="utf-8")
    else:
        sys.stdout.write("".join(lines))


def write_sdf(
    scored: list[tuple[str, str, float]],
    out_path: Path | None,
    include_props: bool,
) -> None:
    if out_path is None:
        raise ValueError("SDF output requires a file path (--out)")
    writer = Chem.SDWriter(str(out_path))
    try:
        for in_smi, cand_smi, sim in scored:
            mol = Chem.MolFromSmiles(cand_smi)
            if mol is None:
                continue
            mol.SetProp("input_smiles", in_smi)
            mol.SetProp("tanimoto", f"{sim:.4f}")
            if include_props:
                from .properties import compute

                try:
                    p = compute(cand_smi)
                    mol.SetProp("MW", f"{p.MW:.2f}")
                    mol.SetProp("LogP", f"{p.LogP:.2f}")
                    mol.SetProp("TPSA", f"{p.TPSA:.2f}")
                    mol.SetProp("HBD", str(p.HBD))
                    mol.SetProp("HBA", str(p.HBA))
                    mol.SetProp("QED", f"{p.QED:.4f}")
                except ValueError:
                    pass
            writer.write(mol)
    finally:
        writer.close()


_WRITERS = {"csv": write_csv, "smi": write_smi, "sdf": write_sdf}


def write_output(
    scored: list[tuple[str, str, float]],
    fmt: str,
    out_path: Path | None,
    include_props: bool,
) -> None:
    _WRITERS[fmt](scored, out_path, include_props)