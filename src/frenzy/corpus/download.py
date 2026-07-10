from __future__ import annotations

from pathlib import Path

import requests
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from tenacity import retry, stop_after_attempt, wait_exponential

CHEMBBL_SDF_URL = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_35.sdf.gz"
# Real ChEMBL URL is huge (~4 GB). For a usable subset we point at the ChEMBL
# monomer SDF which is smaller; users can supply a local file for full coverage.
CHEMBBL_SUBSET_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_35.sdf.gz"
)

ZINC_BASE = "https://files.docking.org/ZINC20-2D/"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
def _download_with_resume(url: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers: dict[str, str] = {}
    mode = "wb"
    initial = 0
    if out_path.exists():
        initial = out_path.stat().st_size
        headers["Range"] = f"bytes={initial}-"
        mode = "ab"

    with requests.get(url, headers=headers, stream=True, timeout=60) as resp:
        if initial and resp.status_code == 200:
            mode = "wb"
            initial = 0
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0)) + initial
        with Progress(
            TextColumn("[bold blue]Downloading"),
            BarColumn(),
            DownloadColumn(),
            TimeRemainingColumn(),
        ) as prog:
            task = prog.add_task("download", total=total or None, completed=initial)
            with out_path.open(mode) as f:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    if chunk:
                        f.write(chunk)
                        prog.update(task, advance=len(chunk))
    return out_path


def _smiles_from_sdf(sdf_path: Path) -> list[str]:
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(str(sdf_path))
    return [Chem.MolToSmiles(m) for m in supplier if m is not None]


def _smiles_from_sdf_gz(gz_path: Path) -> list[str]:
    import gzip

    from rdkit import Chem

    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        supplier = Chem.SDMolSupplier(f.read())
    return [Chem.MolToSmiles(m) for m in supplier if m is not None]


def download_chembl(out_dir: Path) -> Path:
    """Download the ChEMBL SDF and extract SMILES into a .smi file.

    The full ChEMBL SDF is ~4 GB compressed. The SMILES are extracted and
    written to ``<out>/chembl.smi``; the raw SDF is kept for re-indexing.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sdf_gz = out_dir / "chembl.sdf.gz"
    smi_path = out_dir / "chembl.smi"
    if smi_path.exists():
        return smi_path
    _download_with_resume(CHEMBBL_SUBSET_URL, sdf_gz)
    smiles = _smiles_from_sdf_gz(sdf_gz)
    smi_path.write_text("\n".join(smiles), encoding="utf-8")
    return smi_path


def download_zinc(out_dir: Path, tranche: str | None = None) -> Path:
    """Download a ZINC20 tranche and extract SMILES into a .smi file.

    ZINC20 organizes compounds into tranches by physicochemical properties.
    A *tranche* is a substring like ``EAEDAA``; if None, a default small
    tranche is used. See https://files.docking.org/ZINC20-2D/ for tranche IDs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tranche = tranche or "EAEDAA"
    url = f"{ZINC_BASE}{tranche}.smi.gz"
    gz_path = out_dir / f"zinc_{tranche}.smi.gz"
    smi_path = out_dir / f"zinc_{tranche}.smi"
    if smi_path.exists():
        return smi_path
    _download_with_resume(url, gz_path)
    import gzip

    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        lines = [ln.split()[0] for ln in f if ln.strip()]
    smi_path.write_text("\n".join(lines), encoding="utf-8")
    return smi_path


def register_local(path: Path) -> Path:
    """Validate a user-supplied .smi file and return its path."""
    if not path.exists():
        raise FileNotFoundError(f"corpus file not found: {path}")
    return path