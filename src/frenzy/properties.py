from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import QED, Descriptors

_PROPS = {
    "MW": Descriptors.MolWt,
    "LogP": Descriptors.MolLogP,
    "TPSA": Descriptors.TPSA,
    "HBD": Descriptors.NumHDonors,
    "HBA": Descriptors.NumHAcceptors,
}


@dataclass(frozen=True)
class PropertyResult:
    MW: float
    LogP: float
    TPSA: float
    HBD: int
    HBA: int
    QED: float


def compute(smiles: str) -> PropertyResult:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    return PropertyResult(
        MW=_PROPS["MW"](mol),
        LogP=_PROPS["LogP"](mol),
        TPSA=_PROPS["TPSA"](mol),
        HBD=int(_PROPS["HBD"](mol)),
        HBA=int(_PROPS["HBA"](mol)),
        QED=QED.qed(mol),
    )


def compute_many(smiles_list: list[str]) -> list[dict[str, float | int]]:
    results: list[dict[str, float | int]] = []
    for smi in smiles_list:
        try:
            r = compute(smi)
        except ValueError:
            r = PropertyResult(0.0, 0.0, 0.0, 0, 0, 0.0)
        results.append(
            {
                "MW": r.MW,
                "LogP": r.LogP,
                "TPSA": r.TPSA,
                "HBD": r.HBD,
                "HBA": r.HBA,
                "QED": r.QED,
            }
        )
    return results