"""STONED-SELFIES: random SELFIES mutations to generate novel similar compounds.

Based on the STONED-SELFIES method (Nigam et al., 2021). We apply three
mutation types to a SELFIES-encoded molecule:
  - *substitution*: replace a symbol with a random alphabet symbol
  - *addition*: insert a random alphabet symbol at a position
  - *deletion*: remove a symbol
Each mutation is wrapped so the resulting SELFIES always decodes to a valid
molecule (SELFIES guarantees this by construction).
"""

from __future__ import annotations

import selfies as sf
from rdkit import Chem

from ..io import MolEntry

_MUTATION_TYPES = ("sub", "add", "del")
_MAX_MUTATIONS_PER_MOL = 10


def _selfies_symbols(s: str) -> list[str]:
    return list(sf.split_selfies(s))


def _random_symbol(rng, alphabet: list[str]) -> str:
    return alphabet[rng.integers(0, len(alphabet))]


def mutate_selfies(
    selfies_str: str,
    alphabet: list[str],
    rng,
    n_mutations: int = 1,
) -> str | None:
    """Apply *n_mutations* random mutations to a SELFIES string.

    Returns the mutated SELFIES, or None if the string is empty.
    """
    symbols = _selfies_symbols(selfies_str)
    if not symbols:
        return None
    for _ in range(n_mutations):
        op = _MUTATION_TYPES[rng.integers(0, len(_MUTATION_TYPES))]
        pos = int(rng.integers(0, len(symbols)))
        if op == "sub":
            symbols[pos] = _random_symbol(rng, alphabet)
        elif op == "add":
            symbols.insert(pos, _random_symbol(rng, alphabet))
        else:
            if len(symbols) <= 1:
                continue
            symbols.pop(pos)
    return "".join(symbols)


def generate(
    entry: MolEntry,
    n: int,
    seed: int = 0,
    keep_stereo: bool = False,
    alphabet: list[str] | None = None,
) -> list[str]:
    """Generate up to *n* candidate SMILES via STONED-SELFIES mutations.

    Oversamples 4x then dedupes to compensate for invalid/identical decodes.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    if alphabet is None:
        alphabet = sorted(sf.get_semantic_robust_alphabet())

    smi_for_encoding = entry.canon_smiles
    if not keep_stereo:
        mol = Chem.MolFromSmiles(entry.canon_smiles)
        Chem.RemoveStereochemistry(mol)
        smi_for_encoding = Chem.MolToSmiles(mol)

    base_sf = sf.encoder(smi_for_encoding)
    if base_sf is None:
        return []

    seen: set[str] = set()
    seen.add(entry.canon_smiles)
    results: list[str] = []
    attempts = 0
    max_attempts = n * 8

    while len(results) < n and attempts < max_attempts:
        attempts += 1
        n_mut = int(rng.integers(1, _MAX_MUTATIONS_PER_MOL + 1))
        mutated = mutate_selfies(base_sf, alphabet, rng, n_mutations=n_mut)
        if mutated is None:
            continue
        try:
            smi = sf.decoder(mutated)
        except Exception:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol)
        if canon in seen:
            continue
        seen.add(canon)
        results.append(canon)

    return results