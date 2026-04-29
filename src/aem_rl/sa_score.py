"""
Synthetic accessibility (SA) score, mapped onto the 1.0-10.0 scale.

Implements:
    Ertl, P., & Schuffenhauer, A. (2009). Estimation of synthetic accessibility
    score of drug-like molecules based on molecular complexity and fragment
    contributions. *Journal of Cheminformatics*, 1(1), 8.

The fragment-contribution lookup file ``fpscores.pkl.gz`` is required at
runtime; obtain it from the original RDKit Contrib release and place it
alongside this module (or https://github.com/rdkit/rdkit/blob/master/Contrib/SA_Score/sascorer.py).
"""

from __future__ import annotations

import gzip
import math
import os
import os.path as op
import pickle
from typing import Iterable

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


_FSCORES = None


def _read_fragment_scores(name: str = "fpscores") -> None:
    """Load the fragment-contribution lookup table.

    Searches for ``<name>.pkl.gz`` first in the directory of this module,
    then in the current working directory.
    """
    global _FSCORES
    candidates = [
        op.join(op.dirname(__file__), f"{name}.pkl.gz"),
        op.join(os.getcwd(), f"{name}.pkl.gz"),
    ]
    for path in candidates:
        if op.exists(path):
            with gzip.open(path, "rb") as f:
                data = pickle.load(f)
            out = {}
            for entry in data:
                for j in range(1, len(entry)):
                    out[entry[j]] = float(entry[0])
            _FSCORES = out
            return
    raise FileNotFoundError(
        f"Could not find {name}.pkl.gz next to {op.dirname(__file__)} "
        f"or in {os.getcwd()}; download it from the original RDKit Contrib "
        "release."
    )


def _num_bridgeheads_and_spiro(mol):
    return (
        rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
        rdMolDescriptors.CalcNumSpiroAtoms(mol),
    )


def calculate_score(mol) -> float:
    """Compute the SA score for an RDKit ``Mol`` object."""
    if _FSCORES is None:
        _read_fragment_scores()

    fp = rdMolDescriptors.GetMorganFingerprint(mol, 2)
    fps = fp.GetNonzeroElements()
    score1 = 0.0
    nf = 0
    for bit_id, v in fps.items():
        nf += v
        score1 += _FSCORES.get(bit_id, -4) * v
    score1 /= max(nf, 1)

    n_atoms = mol.GetNumAtoms()
    n_chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    ri = mol.GetRingInfo()
    n_bridge, n_spiro = _num_bridgeheads_and_spiro(mol)
    n_macro = sum(1 for ring in ri.AtomRings() if len(ring) > 8)

    size_penalty = n_atoms ** 1.005 - n_atoms
    stereo_penalty = math.log10(n_chiral + 1)
    spiro_penalty = math.log10(n_spiro + 1)
    bridge_penalty = math.log10(n_bridge + 1)
    macro_penalty = math.log10(2) if n_macro > 0 else 0.0

    score2 = -size_penalty - stereo_penalty - spiro_penalty - bridge_penalty - macro_penalty

    score3 = 0.0
    if n_atoms > len(fps):
        score3 = math.log(float(n_atoms) / len(fps)) * 0.5

    sa_raw = score1 + score2 + score3

    # Map onto the 1-10 scale used in the original publication.
    sa = 11.0 - (sa_raw - (-4.0) + 1.0) / (2.5 - (-4.0)) * 9.0
    if sa > 8.0:
        sa = 8.0 + math.log(sa + 1.0 - 9.0)
    if sa > 10.0:
        sa = 10.0
    elif sa < 1.0:
        sa = 1.0
    return sa


def sa_score(smiles: str) -> float:
    """Convenience wrapper that takes a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")
    if _FSCORES is None:
        _read_fragment_scores()
    return calculate_score(mol)


def batch_sa_scores(smiles_list: Iterable[str]):
    """Yield SA scores for an iterable of SMILES strings."""
    if _FSCORES is None:
        _read_fragment_scores()
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        yield calculate_score(mol) if mol is not None else float("nan")
