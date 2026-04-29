"""
General-purpose utilities for SMILES handling, fingerprinting, tokenization,
and cross-validation.

Adapted from the ReLeaSE codebase released with:
    Popova, M., Isayev, O., & Tropsha, A. (2018).
    Deep reinforcement learning for de novo drug design.
    Science Advances, 4(7), eaap7885.
"""

from __future__ import annotations

import csv
import math
import time
import warnings
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem, DataStructs
from sklearn.model_selection import KFold, StratifiedKFold


# ---------------------------------------------------------------------------
# Fingerprints / descriptors
# ---------------------------------------------------------------------------
def mol2image(smi: str, n: int = 2048) -> np.ndarray:
    """Compute an RDKit topological fingerprint for a SMILES string.

    Returns an array of length ``n``; if SMILES parsing fails, returns
    ``[np.nan]`` so the caller can detect the failure.
    """
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return np.array([np.nan])
        fp = Chem.RDKFingerprint(mol, maxPath=4, fpSize=n)
        out = np.zeros(len(fp))
        DataStructs.ConvertToNumpyArray(fp, out)
        return out
    except Exception:
        return np.array([np.nan])


def get_fp(smiles: Sequence[str]) -> Tuple[np.ndarray, List[int], List[int]]:
    """Compute fingerprints for a list of SMILES, returning the valid array,
    the indices that succeeded, and the indices that failed.
    """
    fps: List[np.ndarray] = []
    valid_idx: List[int] = []
    invalid_idx: List[int] = []
    for i, smi in enumerate(smiles):
        v = mol2image(smi)
        if np.isnan(v[0]):
            invalid_idx.append(i)
        else:
            fps.append(v)
            valid_idx.append(i)
    return np.asarray(fps), valid_idx, invalid_idx


def get_desc(smiles: Sequence[str], calc) -> Tuple[np.ndarray, List[int], List[int]]:
    """Apply a callable descriptor calculator (e.g. Mordred) to a list of
    SMILES, returning (descriptors, valid_indices, invalid_indices).
    """
    desc: List[np.ndarray] = []
    valid_idx: List[int] = []
    invalid_idx: List[int] = []
    for i, smi in enumerate(smiles):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                invalid_idx.append(i)
                continue
            desc.append(np.asarray(calc(mol)))
            valid_idx.append(i)
        except Exception:
            invalid_idx.append(i)
    return np.asarray(desc), valid_idx, invalid_idx


def normalize_desc(
    desc_array: np.ndarray,
    desc_mean: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Replace non-finite descriptor entries with the column mean.

    If ``desc_mean`` is None the mean is computed from the input array;
    otherwise the supplied value is used (e.g. the training-set mean during
    inference).
    """
    desc_array = np.asarray(desc_array).reshape(len(desc_array), -1)
    finite_mask = np.isfinite(desc_array)
    if desc_mean is None:
        masked = np.where(finite_mask, desc_array, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            desc_mean = np.nanmean(masked, axis=0)
        desc_mean = np.where(np.isnan(desc_mean), 0.0, desc_mean)
    desc_array = np.where(finite_mask, desc_array, desc_mean)
    return desc_array, desc_mean


# ---------------------------------------------------------------------------
# SMILES sanitisation / canonicalisation
# ---------------------------------------------------------------------------
def sanitize_smiles(
    smiles: Iterable[str],
    canonical: bool = True,
    throw_warning: bool = False,
) -> List[str]:
    """Sanitise SMILES via RDKit; invalid entries become empty strings."""
    out: List[str] = []
    for sm in smiles:
        try:
            mol = Chem.MolFromSmiles(sm, sanitize=True)
            if mol is None:
                raise ValueError("RDKit returned None")
            out.append(Chem.MolToSmiles(mol) if canonical else sm)
        except Exception:
            if throw_warning:
                warnings.warn(f"Unsanitized SMILES: {sm}", UserWarning)
            out.append("")
    return out


def canonical_smiles(
    smiles: Iterable[str],
    sanitize: bool = True,
    throw_warning: bool = False,
) -> List[str]:
    """Return canonical SMILES for each input; invalid entries become ''."""
    out: List[str] = []
    for sm in smiles:
        try:
            mol = Chem.MolFromSmiles(sm, sanitize=sanitize)
            if mol is None:
                raise ValueError("RDKit returned None")
            out.append(Chem.MolToSmiles(mol))
        except Exception:
            if throw_warning:
                warnings.warn(f"{sm} could not be canonicalised", UserWarning)
            out.append("")
    return out


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def save_smi_to_file(filename: str, smiles: Iterable[str], unique: bool = True) -> bool:
    """Write SMILES to a plain-text file, one per line."""
    smiles = list(set(smiles)) if unique else list(smiles)
    with open(filename, "w") as f:
        for s in smiles:
            f.write(s + "\n")
    return True


def read_smi_file(
    filename: str,
    unique: bool = True,
    add_start_end_tokens: bool = False,
) -> Tuple[List[str], bool]:
    """Read SMILES from a plain-text file, one per line."""
    with open(filename, "r") as f:
        lines = [line.rstrip("\n") for line in f]
    if add_start_end_tokens:
        lines = ["<" + s + ">" for s in lines]
    if unique:
        lines = list(set(lines))
    return lines, True


def read_object_property_file(
    path: str,
    delimiter: str = ",",
    cols_to_read: Sequence[int] = (0, 1),
    keep_header: bool = False,
):
    """Read selected columns from a delimited file as parallel lists."""
    with open(path, "r") as f:
        rows = list(csv.reader(f, delimiter=delimiter))
    rows = np.asarray(rows, dtype=object)
    start = 0 if keep_header else 1
    if len(rows) <= start:
        raise ValueError(f"File '{path}' has no data rows.")
    data = [rows[start:, col] for col in cols_to_read]
    return data[0] if len(cols_to_read) == 1 else data


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------
def tokenize(
    smiles: Iterable[str],
    tokens: Optional[Sequence[str]] = None,
) -> Tuple[str, dict, int]:
    """Build a character-level alphabet for a list of SMILES.

    Returns ``(alphabet, token2idx, n_tokens)`` where ``alphabet`` is a string
    (the order is significant for one-hot encoding).
    """
    if tokens is None:
        chars = set("".join(smiles))
        alphabet = "".join(sorted(chars))
    else:
        alphabet = "".join(tokens)
    token2idx = {ch: i for i, ch in enumerate(alphabet)}
    return alphabet, token2idx, len(alphabet)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
def time_since(since: float) -> str:
    """Format an elapsed time as ``"%dm %ds"``."""
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return f"{int(m)}m {int(s)}s"


def cross_validation_split(
    x: Sequence,
    y: Sequence,
    n_folds: int = 5,
    split: str = "random",
    folds: Optional[Sequence] = None,
):
    """Convenience wrapper around ``KFold`` / ``StratifiedKFold``.

    Parameters
    ----------
    x, y : array-like
        Inputs and labels.
    n_folds : int
        Number of folds.
    split : {'random', 'stratified', 'fixed'}
        - 'random'   : ``KFold(shuffle=True)``
        - 'stratified' : ``StratifiedKFold(shuffle=True)``
        - 'fixed'    : reuse a list of pre-defined fold indices
    folds : optional
        Pre-defined fold indices used only when ``split='fixed'``.
    """
    assert len(x) == len(y), "x and y must have the same length"
    x = np.asarray(x)
    y = np.asarray(y)

    if split not in ("random", "stratified", "fixed"):
        raise ValueError("split must be 'random', 'stratified', or 'fixed'")

    if split == "random":
        cv = KFold(n_splits=n_folds, shuffle=True)
        folds = list(cv.split(x, y))
    elif split == "stratified":
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True)
        folds = list(cv.split(x, y))
    elif split == "fixed" and folds is None:
        raise TypeError("'folds' must be a list when split='fixed'")

    cross_val_data: List = []
    cross_val_labels: List = []
    if len(folds) == n_folds:
        for fold in folds:
            cross_val_data.append(x[fold[1]])
            cross_val_labels.append(y[fold[1]])
    elif len(folds) == len(x) and int(np.max(folds)) == n_folds - 1:
        for f in range(n_folds):
            mask = np.where(folds == f)[0]
            cross_val_data.append(x[mask])
            cross_val_labels.append(y[mask])
    else:
        raise ValueError("'folds' has an unexpected shape")
    return cross_val_data, cross_val_labels


def split_and_validate(psmi: str, require_link: bool = True):
    """Split a 'hydrophilic|hydrophobic' SMILES pair and validate it.

    Returns ``(hydrophilic, hydrophobic)`` if both blocks parse with RDKit,
    the first carries positive formal charge, the second is neutral and (if
    ``require_link``) both contain at least one wildcard. Otherwise returns
    ``(None, None)``.
    """
    psmi = psmi.strip().lstrip("<").rstrip(">")
    if "|" not in psmi:
        return None, None
    hydro, phobic = psmi.split("|", 1)
    m1, m2 = Chem.MolFromSmiles(hydro), Chem.MolFromSmiles(phobic)
    if m1 is None or m2 is None:
        return None, None
    if m1.GetFormalCharge() <= 0 or m2.GetFormalCharge() != 0:
        return None, None
    if require_link and (hydro.count("*") == 0 or phobic.count("*") == 0):
        return None, None
    return hydro, phobic
