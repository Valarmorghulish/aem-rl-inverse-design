"""
SMILES augmentation via random atom-order enumeration.

Adapted from:
    Bjerrum, E. J. (2017). SMILES enumeration as data augmentation for neural
    network modeling of molecules. arXiv:1703.07076.

Augmentation lets character-level models see different valid SMILES strings
that decode to the same molecule, which generally improves generalisation
of generative and predictive polymer models (Arus-Pous et al., 2019,
*J. Cheminform.* 11:71).
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Optional, Sequence

import numpy as np
from rdkit import Chem


# ---------------------------------------------------------------------------
# Iterator (kept for compatibility with the original ReLeaSE codebase)
# ---------------------------------------------------------------------------
class Iterator:
    """Abstract iterator producing batches of indices."""

    def __init__(self, n: int, batch_size: int, shuffle: bool, seed: Optional[int]):
        if n < batch_size:
            raise ValueError("Input data length is shorter than batch_size")
        self.n = n
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(n, batch_size, shuffle, seed)

    def reset(self) -> None:
        self.batch_index = 0

    def _flow_index(self, n, batch_size=32, shuffle=False, seed=None):
        self.reset()
        while True:
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)
            if self.batch_index == 0:
                index_array = np.arange(n)
                if shuffle:
                    index_array = np.random.permutation(n)
            current_index = (self.batch_index * batch_size) % n
            if n > current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = n - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (
                index_array[current_index : current_index + current_batch_size],
                current_index,
                current_batch_size,
            )

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)


# ---------------------------------------------------------------------------
# SMILES enumerator
# ---------------------------------------------------------------------------
class SmilesEnumerator:
    """Generate randomized SMILES while preserving role/connection markers.

    The generator-data SMILES strings used in this project carry three kinds
    of markers in addition to the chemistry tokens:
      - outer ``< >`` start/end tokens that delimit a training sample;
      - role prefix ``^`` (hydrophilic motif) or ``~`` (hydrophobic motif);
      - the wildcard ``[*]`` denoting polymerisation connection sites.

    ``randomize_smiles`` strips the outer/role markers, runs RDKit's atom-order
    permutation on the chemical core, then re-attaches the markers and ensures
    bare ``*`` wildcards are written as ``[*]``.
    """

    def __init__(
        self,
        charset: Optional[str] = None,
        pad: int = 256,
        leftpad: bool = True,
        isomeric_smiles: bool = True,
        enum: bool = True,
        canonical: bool = False,
        do_rdkit_randomization: bool = True,
    ):
        self._charset: Optional[str] = None
        if charset is None:
            # Fallback alphabet covering the common SMILES symbols plus the
            # ``< > ^ ~ [*]`` markers used in this project. Callers should
            # always pass a charset built from their own training data.
            self.charset = "<>^~C()c*O=1NS/[\\]F2#liHnPBr+-s\\eIoGa3AZbdKTp4%0"
            logging.warning(
                "SmilesEnumerator initialised with the default charset; "
                "pass an explicit charset built from your data."
            )
        else:
            self.charset = charset

        self.pad = pad
        self.leftpad = leftpad
        self.isomeric_smiles = isomeric_smiles
        self.enumerate = enum
        self.canonical = canonical
        self.do_rdkit_randomization = do_rdkit_randomization

    # ----- charset book-keeping --------------------------------------------
    @property
    def charset(self) -> Optional[str]:
        return self._charset

    @charset.setter
    def charset(self, charset: str) -> None:
        if charset is None:
            raise ValueError("charset cannot be None")
        self._charset = charset
        self._charlen = len(charset)
        self._char_to_int = {c: i for i, c in enumerate(charset)}
        self._int_to_char = {i: c for i, c in enumerate(charset)}

    def fit(self, smiles: Sequence[str], extra_chars: Sequence[str] = (), extra_pad: int = 5) -> None:
        chars = set("".join(smiles))
        self.charset = "".join(chars.union(set(extra_chars)))
        self.pad = max(len(s) for s in smiles) + extra_pad

    # ----- core randomisation ---------------------------------------------
    def randomize_smiles(self, smiles_full: str) -> str:
        """Produce a randomized SMILES while preserving role/end markers.

        On any RDKit failure the original input is returned unchanged.
        """
        original = smiles_full

        outer_prefix = ""
        outer_suffix = ""
        role_prefix = ""
        core = smiles_full

        if core.startswith("<") and core.endswith(">"):
            outer_prefix = "<"
            outer_suffix = ">"
            core = core[1:-1]
        if core.startswith("^"):
            role_prefix, core = "^", core[1:]
        elif core.startswith("~"):
            role_prefix, core = "~", core[1:]

        try:
            mol = Chem.MolFromSmiles(core)
            if mol is None:
                return original
        except Exception:
            return original

        try:
            if self.do_rdkit_randomization:
                order = list(range(mol.GetNumAtoms()))
                np.random.shuffle(order)
                mol = Chem.RenumberAtoms(mol, order)
                processed = Chem.MolToSmiles(
                    mol, canonical=False, isomericSmiles=self.isomeric_smiles
                )
            else:
                processed = Chem.MolToSmiles(
                    mol, canonical=True, isomericSmiles=self.isomeric_smiles
                )
        except Exception:
            return original

        if Chem.MolFromSmiles(processed) is None:
            return original

        # Ensure bare wildcards are written as [*]
        processed = re.sub(r"(?<!\[)\*(?!\])", "[*]", processed)
        return f"{outer_prefix}{role_prefix}{processed}{outer_suffix}"

    # ----- one-hot transforms ---------------------------------------------
    def transform(self, smiles_array: np.ndarray) -> np.ndarray:
        """One-hot encode a numpy array of SMILES strings."""
        one_hot = np.zeros(
            (smiles_array.shape[0], self.pad, self._charlen), dtype=np.int8
        )
        for i, ss in enumerate(smiles_array):
            s = self.randomize_smiles(ss) if self.enumerate else ss
            if any(ch not in self._char_to_int for ch in s):
                logging.warning(
                    "SMILES '%s' contains characters outside the alphabet; "
                    "skipping one-hot encoding.",
                    s,
                )
                continue
            for j, ch in enumerate(s):
                one_hot[i, j, self._char_to_int[ch]] = 1
        return one_hot

    def reverse_transform(self, vect: np.ndarray) -> np.ndarray:
        """Decode one-hot tensors back to SMILES strings."""
        smiles = []
        for v in vect:
            v = v[v.sum(axis=1) == 1]
            smiles.append("".join(self._int_to_char[i] for i in v.argmax(axis=1)))
        return np.asarray(smiles)
