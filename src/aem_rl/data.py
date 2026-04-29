"""
Generator-side data containers.

Adapted from the ReLeaSE codebase released with:
    Popova, M., Isayev, O., & Tropsha, A. (2018).
    Deep reinforcement learning for de novo drug design.
    Science Advances, 4(7), eaap7885.
"""

from __future__ import annotations

import logging
import random
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from .utils import read_object_property_file, read_smi_file, tokenize


class GeneratorData:
    """Character-level dataset for the stack-augmented RNN generator.

    Each training sample is a SMILES string surrounded by ``start_token`` and
    ``end_token`` (default ``<`` and ``>``). Optional role prefixes ``^`` /
    ``~`` are kept as part of the string so the same model can generate
    hydrophilic and hydrophobic motifs based on the prompt.
    """

    def __init__(
        self,
        training_data_path: str,
        tokens: Optional[Sequence[str]] = None,
        start_token: str = "<",
        end_token: str = ">",
        max_len: int = 256,
        use_cuda: Optional[bool] = None,
        **kwargs,
    ):
        kwargs.setdefault("cols_to_read", [0])

        raw = read_object_property_file(training_data_path, **kwargs)
        self.start_token = start_token
        self.end_token = end_token
        self.max_len = max_len
        self.raw_file_contents = [
            f"{start_token}{s}{end_token}" for s in raw if len(s) <= max_len
        ]
        self.file_len = len(self.raw_file_contents)
        if self.file_len == 0:
            raise ValueError(
                f"No SMILES survived max_len filter (={max_len}); "
                f"check the training-data file at '{training_data_path}'."
            )

        self.all_characters, self.char2idx, self.n_characters = tokenize(
            self.raw_file_contents, tokens=tokens
        )

        self.use_cuda = (
            torch.cuda.is_available() if use_cuda is None else use_cuda
        )
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

    # ------------------------------------------------------------------
    def load_dictionary(self, tokens: Sequence[str], char2idx: dict) -> None:
        self.all_characters = "".join(tokens)
        self.char2idx = dict(char2idx)
        self.n_characters = len(self.all_characters)

    def random_chunk(self) -> str:
        return self.raw_file_contents[random.randint(0, self.file_len - 1)]

    def char_tensor(self, string: str) -> torch.Tensor:
        out = torch.zeros(len(string), dtype=torch.long)
        for i, ch in enumerate(string):
            try:
                out[i] = self.char2idx[ch]
            except KeyError:
                logging.warning(
                    "Character '%s' not in alphabet; using index 0 (string='%s')",
                    ch,
                    string,
                )
                out[i] = 0
        return out.to(self.device)

    def random_training_set(
        self, smiles_augmentation=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.random_chunk()
        if smiles_augmentation is not None:
            try:
                aug = smiles_augmentation.randomize_smiles(chunk)
                if aug:
                    chunk = aug
            except Exception as exc:  # noqa: BLE001
                logging.warning("Augmentation failed (%s); using original chunk.", exc)
        inp = self.char_tensor(chunk[:-1])
        target = self.char_tensor(chunk[1:])
        return inp, target

    def update_data(self, path: str) -> bool:
        data, ok = read_smi_file(path, unique=True, add_start_end_tokens=False)
        if not ok:
            logging.error("Failed to read %s", path)
            return False
        self.raw_file_contents = [
            f"{self.start_token}{s}{self.end_token}"
            for s in data
            if len(s) <= self.max_len
        ]
        self.file_len = len(self.raw_file_contents)
        return self.file_len > 0


class PredictorData:
    """Predictor-side dataset wrapper kept from the original ReLeaSE codebase.

    This class is provided for completeness and is not used by the polyBERT
    multitask predictor in this repository, which has its own
    ``transformers.Trainer``-based dataset (see ``predictor.py``).
    """

    def __init__(
        self,
        path: str,
        delimiter: str = ",",
        cols=(0, 1),
        get_features=None,
        has_label: bool = True,
        labels_start: int = 1,
        **kwargs,
    ):
        data = read_object_property_file(path, delimiter, cols_to_read=cols)
        if has_label:
            self.objects = np.asarray(data[0], dtype=str).flatten()
            if len(data) > labels_start:
                y = np.asarray(data[labels_start:], dtype="float32").T
                self.y = y.flatten() if y.ndim == 2 and y.shape[1] == 1 else y
            else:
                self.y = np.array([None] * len(self.objects))
        else:
            self.objects = np.asarray(data[0], dtype=str).flatten()
            self.y = np.array([None] * len(self.objects))

        if get_features is not None:
            self.x, ok_idx, bad_idx = get_features(self.objects, **kwargs)
            self.invalid_objects = self.objects[bad_idx]
            self.objects = self.objects[ok_idx]
            if self.y is not None and len(self.y) > 0:
                self.invalid_y = self.y[bad_idx]
                self.y = self.y[ok_idx]
            else:
                self.invalid_y = None
                self.y = None
        else:
            self.x = self.objects
            self.invalid_objects = None
            self.invalid_y = None
        self.binary_y = None

    def binarize(self, threshold: float) -> None:
        if self.y is not None:
            self.binary_y = np.asarray(self.y >= threshold, dtype="int32")
