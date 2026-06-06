"""Loads per-subject .npy embeddings from a directory into a PyTorch Dataset."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def load_embedding(path: Path) -> np.ndarray:
    """Load a 1-D float32 embedding from a .npy file."""
    arr = np.load(path)
    return arr.astype(np.float32).ravel()


class EmbeddingDataset(Dataset):
    """Dataset of precomputed embeddings indexed by ptid.

    Returns dicts with keys:
        image         float32 tensor of shape (feature_dim,)
        CLSREG_label  long tensor (scalar)
        file_path     ptid string
    """

    def __init__(
        self,
        ptids: list[str],
        emb_by_ptid: dict[str, np.ndarray],
        label_lookup: dict[str, int],
    ) -> None:
        self._ptids = ptids
        self._emb_by_ptid = emb_by_ptid
        self._label_lookup = label_lookup

    def __len__(self) -> int:
        return len(self._ptids)

    def __getitem__(self, idx: int) -> dict:
        ptid = self._ptids[idx]
        emb = self._emb_by_ptid[ptid]
        label = self._label_lookup[ptid]
        return {
            "image": torch.from_numpy(emb),
            "CLSREG_label": torch.tensor(label, dtype=torch.long),
            "file_path": ptid,
        }
