"""EmbeddingDataset: loads precomputed embeddings for asparagus LP.

Reads from a directory of per-subject `.npy` files (produced by embed_all.py or
a participant container) and yields dicts with the same keys as asparagus
ClsRegDataset:

    {"image": Tensor(feature_dim,), "CLSREG_label": Tensor(1,), "file_path": str}

Each `<ptid>.npy` must contain a single 1-D float32 array.

No image transforms are applied — embeddings are already final features.
LinearProbeModule.on_before_batch_transfer will squeeze CLSREG_label to (B,).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def load_embedding(path: Path) -> np.ndarray:
    """Load a 1-D float32 embedding from a `.npy` file."""
    arr = np.asarray(np.load(str(path)), dtype=np.float32).ravel()
    if arr.size == 0:
        raise ValueError(f"{path}: empty embedding")
    return arr


class EmbeddingDataset(Dataset):
    """In-memory dataset backed by precomputed embeddings."""

    def __init__(
        self,
        ptids: list[str],
        embeddings: dict[str, np.ndarray],
        labels: dict[str, int],
    ) -> None:
        self._ptids = ptids
        self._embeddings = embeddings
        self._labels = labels

    def __len__(self) -> int:
        return len(self._ptids)

    def __getitem__(self, idx: int) -> dict:
        ptid = self._ptids[idx]
        emb = torch.from_numpy(self._embeddings[ptid].astype(np.float32))
        label = torch.tensor([self._labels[ptid]], dtype=torch.long)
        return {"image": emb, "CLSREG_label": label, "file_path": ptid}

    @classmethod
    def from_dir(
        cls, emb_dir: Path, ptids: list[str], label_map: dict[str, int]
    ) -> "EmbeddingDataset":
        """Build dataset for a subset of ptids from a directory of `<ptid>.npy` files."""
        emb_by_ptid: dict[str, np.ndarray] = {}
        missing: list[str] = []
        for ptid in ptids:
            path = emb_dir / f"{ptid}.npy"
            if not path.is_file():
                missing.append(ptid)
                continue
            emb_by_ptid[ptid] = load_embedding(path)
        if missing:
            raise KeyError(
                f"{len(missing)} ptid(s) missing from {emb_dir}: {sorted(missing)[:10]}"
                + (" ..." if len(missing) > 10 else "")
            )
        return cls(ptids=ptids, embeddings=emb_by_ptid, labels=label_map)
