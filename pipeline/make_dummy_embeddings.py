"""Create dummy embeddings of multiple sizes for LP/fairness testing.

No NIfTI files or containers required. Each embedding contains class-specific
signal so the linear probe can actually learn (not just random guessing).

Output format matches embed_all.py: one `<ptid>.npy` per subject, each holding
a single 1-D float32 array, so lp_fomo.py reads them identically. When multiple
dims are requested, each goes into its own `dim_<d>/` subdirectory (otherwise
the `<ptid>.npy` filenames would collide).
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def make_dummy_embeddings(
    csv_path: Path, output_dir: Path, dims: tuple[int, ...] = (128, 512, 2048), label_col: str = "label", seed: int = 42
) -> list[Path]:
    rng = np.random.default_rng(seed)
    rows = list(csv.DictReader(csv_path.open()))
    n = len(rows)
    log.info("subjects: %d", n)

    ptids = [row.get("ptid", "") for row in rows]
    labels = [row[label_col] for row in rows]

    # Derive the class set from the CSV (no hardcoded class names), matching how
    # lp_fomo.py builds CLASS_TO_IDX: sorted unique label values -> indices.
    classes = sorted(set(labels))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(class_to_idx)
    log.info("classes (%d): %s", n_classes, classes)

    saved: list[Path] = []
    multi = len(dims) > 1

    for dim in dims:
        # When generating multiple dims, isolate each in its own subdir so the
        # per-subject <ptid>.npy files don't overwrite each other.
        dim_dir = output_dir / f"dim_{dim}" if multi else output_dir
        dim_dir.mkdir(parents=True, exist_ok=True)

        # Base noise
        X = rng.standard_normal((n, dim)).astype(np.float32) * 0.5

        # Add a class-specific signal: each class gets a +2 bump in its own
        # 1/n_classes slice of the dimensions so the LP has something to learn.
        chunk = max(1, dim // n_classes)
        for i, label in enumerate(labels):
            cls = class_to_idx[label]
            X[i, cls * chunk : (cls + 1) * chunk] += 2.0

        # One <ptid>.npy per subject, single 1-D array each (matches embed_all.py).
        for ptid, vec in zip(ptids, X):
            if not ptid:
                continue
            np.save(dim_dir / f"{ptid}.npy", vec.ravel())

        log.info("saved %d embeddings (dim=%d) -> %s", n, dim, dim_dir)
        saved.append(dim_dir)

    return saved


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", required=True, type=Path, help="selected_subjects CSV (e.g. selected_subjects/selected_subjects.csv)"
    )
    parser.add_argument("--out", required=True, type=Path, help="output directory for per-subject <ptid>.npy files")
    parser.add_argument(
        "--dims", nargs="+", type=int, default=[128, 512, 2048], help="embedding sizes to generate (default: 128 512 2048)"
    )
    parser.add_argument("--label-col", default="label", help="CSV column holding the class label (default: label)")
    args = parser.parse_args()
    make_dummy_embeddings(args.csv, args.out, tuple(args.dims), label_col=args.label_col)


if __name__ == "__main__":
    main()
