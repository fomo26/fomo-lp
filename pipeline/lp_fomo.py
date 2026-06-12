"""Standalone linear probe + fairness evaluation on precomputed embeddings.

Reads per-subject `<ptid>.npy` embeddings from a directory, joins them to
labels from the subjects CSV, performs a reproducible stratified
train/val/test split on the subjects, trains a linear probe using the
asparagus LinearProbeModule (DINOv2-style multi-head SGD + CosineAnnealingLR),
then evaluates on the held-out test set and writes a full fairness report.

Usage:
1. First specify FEATURE1_CSV_COL, FEATURE2_CSV_COL, and FEATURE1_BINS in config.py.

2. Then run:
    python pipeline/lp_fomo.py \\
        --embeddings-dir /path/to/embeddings \\
        --csv            /path/to/subjects.csv \\
        --output-dir     results/my_model

Outputs (in --output-dir):
    predictions.json   per-sample label, prediction, and softmax scores
    eval_report.json   overall metrics + per-variable fairness report

Requires:
    asparagus   pip install -e /path/to/asparagus
    fomo-metrics  pip install git+https://github.com/fomo26/fomo-metrics.git
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import lightning as pl
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys
_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_here.parent))

from config import FEATURE1_BINS, FEATURE1_CSV_COL, FEATURE2_CSV_COL
from embedding_dataset import EmbeddingDataset, load_embedding              # noqa: E402
from identity_model import IdentityEncoder                                  # noqa: E402
from fairness_report import bin_continuous_feature, normalize_categorical_feature, build_fairness_report  # noqa: E402
from asparagus.modules.lightning_modules.linear_probe_module import LinearProbeModule


log = logging.getLogger(__name__)

DEFAULT_LRS = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1]
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 64

# ---------------------------------------------------------------------------
# Minimal DataModule so trainer.datamodule.batch_size is always available
# (LinearProbeModule.training_step references it for metric logging)
# ---------------------------------------------------------------------------

class _EmbeddingDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_ds: EmbeddingDataset,
        val_ds: EmbeddingDataset,
        test_ds: EmbeddingDataset,
        batch_size: int,
        seed: int,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self._train_ds = train_ds
        self._val_ds = val_ds
        self._test_ds = test_ds
        self._seed = seed

    def train_dataloader(self) -> DataLoader:
        g = torch.Generator()
        g.manual_seed(self._seed)
        return DataLoader(self._train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=4, generator=g)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._val_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=4)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self._test_ds, batch_size=1, shuffle=False, num_workers=0)


# ---------------------------------------------------------------------------
# LinearProbeModule subclass: adds softmax scores to the test output JSON
# ---------------------------------------------------------------------------

class LinearProbeModuleWithScores(LinearProbeModule):
    """Stock asparagus LP module + per-sample softmax scores in test JSON."""

    def test_step(self, batch, batch_idx):
        x = batch["image"]
        features = self._get_features(x)
        logits = self.heads[self._lr_to_linear_head_name(self.best_head_lr)](features)

        label = batch["CLSREG_label"]
        scores = F.softmax(logits.float(), dim=1).squeeze(0).cpu().tolist()
        # DataLoader's default collate wraps the string file_path in a length-1
        # list; unwrap so the dict key is hashable.
        file_path = batch["file_path"]
        if isinstance(file_path, (list, tuple)):
            file_path = file_path[0]
        self.results[file_path] = {
            "prediction": int(logits.argmax(1).item()),
            "label": int(label.item()),
            "scores": scores,
        }
        self.logits.append(logits.squeeze(0))
        # label is shape (1,) at batch_size=1; squeeze to 0-d so that
        # torch.stack in on_test_epoch_end yields a 1-D (N,) target tensor.
        self.labels.append(label.squeeze(0))



def _stratified_split(
    items: list[dict],
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(seed)
    total = n_train + n_val + n_test

    strata: dict = defaultdict(list)
    for r in items:
        try:
            stratum = int(float(r.get("feature1", "")) // 10) * 10
        except (TypeError, ValueError):
            stratum = -1
        strata[stratum].append(r)

    test_pool: list[dict] = []
    trainval_pool: list[dict] = []
    for decade in sorted(strata):
        members = list(strata[decade])
        rng.shuffle(members)
        n_t = round(len(members) * n_test / total)
        test_pool.extend(members[:n_t])
        trainval_pool.extend(members[n_t:])

    rng.shuffle(test_pool)
    rng.shuffle(trainval_pool)

    delta = len(test_pool) - n_test
    if delta > 0:
        trainval_pool.extend(test_pool[n_test:])
        test_pool = test_pool[:n_test]
    elif delta < 0:
        test_pool.extend(trainval_pool[:-delta])
        trainval_pool = trainval_pool[-delta:]

    val = trainval_pool[:n_val]
    train = trainval_pool[n_val:]
    rng.shuffle(train)
    return train, val, test_pool


# ---------------------------------------------------------------------------
# Core routine
# ---------------------------------------------------------------------------

def run_lp_fomo(
    embeddings_dir: Path,
    csv_path: Path,
    output_dir: Path,
    n_train: int = 80,
    n_val: int = 20,
    n_test: int = 100,
    learning_rates: list[float] = DEFAULT_LRS,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = 42,
) -> dict:
    pl.seed_everything(seed, workers=True)

    # --- Read CSV (single source of truth for ptid / cohort / fairness variables) ---
    with csv_path.open() as f:
        csv_rows = list(csv.DictReader(f))

    ptids: list[str] = [r["ptid"] for r in csv_rows]
    cohorts: list[str] = [r["selected_cohort"] for r in csv_rows]
    feature1_lookup = {r["ptid"]: r.get(FEATURE1_CSV_COL, "") for r in csv_rows}
    feature2_lookup = {r["ptid"]: r.get(FEATURE2_CSV_COL, "") for r in csv_rows}

    unique_cohorts = sorted(set(cohorts))
    CLASS_TO_IDX = {c: i for i, c in enumerate(unique_cohorts)}
    N_CLASSES = len(CLASS_TO_IDX)

    label_lookup = {p: CLASS_TO_IDX[c] for p, c in zip(ptids, cohorts) if c in CLASS_TO_IDX}

    # --- Load embeddings from <ptid>.npy files, in CSV order ---
    emb_by_ptid: dict[str, np.ndarray] = {}
    missing_emb: set[str] = set()
    for ptid in ptids:
        path = embeddings_dir / f"{ptid}.npy"
        if not path.is_file():
            missing_emb.add(ptid)
            continue
        emb_by_ptid[ptid] = load_embedding(path)

    if missing_emb:
        log.warning("%d ptid(s) have no embedding in %s", len(missing_emb), embeddings_dir)

    feature_dims = {v.shape[0] for v in emb_by_ptid.values()}
    if len(feature_dims) != 1:
        raise ValueError(f"embeddings have inconsistent dimensions: {sorted(feature_dims)}")
    feature_dim = feature_dims.pop()

    log.info("embeddings: %s  n=%d  dim=%d  missing=%d",
             embeddings_dir, len(emb_by_ptid), feature_dim, len(missing_emb))

    # --- Stratified split per cohort ---
    by_cohort: dict[str, list[dict]] = defaultdict(list)
    for ptid, cohort in zip(ptids, cohorts):
        by_cohort[cohort].append({"ptid": ptid, "feature1": feature1_lookup[ptid]})

    train_ptids: list[str] = []
    val_ptids: list[str] = []
    test_ptids: list[str] = []
    for cohort in sorted(CLASS_TO_IDX):
        tr, va, te = _stratified_split(by_cohort[cohort], n_train, n_val, n_test, seed)
        train_ptids.extend(r["ptid"] for r in tr)
        val_ptids.extend(r["ptid"] for r in va)
        test_ptids.extend(r["ptid"] for r in te)

    # Train/val embeddings must all be present — missing ones cannot be imputed.
    missing_trainval = [p for p in train_ptids + val_ptids if p in missing_emb]
    if missing_trainval:
        raise FileNotFoundError(
            f"{len(missing_trainval)} train/val ptid(s) missing embeddings: "
            f"{sorted(missing_trainval)[:10]}" + (" ..." if len(missing_trainval) > 10 else "")
        )

    # Missing test embeddings are allowed: they receive worst-case scores in the
    # fairness report (None → opposite-class substitution in compute_ovr_auroc).
    missing_test = [p for p in test_ptids if p in missing_emb]
    if missing_test:
        log.warning(
            "%d test ptid(s) missing embeddings — worst-case scores will be assigned: %s%s",
            len(missing_test), sorted(missing_test)[:10],
            " ..." if len(missing_test) > 10 else "",
        )
    available_test_ptids = [p for p in test_ptids if p not in missing_emb]

    log.info("split: train=%d  val=%d  test=%d (available=%d  missing=%d)",
             len(train_ptids), len(val_ptids), len(test_ptids),
             len(available_test_ptids), len(missing_test))

    train_ds = EmbeddingDataset(train_ptids, emb_by_ptid, label_lookup)
    val_ds = EmbeddingDataset(val_ptids, emb_by_ptid, label_lookup)
    test_ds = EmbeddingDataset(available_test_ptids, emb_by_ptid, label_lookup)

    dm = _EmbeddingDataModule(train_ds, val_ds, test_ds, batch_size=batch_size, seed=seed)

    # --- Build model + LP module ---
    model = IdentityEncoder(feature_dim)

    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / "predictions.json"

    model_module = LinearProbeModuleWithScores(
        model=model,
        learning_rates=learning_rates,
        num_classes=N_CLASSES,
        dimensions="3D",
        loss_weight=None,
        train_transforms=None,
        val_transforms=None,
        test_output_path=str(pred_path),
        weights=None,
        pretrained_target_size=None,
        target_size=None,
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=1,
        check_val_every_n_epoch=5,
        enable_checkpointing=False,
        logger=False,
        num_sanity_val_steps=0,
        use_distributed_sampler=False,
        accumulate_grad_batches=1,
    )

    log.info("starting LP  (epochs=%d  lr_sweep=%s  device=%s)", epochs, learning_rates, accelerator)
    trainer.validate(model=model_module, datamodule=dm)
    trainer.fit(model=model_module, datamodule=dm)
    trainer.test(model=model_module, datamodule=dm)

    # --- Load predictions written by on_test_epoch_end ---
    raw_preds: dict = json.loads(pred_path.read_text())
    best_lr = raw_preds.pop("best_head_lr", None)
    best_head = raw_preds.pop("best_head", None)
    metrics = raw_preds.pop("metrics", {})

    # --- Fairness report ---
    # Include all test ptids; missing embeddings get None scores → compute_ovr_auroc
    # substitutes worst-case scores (0 for the true class, 1/(C-1) for all others).
    y_test = [label_lookup[p] for p in test_ptids]
    y_scores = [raw_preds.get(p, {}).get("scores") for p in test_ptids]

    groups_by_variable = {
        "feature2":     [normalize_categorical_feature(feature2_lookup[p]) for p in test_ptids],
        "feature1_bin": [bin_continuous_feature(feature1_lookup[p], FEATURE1_BINS) for p in test_ptids],
    }
    fairness = build_fairness_report(y_test, y_scores, groups_by_variable)

    report = {
        "embeddings_dir": str(embeddings_dir),
        "feature_dim": feature_dim,
        "n_train": len(train_ptids),
        "n_val": len(val_ptids),
        "n_test": len(test_ptids),
        "best_lr": best_lr,
        "asparagus_metrics": metrics,
        **fairness,
    }

    (output_dir / "eval_report.json").write_text(json.dumps(report, indent=2))

    log.info("predictions  -> %s", pred_path)
    log.info("eval_report  -> %s", output_dir / "eval_report.json")

    log.info("\n=== Overall ===")
    for name, val in fairness["overall"].items():
        log.info("  %s: %s", name, val)

    log.info("\n=== Fairness Score ===")
    for mname, fs in fairness["fairness_score"].items():
        log.info("  %s: %.4f  (vars: %s)", mname, fs["score"] or float("nan"), fs["variables_used"])

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings-dir", required=True, type=Path,
                        help="directory containing one <ptid>.npy per subject (1-D float32 each)")
    parser.add_argument("--csv", required=True, type=Path,
                        help="selected_subjects CSV (ptid, selected_cohort)")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="directory for predictions.json and eval_report.json")
    parser.add_argument("--n-train", type=int, default=80)
    parser.add_argument("--n-val", type=int, default=20)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_lp_fomo(
        embeddings_dir=args.embeddings_dir,
        csv_path=args.csv,
        output_dir=args.output_dir,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
