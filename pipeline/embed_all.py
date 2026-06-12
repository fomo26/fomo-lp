"""Batch-extract embeddings for all subjects in a subjects CSV.

Loads `resenc_unet_b_clsreg` once, then iterates over every row of the CSV,
runs preprocess + encode + global-avg-pool + flatten, and writes one
`<ptid>.npz` per subject into the specified output directory.

Each `<ptid>.npz` contains a single 1-D float32 array (the embedding).

The CSV must have at minimum two columns:
    ptid        unique subject identifier
    nifti_path  absolute path to the subject's NIfTI scan

Failing scans (missing file, not correct NIfTI) are skipped with a warning.
"""
from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path

import numpy as np
import nibabel as nib
import torch

from asparagus.modules.networks.resenc_unet import resenc_unet_b_clsreg
from asparagus.modules.transforms.presets import CPU_clsreg_val_test_transforms_crop


log = logging.getLogger(__name__)


def _strip_prefix(state_dict: dict, prefix: str) -> dict:
    if not any(k.startswith(prefix) for k in state_dict):
        return state_dict
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}


def load_backbone(
    ckpt_path: Path,
    *,
    input_channels: int = 1,
    n_classes: int = 3,
    device: torch.device = torch.device("cpu"),
) -> torch.nn.Module:
    """Build ``resenc_unet_b_clsreg`` and load the (non-classifier) weights from
    a Lightning checkpoint. The decoder/head is left at random init — for embedding
    extraction only the encoder matters.
    """
    model = resenc_unet_b_clsreg(
        input_channels=input_channels,
        output_channels=n_classes,
        dimensions="3D",
        late_fusion=True,
    )
    raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw

    for prefix in ("model.", "module.", ""):
        candidate = _strip_prefix(sd, prefix)
        overlap = set(candidate) & set(model.state_dict())
        if overlap:
            sd = candidate
            break
    model_keys = set(model.state_dict())
    matched = len(set(sd) & model_keys)
    model.load_state_dict(sd, strict=False)
    log.info(
        "load: matched=%d/%d (missing=%d)",
        matched,
        len(model_keys),
        len(model_keys - set(sd)),
    )

    model.eval()
    model.to(device)
    return model


def load_scan(nifti_path: Path) -> torch.Tensor:
    """Load a NIfTI as a float32 tensor of shape (1, D, H, W) (single modality, no batch)."""
    img = nib.load(str(nifti_path))
    arr = np.asarray(img.get_fdata(), dtype=np.float32)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {arr.shape} from {nifti_path}")
    return torch.from_numpy(arr).unsqueeze(0)  # (1, D, H, W)


def preprocess(
    image: torch.Tensor,
    target_size: tuple[int, int, int] = (160, 160, 160),
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """Apply the same CPU val/test transforms the asparagus LP module uses."""
    transform = CPU_clsreg_val_test_transforms_crop(target_size=target_size, normalize=normalize)
    data_dict = {"image": image, "transforms_applied": {}}
    return transform(data_dict)["image"]


@torch.no_grad()
def embed_scan(
    model: torch.nn.Module,
    image: torch.Tensor,
    *,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Run encoder + global pool + flatten on a (N, D, H, W) tensor.

    Returns a 1-D tensor of size (feature_dim,) where feature_dim = N_modalities * 320
    for resenc_unet_b with late fusion.
    """
    x = image.unsqueeze(0).to(device)  # (1, N, D, H, W)
    skips = model._encode(x)
    deepest = skips[-1] if isinstance(skips, list) else skips
    pooled = torch.nn.functional.adaptive_avg_pool3d(deepest, (1, 1, 1))
    return pooled.flatten(1).squeeze(0).cpu()


def embed_all(
    csv_path: Path,
    ckpt_path: Path,
    out_dir: Path,
    *,
    target_size: tuple[int, int, int] = (160, 160, 160),
    log_every: int = 25,
) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device: %s", device)

    log.info("loading backbone from %s ...", ckpt_path)
    model = load_backbone(ckpt_path, device=device)

    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    log.info("subjects in csv: %d", len(rows))

    out_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    failures: list[tuple[str, str]] = []
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        ptid = row.get("ptid", "")
        scan = row.get("nifti_path") or ""
        scan_path = Path(scan)

        try:
            if not ptid:
                raise ValueError("missing ptid in csv row")
            if not scan_path.is_file():
                raise FileNotFoundError(scan_path)
            image = preprocess(load_scan(scan_path), target_size=target_size)
            vec = embed_scan(model, image, device=device).numpy().astype(np.float32).ravel()
        except Exception as e:  # noqa: BLE001
            log.warning("[%d/%d] %s  SKIP: %s", i, len(rows), ptid, e)
            failures.append((ptid, str(e)))
            continue

        np.savez(out_dir / f"{ptid}.npz", vec)
        ok += 1

        if i % log_every == 0 or i == len(rows):
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            eta = (len(rows) - i) / rate if rate > 0 else 0.0
            log.info("[%d/%d] ok=%d skip=%d  %.2fs/scan  eta=%.0fs",
                     i, len(rows), ok, len(failures), 1 / rate if rate else 0, eta)

    if ok == 0:
        raise RuntimeError("no embeddings produced — every scan failed")

    log.info("saved %d embeddings -> %s", ok, out_dir)
    if failures:
        log.warning("%d subjects skipped:", len(failures))
        for ptid, err in failures[:20]:
            log.warning("  %s : %s", ptid, err)
        if len(failures) > 20:
            log.warning("  ... and %d more", len(failures) - 20)

    return out_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True, type=Path,
                        help="subjects CSV (must include columns: ptid, nifti_path)")
    parser.add_argument("--ckpt", required=True, type=Path,
                        help="lightning checkpoint with resenc_unet_b weights")
    parser.add_argument("--out-dir", required=True, type=Path,
                        help="output directory; writes one <ptid>.npz per subject")
    parser.add_argument("--target-size", type=int, nargs=3, default=[160, 160, 160])
    args = parser.parse_args()

    embed_all(args.csv, args.ckpt, args.out_dir, target_size=tuple(args.target_size))


if __name__ == "__main__":
    main()