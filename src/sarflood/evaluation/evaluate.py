"""Full evaluation of a checkpoint on a dataset split: metrics, calibration,
risk-coverage, and optional uncertainty maps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.dataset import ETCIFloodDataset
from ..models.build import build_model
from ..models.uncertainty_inference import (
    stochastic_forward_passes, summarize_passes,
)
from ..training.metrics import SegmentationMetrics
from ..uncertainty.calibration import brier_score, expected_calibration_error
from ..uncertainty.risk_coverage import aurc, risk_coverage_curve, sparsification_error


def load_checkpoint(path: str | Path, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg["model"], in_channels=len(cfg["data"]["bands"]))
    model.load_state_dict(ckpt["model_state"])
    return model.to(device), cfg


@torch.no_grad()
def evaluate(
    checkpoint: str | Path,
    regions: list[str],
    device: str | None = None,
    mc_passes: int = 0,
    batch_size: int = 16,
    save_maps_dir: str | Path | None = None,
) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_checkpoint(checkpoint, device)
    model.eval()

    ds = ETCIFloodDataset(cfg["data"]["root"], regions, cfg["data"]["bands"], rotation_aug=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

    metrics = SegmentationMetrics()
    all_probs, all_labels, all_unc = [], [], []
    map_dir = Path(save_maps_dir) if save_maps_dir else None
    if map_dir:
        map_dir.mkdir(parents=True, exist_ok=True)

    for bi, batch in enumerate(loader):
        img = batch["image"].to(device)
        mask = batch["mask"].numpy()
        if mc_passes > 0:
            passes = stochastic_forward_passes(model, img, mc_passes)  # (N,B,1,H,W)
            summary = summarize_passes(passes)  # reduce over passes -> (B,1,H,W)
            probs = summary["mean"].cpu().numpy()
            unc = summary["entropy"].cpu().numpy()
        else:
            probs = torch.sigmoid(model(img)).cpu().numpy()
            unc = np.zeros_like(probs)
        for i in range(len(img)):
            metrics.update(probs[i], mask[i])
            all_probs.append(probs[i].ravel())
            all_labels.append(mask[i].ravel())
            all_unc.append(unc[i].ravel())
            if map_dir:
                np.savez_compressed(
                    map_dir / f"{batch['id'][i]}.npz",
                    prob=probs[i, 0], label=mask[i, 0], uncertainty=unc[i, 0],
                )

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    unc = np.concatenate(all_unc)

    result = {"metrics": metrics.compute(), "per_tile_iou": metrics.per_tile_iou}
    result["ece"] = expected_calibration_error(probs, labels)
    result["brier"] = brier_score(probs, labels)
    if mc_passes > 0:
        cov, risk = risk_coverage_curve(probs, labels, unc)
        cov_s, risk_m, risk_o, se = sparsification_error(probs, labels, unc)
        result["aurc"] = aurc(cov, risk)
        result["sparsification_error"] = se
        result["risk_coverage"] = {"coverage": cov.tolist(), "risk": risk.tolist(),
                                   "risk_oracle": risk_o.tolist()}
    return result
