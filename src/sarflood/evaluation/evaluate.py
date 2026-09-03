"""Full checkpoint evaluation: segmentation, calibration, and selective prediction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import DataLoader

from ..data.dataset import ETCIFloodDataset
from ..models.build import build_model
from ..models.uncertainty_inference import (
    confidence_uncertainty,
    deterministic_entropy,
    stochastic_forward_passes,
    summarize_passes,
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


def _selective_metrics(probs, labels, uncertainty) -> dict:
    cov, risk = risk_coverage_curve(probs, labels, uncertainty)
    _, _, risk_oracle, se = sparsification_error(probs, labels, uncertainty)
    return {
        "aurc": aurc(cov, risk),
        "sparsification_error": se,
        "coverage": cov.tolist(),
        "risk": risk.tolist(),
        "risk_oracle": risk_oracle.tolist(),
    }


def _boundary_mask(labels: np.ndarray, tolerance: int = 3) -> np.ndarray:
    """Return a narrow evaluation band around class boundaries."""
    label = labels.astype(bool)
    dilated = ndimage.binary_dilation(label, iterations=tolerance)
    eroded = ndimage.binary_erosion(label, iterations=tolerance)
    return dilated ^ eroded


def _calibration_breakdown(probs: np.ndarray, labels: np.ndarray) -> dict:
    """Report calibration globally and on flood/non-flood/boundary subsets."""
    labels_bool = labels.astype(bool)
    result = {
        "overall": {
            "ece": expected_calibration_error(probs, labels),
            "brier": brier_score(probs, labels),
            "n": int(len(probs)),
        }
    }
    for name, subset in (("flood", labels_bool), ("non_flood", ~labels_bool)):
        if subset.any():
            result[name] = {
                "ece": expected_calibration_error(probs[subset], labels[subset]),
                "brier": brier_score(probs[subset], labels[subset]),
                "n": int(subset.sum()),
            }
    return result


@torch.no_grad()
def evaluate(
    checkpoint: str | Path,
    regions: list[str],
    device: str | None = None,
    mc_passes: int = 0,
    batch_size: int = 16,
    save_maps_dir: str | Path | None = None,
    uq_max_pixels: int = 2_000_000,
    seed: int = 42,
) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_checkpoint(checkpoint, device)
    model.eval()

    ds = ETCIFloodDataset(
        cfg["data"]["root"], regions, cfg["data"]["bands"], rotation_aug=False,
        image_size=cfg["data"].get("image_size", 256),
        ratio_clip=cfg["data"].get("ratio_clip", 10.0),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

    metrics = SegmentationMetrics()
    sampled_probs, sampled_labels = [], []
    sampled_uncertainties: dict[str, list[np.ndarray]] = {
        "deterministic_entropy": [],
        "deterministic_confidence": [],
    }
    if mc_passes > 0:
        sampled_uncertainties.update({
            "predictive_entropy": [],
            "expected_entropy": [],
            "mutual_information": [],
            "variance": [],
        })

    rng = np.random.default_rng(seed)
    pixels_per_batch = max(1, int(np.ceil(uq_max_pixels / max(len(loader), 1))))
    map_dir = Path(save_maps_dir) if save_maps_dir else None
    if map_dir:
        map_dir.mkdir(parents=True, exist_ok=True)

    for batch in loader:
        img = batch["image"].to(device)
        mask = batch["mask"].numpy()

        if mc_passes > 0:
            passes = stochastic_forward_passes(model, img, mc_passes)
            summary = summarize_passes(passes)
            mean_t = summary["mean"]
            probs = mean_t.cpu().numpy()
            uncertainty_maps = {
                "deterministic_entropy": deterministic_entropy(mean_t).cpu().numpy(),
                "deterministic_confidence": confidence_uncertainty(mean_t).cpu().numpy(),
                "predictive_entropy": summary["predictive_entropy"].cpu().numpy(),
                "expected_entropy": summary["expected_entropy"].cpu().numpy(),
                "mutual_information": summary["mutual_information"].cpu().numpy(),
                "variance": summary["variance"].cpu().numpy(),
            }
        else:
            prob_t = torch.sigmoid(model(img))
            probs = prob_t.cpu().numpy()
            uncertainty_maps = {
                "deterministic_entropy": deterministic_entropy(prob_t).cpu().numpy(),
                "deterministic_confidence": confidence_uncertainty(prob_t).cpu().numpy(),
            }

        flat_probs, flat_labels = probs.ravel(), mask.ravel()
        sample_size = min(pixels_per_batch, len(flat_probs))
        sample_index = rng.choice(len(flat_probs), size=sample_size, replace=False)
        sampled_probs.append(flat_probs[sample_index])
        sampled_labels.append(flat_labels[sample_index])
        for name, values in uncertainty_maps.items():
            sampled_uncertainties[name].append(values.ravel()[sample_index])

        for i in range(len(img)):
            metrics.update(probs[i], mask[i])
            if map_dir:
                payload = {
                    "prob": probs[i, 0],
                    "label": mask[i, 0],
                }
                payload.update({name: values[i, 0] for name, values in uncertainty_maps.items()})
                np.savez_compressed(map_dir / f"{batch['id'][i]}.npz", **payload)

    probs = np.concatenate(sampled_probs)
    labels = np.concatenate(sampled_labels)
    uncertainty = {name: np.concatenate(parts) for name, parts in sampled_uncertainties.items()}

    if len(probs) > uq_max_pixels:
        keep = rng.choice(len(probs), size=uq_max_pixels, replace=False)
        probs, labels = probs[keep], labels[keep]
        uncertainty = {name: values[keep] for name, values in uncertainty.items()}

    result = {
        "metrics": metrics.compute(),
        "per_tile_iou": metrics.per_tile_iou,
        "uq_sample_pixels": int(len(probs)),
        "calibration": _calibration_breakdown(probs, labels),
        # Backwards-compatible top-level values.
        "ece": expected_calibration_error(probs, labels),
        "brier": brier_score(probs, labels),
        "selective_prediction": {},
    }

    for name, unc in uncertainty.items():
        result["selective_prediction"][name] = _selective_metrics(probs, labels, unc)

    return result
