"""Full checkpoint evaluation: segmentation, calibration, and selective prediction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
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


def _calibration_breakdown(probs: np.ndarray, labels: np.ndarray) -> dict:
    """Report calibration globally and separately for positive/negative pixels."""
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
    sampled_labels, sampled_det_probs, sampled_eval_probs = [], [], []
    sampled_uncertainties: dict[str, list[np.ndarray]] = {
        "deterministic_entropy": [],
        "deterministic_confidence": [],
    }
    if mc_passes > 0:
        sampled_uncertainties.update({
            "mc_predictive_entropy": [],
            "mc_expected_entropy": [],
            "mc_mutual_information": [],
            "mc_variance": [],
        })

    rng = np.random.default_rng(seed)
    pixels_per_batch = max(1, int(np.ceil(uq_max_pixels / max(len(loader), 1))))
    map_dir = Path(save_maps_dir) if save_maps_dir else None
    if map_dir:
        map_dir.mkdir(parents=True, exist_ok=True)

    for batch in loader:
        img = batch["image"].to(device)
        mask = batch["mask"].numpy()

        # True one-pass deterministic baseline: dropout disabled, BatchNorm frozen.
        model.eval()
        det_prob_t = torch.sigmoid(model(img))
        det_probs = det_prob_t.cpu().numpy()

        uncertainty_maps = {
            "deterministic_entropy": deterministic_entropy(det_prob_t).cpu().numpy(),
            "deterministic_confidence": confidence_uncertainty(det_prob_t).cpu().numpy(),
        }

        if mc_passes > 0:
            passes = stochastic_forward_passes(model, img, mc_passes)
            summary = summarize_passes(passes)
            eval_prob_t = summary["mean"]
            eval_probs = eval_prob_t.cpu().numpy()
            uncertainty_maps.update({
                "mc_predictive_entropy": summary["predictive_entropy"].cpu().numpy(),
                "mc_expected_entropy": summary["expected_entropy"].cpu().numpy(),
                "mc_mutual_information": summary["mutual_information"].cpu().numpy(),
                "mc_variance": summary["variance"].cpu().numpy(),
            })
        else:
            eval_probs = det_probs

        flat_labels = mask.ravel()
        flat_det_probs = det_probs.ravel()
        flat_eval_probs = eval_probs.ravel()
        sample_size = min(pixels_per_batch, len(flat_labels))
        sample_index = rng.choice(len(flat_labels), size=sample_size, replace=False)

        sampled_labels.append(flat_labels[sample_index])
        sampled_det_probs.append(flat_det_probs[sample_index])
        sampled_eval_probs.append(flat_eval_probs[sample_index])
        for name, values in uncertainty_maps.items():
            sampled_uncertainties[name].append(values.ravel()[sample_index])

        # Report segmentation performance of the actual prediction used by the
        # evaluated inference mode: deterministic when mc_passes=0, MC mean otherwise.
        for i in range(len(img)):
            metrics.update(eval_probs[i], mask[i])
            if map_dir:
                payload = {
                    "prob": eval_probs[i, 0],
                    "deterministic_prob": det_probs[i, 0],
                    "label": mask[i, 0],
                }
                payload.update({name: values[i, 0] for name, values in uncertainty_maps.items()})
                np.savez_compressed(map_dir / f"{batch['id'][i]}.npz", **payload)

    labels = np.concatenate(sampled_labels)
    det_probs = np.concatenate(sampled_det_probs)
    eval_probs = np.concatenate(sampled_eval_probs)
    uncertainty = {name: np.concatenate(parts) for name, parts in sampled_uncertainties.items()}

    if len(labels) > uq_max_pixels:
        keep = rng.choice(len(labels), size=uq_max_pixels, replace=False)
        labels, det_probs, eval_probs = labels[keep], det_probs[keep], eval_probs[keep]
        uncertainty = {name: values[keep] for name, values in uncertainty.items()}

    deterministic_calibration = _calibration_breakdown(det_probs, labels)
    result = {
        "metrics": metrics.compute(),
        "per_tile_iou": metrics.per_tile_iou,
        "inference_mode": "mc_mean" if mc_passes > 0 else "deterministic",
        "uq_sample_pixels": int(len(labels)),
        "calibration": {"deterministic": deterministic_calibration},
        "ece": deterministic_calibration["overall"]["ece"],
        "brier": deterministic_calibration["overall"]["brier"],
        "selective_prediction": {
            "deterministic_entropy": _selective_metrics(
                det_probs, labels, uncertainty["deterministic_entropy"]
            ),
            "deterministic_confidence": _selective_metrics(
                det_probs, labels, uncertainty["deterministic_confidence"]
            ),
        },
    }

    if mc_passes > 0:
        result["calibration"]["mc_mean"] = _calibration_breakdown(eval_probs, labels)
        for name in (
            "mc_predictive_entropy",
            "mc_expected_entropy",
            "mc_mutual_information",
            "mc_variance",
        ):
            result["selective_prediction"][name] = _selective_metrics(
                eval_probs, labels, uncertainty[name]
            )

    return result
