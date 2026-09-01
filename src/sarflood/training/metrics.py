"""Segmentation metrics: accuracy, precision, recall, F1, IoU, kappa, boundary F1.

Confusion-matrix accumulation across a dataloader so metrics match the
pooled-over-dataset convention used by Ghosh et al. (2024); per-tile IoU is
also tracked for Wilcoxon tests.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


class SegmentationMetrics:
    def __init__(self, threshold: float = 0.5, boundary_tolerance: int = 3):
        self.threshold = threshold
        self.boundary_tolerance = boundary_tolerance
        self.reset()

    def reset(self):
        self.tp = self.fp = self.fn = self.tn = 0
        self.per_tile_iou: list[float] = []
        self._btp = self._bfp = self._bfn = 0

    @staticmethod
    def _boundary(mask: np.ndarray, tolerance: int) -> np.ndarray:
        eroded = ndimage.binary_erosion(mask, iterations=tolerance)
        return mask & ~eroded

    def update(self, probs: np.ndarray, target: np.ndarray):
        """probs: (1,H,W) float; target: (1,H,W) or (H,W) in {0,1}."""
        pred = (np.asarray(probs).squeeze() >= self.threshold)
        gt = np.asarray(target).squeeze().astype(bool)
        self.tp += int((pred & gt).sum())
        self.fp += int((pred & ~gt).sum())
        self.fn += int((~pred & gt).sum())
        self.tn += int((~pred & ~gt).sum())
        inter = (pred & gt).sum()
        union = (pred | gt).sum()
        self.per_tile_iou.append(float(inter / union) if union > 0 else float("nan"))
        # boundary F1
        pb, gb = self._boundary(pred, self.boundary_tolerance), self._boundary(gt, self.boundary_tolerance)
        self._btp += int((pb & ndimage.binary_dilation(gb, iterations=self.boundary_tolerance)).sum())
        self._bfp += int((pb & ~ndimage.binary_dilation(gb, iterations=self.boundary_tolerance)).sum())
        self._bfn += int((gb & ~ndimage.binary_dilation(pb, iterations=self.boundary_tolerance)).sum())

    def compute(self) -> dict[str, float]:
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        eps = 1e-9
        acc = (tp + tn) / (tp + fp + fn + tn + eps)
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)
        iou = tp / (tp + fp + fn + eps)
        po = acc
        pe = ((tp + fp) * (tp + fn) + (fp + tn) * (fn + tn)) / ((tp + fp + fn + tn) ** 2 + eps)
        kappa = (po - pe) / (1 - pe + eps)
        bprec = self._btp / (self._btp + self._bfp + eps)
        brec = self._btp / (self._btp + self._bfn + eps)
        bf1 = 2 * bprec * brec / (bprec + brec + eps)
        tile_ious = np.array(self.per_tile_iou)
        return {
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "iou": iou, "kappa": kappa, "boundary_f1": bf1,
            "miou_tiles": float(np.nanmean(tile_ious)) if len(tile_ious) else float("nan"),
        }


def eval_score(metrics: dict[str, float]) -> float:
    """Model-selection score used by Ghosh et al.: F1 + mIoU."""
    return metrics["f1"] + metrics["iou"]
