"""Segmentation metrics: accuracy, precision, recall, F1, IoU, kappa, boundary F1.

Confusion-matrix accumulation provides pooled metrics, while per-tile IoU is
tracked separately for model selection and paired statistical comparisons.
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
        """Update metrics for exactly one tile.

        ``probs`` may be shaped ``(1,H,W)`` or ``(H,W)``; target follows the same
        convention. Batch-shaped arrays should be iterated by the caller so that
        per-tile IoU and boundary morphology remain well defined.
        """
        pred = (np.asarray(probs).squeeze() >= self.threshold)
        gt = np.asarray(target).squeeze().astype(bool)
        if pred.ndim != 2 or gt.ndim != 2:
            raise ValueError(
                f"SegmentationMetrics.update expects one 2-D tile; got {pred.shape} and {gt.shape}"
            )
        self.tp += int((pred & gt).sum())
        self.fp += int((pred & ~gt).sum())
        self.fn += int((~pred & gt).sum())
        self.tn += int((~pred & ~gt).sum())
        inter = (pred & gt).sum()
        union = (pred | gt).sum()
        self.per_tile_iou.append(float(inter / union) if union > 0 else float("nan"))

        pb = self._boundary(pred, self.boundary_tolerance)
        gb = self._boundary(gt, self.boundary_tolerance)
        gb_dilated = ndimage.binary_dilation(gb, iterations=self.boundary_tolerance)
        pb_dilated = ndimage.binary_dilation(pb, iterations=self.boundary_tolerance)
        self._btp += int((pb & gb_dilated).sum())
        self._bfp += int((pb & ~gb_dilated).sum())
        self._bfn += int((gb & ~pb_dilated).sum())

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
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "iou": iou,
            "kappa": kappa,
            "boundary_f1": bf1,
            "miou_tiles": float(np.nanmean(tile_ious)) if len(tile_ious) else float("nan"),
        }


def eval_score(metrics: dict[str, float]) -> float:
    """Reference-aligned model-selection score: pooled F1 + mean per-tile IoU."""
    return metrics["f1"] + metrics["miou_tiles"]
