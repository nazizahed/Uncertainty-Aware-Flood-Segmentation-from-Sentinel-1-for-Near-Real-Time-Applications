"""Paired statistics for model comparison.

Primary: per-tile Wilcoxon signed-rank on IoU (valid under spatial autocorrelation
within tiles, unlike pixel-level tests). Secondary: block bootstrap CIs.
McNemar is included ONLY for direct comparability with Ghosh et al. (2024) —
do not use it as evidence on its own (pixel pseudo-replication).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def wilcoxon_per_tile(iou_a: np.ndarray, iou_b: np.ndarray) -> dict:
    """Paired Wilcoxon signed-rank on per-tile IoU. NaN tiles (empty union) dropped pairwise."""
    a, b = np.asarray(iou_a, float), np.asarray(iou_b, float)
    valid = ~(np.isnan(a) | np.isnan(b))
    a, b = a[valid], b[valid]
    nonzero = a != b
    if nonzero.sum() < 10:
        return {"n": int(nonzero.sum()), "p_value": float("nan"), "median_delta": float("nan")}
    res = stats.wilcoxon(a[nonzero], b[nonzero])
    return {
        "n": int(nonzero.sum()),
        "p_value": float(res.pvalue),
        "median_delta": float(np.median(a - b)),
        "mean_delta": float(np.mean(a - b)),
    }


def block_bootstrap_iou_delta(
    iou_a: np.ndarray, iou_b: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> dict:
    """Bootstrap CI for mean per-tile IoU difference (a - b), resampling tiles."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(iou_a, float), np.asarray(iou_b, float)
    valid = ~(np.isnan(a) | np.isnan(b))
    d = (a - b)[valid]
    if not len(d):
        return {"mean_delta": float("nan"), "ci95": [float("nan"), float("nan")]}
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return {
        "mean_delta": float(d.mean()),
        "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
    }


def mcnemar_pvalue(correct_a: np.ndarray, correct_b: np.ndarray) -> float:
    """McNemar exact test on pixel-wise correctness — for baseline comparability only."""
    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return float(min(1.0, 2 * stats.binom.cdf(k, n, 0.5)))
