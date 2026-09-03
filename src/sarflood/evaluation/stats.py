"""Paired statistics for model comparison.

Primary inference is based on per-tile paired differences. For confidence
intervals, ``grouped_bootstrap_iou_delta`` can resample higher-level independent
units (for example Sentinel-1 scenes or flood events) rather than pretending
individual adjacent tiles are independent blocks.

Pixel-level McNemar is retained only for direct comparability with earlier work;
it should not be used as standalone evidence because spatial autocorrelation
causes severe pixel pseudo-replication.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def wilcoxon_per_tile(iou_a: np.ndarray, iou_b: np.ndarray) -> dict:
    """Paired Wilcoxon signed-rank on per-tile IoU; NaNs are dropped pairwise."""
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


def tile_bootstrap_iou_delta(
    iou_a: np.ndarray, iou_b: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> dict:
    """Ordinary bootstrap CI over paired tiles.

    Use only when tiles can reasonably be treated as independent sampling units.
    For adjacent tiles from common scenes/events, prefer grouped_bootstrap_iou_delta.
    """
    rng = np.random.default_rng(seed)
    a, b = np.asarray(iou_a, float), np.asarray(iou_b, float)
    valid = ~(np.isnan(a) | np.isnan(b))
    d = (a - b)[valid]
    if not len(d):
        return {"mean_delta": float("nan"), "ci95": [float("nan"), float("nan")], "n": 0}
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return {
        "mean_delta": float(d.mean()),
        "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "n": int(len(d)),
    }


def grouped_bootstrap_iou_delta(
    iou_a: np.ndarray,
    iou_b: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Cluster/group bootstrap CI for mean paired IoU difference.

    ``groups`` should identify a higher-level unit shared by spatially correlated
    tiles, such as source scene, flood event, or acquisition. Groups are sampled
    with replacement and all valid tiles belonging to a selected group are kept.
    """
    a, b = np.asarray(iou_a, float), np.asarray(iou_b, float)
    groups = np.asarray(groups)
    if not (len(a) == len(b) == len(groups)):
        raise ValueError("iou_a, iou_b, and groups must have identical length")

    valid = ~(np.isnan(a) | np.isnan(b))
    d, g = (a - b)[valid], groups[valid]
    if not len(d):
        return {
            "mean_delta": float("nan"),
            "ci95": [float("nan"), float("nan")],
            "n_tiles": 0,
            "n_groups": 0,
        }

    unique_groups = np.unique(g)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sampled_parts = [d[g == group] for group in sampled_groups]
        boots.append(np.concatenate(sampled_parts).mean())
    boots = np.asarray(boots)
    return {
        "mean_delta": float(d.mean()),
        "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "n_tiles": int(len(d)),
        "n_groups": int(len(unique_groups)),
    }


# Backwards-compatible name, but no longer mislabeled as a true block bootstrap.
def block_bootstrap_iou_delta(
    iou_a: np.ndarray, iou_b: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> dict:
    return tile_bootstrap_iou_delta(iou_a, iou_b, n_boot=n_boot, seed=seed)


def mcnemar_pvalue(correct_a: np.ndarray, correct_b: np.ndarray) -> float:
    """McNemar exact test on pixel-wise correctness — comparability only."""
    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return float(min(1.0, 2 * stats.binom.cdf(k, n, 0.5)))
