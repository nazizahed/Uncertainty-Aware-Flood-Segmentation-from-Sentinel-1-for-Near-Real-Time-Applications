"""Calibration metrics: Expected Calibration Error, Brier score, temperature scaling."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """ECE over all pixels. probs/labels: flattened arrays."""
    probs, labels = probs.ravel(), labels.ravel().astype(bool)
    edges = np.linspace(0, 1, n_bins + 1)
    ece, n = 0.0, len(probs)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (probs > lo) & (probs <= hi)
        if m.any():
            ece += m.sum() / n * abs(labels[m].mean() - probs[m].mean())
    return float(ece)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs.ravel() - labels.ravel()) ** 2))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit a scalar temperature on validation logits (Guo et al. 2017)."""
    logits, labels = logits.ravel().astype(np.float64), labels.ravel().astype(np.float64)

    def nll(t):
        z = np.clip(logits / t, -30, 30)
        return np.mean(np.logaddexp(0, z) - labels * z)

    res = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    return float(res.x)


def reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15):
    """(bin_centers, observed_freq, predicted_conf, count) for reliability diagrams."""
    probs, labels = probs.ravel(), labels.ravel().astype(bool)
    edges = np.linspace(0, 1, n_bins + 1)
    centers, obs, conf, cnt = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (probs > lo) & (probs <= hi)
        if m.any():
            centers.append((lo + hi) / 2)
            obs.append(labels[m].mean())
            conf.append(probs[m].mean())
            cnt.append(int(m.sum()))
    return np.array(centers), np.array(obs), np.array(conf), np.array(cnt)
