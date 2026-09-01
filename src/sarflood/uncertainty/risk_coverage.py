"""Selective-prediction analysis: risk-coverage curves, AURC, sparsification error.

This is the correct formulation of "filter the top-X% most uncertain pixels":
the model *abstains* on high-uncertainty pixels, and we measure how the error on the
remaining (covered) pixels falls as coverage decreases. An oracle ranks pixels by true
error; the gap between the model's ranking and the oracle's is the sparsification error.
"""

from __future__ import annotations

import numpy as np

_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy>=2.0 renamed trapz


def risk_coverage_curve(
    probs: np.ndarray, labels: np.ndarray, uncertainty: np.ndarray, n_steps: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (coverage, risk): coverage from 1.0 down, risk = error rate on covered pixels."""
    probs, labels, unc = probs.ravel(), labels.ravel().astype(bool), uncertainty.ravel()
    errors = ((probs >= 0.5) != labels).astype(np.float64)
    order = np.argsort(-unc)            # most uncertain first
    errors_sorted = errors[order]
    n = len(errors)
    coverages = np.linspace(1.0, 1.0 / n_steps, n_steps)
    risks = []
    for c in coverages:
        k = max(int(round(c * n)), 1)
        kept = errors_sorted[n - k:]    # drop the most uncertain (n-k) pixels
        risks.append(kept.mean())
    return coverages, np.array(risks)


def aurc(coverages: np.ndarray, risks: np.ndarray) -> float:
    """Area Under the Risk-Coverage curve (lower is better)."""
    order = np.argsort(coverages)
    return float(_trapezoid(risks[order], coverages[order]))


def sparsification_error(
    probs: np.ndarray, labels: np.ndarray, uncertainty: np.ndarray, n_steps: int = 50
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Model vs. oracle risk-coverage curves and the sparsification error (area gap)."""
    cov, risk_model = risk_coverage_curve(probs, labels, uncertainty, n_steps)
    oracle_unc = np.abs(probs - labels)  # oracle: uncertainty == true error magnitude
    _, risk_oracle = risk_coverage_curve(probs, labels, oracle_unc, n_steps)
    order = np.argsort(cov)
    se = float(_trapezoid((risk_model - risk_oracle)[order], cov[order]))
    return cov, risk_model, risk_oracle, se
