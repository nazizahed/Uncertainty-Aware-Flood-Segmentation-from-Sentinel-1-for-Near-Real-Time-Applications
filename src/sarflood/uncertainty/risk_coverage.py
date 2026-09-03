"""Selective-prediction analysis: risk-coverage curves and sparsification error.

The model abstains on high-uncertainty pixels and risk is measured on the
remaining pixels. The default risk is zero-one segmentation error after applying
the classification threshold. The oracle therefore ranks actual classification
errors ahead of correct predictions, matching the risk being optimized.
"""

from __future__ import annotations

import numpy as np

_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def _classification_errors(
    probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    return ((probs.ravel() >= threshold) != labels.ravel().astype(bool)).astype(np.float64)


def risk_coverage_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    uncertainty: np.ndarray,
    n_steps: int = 50,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return coverage and zero-one risk on retained pixels.

    Pixels are sorted by descending uncertainty and the most uncertain pixels are
    progressively removed. Lower risk at a given coverage is better.
    """
    unc = uncertainty.ravel()
    errors = _classification_errors(probs, labels, threshold)
    if len(unc) != len(errors):
        raise ValueError("uncertainty, probabilities, and labels must contain the same number of pixels")
    if not len(errors):
        raise ValueError("risk-coverage requires at least one pixel")

    order = np.argsort(-unc, kind="stable")
    errors_sorted = errors[order]
    n = len(errors)
    coverages = np.linspace(1.0, 1.0 / n_steps, n_steps)
    risks = []
    for coverage in coverages:
        k = max(int(round(coverage * n)), 1)
        kept = errors_sorted[n - k:]
        risks.append(kept.mean())
    return coverages, np.asarray(risks)


def aurc(coverages: np.ndarray, risks: np.ndarray) -> float:
    """Area under the risk-coverage curve; lower is better."""
    order = np.argsort(coverages)
    return float(_trapezoid(risks[order], coverages[order]))


def sparsification_error(
    probs: np.ndarray,
    labels: np.ndarray,
    uncertainty: np.ndarray,
    n_steps: int = 50,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compare model ranking against the oracle for zero-one classification risk."""
    cov, risk_model = risk_coverage_curve(
        probs, labels, uncertainty, n_steps=n_steps, threshold=threshold
    )
    errors = _classification_errors(probs, labels, threshold)
    # Any ordering that removes erroneous predictions before correct predictions
    # is optimal for zero-one risk. Using errors themselves gives that oracle.
    _, risk_oracle = risk_coverage_curve(
        probs, labels, errors, n_steps=n_steps, threshold=threshold
    )
    order = np.argsort(cov)
    se = float(_trapezoid((risk_model - risk_oracle)[order], cov[order]))
    return cov, risk_model, risk_oracle, max(se, 0.0)
