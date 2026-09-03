"""MC-dropout and deep-ensemble inference with uncertainty decomposition."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .build import enable_mc_dropout


@torch.no_grad()
def stochastic_forward_passes(
    model: nn.Module, image: torch.Tensor, n_passes: int
) -> torch.Tensor:
    """N stochastic forward passes with dropout active.

    Returns ``(N, B, 1, H, W)`` per-pass flood probabilities.
    """
    if n_passes < 2:
        raise ValueError("MC-dropout uncertainty requires at least two stochastic passes")
    enable_mc_dropout(model)
    probs = []
    for _ in range(n_passes):
        logits = model(image)
        probs.append(torch.sigmoid(logits))
    return torch.stack(probs, dim=0)


@torch.no_grad()
def ensemble_forward_passes(
    models: list[nn.Module], image: torch.Tensor
) -> torch.Tensor:
    """One forward pass per ensemble member. Returns ``(M, B, 1, H, W)``."""
    if len(models) < 2:
        raise ValueError("Ensemble uncertainty requires at least two models")
    probs = []
    for m in models:
        m.eval()
        probs.append(torch.sigmoid(m(image)))
    return torch.stack(probs, dim=0)


def _binary_entropy(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-7, 1 - 1e-7)
    return -(p * p.log() + (1 - p) * (1 - p).log())


def summarize_passes(probs: torch.Tensor) -> dict[str, torch.Tensor]:
    """Predictive statistics from stacked probability maps ``(N,B,1,H,W)``.

    Returned quantities:
    - ``mean``: predictive mean flood probability.
    - ``variance``: disagreement variance across stochastic passes/members.
    - ``predictive_entropy``: entropy of the predictive mean; total predictive uncertainty.
    - ``expected_entropy``: mean entropy of individual predictions; a data/conditional
      uncertainty proxy.
    - ``mutual_information``: predictive entropy minus expected entropy, a standard
      epistemic/model-uncertainty proxy for Bayesian ensembles and MC dropout.
    - ``entropy``: backwards-compatible alias for ``predictive_entropy``.
    """
    if probs.ndim != 5:
        raise ValueError(f"Expected stacked probabilities with shape (N,B,1,H,W), got {probs.shape}")
    mean = probs.mean(dim=0)
    variance = probs.var(dim=0, unbiased=False)
    predictive_entropy = _binary_entropy(mean)
    expected_entropy = _binary_entropy(probs).mean(dim=0)
    mutual_information = (predictive_entropy - expected_entropy).clamp_min(0.0)
    return {
        "mean": mean,
        "variance": variance,
        "predictive_entropy": predictive_entropy,
        "expected_entropy": expected_entropy,
        "mutual_information": mutual_information,
        "entropy": predictive_entropy,
    }


def deterministic_entropy(probs: torch.Tensor) -> torch.Tensor:
    """Entropy baseline requiring only one deterministic forward pass."""
    return _binary_entropy(probs)


def confidence_uncertainty(probs: torch.Tensor) -> torch.Tensor:
    """Simple deterministic uncertainty in [0, 1], maximal at p=0.5."""
    return 1.0 - torch.abs(2.0 * probs - 1.0)


def passes_to_numpy(probs: torch.Tensor) -> np.ndarray:
    """Convert stacked passes to NumPy without dropping batch dimensions."""
    return probs.cpu().numpy()
