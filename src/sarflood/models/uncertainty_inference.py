"""MC-dropout and deep-ensemble inference: predictive mean, epistemic variance, entropy."""

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
    probs = []
    for m in models:
        m.eval()
        probs.append(torch.sigmoid(m(image)))
    return torch.stack(probs, dim=0)


def summarize_passes(probs: torch.Tensor) -> dict[str, torch.Tensor]:
    """Predictive statistics from stacked probability maps (N,1,H,W).

    - mean:     predictive mean (flood probability)
    - variance: epistemic uncertainty (disagreement between passes/members)
    - entropy:  predictive entropy of the mean (total uncertainty)
    """
    mean = probs.mean(dim=0)
    var = probs.var(dim=0, unbiased=False)
    p = mean.clamp(1e-7, 1 - 1e-7)
    entropy = -(p * p.log() + (1 - p) * (1 - p).log())
    return {"mean": mean, "variance": var, "entropy": entropy}


def passes_to_numpy(probs: torch.Tensor) -> np.ndarray:
    """Convert stacked passes to NumPy without dropping batch dimensions."""
    return probs.cpu().numpy()
