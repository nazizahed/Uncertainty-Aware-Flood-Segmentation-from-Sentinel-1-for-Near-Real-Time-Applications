"""Loss functions: BCE + soft Dice, matching Ghosh et al. (2024) and Zhou et al. (2018)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = tuple(range(1, probs.ndim))
        inter = (probs * target).sum(dims)
        denom = probs.sum(dims) + target.sum(dims)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    """BCE + Dice, summed (per baseline). For UNet++ deep supervision, pass each
    of the four decoder-level outputs separately and sum — see training loop."""

    def __init__(self):
        super().__init__()
        self.dice = SoftDiceLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target)
        return bce + self.dice(logits, target)
