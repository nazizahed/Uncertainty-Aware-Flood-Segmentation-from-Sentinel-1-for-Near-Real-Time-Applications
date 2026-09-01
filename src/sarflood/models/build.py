"""Model factory: UNet / UNet++ with swappable encoders, plus decoder dropout injection.

Encoder names follow segmentation-models-pytorch conventions, e.g.:
  resnet34, inceptionv3, efficientnet-b0 .. efficientnet-b7,
  timm-mobilenetv3_large_100, mit_b0 (SegFormer encoder)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

ARCHS = {"unet": smp.Unet, "unetplusplus": smp.UnetPlusPlus}


def build_model(model_cfg: dict, in_channels: int) -> nn.Module:
    arch = model_cfg.get("arch", "unet")
    if arch not in ARCHS:
        raise ValueError(f"unknown arch '{arch}', choose from {list(ARCHS)}")
    model = ARCHS[arch](
        encoder_name=model_cfg["encoder"],
        encoder_weights=model_cfg.get("encoder_weights", "imagenet"),
        in_channels=in_channels,
        classes=1,
    )
    p = float(model_cfg.get("dropout", 0.0))
    if p > 0:
        inject_decoder_dropout(model, p)
    return model


def inject_decoder_dropout(model: nn.Module, p: float) -> nn.Module:
    """Attach a Dropout2d to every decoder block output.

    Dropout is placed in the *decoder only*, on purpose: EfficientNet/MobileNet encoders
    rely on BatchNorm, and injecting dropout throughout the encoder degrades calibration
    (known MC-dropout pitfall). Decoder-side placement is the standard, stable variant.
    Each decoder block gets a `.dropout` attribute so enable_mc_dropout() can find them.
    """
    if not hasattr(model, "decoder"):
        raise ValueError("model has no .decoder; dropout injection not supported")
    blocks = model.decoder.blocks
    # Unet: ModuleList; UnetPlusPlus: ModuleDict — iterate the actual blocks
    iterable = blocks.values() if isinstance(blocks, nn.ModuleDict) else blocks
    for block in iterable:
        block.dropout = nn.Dropout2d(p)
        block.register_forward_hook(lambda m, inp, out: m.dropout(out))
    return model


def enable_mc_dropout(model: nn.Module) -> None:
    """Put the model in eval mode except dropout modules, which stay active.

    Use for MC-dropout inference: deterministic BatchNorm statistics,
    stochastic decoder dropout.
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model: nn.Module) -> float:
    return count_parameters(model) * 4 / 1e6  # fp32
