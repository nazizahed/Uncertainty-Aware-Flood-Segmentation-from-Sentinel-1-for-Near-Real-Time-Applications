"""Training loop: Adam, BCE+Dice, AMP, model selection by F1+mean tile IoU."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from ..data.dataset import build_dataloaders
from ..models.build import build_model, count_parameters, model_size_mb
from .losses import BCEDiceLoss
from .metrics import SegmentationMetrics, eval_score


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def validate(model, loader, device, loss_fn) -> tuple[dict, float]:
    """Validate with genuinely per-tile metrics."""
    model.eval()
    metrics = SegmentationMetrics()
    total_loss, n = 0.0, 0
    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        logits = model(img)
        total_loss += loss_fn(logits, mask).item() * img.size(0)
        n += img.size(0)
        probs = torch.sigmoid(logits).cpu().numpy()
        targets = mask.cpu().numpy()
        for i in range(img.size(0)):
            metrics.update(probs[i], targets[i])
    return metrics.compute(), total_loss / max(n, 1)


def train(cfg: dict) -> Path:
    set_seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(cfg.get("output_dir", "runs")) / cfg["experiment_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    train_loader, val_loader = build_dataloaders(cfg)
    in_channels = len(cfg["data"]["bands"])
    model = build_model(cfg["model"], in_channels).to(device)
    n_params, size_mb = count_parameters(model), model_size_mb(model)

    tcfg = cfg["training"]
    accumulation_steps = max(1, int(tcfg.get("gradient_accumulation_steps", 1)))
    batch_size = int(tcfg["batch_size"])
    effective_batch = batch_size * accumulation_steps
    print(
        f"model: {cfg['model']['arch']}/{cfg['model']['encoder']}  "
        f"params={n_params/1e6:.2f}M  size={size_mb:.1f}MB (fp32)  "
        f"batch={batch_size} x accumulation={accumulation_steps} "
        f"=> effective_batch={effective_batch}"
    )

    opt = torch.optim.Adam(
        model.parameters(),
        lr=tcfg["lr"],
        weight_decay=tcfg.get("weight_decay", 0.0),
    )
    scheduler = None
    if tcfg.get("scheduler") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tcfg["epochs"])
    loss_fn = BCEDiceLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=tcfg.get("amp", True) and device == "cuda")

    log_path = out_dir / "log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "val_loss", "accuracy", "precision", "recall",
             "f1", "iou", "miou_tiles", "kappa", "boundary_f1", "eval_score", "seconds"]
        )

    best_score = -np.inf
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        t0, train_loss, n = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        n_batches = len(train_loader)

        for batch_index, batch in enumerate(
            tqdm(train_loader, desc=f"epoch {epoch}/{tcfg['epochs']}", leave=False), start=1
        ):
            img = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(img)
                raw_loss = loss_fn(logits, mask)
                loss = raw_loss / accumulation_steps

            scaler.scale(loss).backward()
            should_step = (batch_index % accumulation_steps == 0) or (batch_index == n_batches)
            if should_step:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            train_loss += raw_loss.item() * img.size(0)
            n += img.size(0)

        if scheduler is not None:
            scheduler.step()

        val_metrics, val_loss = validate(model, val_loader, device, loss_fn)
        score = eval_score(val_metrics)
        row = [epoch, train_loss / max(n, 1), val_loss,
               *[round(val_metrics[k], 5) for k in
                 ("accuracy", "precision", "recall", "f1", "iou", "miou_tiles", "kappa", "boundary_f1")],
               round(score, 5), round(time.time() - t0, 1)]
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        print(
            f"epoch {epoch}: val_loss={val_loss:.4f} pooled_IoU={val_metrics['iou']:.4f} "
            f"mIoU_tiles={val_metrics['miou_tiles']:.4f} F1={val_metrics['f1']:.4f} "
            f"score={score:.4f}"
        )

        if score > best_score:
            best_score = score
            torch.save(
                {"model_state": model.state_dict(), "config": cfg,
                 "epoch": epoch, "val_metrics": val_metrics},
                out_dir / "best.pt",
            )
        torch.save(
            {"model_state": model.state_dict(), "config": cfg, "epoch": epoch},
            out_dir / "last.pt",
        )

    print(f"done. best F1+mean-tile-IoU={best_score:.4f}  checkpoints in {out_dir}")
    return out_dir
