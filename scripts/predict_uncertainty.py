#!/usr/bin/env python
"""Generate uncertainty maps (MC dropout or deep ensemble) for a dataset split.

Usage:
  python scripts/predict_uncertainty.py --checkpoint runs/<run>/best.pt --regions florence \
      --method mc_dropout --passes 20 --out outputs/uncertainty/florence
  python scripts/predict_uncertainty.py --checkpoints runs/seed{1..5}/best.pt \
      --regions spain2019 --method ensemble --out outputs/uncertainty/spain2019
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sarflood.data.dataset import ETCIFloodDataset  # noqa: E402
from sarflood.evaluation.evaluate import load_checkpoint  # noqa: E402
from sarflood.models.uncertainty_inference import (  # noqa: E402
    ensemble_forward_passes, stochastic_forward_passes, summarize_passes,
)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="single checkpoint (mc_dropout)")
    ap.add_argument("--checkpoints", nargs="+", default=None, help="members (ensemble)")
    ap.add_argument("--regions", nargs="+", required=True)
    ap.add_argument("--method", choices=["mc_dropout", "ensemble"], required=True)
    ap.add_argument("--passes", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.method == "ensemble":
        assert args.checkpoints, "ensemble requires --checkpoints"
        models, cfg = [], None
        for p in args.checkpoints:
            m, cfg = load_checkpoint(p, device)
            models.append(m)
    else:
        assert args.checkpoint, "mc_dropout requires --checkpoint"
        model, cfg = load_checkpoint(args.checkpoint, device)

    ds = ETCIFloodDataset(cfg["data"]["root"], args.regions, cfg["data"]["bands"],
                          rotation_aug=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    for batch in loader:
        img = batch["image"].to(device)
        if args.method == "ensemble":
            passes = ensemble_forward_passes(models, img)          # (M,B,1,H,W)
        else:
            passes = stochastic_forward_passes(model, img, args.passes)
        s = summarize_passes(passes)
        mean, var, ent = s["mean"].cpu().numpy(), s["variance"].cpu().numpy(), s["entropy"].cpu().numpy()
        for i, tile_id in enumerate(batch["id"]):
            np.savez_compressed(
                out / f"{tile_id}.npz",
                flood_prob=mean[i, 0].astype(np.float32),
                epistemic_var=var[i, 0].astype(np.float32),
                entropy=ent[i, 0].astype(np.float32),
                label=batch["mask"][i, 0].numpy(),
            )
    print(f"wrote {len(ds)} tiles to {out}")


if __name__ == "__main__":
    main()
