#!/usr/bin/env python
"""Evaluate a checkpoint: pooled metrics + calibration + optional MC-dropout UQ.

Usage:
  python scripts/evaluate.py --checkpoint runs/<run>/best.pt --regions florence
  python scripts/evaluate.py --checkpoint runs/<run>/best.pt --regions spain2019 --mc-passes 20 \
      --save-maps outputs/maps/spain2019
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sarflood.evaluation.evaluate import evaluate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--regions", nargs="+", required=True)
    ap.add_argument("--mc-passes", type=int, default=0,
                    help="N stochastic passes; 0 = deterministic eval")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--save-maps", default=None)
    ap.add_argument("--out", default=None, help="write results JSON here")
    args = ap.parse_args()

    result = evaluate(args.checkpoint, args.regions, mc_passes=args.mc_passes,
                      batch_size=args.batch_size, save_maps_dir=args.save_maps)
    printable = {k: v for k, v in result.items() if k != "per_tile_iou"}
    print(json.dumps(printable, indent=2, default=float))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, default=float)


if __name__ == "__main__":
    main()
