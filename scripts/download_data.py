#!/usr/bin/env python
"""Download the ETCI 2021 dataset.

Official access is via the competition page (CodaLab registration):
    https://nasa-impact.github.io/etci2021/

For convenience this script pulls the community mirror on Hugging Face
(blanchon/ETCI-2021-Flood-Detection, 5.6 GB, 66,810 tiles) which preserves the
original folder layout:

    python scripts/download_data.py --out data/etci

NOTE on terms of use: any publication using this dataset must include the
ETCI acknowledgement (see the competition page). For full compliance, register
on CodaLab as well.
"""

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/etci")
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("pip install huggingface_hub first")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"downloading ETCI 2021 mirror to {out} (5.6 GB, this takes a while)...")
    snapshot_download(
        repo_id="blanchon/ETCI-2021-Flood-Detection",
        repo_type="dataset",
        local_dir=str(out),
    )
    print(f"done. point your config's data.root at: {out}")
    print("(the loader auto-discovers <region>_<datetime>/ folders, no reorganization needed)")


if __name__ == "__main__":
    main()
