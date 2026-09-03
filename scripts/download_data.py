#!/usr/bin/env python
"""Download the ETCI 2021 dataset with a notebook-friendly fallback.

Official access is via the competition page (CodaLab registration):
    https://nasa-impact.github.io/etci2021/

For convenience this script pulls the public community mirror on Hugging Face:
    blanchon/ETCI-2021-Flood-Detection

The mirror is about 5.3 GB and contains many small files. Hugging Face currently
stores it with Xet; some Colab/Kaggle sessions can fail through that backend.
This helper first tries the normal Hub path and, if it fails, retries with Xet
disabled so the regular HTTP download path is used. Existing partial downloads
are reused by snapshot_download, so rerunning the command is safe.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO_ID = "blanchon/ETCI-2021-Flood-Detection"


def _looks_complete(root: Path) -> bool:
    """Cheap structural check; avoids treating an empty partial directory as data."""
    data = root / "data"
    if not data.exists():
        return False
    expected = (data / "train", data / "test", data / "test_internal")
    return all(p.exists() and any(p.iterdir()) for p in expected)


def _download(out: Path, disable_xet: bool) -> None:
    if disable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    else:
        os.environ.pop("HF_HUB_DISABLE_XET", None)

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(out),
        max_workers=4,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/etci")
    ap.add_argument(
        "--force-http",
        action="store_true",
        help="Disable Hugging Face Xet and use the regular HTTP path immediately.",
    )
    args = ap.parse_args()

    try:
        import huggingface_hub  # noqa: F401
    except ImportError as exc:
        raise SystemExit("huggingface_hub is missing; install the project data dependencies first") from exc

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if _looks_complete(out):
        print(f"ETCI structure already present at {out}; nothing to download.")
        return

    print(f"Downloading ETCI 2021 mirror to {out} (~5.3 GB; many small files).")
    print("Partial files from an interrupted attempt will be reused where possible.")

    if args.force_http:
        print("Using regular HTTP path (Xet disabled).")
        _download(out, disable_xet=True)
    else:
        try:
            _download(out, disable_xet=False)
        except Exception as first_error:
            print("\nInitial Hugging Face download failed:")
            print(f"  {type(first_error).__name__}: {first_error}")
            print("Retrying with HF_HUB_DISABLE_XET=1 (regular HTTP path)...")
            try:
                _download(out, disable_xet=True)
            except Exception as second_error:
                raise SystemExit(
                    "ETCI download failed through both Hugging Face backends.\n"
                    f"First error: {type(first_error).__name__}: {first_error}\n"
                    f"HTTP fallback error: {type(second_error).__name__}: {second_error}\n"
                    "The partial directory may be kept; rerunning can resume it."
                ) from second_error

    if not _looks_complete(out):
        raise SystemExit(
            f"Download command returned but the expected ETCI train/test/test_internal structure was not found under {out}."
        )

    print(f"Done. Point data.root at: {out}")
    print("The loader recursively discovers the event folders; no reorganization is needed.")


if __name__ == "__main__":
    main()
