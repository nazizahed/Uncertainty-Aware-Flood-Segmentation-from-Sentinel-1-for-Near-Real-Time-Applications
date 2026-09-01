"""Synthetic ETCI-layout tests for discovery, channels, rotations, and sampling."""

from pathlib import Path

import numpy as np
from PIL import Image

from sarflood.data.dataset import ETCIFloodDataset, StratifiedFloodBatchSampler


def _write_tile(folder: Path, name: str, array: np.ndarray) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8)).save(folder / name)


def test_dataset_discovers_mirror_layout_and_builds_channels(tmp_path):
    tile_root = tmp_path / "data" / "train" / "north_alabama_20200101t000000" / "tiles"
    for index, flooded in enumerate((False, True)):
        prefix = f"north_alabama_20200101t000000_x-{index}_y-0"
        vv = np.full((16, 16), 180 + index, dtype=np.uint8)
        vh = np.full((16, 16), 80 + index, dtype=np.uint8)
        label = np.zeros((16, 16), dtype=np.uint8)
        if flooded:
            label[4:12, 4:12] = 255
        _write_tile(tile_root / "vv", f"{prefix}_vv.png", vv)
        _write_tile(tile_root / "vh", f"{prefix}_vh.png", vh)
        _write_tile(tile_root / "flood_label", f"{prefix}_vv.png", label)

    dataset = ETCIFloodDataset(
        tmp_path,
        regions=["north_alabama"],
        bands=["vv", "vh", "ratio"],
        rotation_aug=True,
        image_size=16,
    )
    assert len(dataset) == 8
    sample = dataset[0]
    assert sample["image"].shape == (3, 16, 16)
    assert sample["mask"].shape == (1, 16, 16)
    assert np.isfinite(sample["image"].numpy()).all()
    assert np.abs(sample["image"][2].numpy()).max() <= 1
    assert dataset.flood_flags.tolist() == [False] * 4 + [True] * 4

    sampler = StratifiedFloodBatchSampler(dataset.flood_flags, batch_size=4, flood_fraction=0.5)
    for batch in sampler:
        assert dataset.flood_flags[batch].sum() >= 2
