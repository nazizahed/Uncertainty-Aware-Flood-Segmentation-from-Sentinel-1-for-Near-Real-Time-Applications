"""ETCI 2021 Sentinel-1 flood dataset discovery and loading.

The official release and the documented community mirror use slightly different
parent directories, but both contain event folders with ``vv``, ``vh``, and
``flood_label`` subdirectories (sometimes below an additional ``tiles`` level).
This module discovers either layout without assuming a fixed train/test parent.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler


VALID_BANDS = {"vv", "vh", "ratio"}


def _normalise_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _tile_key(path: Path) -> str:
    """Return the common tile identifier used to pair VV, VH, and labels."""
    return re.sub(
        r"_(?:vv|vh|flood|flood_label|water_body|water_body_label)$",
        "",
        path.stem.lower(),
    )


def _event_name(label_dir: Path) -> str:
    parent = label_dir.parent
    return parent.parent.name if parent.name.lower() == "tiles" else parent.name


@dataclass(frozen=True)
class TileRecord:
    tile_id: str
    vv: Path
    vh: Path
    flood_label: Path


class ETCIFloodDataset(Dataset):
    """Paired VV/VH imagery and binary flood masks from ETCI 2021.

    Parameters
    ----------
    root:
        Dataset root. Event directories may occur anywhere below this path.
    regions:
        Region prefixes such as ``north_alabama`` or ``florence``. Punctuation
        and case are ignored while matching event-directory names.
    bands:
        Ordered subset of ``vv``, ``vh``, and ``ratio``.
    rotation_aug:
        If true, expose the original tile plus deterministic 90, 180, and
        270-degree rotations, matching the baseline augmentation protocol.
    image_size:
        Optional square output size. The official tiles are already 256 x 256.
    ratio_clip:
        Symmetric clipping bound for the stabilized polarization ratio. The
        clipped ratio is divided by this value, yielding a channel in [-1, 1].
    """

    def __init__(
        self,
        root: str | Path,
        regions: Sequence[str],
        bands: Sequence[str] = ("vv", "vh", "ratio"),
        rotation_aug: bool = False,
        image_size: int | None = 256,
        ratio_clip: float = 10.0,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.regions = tuple(regions)
        self.bands = tuple(bands)
        self.rotation_aug = rotation_aug
        self.image_size = image_size
        self.ratio_clip = float(ratio_clip)

        invalid = set(self.bands) - VALID_BANDS
        if invalid:
            raise ValueError(f"Unsupported bands: {sorted(invalid)}")
        if not self.bands:
            raise ValueError("At least one input band is required")
        if self.ratio_clip <= 0:
            raise ValueError("ratio_clip must be positive")
        if not self.root.exists():
            raise FileNotFoundError(f"ETCI root does not exist: {self.root}")

        self.records = self._discover_records()
        if not self.records:
            requested = ", ".join(self.regions)
            raise FileNotFoundError(
                f"No paired ETCI tiles found below {self.root} for regions: {requested}"
            )
        self._base_flood_flags: np.ndarray | None = None

    def _discover_records(self) -> list[TileRecord]:
        requested = {_normalise_name(region) for region in self.regions}
        records: list[TileRecord] = []

        for label_dir in sorted(self.root.rglob("flood_label")):
            if not label_dir.is_dir():
                continue
            event = _event_name(label_dir)
            event_normalised = _normalise_name(event)
            if not any(event_normalised.startswith(region) for region in requested):
                continue

            tile_root = label_dir.parent
            vv_dir, vh_dir = tile_root / "vv", tile_root / "vh"
            if not vv_dir.is_dir() or not vh_dir.is_dir():
                continue

            vv_files = {_tile_key(path): path for path in vv_dir.glob("*.png")}
            vh_files = {_tile_key(path): path for path in vh_dir.glob("*.png")}
            label_files = {_tile_key(path): path for path in label_dir.glob("*.png")}
            common = sorted(vv_files.keys() & vh_files.keys() & label_files.keys())
            if not common and label_files:
                raise ValueError(f"Could not pair VV, VH, and flood labels in {tile_root}")

            for key in common:
                records.append(
                    TileRecord(
                        tile_id=f"{event}/{key}",
                        vv=vv_files[key],
                        vh=vh_files[key],
                        flood_label=label_files[key],
                    )
                )
        return records

    def __len__(self) -> int:
        multiplier = 4 if self.rotation_aug else 1
        return len(self.records) * multiplier

    def _index_and_rotation(self, index: int) -> tuple[int, int]:
        if self.rotation_aug:
            return index // 4, index % 4
        return index, 0

    def _read_image(self, path: Path, *, mask: bool = False) -> np.ndarray:
        with Image.open(path) as image:
            image = image.convert("L")
            if self.image_size and image.size != (self.image_size, self.image_size):
                resampling = Image.Resampling.NEAREST if mask else Image.Resampling.BILINEAR
                image = image.resize((self.image_size, self.image_size), resampling)
            array = np.asarray(image)
        if mask:
            return (array > 0).astype(np.float32)
        return array.astype(np.float32) / 255.0

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        base_index, rotations = self._index_and_rotation(index)
        record = self.records[base_index]
        vv = self._read_image(record.vv)
        vh = self._read_image(record.vh)
        mask = self._read_image(record.flood_label, mask=True)

        channels = []
        for band in self.bands:
            if band == "vv":
                channels.append(vv)
            elif band == "vh":
                channels.append(vh)
            else:
                difference = vv - vh
                epsilon = 1.0 / 255.0
                safe_difference = np.where(
                    np.abs(difference) < epsilon,
                    np.where(difference < 0, -epsilon, epsilon),
                    difference,
                )
                ratio = (vv + vh) / safe_difference
                channels.append(np.clip(ratio, -self.ratio_clip, self.ratio_clip) / self.ratio_clip)

        image = np.stack(channels).astype(np.float32, copy=False)
        if rotations:
            image = np.rot90(image, k=rotations, axes=(-2, -1)).copy()
            mask = np.rot90(mask, k=rotations, axes=(-2, -1)).copy()

        return {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(mask[None, ...]),
            "id": record.tile_id.replace("/", "__"),
        }

    @property
    def flood_flags(self) -> np.ndarray:
        """Boolean flood-presence flag for every exposed sample."""
        if self._base_flood_flags is None:
            self._base_flood_flags = np.array(
                [self._read_image(record.flood_label, mask=True).any() for record in self.records],
                dtype=bool,
            )
        return np.repeat(self._base_flood_flags, 4) if self.rotation_aug else self._base_flood_flags


class StratifiedFloodBatchSampler(Sampler[list[int]]):
    """Sample batches with at least a requested fraction of flood-positive tiles."""

    def __init__(
        self,
        flood_flags: np.ndarray,
        batch_size: int,
        flood_fraction: float = 0.5,
        seed: int = 42,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not 0 < flood_fraction <= 1:
            raise ValueError("flood_fraction must be in (0, 1]")
        flags = np.asarray(flood_flags, dtype=bool)
        self.positive = np.flatnonzero(flags)
        self.negative = np.flatnonzero(~flags)
        if not len(self.positive) or not len(self.negative):
            raise ValueError("Stratified sampling requires both positive and negative tiles")
        self.batch_size = batch_size
        self.positive_per_batch = min(math.ceil(batch_size * flood_fraction), batch_size)
        self.negative_per_batch = batch_size - self.positive_per_batch
        self.n_batches = math.ceil(len(flags) / batch_size)
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(self.n_batches):
            parts = [
                rng.choice(self.positive, self.positive_per_batch, replace=True),
            ]
            if self.negative_per_batch:
                parts.append(rng.choice(self.negative, self.negative_per_batch, replace=True))
            batch = np.concatenate(parts)
            rng.shuffle(batch)
            yield batch.tolist()


def build_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    """Construct training and Florence-held-out validation dataloaders."""
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    common = {
        "root": data_cfg["root"],
        "bands": data_cfg["bands"],
        "image_size": data_cfg.get("image_size", 256),
        "ratio_clip": data_cfg.get("ratio_clip", 10.0),
    }
    train_dataset = ETCIFloodDataset(
        regions=data_cfg["regions"],
        rotation_aug=data_cfg.get("rotation_aug", False),
        **common,
    )
    val_dataset = ETCIFloodDataset(
        regions=data_cfg["val_regions"],
        rotation_aug=False,
        **common,
    )

    batch_size = int(train_cfg["batch_size"])
    workers = int(train_cfg.get("num_workers", 0))
    sampler = StratifiedFloodBatchSampler(
        train_dataset.flood_flags,
        batch_size=batch_size,
        flood_fraction=float(data_cfg.get("stratified_flood_fraction", 0.5)),
        seed=int(cfg.get("seed", 42)),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader
