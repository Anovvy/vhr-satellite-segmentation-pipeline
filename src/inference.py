"""Sliding-window inference for georeferenced RGB rasters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import torch
from affine import Affine
from rasterio.crs import CRS
from rasterio.windows import Window

from . import NUM_CLASSES
from .models import forward_logits


IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


@dataclass(frozen=True)
class RasterMetadata:
    width: int
    height: int
    transform: Affine
    crs: CRS | None
    profile: dict


def _pad_tile(image: np.ndarray, tile_size: int) -> np.ndarray:
    height, width = image.shape[:2]
    if height == tile_size and width == tile_size:
        return image
    pad_h = tile_size - height
    pad_w = tile_size - width
    # NumPy reflect padding cannot extend a one-pixel axis, so edge padding is
    # used only for that rare border case.
    mode = "reflect" if height > 1 and width > 1 else "edge"
    return np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)


def _to_tensor(image: np.ndarray, input_scale: float) -> torch.Tensor:
    image = image.astype(np.float32) / float(input_scale)
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    chw = np.moveaxis(image, -1, 0)
    return torch.from_numpy(np.ascontiguousarray(chw)).unsqueeze(0)


def predict_raster(
    raster_path: str | Path,
    model: torch.nn.Module,
    model_name: str,
    device: torch.device,
    *,
    tile_size: int = 512,
    stride: int = 256,
    num_classes: int = NUM_CLASSES,
    input_scale: float = 255.0,
) -> tuple[np.ndarray, RasterMetadata]:
    """Predict a large raster using overlap-averaged class probabilities.

    The implementation mirrors the refined notebooks: the first three raster
    bands are treated as RGB, incomplete border windows use reflect padding,
    and probabilities from overlapping windows are averaged before argmax.
    """
    raster_path = Path(raster_path)
    if not raster_path.is_file():
        raise FileNotFoundError(f"Input raster not found: {raster_path}")
    if tile_size < 2:
        raise ValueError("tile_size must be at least 2.")
    if stride < 1 or stride > tile_size:
        raise ValueError("stride must be between 1 and tile_size.")
    if input_scale <= 0:
        raise ValueError("input_scale must be greater than zero.")

    with rasterio.open(raster_path) as src:
        if src.count < 3:
            raise ValueError("The input raster must contain at least three RGB bands.")

        metadata = RasterMetadata(
            width=src.width,
            height=src.height,
            transform=src.transform,
            crs=src.crs,
            profile=src.profile.copy(),
        )
        probability_sum = np.zeros(
            (num_classes, src.height, src.width), dtype=np.float32
        )
        observation_count = np.zeros((src.height, src.width), dtype=np.uint16)

        total_windows = len(range(0, src.height, stride)) * len(
            range(0, src.width, stride)
        )
        completed = 0

        with torch.inference_mode():
            for y in range(0, src.height, stride):
                for x in range(0, src.width, stride):
                    width = min(tile_size, src.width - x)
                    height = min(tile_size, src.height - y)
                    window = Window(x, y, width, height)
                    image = src.read(indexes=(1, 2, 3), window=window)
                    image = np.moveaxis(image, 0, -1)

                    if np.max(image) != 0:
                        padded = _pad_tile(image, tile_size)
                        batch = _to_tensor(padded, input_scale).to(device)
                        logits = forward_logits(
                            model, model_name, batch, (tile_size, tile_size)
                        )
                        probabilities = (
                            torch.softmax(logits, dim=1)
                            .squeeze(0)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        probability_sum[:, y : y + height, x : x + width] += (
                            probabilities[:, :height, :width]
                        )
                        observation_count[y : y + height, x : x + width] += 1

                    completed += 1
                    if completed % 100 == 0 or completed == total_windows:
                        print(f"  windows: {completed}/{total_windows}", end="\r")

    print()
    safe_count = np.maximum(observation_count, 1)
    probability_sum /= safe_count[np.newaxis, :, :]
    prediction = np.argmax(probability_sum, axis=0).astype(np.uint8)
    prediction[observation_count == 0] = 0
    return prediction, metadata


def save_prediction_raster(
    prediction: np.ndarray,
    metadata: RasterMetadata,
    output_path: str | Path,
) -> Path:
    """Write a single-band uint8 classified GeoTIFF."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile = metadata.profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        compress="deflate",
        predictor=2,
        nodata=None,
    )
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(prediction.astype(np.uint8), 1)
        dst.set_band_description(1, "land_cover_class_id")
    return output_path

