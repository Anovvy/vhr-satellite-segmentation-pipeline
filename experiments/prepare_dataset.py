#!/usr/bin/env python3
"""Create 512 px image/mask chips from a raster and wall-to-wall labels.

This portable script documents the original dataset-preparation experiment.
The proprietary source imagery and labels are not distributed.

Class 0 is a semantic Unclassified/Other class, not training background. The
original masks classify every pixel into IDs 0-4.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window


CLASS_MAPPING = {
    "Unclassified": 0,
    "Bangunan Permukiman/campuran": 1,
    "Bangunan bukan permukiman": 2,
    "Padang Rumput": 3,
    "Hutan Kota, jalur hijau dan taman kota": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raster", type=Path, required=True)
    parser.add_argument("--vector", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-field", default="Dataset_Cl")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument(
        "--skip-empty-masks",
        action="store_true",
        help=(
            "Skip all-zero masks. Avoid this option when a valid chip contains "
            "only the Unclassified/Other class."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    images_dir = args.output_dir / "images"
    masks_dir = args.output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    labels = gpd.read_file(args.vector)
    if args.class_field not in labels.columns:
        raise ValueError(
            f"Class field '{args.class_field}' not found. Fields: {list(labels.columns)}"
        )
    labels = labels[labels[args.class_field].isin(CLASS_MAPPING)].copy()
    labels["class_id"] = labels[args.class_field].map(CLASS_MAPPING)

    count = 0
    with rasterio.open(args.raster) as source:
        if source.count < 3:
            raise ValueError("Input raster must contain at least three RGB bands.")
        if labels.crs != source.crs:
            labels = labels.to_crs(source.crs)

        for y in range(0, source.height, args.tile_size):
            for x in range(0, source.width, args.tile_size):
                if x + args.tile_size > source.width or y + args.tile_size > source.height:
                    continue

                window = Window(x, y, args.tile_size, args.tile_size)
                image = source.read(indexes=(1, 2, 3), window=window)
                if np.max(image) == 0:
                    continue

                bounds = rasterio.windows.bounds(window, source.transform)
                min_x, min_y, max_x, max_y = bounds
                tile_labels = labels.cx[min_x:max_x, min_y:max_y]
                pairs = (
                    (geometry, int(value))
                    for geometry, value in zip(
                        tile_labels.geometry, tile_labels.class_id
                    )
                )
                mask = rasterize(
                    pairs,
                    out_shape=(args.tile_size, args.tile_size),
                    transform=source.window_transform(window),
                    fill=0,
                    dtype=np.uint8,
                )
                if args.skip_empty_masks and not np.any(mask):
                    continue

                filename = f"tile_y{y}_x{x}.png"
                rgb = np.moveaxis(image, 0, -1)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(images_dir / filename), bgr)
                cv2.imwrite(str(masks_dir / filename), mask)
                count += 1

    print(f"Created {count} image/mask pairs under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
