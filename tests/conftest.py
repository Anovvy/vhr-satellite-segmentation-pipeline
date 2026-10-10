"""Shared pytest fixtures for geospatial testing."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import torch
import torch.nn as nn
from affine import Affine
from rasterio.crs import CRS
from shapely.geometry import Polygon

from src import NUM_CLASSES
from src.inference import RasterMetadata


class DeterministicFiveClassModel(nn.Module):
    """Deterministic 5-class mock model for fast, network-free tests.

    Returns logits shaped (B, 5, H, W) where class logits are deterministically
    assigned based on the input spatial position or channel intensity.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes

    def forward(
        self,
        x: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
    ):
        tensor = pixel_values if pixel_values is not None else x
        if tensor is None:
            raise ValueError("Either x or pixel_values must be provided.")
        batch_size, _, height, width = tensor.shape
        device = tensor.device
        logits = torch.zeros(
            (batch_size, self.num_classes, height, width), device=device
        )
        grid_x = torch.arange(width, device=device).unsqueeze(0).expand(height, width)
        for class_id in range(self.num_classes):
            mask = (grid_x % self.num_classes) == class_id
            logits[:, class_id, :, :] = torch.where(mask, 10.0, -10.0)

        if pixel_values is not None:

            class OutputWrapper:
                def __init__(self, logits_tensor):
                    self.logits = logits_tensor

            return OutputWrapper(logits)
        return logits


@pytest.fixture
def mock_model() -> nn.Module:
    """Fixture providing a deterministic 5-class model instance."""
    return DeterministicFiveClassModel(num_classes=5)


def create_synthetic_geotiff(
    path: Path,
    width: int = 128,
    height: int = 128,
    crs: str = "EPSG:32748",
    transform: Affine | None = None,
    bands: int = 3,
    dtype: str = "uint8",
) -> Path:
    """Helper to write a synthetic georeferenced GeoTIFF."""
    if transform is None:
        # 0.3 meter resolution typical for VHR
        transform = Affine(0.3, 0.0, 700000.0, 0.0, -0.3, 9300000.0)

    # Generate deterministic RGB gradient pattern
    data = np.zeros((bands, height, width), dtype=dtype)
    for b in range(bands):
        for y in range(height):
            for x in range(width):
                data[b, y, x] = int(((x + y * 2 + b * 50) % 254) + 1)

    profile = {
        "driver": "GTiff",
        "count": bands,
        "dtype": dtype,
        "width": width,
        "height": height,
        "crs": CRS.from_user_input(crs) if crs else None,
        "transform": transform,
        "compress": "deflate",
    }

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)

    return path


@pytest.fixture
def synthetic_raster_32748(tmp_path: Path) -> Path:
    """Standard 128x128 3-band GeoTIFF with UTM CRS (EPSG:32748)."""
    raster_path = tmp_path / "synthetic_32748.tif"
    return create_synthetic_geotiff(
        raster_path, width=128, height=128, crs="EPSG:32748"
    )


@pytest.fixture
def synthetic_raster_odd_size(tmp_path: Path) -> Path:
    """Odd-size 73x59 3-band GeoTIFF to test boundary and padding conditions."""
    raster_path = tmp_path / "synthetic_odd.tif"
    return create_synthetic_geotiff(raster_path, width=73, height=59, crs="EPSG:32748")


@pytest.fixture
def synthetic_raster_4326(tmp_path: Path) -> Path:
    """Geographic CRS (EPSG:4326) 64x64 3-band GeoTIFF."""
    raster_path = tmp_path / "synthetic_4326.tif"
    transform = Affine(0.00001, 0.0, 106.8, 0.0, -0.00001, -6.2)
    return create_synthetic_geotiff(
        raster_path, width=64, height=64, crs="EPSG:4326", transform=transform
    )


@pytest.fixture
def synthetic_reference_gpkg(tmp_path: Path) -> Path:
    """GeoPackage reference vector with geometries for classes 0 to 4 in EPSG:32748."""
    # Matches top-left 700000.0, 9300000.0 with 0.3m per pixel
    # 128x128 pixels span: X from 700000.0 to 700038.4, Y from 9300000.0 down to 9299961.6
    polygons = [
        # Class 0: Unclassified / Other
        Polygon(
            [
                (700000.0, 9300000.0),
                (700006.0, 9300000.0),
                (700006.0, 9299990.0),
                (700000.0, 9299990.0),
            ]
        ),
        # Class 1: Residential
        Polygon(
            [
                (700007.0, 9300000.0),
                (700015.0, 9300000.0),
                (700015.0, 9299990.0),
                (700007.0, 9299990.0),
            ]
        ),
        # Class 2: Non-residential
        Polygon(
            [
                (700016.0, 9300000.0),
                (700024.0, 9300000.0),
                (700024.0, 9299990.0),
                (700016.0, 9299990.0),
            ]
        ),
        # Class 3: Grassland
        Polygon(
            [
                (700000.0, 9299980.0),
                (700015.0, 9299980.0),
                (700015.0, 9299970.0),
                (700000.0, 9299970.0),
            ]
        ),
        # Class 4: Urban Forest
        Polygon(
            [
                (700016.0, 9299980.0),
                (700030.0, 9299980.0),
                (700030.0, 9299970.0),
                (700016.0, 9299970.0),
            ]
        ),
    ]
    gdf = gpd.GeoDataFrame(
        {
            "class_id": [0, 1, 2, 3, 4],
            "category": [
                "Unclassified / Other",
                "Residential / Mixed-use Built-up",
                "Non-residential Built-up",
                "Grassland",
                "Urban Forest / Green Belt / Urban Park",
            ],
            "geometry": polygons,
        },
        crs="EPSG:32748",
    )
    gpkg_path = tmp_path / "reference_synthetic.gpkg"
    gdf.to_file(gpkg_path, driver="GPKG")
    return gpkg_path


@pytest.fixture
def sample_metadata() -> RasterMetadata:
    """Sample metadata object for tests."""
    return RasterMetadata(
        width=128,
        height=128,
        transform=Affine(0.3, 0.0, 700000.0, 0.0, -0.3, 9300000.0),
        crs=CRS.from_epsg(32748),
        profile={"driver": "GTiff", "count": 1, "dtype": "uint8"},
    )
