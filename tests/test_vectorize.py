"""Tests for vectorization to GeoPackage and Shapefile."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from rasterio.features import rasterize

from src.inference import RasterMetadata
from src.vectorize import vectorize_prediction


def test_vectorize_gpkg_and_shp_roundtrip(sample_metadata: RasterMetadata, tmp_path: Path):
    """Vectorize a prediction with all 5 classes and test rasterize-back roundtrip."""
    # Create simple 128x128 grid where quadrant/bands correspond to classes
    pred = np.zeros((128, 128), dtype=np.uint8)
    pred[:64, :64] = 0
    pred[:64, 64:] = 1
    pred[64:, :64] = 2
    pred[64:96, 64:] = 3
    pred[96:, 64:] = 4

    gpkg_file = tmp_path / "test_vec.gpkg"
    out_path = vectorize_prediction(pred, sample_metadata, gpkg_file, include_unclassified=True)
    assert out_path.is_file()

    gdf = gpd.read_file(out_path)
    assert len(gdf) >= 5
    assert gdf.crs == sample_metadata.crs
    assert set(gdf["class_id"].unique()) == {0, 1, 2, 3, 4}
    assert all(gdf.geometry.is_valid)

    # Test roundtrip: rasterize the polygons back and compare
    shapes = ((geom, int(cid)) for geom, cid in zip(gdf.geometry, gdf["class_id"], strict=False))
    rasterized_back = rasterize(
        shapes,
        out_shape=(sample_metadata.height, sample_metadata.width),
        transform=sample_metadata.transform,
        fill=0,
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(rasterized_back, pred)


def test_vectorize_exclude_unclassified(sample_metadata: RasterMetadata, tmp_path: Path):
    """Ensure class 0 is omitted when include_unclassified=False."""
    pred = np.zeros((64, 64), dtype=np.uint8)
    pred[10:30, 10:30] = 1

    gpkg_file = tmp_path / "no_unclassified.gpkg"
    vectorize_prediction(pred, sample_metadata, gpkg_file, include_unclassified=False)

    gdf = gpd.read_file(gpkg_file)
    assert 0 not in gdf["class_id"].values
    assert 1 in gdf["class_id"].values


def test_vectorize_invalid_extension_raises(sample_metadata: RasterMetadata, tmp_path: Path):
    """Non-gpkg, non-shp formats must raise ValueError."""
    pred = np.ones((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError, match=r"\.gpkg or \.shp"):
        vectorize_prediction(pred, sample_metadata, tmp_path / "invalid.geojson")
