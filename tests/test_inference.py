"""Tests for sliding-window inference, padding, and GeoTIFF export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch

from src.inference import (
    _pad_tile,
    _to_tensor,
    predict_raster,
    save_prediction_raster,
)


def test_pad_tile_exact_size():
    """Tiles that already match tile_size must remain unpadded."""
    arr = np.ones((512, 512, 3), dtype=np.uint8)
    padded = _pad_tile(arr, 512)
    assert padded.shape == (512, 512, 3)
    np.testing.assert_array_equal(arr, padded)


def test_pad_tile_reflect_and_edge():
    """Incomplete border tile is padded to square tile_size."""
    arr = np.ones((100, 80, 3), dtype=np.uint8) * 42
    padded = _pad_tile(arr, 128)
    assert padded.shape == (128, 128, 3)
    # Check top-left contains the original values
    np.testing.assert_array_equal(padded[:100, :80, :], arr)

    # 1-pixel edge case triggers "edge" mode
    arr_1px = np.ones((1, 5, 3), dtype=np.uint8)
    padded_1px = _pad_tile(arr_1px, 8)
    assert padded_1px.shape == (8, 8, 3)


def test_to_tensor_normalization():
    """Verify ImageNet normalization math on input numpy array."""
    arr = np.full((32, 32, 3), 255, dtype=np.uint8)
    tensor = _to_tensor(arr, input_scale=255.0)
    assert tensor.shape == (1, 3, 32, 32)
    assert tensor.dtype == torch.float32
    # With 255 / 255.0 = 1.0:
    # (1.0 - 0.485) / 0.229 = ~2.2489
    expected_r = (1.0 - 0.485) / 0.229
    assert tensor[0, 0, 0, 0].item() == pytest.approx(expected_r, rel=1e-3)


def test_predict_raster_standard(
    synthetic_raster_32748: Path, mock_model, tmp_path: Path
):
    """Predict standard UTM raster with overlap and assert spatial properties."""
    device = torch.device("cpu")
    pred, metadata = predict_raster(
        synthetic_raster_32748,
        mock_model,
        "segformer",
        device,
        tile_size=64,
        stride=32,
    )

    assert pred.shape == (128, 128)
    assert pred.dtype == np.uint8
    assert metadata.width == 128
    assert metadata.height == 128
    assert metadata.crs is not None
    assert metadata.crs.to_epsg() == 32748

    # Save to disk and verify GeoTIFF integrity
    out_file = tmp_path / "pred_standard.tif"
    saved = save_prediction_raster(pred, metadata, out_file)
    assert saved.is_file()

    with rasterio.open(saved) as ds:
        assert ds.count == 1
        assert ds.dtypes[0] == "uint8"
        assert ds.width == 128
        assert ds.height == 128
        assert ds.crs.to_epsg() == 32748
        assert ds.transform == metadata.transform
        assert ds.descriptions[0] == "land_cover_class_id"
        read_pred = ds.read(1)
        np.testing.assert_array_equal(read_pred, pred)


def test_predict_raster_odd_and_small_size(synthetic_raster_odd_size: Path, mock_model):
    """Verify inference on odd-dimension (73x59) raster smaller than tile_size."""
    device = torch.device("cpu")
    pred, metadata = predict_raster(
        synthetic_raster_odd_size,
        mock_model,
        "unet",
        device,
        tile_size=128,
        stride=64,
    )

    assert pred.shape == (59, 73)
    assert metadata.width == 73
    assert metadata.height == 59


def test_predict_raster_parameter_validation(synthetic_raster_32748: Path, mock_model):
    """Invalid tile_size or stride parameters must raise informative ValueErrors."""
    device = torch.device("cpu")

    with pytest.raises(ValueError, match="tile_size must be at least 2"):
        predict_raster(
            synthetic_raster_32748, mock_model, "segformer", device, tile_size=1
        )

    with pytest.raises(ValueError, match="stride must be between 1 and tile_size"):
        predict_raster(
            synthetic_raster_32748,
            mock_model,
            "segformer",
            device,
            tile_size=64,
            stride=0,
        )

    with pytest.raises(ValueError, match="stride must be between 1 and tile_size"):
        predict_raster(
            synthetic_raster_32748,
            mock_model,
            "segformer",
            device,
            tile_size=64,
            stride=65,
        )

    with pytest.raises(ValueError, match="input_scale must be greater than zero"):
        predict_raster(
            synthetic_raster_32748,
            mock_model,
            "segformer",
            device,
            tile_size=64,
            stride=32,
            input_scale=-1.0,
        )
