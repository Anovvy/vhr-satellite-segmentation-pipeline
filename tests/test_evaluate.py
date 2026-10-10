"""Tests for reference rasterization, confusion matrix, and evaluation metrics."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from affine import Affine
from rasterio.crs import CRS
from shapely.geometry import box

from src import REFERENCE_NODATA
from src.evaluate import (
    evaluate_prediction,
    rasterize_reference,
    save_metrics,
)
from src.inference import RasterMetadata


def test_evaluate_prediction_exact_metrics():
    """Verify exact hand-calculated confusion matrix, IoU, and F1 scores."""
    # Create a 3x3 array:
    # GT:
    # 0 1 2
    # 3 4 255 (255 is REFERENCE_NODATA, must be excluded)
    # 0 0 1
    # Pred:
    # 0 1 2
    # 3 0 4 (position with GT 255 is ignored)
    # 1 0 1
    gt = np.array(
        [
            [0, 1, 2],
            [3, 4, 255],
            [0, 0, 1],
        ],
        dtype=np.uint8,
    )
    pred = np.array(
        [
            [0, 1, 2],
            [3, 0, 4],
            [1, 0, 1],
        ],
        dtype=np.uint8,
    )

    metrics, compatibility, cm = evaluate_prediction(pred, gt)

    # Total evaluated pixels: 8 (excluding the 255 pixel)
    assert metrics["evaluated_pixels"] == 8

    # Ground truth valid pixels:
    # y_true: [0, 1, 2, 3, 4, 0, 0, 1]
    # y_pred: [0, 1, 2, 3, 0, 1, 0, 1]
    # Matches:
    # idx 0: 0 == 0 (match)
    # idx 1: 1 == 1 (match)
    # idx 2: 2 == 2 (match)
    # idx 3: 3 == 3 (match)
    # idx 4: 4 != 0 (mismatch)
    # idx 5: 0 != 1 (mismatch)
    # idx 6: 0 == 0 (match)
    # idx 7: 1 == 1 (match)
    # Matches = 6 out of 8 -> accuracy = 6/8 = 0.75
    assert metrics["overall_accuracy"] == pytest.approx(0.75)

    # Class 0:
    # true=3 (idx 0, 5, 6), pred=3 (idx 0, 4, 6)
    # TP = 2 (idx 0, 6)
    # FP = 1 (idx 4)
    # FN = 1 (idx 5)
    # precision = 2 / 3, recall = 2 / 3, IoU = 2 / (3 + 3 - 2) = 2/4 = 0.5
    class_0_metric = next(r for r in metrics["per_class"] if r["class_id"] == 0)
    assert class_0_metric["iou"] == pytest.approx(0.5)
    assert class_0_metric["precision"] == pytest.approx(2 / 3)
    assert class_0_metric["recall"] == pytest.approx(2 / 3)

    # Compatibility map values:
    # 0 for nodata, 1 for match, 2 for mismatch
    assert compatibility[1, 2] == 0  # was 255 in GT
    assert compatibility[0, 0] == 1  # match
    assert compatibility[1, 1] == 2  # mismatch (gt 4 vs pred 0)


def test_semantic_class_zero_is_not_treated_as_nodata():
    """Ensure semantic class 0 is fully evaluated and distinct from REFERENCE_NODATA (255)."""
    gt = np.array([[0, 0], [255, 255]], dtype=np.uint8)
    pred = np.array([[0, 1], [0, 1]], dtype=np.uint8)

    metrics, compatibility, _ = evaluate_prediction(pred, gt)
    assert metrics["evaluated_pixels"] == 2
    # Only the top 2 pixels were evaluated
    c0 = next(r for r in metrics["per_class"] if r["class_id"] == 0)
    assert c0["support_pixels"] == 2
    assert c0["recall"] == pytest.approx(0.5)  # 1 hit out of 2


def test_evaluate_unmatched_dimensions_raises():
    """Arrays of differing shapes must raise ValueError."""
    pred = np.zeros((10, 10), dtype=np.uint8)
    gt = np.zeros((10, 12), dtype=np.uint8)
    with pytest.raises(ValueError, match="equal shapes"):
        evaluate_prediction(pred, gt)


def test_evaluate_invalid_class_ids_raises():
    """Out-of-range class IDs (e.g. 5, 99) must raise ValueError."""
    pred = np.array([[0, 5]], dtype=np.uint8)
    gt = np.array([[0, 1]], dtype=np.uint8)
    with pytest.raises(ValueError, match="unsupported class IDs"):
        evaluate_prediction(pred, gt)

    pred_ok = np.array([[0, 1]], dtype=np.uint8)
    gt_bad = np.array([[0, 9]], dtype=np.uint8)
    with pytest.raises(ValueError, match="unsupported class IDs"):
        evaluate_prediction(pred_ok, gt_bad)


def test_evaluate_empty_labels_raises():
    """All pixels having REFERENCE_NODATA (255) must raise ValueError."""
    pred = np.zeros((5, 5), dtype=np.uint8)
    gt = np.full((5, 5), REFERENCE_NODATA, dtype=np.uint8)
    with pytest.raises(ValueError, match="no covered pixels"):
        evaluate_prediction(pred, gt)


def test_rasterize_reference_reprojection(tmp_path: Path):
    """Reference vector with different CRS should reproject to match raster metadata CRS."""
    meta = RasterMetadata(
        width=100,
        height=100,
        transform=Affine(0.3, 0.0, 700000.0, 0.0, -0.3, 9300000.0),
        crs=CRS.from_epsg(32748),
        profile={},
    )

    # Create polygon in EPSG:32748 then transform to EPSG:4326
    geom_utm = box(700005.0, 9299975.0, 700020.0, 9299990.0)
    gdf_utm = gpd.GeoDataFrame(
        [{"class_id": 2, "geometry": geom_utm}], crs="EPSG:32748"
    )
    gdf_4326 = gdf_utm.to_crs("EPSG:4326")

    ref_path = tmp_path / "ref_4326.gpkg"
    gdf_4326.to_file(ref_path, driver="GPKG")

    rasterized = rasterize_reference(ref_path, meta, class_field="class_id")
    assert rasterized.shape == (100, 100)
    # Must contain class 2 and REFERENCE_NODATA
    assert 2 in rasterized
    assert REFERENCE_NODATA in rasterized


def test_save_metrics_output(tmp_path: Path):
    """Verify save_metrics writes json, per-class csv, and confusion matrix csv."""
    pred = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    gt = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    metrics, _, _ = evaluate_prediction(pred, gt)

    out_dir = tmp_path / "metrics_out"
    saved = save_metrics(metrics, out_dir)

    assert saved["json"].is_file()
    assert saved["per_class_csv"].is_file()
    assert saved["confusion_matrix_csv"].is_file()

    with saved["json"].open() as f:
        data = json.load(f)
        assert data["overall_accuracy"] == 1.0
