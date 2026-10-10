"""End-to-end integration tests using synthetic rasters and reference vectors."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import rasterio
import torch

from src.evaluate import evaluate_prediction, rasterize_reference, save_metrics
from src.inference import predict_raster, save_prediction_raster
from src.provenance import generate_run_manifest, save_run_manifest
from src.vectorize import vectorize_prediction


def test_synthetic_end_to_end_pipeline(
    synthetic_raster_32748: Path,
    synthetic_reference_gpkg: Path,
    mock_model,
    tmp_path: Path,
):
    """Run full deterministic synthetic pipeline and verify all outputs and provenance."""
    out_dir = tmp_path / "e2e_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    # 1. Inference
    pred, metadata = predict_raster(
        synthetic_raster_32748,
        mock_model,
        "segformer",
        device,
        tile_size=64,
        stride=32,
    )
    assert pred.shape == (128, 128)

    # 2. Save prediction raster
    raster_file = save_prediction_raster(pred, metadata, out_dir / "prediction.tif")
    assert raster_file.is_file()
    with rasterio.open(raster_file) as ds:
        assert ds.crs == metadata.crs
        assert ds.width == 128
        assert ds.height == 128

    # 3. Vectorize prediction
    vector_file = out_dir / "prediction.gpkg"
    vectorize_prediction(pred, metadata, vector_file, include_unclassified=True)
    assert vector_file.is_file()
    gdf = gpd.read_file(vector_file)
    assert len(gdf) > 0
    assert gdf.crs == metadata.crs
    assert all(gdf.geometry.is_valid)

    # 4. Reference rasterization and evaluation
    gt = rasterize_reference(synthetic_reference_gpkg, metadata, class_field="class_id")
    assert gt.shape == (128, 128)

    metrics, compatibility, cm = evaluate_prediction(pred, gt)
    assert metrics["evaluated_pixels"] > 0
    saved_metrics = save_metrics(metrics, out_dir)
    assert saved_metrics["json"].is_file()
    assert saved_metrics["per_class_csv"].is_file()
    assert saved_metrics["confusion_matrix_csv"].is_file()

    # 5. Provenance manifest
    output_files = {
        "raster": raster_file,
        "vector": vector_file,
        "metrics_json": saved_metrics["json"],
    }
    params = {"tile_size": 64, "stride": 32, "input_scale": 255.0}
    fake_weights = tmp_path / "mock_weights.pth"
    fake_weights.write_bytes(b"mock weights bytes")

    manifest = generate_run_manifest(
        "segformer",
        fake_weights,
        synthetic_raster_32748,
        metadata,
        output_files,
        params,
        "cpu",
        tmp_path,
    )
    manifest_file = save_run_manifest(manifest, out_dir / "run_manifest.json")
    assert manifest_file.is_file()

    with manifest_file.open("r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    assert manifest_data["manifest_version"] == "1.0"
    assert manifest_data["model"]["name"] == "segformer"
    assert manifest_data["input_raster"]["crs"] == "EPSG:32748"
    assert "outputs" in manifest_data
    assert "raster" in manifest_data["outputs"]
    assert "sha256" in manifest_data["outputs"]["raster"]
