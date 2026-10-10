# Geospatial Reproducibility & Provenance Guide

This guide describes how to achieve deterministic, reproducible inference results and verify data provenance in the VHR Satellite Segmentation Pipeline.

---

## 1. Run Manifest (`run_manifest.json`)

Every execution of the inference pipeline automatically writes a `run_manifest.json` file inside the designated model output folder (e.g. `outputs/segformer/run_manifest.json`).

### Manifest Structure

```json
{
  "manifest_version": "1.0",
  "timestamp_utc": "2026-10-10T12:00:00+00:00",
  "git": {
    "commit": "4475cf2af80d4ce5d2b20ff9aac6c83b93be67fc",
    "dirty": false
  },
  "runtime": {
    "python_version": "3.11.8",
    "platform": "Linux-6.1.0-x86_64",
    "device": "cpu",
    "image_digest": "sha256:abc123..."
  },
  "dependencies": {
    "torch": "2.2.2",
    "rasterio": "1.3.9",
    "geopandas": "0.14.3",
    "transformers": "4.39.2"
  },
  "model": {
    "name": "segformer",
    "weights_filename": "segformer_mit_b0.pth",
    "weights_sha256": "b313598c922fc8f53c1a68b1a96dc9ede567488c38b13767191eeeefe71ffea0"
  },
  "input_raster": {
    "filename": "imagery_demo.tif",
    "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "width": 1024,
    "height": 1024,
    "crs": "EPSG:32748",
    "transform": [0.3, 0.0, 700000.0, 0.0, -0.3, 9300000.0]
  },
  "parameters": {
    "tile_size": 512,
    "stride": 256,
    "input_scale": 255.0,
    "vector_format": "gpkg",
    "exclude_unclassified_vector": false,
    "skip_evaluation": false
  },
  "outputs": {
    "raster": {
      "filename": "prediction.tif",
      "sha256": "...",
      "size_bytes": 1048576
    },
    "vector": {
      "filename": "prediction.gpkg",
      "sha256": "...",
      "size_bytes": 204800
    }
  }
}
```

### Privacy & Path Sanitization
- Local filesystem absolute paths (e.g., `/home/username/...` or `C:\Users\username\...`) are stripped from the manifest.
- Only relative basenames and cryptographic SHA-256 digests are recorded, preventing information leakage.

---

## 2. Verifying Model Checksums

Standard checkpoints are verified against `weights/checksums.sha256`.

To verify locally:
```bash
# Linux
sha256sum -c weights/checksums.sha256

# Windows PowerShell
Get-FileHash weights\*.pth -Algorithm SHA256
```

Or run the built-in diagnostic:
```bash
python scripts/run_demo.py --doctor
```

---

## 3. QGIS Visualization & Inspection

To inspect the generated outputs in QGIS:

1. **Add Prediction Raster**:
   - In QGIS, navigate to **Layer > Add Layer > Add Raster Layer...**
   - Select `outputs/segformer/prediction.tif`.
   - Open **Layer Properties > Symbology**:
     - Render type: **Paletted / Unique values**.
     - Class 0: `#000000` (Unclassified / Other)
     - Class 1: `#ffff00` (Residential)
     - Class 2: `#ff9f1a` (Non-residential)
     - Class 3: `#90ee90` (Grassland)
     - Class 4: `#006400` (Urban Forest)

2. **Add Prediction Vectors**:
   - In QGIS, navigate to **Layer > Add Layer > Add Vector Layer...**
   - Select `outputs/segformer/prediction.gpkg`.
   - Open **Layer Properties > Symbology > Categorized**:
     - Value: `class_id` or `category`.
     - Assign matching color styles.

3. **Overlay & Alignment**:
   - Check alignment against the original `demo/imagery_demo.tif`. The georeferenced coordinates and Affine transform match with zero spatial displacement.
