# CLI Execution & Parameter Reference

This guide covers advanced CLI parameters, custom raster processing, and batch model comparisons using the containerized execution environment.

---

## 1. Command Quick Reference

| Task | Linux / macOS (`run.sh`) | Windows PowerShell (`run.ps1`) |
|---|---|---|
| **SegFormer Demo** | `./run.sh --model segformer --demo` | `.\run.ps1 -Model segformer -Demo` |
| **U-Net Demo** | `./run.sh --model unet --demo` | `.\run.ps1 -Model unet -Demo` |
| **DeepLabV3+ Demo** | `./run.sh --model deeplabv3plus --demo` | `.\run.ps1 -Model deeplabv3plus -Demo` |
| **All Models Comparison** | `./run.sh --model all --demo` | `.\run.ps1 -Model all -Demo` |
| **Prediction Only (No Ref)**| `./run.sh --model segformer --demo --skip-evaluation` | `.\run.ps1 -Model segformer -Demo -SkipEvaluation` |
| **GPU Acceleration** | `./run.sh --gpu --model segformer --demo` | `.\run.ps1 -Gpu -Model segformer -Demo` |
| **Shapefile Vectors** | `./run.sh --model segformer --demo --vector-format shp` | `.\run.ps1 -Model segformer -Demo -VectorFormat shp` |
| **Raster Only (No Vectors)**| `./run.sh --model segformer --demo --vector-format none`| `.\run.ps1 -Model segformer -Demo -VectorFormat none` |
| **System Diagnostics** | `./run.sh --doctor` | `.\run.ps1 -Doctor` |
| **Run Pytest Suite** | `./run.sh --test` | `.\run.ps1 -Test` |

*Equivalent standard docker-compose command: `docker compose run --rm cpu <args>` (or `gpu`).*

---

## 2. Advanced Parameter Reference

| Argument | Type | Default | Description |
|---|---|---:|---|
| `--model` | string | `segformer` | Model architecture: `segformer`, `unet`, `deeplabv3plus`, or `all`. |
| `--tile-size` | int | `512` | Window tile width and height in pixels. |
| `--stride` | int | `256` | Step distance between neighboring windows (50% overlap). |
| `--input-scale` | float | `255.0` | Radiometric scaling factor applied before ImageNet normalization. |
| `--vector-format` | string | `gpkg` | Vector polygon format: `gpkg` (GeoPackage), `shp` (Shapefile), or `none`. |
| `--all-touched` | flag | `false` | When true, rasterizes every pixel touched by reference polygons. |
| `--exclude-unclassified-vector`| flag | `false`| Excludes Class 0 (Unclassified) from output vector polygon layers. |
| `--skip-evaluation` | flag | `false` | Skips reference rasterization and metrics; generates `prediction.png`. |
| `--weights-dir` | path | `/models` | Directory holding standardized `.pth` model checkpoints. |
| `--output-dir` | path | `/outputs` | Destination folder for prediction raster, vector, and metrics. |
| `--device` | string | `auto` | Execution device: `auto`, `cpu`, or `cuda`. |

---

## 3. Running Custom Satellite Rasters

To segment custom VHR imagery outside the demo dataset, mount your host data folder to `/data` in the container.

### Imagery-Only (Prediction Mode)
```bash
docker run --rm \
  -v "/path/to/my_data:/data:ro" \
  -v "$PWD/weights:/models:ro" \
  -v "$PWD/outputs:/outputs" \
  vhr-seg:dev-cpu \
  --model segformer \
  --raster /data/custom_vhr.tif \
  --weights-dir /models \
  --output-dir /outputs \
  --skip-evaluation
```

### With Reference Vector Evaluation
```bash
docker run --rm \
  -v "/path/to/my_data:/data:ro" \
  -v "$PWD/weights:/models:ro" \
  -v "$PWD/outputs:/outputs" \
  vhr-seg:dev-cpu \
  --model segformer \
  --raster /data/custom_vhr.tif \
  --reference /data/ground_truth.gpkg \
  --reference-field class_id \
  --weights-dir /models \
  --output-dir /outputs
```

### Input Requirements:
1. **Raster Bands**: Bands 1, 2, and 3 must represent Red, Green, and Blue.
2. **Spatial CRS**: Raster and vector layers must contain valid Coordinate Reference Systems.
3. **Radiometry**: Pixel values should be standard 8-bit RGB (0–255) for ImageNet normalization.
4. **Resolution**: Trained on 30 cm VHR imagery. Substantial resolution deviations (e.g. 10 m Sentinel-2) require retraining.
5. **Reference Classes**: Reference polygon attributes must match class IDs `0–4` or supported English/Indonesian taxonomy names.
