# Docker Execution Guide

This document describes how to build and run containerized batch inference for the VHR Satellite Segmentation Pipeline using Docker for both CPU and NVIDIA GPU environments.

---

## 1. Container Architecture

The containerized system enforces strict separation of concerns:
- **Inference Engine**: Encapsulated within the container image (OS, Python 3.11, PyTorch, GDAL/Rasterio, Transformers, SMP).
- **Satellite Rasters & Labels**: Mounted **read-only** (`:ro`) at `/data`.
- **Model Checkpoints**: Mounted **read-only** (`:ro`) at `/models`. Checkpoint binaries are not bundled into the image to maintain lightweight OCI artifacts.
- **Output Artifacts**: Mounted **read-write** at `/outputs`. All generated GeoTIFFs, GeoPackages, metrics, figures, and provenance manifests persist directly on the host.
- **Security**: The container runs under a non-root `appuser` (UID 1000).

---

## 2. Recommended: Running via Docker Compose or Helper Scripts

The easiest and officially recommended way to interact with the containerized pipeline is using `docker compose` or the cross-platform runner scripts.

### Using Cross-Platform Helper Scripts
The repository includes automated runner scripts that handle image building, volume mounting, and command routing:

- **Linux / macOS**: `./run.sh [options]`
- **Windows PowerShell**: `.\run.ps1 [options]`

#### Check System & Weight Manifest Health:
```bash
./run.sh --doctor        # Linux / macOS
.\run.ps1 -Doctor        # Windows PowerShell
```

#### Run SegFormer Demo:
```bash
./run.sh --model segformer --demo        # Linux / macOS
.\run.ps1 -Model segformer -Demo        # Windows PowerShell
```

#### Run All Geospatial Unit Tests:
```bash
./run.sh --test          # Linux / macOS
.\run.ps1 -Test          # Windows PowerShell
```

### Using Docker Compose Directly

```bash
# Run doctor check
docker compose run --rm doctor

# Run test suite
docker compose run --rm test

# Run CPU inference demo
docker compose run --rm cpu --model segformer --demo

# Run GPU inference demo (requires NVIDIA Container Toolkit)
docker compose run --rm gpu --model segformer --demo --device cuda
```

---

## 3. Building the Images Manually

If not using Docker Compose, you can build images directly:

### CPU Image
```bash
docker build -f docker/Dockerfile.cpu -t vhr-seg:dev-cpu .
```

### GPU Image (CUDA 12.1)
```bash
docker build -f docker/Dockerfile.gpu -t vhr-seg:dev-gpu .
```

---

## 4. Running Environment Doctor via Direct Docker Run

### Linux / macOS (Bash)
```bash
docker run --rm \
  -v "$PWD/weights:/models:ro" \
  vhr-seg:dev-cpu \
  --doctor --weights-dir /models
```

### Windows (PowerShell)
```powershell
docker run --rm `
  -v "${PWD}/weights:/models:ro" `
  vhr-seg:dev-cpu `
  --doctor --weights-dir /models
```

---

## 5. Running Inference via Direct Docker Run

### CPU Inference Example

#### Linux / macOS (Bash)
```bash
docker run --rm \
  -v "$PWD/demo:/data:ro" \
  -v "$PWD/weights:/models:ro" \
  -v "$PWD/outputs:/outputs" \
  vhr-seg:dev-cpu \
  --model segformer \
  --raster /data/imagery_demo.tif \
  --weights-dir /models \
  --output-dir /outputs \
  --device cpu \
  --skip-evaluation
```

#### Windows (PowerShell)
```powershell
docker run --rm `
  -v "${PWD}/demo:/data:ro" `
  -v "${PWD}/weights:/models:ro" `
  -v "${PWD}/outputs:/outputs" `
  vhr-seg:dev-cpu `
  --model segformer `
  --raster /data/imagery_demo.tif `
  --weights-dir /models `
  --output-dir /outputs `
  --device cpu `
  --skip-evaluation
```

### GPU Inference Example (Requires NVIDIA Container Toolkit / WSL2)

#### Linux (Bash)
```bash
docker run --rm --gpus all \
  -v "$PWD/demo:/data:ro" \
  -v "$PWD/weights:/models:ro" \
  -v "$PWD/outputs:/outputs" \
  vhr-seg:dev-gpu \
  --model segformer \
  --raster /data/imagery_demo.tif \
  --weights-dir /models \
  --output-dir /outputs \
  --device cuda \
  --skip-evaluation
```

#### Windows PowerShell (WSL2 with NVIDIA Driver)
```powershell
docker run --rm --gpus all `
  -v "${PWD}/demo:/data:ro" `
  -v "${PWD}/weights:/models:ro" `
  -v "${PWD}/outputs:/outputs" `
  vhr-seg:dev-gpu `
  --model segformer `
  --raster /data/imagery_demo.tif `
  --weights-dir /models `
  --output-dir /outputs `
  --device cuda `
  --skip-evaluation
```

---

## 6. With Reference Evaluation

To compute accuracy, IoU, confusion matrix, and compatibility map:

```bash
docker run --rm \
  -v "$PWD/demo:/data:ro" \
  -v "$PWD/weights:/models:ro" \
  -v "$PWD/outputs:/outputs" \
  vhr-seg:dev-cpu \
  --model segformer \
  --raster /data/imagery_demo.tif \
  --reference /data/reference_demo.gpkg \
  --weights-dir /models \
  --output-dir /outputs \
  --device cpu
```

The resulting files in `outputs/segformer/` include:
- `prediction.tif` (Georeferenced GeoTIFF)
- `prediction.gpkg` (Polygonized GeoPackage)
- `metrics.json` & `metrics_per_class.csv`
- `confusion_matrix.csv`
- `evaluation.png`
- `run_manifest.json` (Provenance and reproducibility record)
