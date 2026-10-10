# Demo Commands and Advanced Usage

Start with the [Docker Quickstart guide](../README.md#docker-quickstart-recommended). All commands below run entirely inside pre-configured Docker containers—no manual host Python, CUDA, or GDAL environment setup required.

For convenience, cross-platform helper scripts are provided in the repo root:
- **Linux / macOS**: `./run.sh [args]`
- **Windows PowerShell**: `.\run.ps1 [args]`
- **Standard Docker Compose**: `docker compose run --rm cpu [args]`

---

## 1. Run U-Net Demo

```bash
# Linux / macOS
./run.sh --model unet --demo

# Windows PowerShell
.\run.ps1 -Model unet -Demo

# Docker Compose directly
docker compose run --rm cpu --model unet --demo
```

## 2. Run DeepLabV3+ Demo

```bash
# Linux / macOS
./run.sh --model deeplabv3plus --demo

# Windows PowerShell
.\run.ps1 -Model deeplabv3plus -Demo

# Docker Compose directly
docker compose run --rm cpu --model deeplabv3plus --demo
```

## 3. Run SegFormer Demo

```bash
# Linux / macOS
./run.sh --model segformer --demo

# Windows PowerShell
.\run.ps1 -Model segformer -Demo

# Docker Compose directly
docker compose run --rm cpu --model segformer --demo
```

## 4. Run All Models in Sequence

```bash
# Linux / macOS
./run.sh --model all --demo --output-dir /outputs/demo_comparison

# Windows PowerShell
.\run.ps1 -Model all -Demo -OutputDir outputs/demo_comparison

# Docker Compose directly
docker compose run --rm cpu --model all --demo --output-dir /outputs/demo_comparison
```

The models run one after another. Each model receives a separate output folder inside `outputs/`.

---

## 5. Run Without a Reference Layer

Reference vector data is optional. To run SegFormer on imagery without computing evaluation metrics:

```bash
# Linux / macOS
./run.sh --model segformer --demo --skip-evaluation

# Windows PowerShell
.\run.ps1 -Model segformer -Demo -SkipEvaluation
```

For a custom image, mount the file into `/data` or specify its path inside the container:

```bash
# Linux / macOS
docker compose run --rm -v "/host/path/to/raster.tif:/data/custom.tif:ro" cpu \
  --model segformer --raster /data/custom.tif --skip-evaluation
```

Imagery-only runs create `prediction.tif`, the selected vector output, and `prediction.png`. Metric files, a confusion matrix, and `evaluation.png` are created only when a reference vector is supplied.

---

## 6. Select CPU or GPU (CUDA) Execution

Automatic device selection is handled by the container targets:

### CPU Execution (Default)
```bash
./run.sh --model segformer --demo --device cpu
```

### GPU Execution (NVIDIA CUDA)
Requires NVIDIA Container Toolkit on Linux or WSL2 on Windows:

```bash
# Linux / macOS
./run.sh --gpu --model segformer --demo --device cuda

# Windows PowerShell
.\run.ps1 -Gpu -Model segformer -Demo -Device cuda

# Docker Compose directly
docker compose run --rm gpu --model segformer --demo --device cuda
```

---

## 7. Vector Output Formats

Generate ESRI Shapefile instead of GeoPackage:

```bash
./run.sh --model segformer --demo --vector-format shp
```

Shapefile output consists of several files. Keep `.shp`, `.shx`, `.dbf`, and `.prj` sidecars together.

Skip vector polygonization entirely (raster only):

```bash
./run.sh --model segformer --demo --vector-format none
```

---

## 8. Change the Output Folder

```bash
./run.sh --model segformer --demo --output-dir /outputs/demo_run_02
```

Use a new folder name to preserve earlier outputs.

---

## 9. Advanced Settings Reference

The default settings match the refined research notebooks.

| Option | Default | Meaning |
|---|---:|---|
| `--tile-size` | 512 | Tile width and height in pixels |
| `--stride` | 256 | Distance between neighboring tiles |
| `--input-scale` | 255 | Divisor before ImageNet normalization |
| `--vector-format` | `gpkg` | Polygon output format (`gpkg`, `shp`, `none`) |
| `--all-touched` | Off | Rasterize every pixel touched by a reference polygon |
| `--exclude-unclassified-vector` | Off | Remove class `0` from polygon output |
| `--skip-evaluation` | Off | Ignore reference and create predictions only |

Changing tile size, stride, input scaling, or reference rasterization can change model predictions and reported metrics. Use identical settings for model comparison.

---

## 10. Run a Custom Raster

Custom data is executed by mounting custom host directories to `/data` in the container:

```bash
docker run --rm \
  -v "/path/to/imagery.tif:/data/input.tif:ro" \
  -v "$PWD/weights:/models:ro" \
  -v "$PWD/outputs:/outputs" \
  vhr-seg:dev-cpu \
  --model segformer \
  --raster /data/input.tif \
  --weights-dir /models \
  --output-dir /outputs \
  --skip-evaluation
```

Imagery with reference evaluation:

```bash
docker run --rm \
  -v "/path/to/imagery.tif:/data/input.tif:ro" \
  -v "/path/to/reference.gpkg:/data/reference.gpkg:ro" \
  -v "$PWD/weights:/models:ro" \
  -v "$PWD/outputs:/outputs" \
  vhr-seg:dev-cpu \
  --model segformer \
  --raster /data/input.tif \
  --reference /data/reference.gpkg \
  --reference-field class_id \
  --weights-dir /models \
  --output-dir /outputs
```

Requirements:
1. Georeferenced raster with RGB in bands 1–3.
2. Pixel values compatible with the training preprocessing (0–255).
3. Valid raster and vector coordinate reference systems (CRS).
4. For evaluation: reference class IDs from `0` to `4`.
5. For evaluation: spatial overlap between imagery and reference polygons.

---

## 11. Reproducibility & Provenance Record

Every run automatically produces a tamper-evident `run_manifest.json` inside the output folder, recording:
- Exact CLI invocation flags and runtime arguments;
- SHA-256 checksums of input raster, reference vector, and model weights;
- Full hardware and library environment snapshot (OS, Python, PyTorch, GDAL, CUDA status);
- Output file paths and evaluation summary.

## Metric Scope

See [results.md](results.md) for the distinction between training validation, private-scene-imagery evaluation, and public demo evaluation.
