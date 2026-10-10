# VHR Satellite Segmentation Pipeline

Deep-learning segmentation of selected urban land-cover classes from 30 cm very-high-resolution RGB satellite imagery using U-Net, DeepLabV3+, and SegFormer.

<img width="639" height="360" alt="VHR_edit-ezgif com-video-to-gif-converter" src="https://github.com/user-attachments/assets/5be58630-8c42-4218-b8da-5e2d6a586401" />

## Description

This project maps four target classes from an RGB satellite image:

1. residential / mixed-use built-up;
2. non-residential built-up;
3. grassland;
4. urban forest, green belt, and urban park.

Every remaining object is assigned to `Unclassified / Other`. Examples include water bodies, plantations, open land, and other objects outside the four target classes. Class `0` is therefore a valid semantic class, not a training background label.

## Intended Use and Geographic Limitation

The models were trained using dense residential landscapes in Jakarta. Building density, roof materials, street patterns, vegetation structure, illumination, season, sensor characteristics, and image resolution can differ substantially in another location.

Prediction quality may decrease in suburban, rural, agricultural, industrial, or geographically distant areas. A visually unfamiliar object can also be assigned to the closest learned class. **Testing on a small labeled sample is strongly recommended before large-area processing** outside dense Jakarta-like urban environments.

## Main Features

### 1. Three segmentation models

The same workflow supports U-Net, DeepLabV3+, and SegFormer.

### 2. Large-raster inference

The raster is processed with `512 × 512` tiles, a `256` pixel stride, 50% overlap, reflect padding at image edges, and probability averaging inside overlap areas.

### 3. Reference evaluation

Inference can run with imagery only. A reference vector is required only for accuracy, precision, recall, F1-score, IoU, and confusion-matrix calculation. Reference-covered pixels from all five classes, including `Unclassified / Other`, are included in evaluation.

### 4. GIS-ready output

The classification is saved as a georeferenced GeoTIFF and converted into a GeoPackage (.gpkg) polygon layer. Shapefile (.shp) output is also supported.

### 5. Visual results

Prediction-only mode creates a classified-map PNG. Evaluation mode creates ground-truth, prediction, and compatibility panels.

### 6. Containerized batch inference & environment doctor

Separate CPU and NVIDIA GPU Docker containers (`docker/Dockerfile.cpu`, `docker/Dockerfile.gpu`) with an offline-friendly system diagnostic (`--doctor`) for libraries, GDAL/CRS engines, compute devices, and checkpoint verification.

### 7. Geospatial validation & reproducibility provenance

Automated GitHub Actions CI testing spatial geometry, CRS alignment, and confusion-matrix values. Every inference run automatically generates an immutable `run_manifest.json` provenance record.

## Models

| Model | Backbone | Command name | Weight filename |
|---|---|---|---|
| U-Net | ResNet34 | `unet` | `unet_resnet34.pth` |
| DeepLabV3+ | ResNet50 | `deeplabv3plus` | `deeplabv3plus_resnet50.pth` |
| SegFormer | MiT-B0 | `segformer` | `segformer_mit_b0.pth` |

## Class Definitions

| Class ID | Class name | Definition and examples |
|---:|---|---|
| 0 | Unclassified / Other | Valid semantic catch-all for objects outside classes 1–4, including water bodies, plantations, open land, and other unsupported objects |
| 1 | Residential / Mixed-use Built-up | Built-up land used primarily for housing or residential activity, including mixed residential use |
| 2 | Non-residential Built-up | Built-up land not used as housing, such as hospitals, government offices, schools, and other institutional or service buildings |
| 3 | Grassland / Yard | Grass-covered land and similar low vegetation |
| 4 | Urban Forest / Green Belt / Urban Park | Tree-dominated urban green areas, green corridors, and city parks |

The model contains five output channels. Four thematic targets use IDs `1–4`, while class `0` collects all remaining land-cover objects.

## Repository Structure

```text
vhr-satellite-segmentation-pipeline/
├── README.md                          # Project overview
├── LICENSE                            
├── docker-compose.yml                 # Multi-service container orchestration
├── run.sh                             # Linux/macOS Docker runner script
├── run.ps1                            # Windows PowerShell Docker runner script
├── pyproject.toml                     # Ruff and pytest configuration
├── requirements.txt                   # Baseline dependencies
├── .gitignore                         
├── .dockerignore                      
│
├── .github/workflows/                 # GitHub Actions CI/CD
│   ├── ci.yml                         # Automated lint, geospatial pytest & Docker smoke
│   └── release-image.yml              # Gated GHCR image delivery for release tags
│
├── docker/                            # Production container runtimes
│   ├── Dockerfile.cpu                 # Optimized CPU inference container
│   └── Dockerfile.gpu                 # CUDA-accelerated GPU inference container
│
├── requirements/                      # Split dependency constraints
│   ├── requirements-cpu.txt           # CPU wheels with PyTorch CPU index
│   ├── requirements-gpu.txt           # GPU wheels with CUDA PyTorch index
│   └── requirements-dev.txt           # Development and testing tools
│
├── scripts/
│   ├── doctor.py                      # Environment and runtime diagnostic
│   └── run_demo.py                    # Main batch inference CLI
│
├── src/                               # Reusable inference pipeline
│   ├── __init__.py                    
│   ├── doctor.py                      # System health check engine
│   ├── evaluate.py                    # Calculate metrics evaluation
│   ├── inference.py                   # Tiling, padding, prediction, and GeoTIFF export
│   ├── models.py                      # Model construction and checkpoint loading
│   ├── provenance.py                  # Cryptographic run manifest generator
│   ├── vectorize.py                   # Classification raster to vector
│   └── visualize.py                   # Prediction and evaluation PNG figures
│
├── tests/                             # Geospatial test suite
│   ├── conftest.py                    # Synthetic GeoTIFF & mock fixtures
│   ├── test_cli.py                    # CLI parameter validation
│   ├── test_end_to_end.py             # End-to-end integration pipeline
│   ├── test_evaluate.py               # Confusion matrix & metrics
│   ├── test_inference.py              # Spatial padding & georeferencing
│   ├── test_models.py                 # Offline model building & loading
│   └── test_vectorize.py              # Vector geometry & roundtrips
│
├── demo/                              # Demonstration assets
│   ├── imagery_demo.tif               # Demo RGB satellite raster
│   ├── reference_demo.gpkg            # Demo evaluation reference
│   └── README.md                      
│
├── weights/                           # Locations for trained checkpoints
│   ├── checksums.sha256               # Cryptographic SHA-256 release manifest
│   ├── unet_resnet34.pth              # U-Net checkpoint
│   ├── deeplabv3plus_resnet50.pth     # DeepLabV3+ checkpoint
│   ├── segformer_mit_b0.pth           # SegFormer checkpoint
│   └── README.md                      
│
├── results/                           # Experiment reports
│   ├── benchmark.csv                  # Private-scene evaluation summary
│   └── figures/                       # Training and evaluation figures
│       ├── grafik_training_unet.png
│       ├── grafik_training_deeplab.png
│       ├── grafik_training_segformer.png
│       ├── unet_evaluation_research.png
│       ├── deeplabv3plus_evaluation_research.png
│       ├── segformer_evaluation_research.png
│       └── README.md
│
├── experiments/                       # Records of the experiments
│   ├── prepare_dataset.py             # Data Preparation
│   ├── train_model.py                 
│   ├── train_unet.py                  
│   ├── train_deeplabv3plus.py         
│   └── train_segformer.py             
│
├── docs/                              # Technical documentation
│   ├── ci-cd.md                       # CI/CD and security architecture
│   ├── docker.md                      # Docker container execution guide
│   ├── methodology.md                 # Methodology detail
│   ├── reproducibility.md             # Reproducibility & provenance guide
│   ├── results.md                     # Training and inference result interpretation
│   └── usage.md                       # Additional commands and input checks
│
└── outputs/                           # Generated automatically after demo inference
    └── <model_name>/                  
```
## Quickstart Guide (Docker-Only)

This project is fully containerized. **No manual installation of Python, virtual environments, GDAL, or CUDA is required on the host system.** Everything runs reproducibly inside Docker.

### Prerequisites

Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/macOS) or Docker Engine with Docker Compose (Linux). If using an NVIDIA GPU, enable the NVIDIA Container Toolkit / WSL2 GPU support.

### Step 1: Clone the repository

```bash
git clone https://github.com/Anovvy/vhr-satellite-segmentation-pipeline.git
cd vhr-satellite-segmentation-pipeline
```

### Step 2: Add the trained weights

Download the model files from the repository's GitHub Release and place the files inside `weights/`:
- `weights/segformer_mit_b0.pth` (SegFormer)
- `weights/unet_resnet34.pth` (U-Net)
- `weights/deeplabv3plus_resnet50.pth` (DeepLabV3+)

*(Only one weight file is needed for single-model inference; all three are needed for `--model all`.)*

### Step 3: Verify environment and checkpoints (Doctor)

Run the containerized diagnostic tool to verify the container runtime, GDAL raster engine, compute devices, and weights integrity:

```bash
# Docker Compose:
docker compose run --rm doctor

# Or using helper scripts:
.\run.ps1 --doctor    # Windows PowerShell
./run.sh --doctor     # Linux / macOS
```

### Step 4A: Run the demo with evaluation

Process `demo/imagery_demo.tif` and calculate evaluation metrics against `demo/reference_demo.gpkg`:

```bash
# CPU inference (Docker Compose):
docker compose run --rm cpu --model segformer --demo

# GPU inference (Docker Compose with NVIDIA GPU):
docker compose run --rm gpu --model segformer --demo

# Or using helper scripts:
.\run.ps1 --model segformer --demo          # Windows PowerShell (CPU)
.\run.ps1 -Gpu --model segformer --demo     # Windows PowerShell (GPU)
./run.sh --model segformer --demo           # Linux / macOS (CPU)
./run.sh --gpu --model segformer --demo      # Linux / macOS (GPU)
```

### Step 4B: Run the demo without evaluation (prediction only)

This mode uses `demo/imagery_demo.tif` and the model weights, but skips reference rasterization and metric calculations:

```bash
docker compose run --rm cpu --model segformer --demo --skip-evaluation
```

## Outputs

The default root output directory is `outputs/`. Each model receives a separate subfolder:

```text
outputs/
└── segformer/
```

### With reference evaluation

Command:

```bash
python scripts/run_demo.py --model segformer --demo
```

Default files:

| File | Content |
|---|---|
| `outputs/segformer/prediction.tif` | Georeferenced classification raster with class IDs `0–4` |
| `outputs/segformer/prediction.gpkg` | Raw classification polygons and class attributes | 
| `outputs/segformer/metrics.json` | Overall accuracy, mean IoU, macro scores, and per-class metrics | 
| `outputs/segformer/metrics_per_class.csv` | Precision, recall, F1-score, IoU, and support for classes `0–4` |
| `outputs/segformer/confusion_matrix.csv` | Reference class versus predicted class |
| `outputs/segformer/evaluation.png` | Ground truth, prediction, and green/red compatibility map |
| `outputs/segformer/run_manifest.json` | Provenance manifest with git commit, library versions, parameters, and cryptographic SHA-256 digests |

### Without reference evaluation

Command:

```bash
python scripts/run_demo.py --model segformer --demo --skip-evaluation
```

Default files:

| File | Content |
|---|---|
| `outputs/segformer/prediction.tif` | Georeferenced classification raster with class IDs `0–4` |
| `outputs/segformer/prediction.gpkg` | Raw classification polygons and class attributes | 
| `outputs/segformer/prediction.png` | Classified-map preview and legend |
| `outputs/segformer/run_manifest.json` | Provenance manifest with git commit, library versions, parameters, and cryptographic SHA-256 digests |

Metric files and `evaluation.png` are not created without a reference vector.

Existing files with the same names can be replaced during another run. Use `--output-dir` to preserve earlier results.

## Run Another Model

U-Net:

```bash
docker compose run --rm cpu --model unet --demo
```

DeepLabV3+:

```bash
docker compose run --rm cpu --model deeplabv3plus --demo
```

All models comparison:

```bash
docker compose run --rm cpu --model all --demo --output-dir /outputs/demo_comparison
```

All models without evaluation:

```bash
docker compose run --rm cpu --model all --demo --skip-evaluation --output-dir /outputs/demo_predictions
```

The all-model commands create separate `unet`, `deeplabv3plus`, and `segformer` folders inside the selected output directory. *(Windows PowerShell users can use `.\run.ps1 --model unet --demo`, and Linux users can use `./run.sh --model unet --demo`)*.

## Run Custom Imagery

Reference evaluation remains optional for imagery outside the bundled demo. To process custom imagery, bind-mount the host directory containing your GeoTIFF raster into `/data` inside the container:

### Imagery only

Linux / macOS:

```bash
docker compose run --rm \
  -v "/path/to/my_data:/data:ro" \
  cpu \
  --model segformer \
  --raster /data/image.tif \
  --output-dir /outputs
```

Windows PowerShell:

```powershell
docker compose run --rm `
  -v "C:\path\to\my_data:/data:ro" `
  cpu `
  --model segformer `
  --raster /data/image.tif `
  --output-dir /outputs
```

### Imagery with reference evaluation

```bash
docker compose run --rm \
  -v "/path/to/my_data:/data:ro" \
  cpu \
  --model segformer \
  --raster /data/image.tif \
  --reference /data/reference.gpkg \
  --reference-field class_id \
  --output-dir /outputs
```

Custom imagery should use RGB bands 1–3, compatible radiometry, and a spatial resolution close to the 30 cm training imagery. Prediction quality outside dense Jakarta-like residential landscapes requires independent checking.

## Automated Testing & CI/CD

Run the entire geospatial test suite inside the container with zero host dependencies:

```bash
# Run test suite inside Docker:
docker compose run --rm test

# Or using helper scripts:
.\run.ps1 --test      # Windows PowerShell
./run.sh --test       # Linux / macOS
```

- **CI Pipeline**: Evaluated on every pull request and push to `main` across Python 3.10 and 3.11 with container smoke tests. Details in [docs/ci-cd.md](docs/ci-cd.md).
- **Reproducibility & Provenance**: Cryptographic manifest structure and QGIS verification guide in [docs/reproducibility.md](docs/reproducibility.md).
- **Docker Architecture**: Complete containerization design and GPU setup details in [docs/docker.md](docs/docker.md).

## Original Training Reports

The following plots document the original training runs. Training validation and final inference evaluation use different data and different metric rules, so the scores must be read separately.

| Model | Recorded epochs | Final validation accuracy | Final validation mIoU |
|---|---:|---:|---:|
| U-Net | 22 | 75.8% | 51.6% |
| DeepLabV3+ | 23 | 76.0% | 51.0% |
| SegFormer | 46 | 76.1% | 53.8% |

### U-Net

![U-Net training curves](results/figures/grafik_training_unet.png)

### DeepLabV3+

![DeepLabV3+ training curves](results/figures/grafik_training_deeplab.png)

### SegFormer

![SegFormer training curves](results/figures/grafik_training_segformer.png)

## Original Evaluation Results

The refined evaluation notebooks report the following results on the same labeled Jakarta research scene:

| Model | Overall accuracy | Mean IoU |
|---|---:|---:|
| U-Net | 65.65% | 50.57% |
| DeepLabV3+ | 60.00% | 42.45% |
| SegFormer | **71.20%** | **56.03%** |

These historical private-scene values were produced by the notebooks using class IDs `1–4`. The current repository evaluation includes all reference-covered classes `0–4`, so **demo metrics must not be compared directly with this historical table**. SegFormer records the highest overall accuracy and mean IoU under the historical protocol.

The figures below were generated from non-redistributable research data. The public demo uses a different legally redistributable scene and produces separate demo metrics.

### U-Net evaluation

![U-Net evaluation on the private research scene](results/figures/unet_evaluation_research.png)

### DeepLabV3+ evaluation

![DeepLabV3+ evaluation on the private research scene](results/figures/deeplabv3plus_evaluation_research.png)

### SegFormer evaluation

![SegFormer evaluation on the private research scene](results/figures/segformer_evaluation_research.png)

Detailed per-class values are available in [docs/results.md](docs/results.md) and [results/benchmark.csv](results/benchmark.csv).

## Method Summary

```mermaid
flowchart TD
    A[RGB GeoTIFF] --> B[Overlapping tile inference]
    B --> C[Probability blending]
    C --> D[Classification outputs]
    D --> E[Prediction-only figure]
    D --> F[Optional reference evaluation]
    F --> G[Metrics and compatibility figure]
```

The full technical explanation is available in [docs/methodology.md](docs/methodology.md). Additional commands are available in [docs/usage.md](docs/usage.md).

## Scope and Limitations

- Training data represents dense residential landscapes in Jakarta. Large geographic or morphological differences can cause domain shift and misclassification.
- The original research used 30 cm Pleiades RGB imagery. A different resolution or radiometric profile changes the visual patterns received by the model.
- The taxonomy contains four target classes. Water, plantations, open land, roads, and unsupported objects are not mapped into separate thematic classes; these objects become `Unclassified / Other`.
- Original commercial imagery and training labels are not distributed.
- Training scripts preserve the experiment methodology; the main reproducible workflow starts from trained weights.

## License

Source code is released under the [MIT License](LICENSE). Satellite imagery, reference labels, evaluation figures, and trained weights can have separate usage or redistribution conditions.
