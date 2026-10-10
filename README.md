# VHR Satellite Segmentation Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Inference-Docker%20Container-blue.svg)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.11-green.svg)](https://www.python.org/)

Production-ready, fully containerized deep-learning pipeline for urban land-cover segmentation from 30 cm very-high-resolution (VHR) RGB satellite imagery using **U-Net**, **DeepLabV3+**, and **SegFormer**.

<img width="639" height="360" alt="VHR Pipeline Animation" src="https://github.com/user-attachments/assets/5be58630-8c42-4218-b8da-5e2d6a586401" />

---

## Overview & Architecture

The pipeline processes large satellite rasters via sliding-window tile inference (`512 × 512` px, `256` px stride, 50% overlap, reflect padding, and Gaussian/mean probability blending), outputting georeferenced GeoTIFFs, polygonized GeoPackages, confusion matrices, and cryptographic provenance manifests.

### Supported Models

| Architecture | Backbone | Parameter Weights | Inference Target |
|---|---|---|---|
| **SegFormer** | MiT-B0 | `weights/segformer_mit_b0.pth` | Best overall accuracy & spatial boundary fidelity |
| **U-Net** | ResNet34 | `weights/unet_resnet34.pth` | Lightweight baseline for fast urban segmentation |
| **DeepLabV3+** | ResNet50 | `weights/deeplabv3plus_resnet50.pth` | Atrous spatial pyramid pooling for context capture |

### Class Taxonomy

| ID | Class Name | Description |
|---:|---|---|
| `0` | **Unclassified / Other** | Semantic catch-all (water bodies, bare soil, plantations, non-target features) |
| `1` | **Residential / Mixed-use** | Housing settlements, residential mixed-use buildings |
| `2` | **Non-residential Built-up** | Commercial, industrial, educational, and governmental facilities |
| `3` | **Grassland / Yard** | Open lawns, low herbaceous vegetation, sports grounds |
| `4` | **Urban Forest / Park** | Dense tree canopy, protected green belts, city parklands |

> [!NOTE]
> Models were trained on dense urban landscapes in Jakarta. Testing on local validation chips is recommended prior to large-scale deployment on visually or geographically distinct landscapes.

---

## Repository Structure

```text
vhr-satellite-segmentation-pipeline/
├── docker-compose.yml       # Multi-service container orchestration (cpu, gpu, test, doctor)
├── run.sh / run.ps1         # Cross-platform runner scripts (Linux/macOS & Windows PowerShell)
├── pyproject.toml           # Linter (Ruff) and test engine (Pytest) configurations
│
├── docker/                  # Production Dockerfiles (Dockerfile.cpu & Dockerfile.gpu)
├── scripts/                 # CLI entrypoint (run_demo.py) and environment doctor (doctor.py)
├── src/                     # Core inference, vectorization, evaluation, models & provenance
├── tests/                   # Geospatial pytest test suite (29 tests)
│
├── demo/                    # Redistributable sample raster & reference ground-truth
├── weights/                 # Checkpoint directory with SHA-256 integrity manifest
├── results/                 # Original experiment benchmark metrics & training figures
├── docs/                    # Technical architecture, methodology, and operations guides
└── outputs/                 # Default persistence destination for generated artifacts
```

---

## Quickstart Guide (Docker-Only)

This repository is **100% containerized**. No manual installation of Python, GDAL, CUDA, or virtual environments (`.venv`) is required on the host system.

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows / macOS) or Docker Engine with Docker Compose (Linux).
- *(Optional for GPU)*: NVIDIA GPU with NVIDIA Container Toolkit / WSL2 GPU acceleration.

<details>
<summary><b>Docker Desktop Setup Guide for Windows (Click to expand)</b></summary>

Setting up Docker Desktop on Windows for the first time? Follow these quick steps to get started smoothly:

1. **Verify Virtualization:** Ensure Hardware Virtualization (VT-x / AMD-V) is enabled in your BIOS/UEFI. Check this in Windows Task Manager > **Performance** > **CPU** > **Virtualization: Enabled**.
2. **Install WSL 2 (Windows Subsystem for Linux):**
   - Open **PowerShell** as Administrator.
   - Run:
     ```powershell
     wsl --install
     ```
   - If WSL is already installed, update it to the latest version:
     ```powershell
     wsl --update
     ```
   - Restart your computer if prompted.
   - *Official Reference:* [Microsoft WSL Installation Guide](https://learn.microsoft.com/en-us/windows/wsl/install).
3. **Download & Install Docker Desktop:**
   - Download the installer from the [Official Docker Desktop for Windows Guide](https://docs.docker.com/desktop/setup/install/windows-install/).
   - Run `Docker Desktop Installer.exe` and make sure the option **"Use WSL 2 instead of Hyper-V"** is checked.
4. **Skip Account Sign-In / Survey (Optional):**
   - When Docker Desktop launches and prompts you to sign in or create an account, **you do NOT need to create an account** for personal/local usage.
   - Simply click **"Continue without signing in"** (or "Skip"), then skip the survey.
5. **Keep Docker Desktop Running:**
   - Docker Desktop must be running (showing a green whale icon in the Windows taskbar/system tray) whenever you execute AI inference in QGIS.
   - *Tip:* In Docker Desktop, go to **Settings (Gear Icon) > General** and check **"Start Docker Desktop when you log in"** so it automatically starts in the background without needing manual launch every time.

</details>

<details>
<summary><b>Docker Desktop Setup Guide for macOS (Click to expand)</b></summary>

Follow these steps to set up Docker Desktop on macOS:

1. **Identify Your Mac Processor:**
   - Click the Apple logo () in the top-left corner > **About This Mac**.
   - Check whether your Mac has an **Apple Chip** (M1, M2, M3, M4) or an **Intel Processor**.
2. **Download & Install Docker Desktop:**
   - Visit the [Official Docker Desktop for Mac Guide](https://docs.docker.com/desktop/setup/install/mac-install/).
   - Click **"Mac with Apple chip"** or **"Mac with Intel chip"** according to your hardware.
   - Double-click the downloaded `Docker.dmg`, then drag and drop the **Docker** icon into your **Applications** folder.
   - Open **Docker** from Applications (or Spotlight `Cmd + Space`).
   - When macOS prompts for system permissions, authorize with your Mac password or Touch ID to allow Docker to install its helper tools.
3. **Skip Account Sign-In / Survey (Optional):**
   - Click **"Continue without signing in"** (or "Skip") and dismiss the survey. A Docker account is not needed for local usage.
4. **Enable Rosetta 2 (Recommended for Apple Silicon):**
   - To ensure compatibility with x86/amd64 containers, install Apple's Rosetta 2 via Terminal:
     ```bash
     softwareupdate --install-rosetta
     ```
   - In Docker Desktop, go to **Settings (Gear Icon) > General**, and verify that **"Use Rosetta for x86/amd64 emulation on Apple Silicon"** is enabled.
5. **Keep Docker Desktop Running:**
   - Docker must be active (look for the whale icon in the top macOS menu bar) before running PalmAI tasks.
   - *Tip:* In Docker Desktop **Settings > General**, check **"Start Docker Desktop when you log in"**.
6. **💡 Tip for Apple Silicon (M1/M2/M3/M4) - Apple Metal (MPS) Acceleration:**
   - Docker on macOS runs Linux containers in CPU mode.
   - If you want native **Apple Silicon GPU acceleration (Metal Performance Shaders / MPS)** for much faster inference, you can use PalmAI's built-in **Virtualenv backend** instead of Docker! See [Step 2: Configure Inference Backend](#step-2-configure-inference-backend).

</details>

### 1. Download Checkpoint Weights
Place the required model checkpoint(s) in `weights/` (e.g. from GitHub Releases):
- `weights/segformer_mit_b0.pth`

### 2. Verify System & Checkpoint Health
Run the built-in diagnostic doctor to verify container runtime, GDAL raster drivers, and checkpoint integrity:

```bash
# Windows PowerShell
.\run.ps1 -Doctor

# Linux / macOS
./run.sh --doctor

# Standard Docker Compose
docker compose run --rm doctor
```

### 3. Run Batch Inference Demo

```bash
# Run SegFormer demo with spatial reference evaluation (CPU)
.\run.ps1 -Model segformer -Demo          # Windows PowerShell
./run.sh --model segformer --demo         # Linux / macOS

# Run with NVIDIA GPU acceleration (CUDA)
.\run.ps1 -Gpu -Model segformer -Demo     # Windows PowerShell
./run.sh --gpu --model segformer --demo    # Linux / macOS
```

### 4. Run Test Suite
Validate geospatial geometry handling, CRS alignment, and model construction:

```bash
.\run.ps1 -Test                           # Windows PowerShell
./run.sh --test                           # Linux / macOS
```

---

## Output Artifacts

Every completed run writes standardized outputs to `outputs/<model_name>/` on the host:

| Artifact | File | Format | Description |
|---|---|---|---|
| **Classified Raster** | `prediction.tif` | GeoTIFF | Georeferenced single-band raster (Class IDs `0–4`) matching source CRS |
| **Polygon Vectors** | `prediction.gpkg` | GeoPackage | Polygonized boundaries with `class_id` and category attributes |
| **Evaluation Metrics** | `metrics.json` | JSON | Overall Accuracy, Mean IoU, Macro F1, Precision, and Recall |
| **Per-Class Report** | `metrics_per_class.csv`| CSV | Tabular metrics breakdown across classes `0–4` |
| **Error Matrix** | `confusion_matrix.csv` | CSV | Pixel-level cross-tabulation of reference vs predicted classes |
| **Diagnostic Plot** | `evaluation.png` | PNG | Visual comparison: Ground Truth, Prediction, and Compatibility Map |
| **Audit Provenance** | `run_manifest.json` | JSON | Cryptographic audit record (SHA-256 hashes, CLI flags, library versions) |

*Note: For imagery-only runs (`--skip-evaluation`), evaluation figures and CSV reports are omitted, while `prediction.png` is generated.*

---

## Benchmark Summary

Evaluated on the labeled Jakarta research scene (four target classes protocol):

| Model Architecture | Overall Accuracy | Mean IoU | Macro F1 | Key Characteristic |
|---|---:|---:|---:|---|
| **SegFormer (MiT-B0)** | **71.20%** | **56.03%** | **71.08%** | Superior boundary retention & highest non-residential IoU |
| **U-Net (ResNet34)** | 65.65% | 50.57% | 66.08% | Fast, sharp structural predictions on residential clusters |
| **DeepLabV3+ (ResNet50)** | 60.00% | 42.45% | 58.34% | Broad contextual aggregation across contiguous land parcels |

*Detailed per-class curves, loss trajectories, and visual error masks are documented in [docs/results.md](docs/results.md).*

---

## Technical Documentation

Detailed guides are modularly organized in [`docs/`](docs/):

- 📖 **[Execution & CLI Guide](docs/usage.md)**: Full CLI arguments reference, custom raster ingestion, and batch model comparisons.
- 🐳 **[Docker Architecture](docs/docker.md)**: Volume mount mappings, UID 1000 security profile, and GPU setup.
- 📐 **[Methodology & Theory](docs/methodology.md)**: Sliding-window probability blending, loss functions, and spatial metrics formulas.
- 📊 **[Research Results & Figures](docs/results.md)**: Training curves, loss progression, per-class evaluations, and benchmark records.
- 🛡️ **[Reproducibility & Provenance](docs/reproducibility.md)**: Run manifest specification, SHA-256 hashing, and QGIS styling guide.
- 🚀 **[CI/CD & Release Architecture](docs/ci-cd.md)**: GitHub Actions test pipelines, multi-python matrix, and automated image publishing.

---

## License

Source code is licensed under the [MIT License](LICENSE). Checkpoints and satellite assets remain subject to their respective distribution terms.
