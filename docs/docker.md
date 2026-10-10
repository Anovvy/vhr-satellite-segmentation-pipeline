# Docker Container Architecture & Operations

This document details the containerized architecture, security model, volume mount specifications, and GPU acceleration setup for the VHR Satellite Segmentation Pipeline.

---

## 1. Container Isolation & Volume Mount Model

To guarantee reproducible, stateless execution without bundling heavy binary data into OCI images, the runtime strictly decouples code, models, and data through container volume mounts:

```text
[Host Filesystem]                             [Container Filesystem]
./demo/ (or custom data)  ==== (ro mount) ====>  /data/
./weights/                ==== (ro mount) ====>  /models/ (or /app/weights/)
./outputs/                ==== (rw mount) ====>  /outputs/ (or /app/outputs/)
```

| Mount Point | Access | Description |
|---|---|---|
| `/data` | Read-Only (`:ro`) | Input satellite rasters (GeoTIFF) and ground-truth vectors (GeoPackage/Shapefile). |
| `/models` | Read-Only (`:ro`) | Checkpoint binary weights (`.pth`). Images remain lightweight and vendor-agnostic. |
| `/outputs` | Read-Write (`:rw`) | Destination for generated GeoTIFFs, GeoPackages, metrics, figures, and provenance manifests. |

### Security Model
The container executes under a dedicated non-root user `appuser` (UID 1000, GID 1000) with least-privilege principles to prevent privilege escalation on the host system.

---

## 2. Docker Compose Services

The project defines 4 specialized services in [`docker-compose.yml`](../docker-compose.yml):

```yaml
services:
  cpu:     # Default CPU inference runtime (Python 3.11, PyTorch CPU, GDAL/Rasterio)
  gpu:     # CUDA 12.1 accelerated inference runtime (NVIDIA Container Toolkit required)
  doctor:  # Offline system & weights integrity diagnostic runner
  test:    # Automated geospatial pytest runner (29 tests)
```

### Quick Commands:
```bash
docker compose run --rm doctor                                   # Diagnostics
docker compose run --rm test                                     # Test suite
docker compose run --rm cpu --model segformer --demo             # CPU inference
docker compose run --rm gpu --model segformer --demo --device cuda # GPU inference
```

---

## 3. NVIDIA GPU Setup (CUDA 12.1)

To run GPU-accelerated inference:

### On Linux
1. Install NVIDIA drivers (`>= 525.60.13`).
2. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
3. Restart Docker daemon: `sudo systemctl restart docker`.
4. Run: `./run.sh --gpu --model segformer --demo --device cuda`.

### On Windows (WSL2)
1. Install the latest NVIDIA Windows Driver.
2. Ensure Docker Desktop has **Use the WSL 2 based engine** enabled under Settings > General.
3. Run: `.\run.ps1 -Gpu -Model segformer -Demo -Device cuda`.

---

## 4. Manual Image Building

If building outside Docker Compose:

```bash
# Build CPU Image
docker build -f docker/Dockerfile.cpu -t vhr-seg:dev-cpu .

# Build GPU Image (CUDA 12.1)
docker build -f docker/Dockerfile.gpu -t vhr-seg:dev-gpu .
```
