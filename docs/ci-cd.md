# Continuous Integration & Delivery Architecture

This document outlines the CI/CD pipeline, security boundaries, and recommended GitHub branch protection rules for the VHR Satellite Segmentation Pipeline.

---

## 1. Overview and CD Boundary

- **Continuous Integration (CI)**: Automatically tests every pull request and push to `main` for code hygiene, spatial correctness regressions, and Docker container buildability.
- **Continuous Delivery (CD)**: Builds and publishes versioned, attested OCI container images (CPU and GPU) to GitHub Packages (GHCR) upon signed and approved release tags (`v*.*.*`).
- **Deployment Scope**: The repository produces batch-inference CLI tools and container images. Automatic rollout to production servers (Continuous Deployment) is intentionally **out of scope**.

---

## 2. CI Quality Gates (`.github/workflows/ci.yml`)

The CI workflow triggers on:
- `pull_request` targeting `main`
- `push` to `main`
- Manual invocation via `workflow_dispatch`

### Jobs:

1. **`lint` (Lint & Format Check)**:
   - Python 3.11 runner.
   - Enforces Ruff style guidelines (`ruff check`) and code formatting (`ruff format --check`).

2. **`pytest-geospatial` (Geospatial Pytest Matrix)**:
   - Matrix testing across supported Python runtimes: **3.10** and **3.11**.
   - Executes offline geospatial assertions on synthetic fixtures (GeoTIFF metadata, CRS reprojection, confusion matrix, metric calculations, vectorization roundtrips).
   - Zero external network dependencies, no GPU required, runs deterministically under 2 minutes.

3. **`docker-cpu-smoke` (Docker Build & Container Smoke Test)**:
   - Builds `docker/Dockerfile.cpu`.
   - Executes container `--help` and environment `--doctor` diagnostics inside the isolated Linux environment.
   - Verifies GDAL/Rasterio C-libraries and PyTorch CPU wheel compatibility.

---

## 3. Release Delivery Workflow (`.github/workflows/release-image.yml`)

Triggered exclusively by approved semver tag pushes (`v*.*.*`):
1. **Re-validation Gate**: Reruns linting and test suite.
2. **Build & Attestation**:
   - Builds both CPU (`vX.Y.Z-cpu`) and GPU (`vX.Y.Z-gpu`) OCI images via Docker Buildx.
   - Attaches OCI standard metadata labels (`org.opencontainers.image.source`, `revision`, `version`).
   - Publishes to `ghcr.io/anovvy/vhr-satellite-segmentation-pipeline`.
   - Generates immutable SHA-256 image digests.

---

## 4. Security Model & Least Privilege

- **Untrusted Pull Requests**:
  - Run strictly under top-level `permissions: { contents: read }`.
  - Never receive GHCR write permissions (`packages: write`) or access to repository secrets.
  - Dangerous triggers like `pull_request_target` are strictly avoided.
- **Publishing Permissions**:
  - `packages: write` is isolated strictly to release publishing jobs on protected tags.

---

## 5. Recommended GitHub Branch Protection Settings

Repository owners should configure branch protection on `main`:
1. Navigate to **Settings > Branches > Branch protection rules** (or Rulesets).
2. For branch pattern `main`:
   - [x] **Require a pull request before merging**
   - [x] **Require approvals** (minimum 1 review)
   - [x] **Dismiss stale pull request approvals when new commits are pushed**
   - [x] **Require status checks to pass before merging**:
     - `Lint & Format Check`
     - `Geospatial Pytest (Python 3.10)`
     - `Geospatial Pytest (Python 3.11)`
     - `Docker CPU Build & Smoke Test`
   - [x] **Require branches to be up to date before merging**
   - [x] **Do not allow bypassing the above settings**
