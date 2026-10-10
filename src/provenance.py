"""Reproducibility and provenance tracking for VHR segmentation runs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .inference import RasterMetadata


def get_git_info(repo_root: Path) -> dict[str, Any]:
    """Retrieve git commit SHA and status without crashing outside a git repo."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return {"commit": commit, "dirty": bool(status)}
    except Exception:
        return {"commit": "unknown", "dirty": False}


def file_sha256(path: Path | str) -> str:
    """Compute SHA-256 hash of a file."""
    p = Path(path)
    if not p.is_file():
        return "missing"
    hasher = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def generate_run_manifest(
    model_name: str,
    weights_path: Path,
    raster_path: Path,
    metadata: RasterMetadata,
    output_files: dict[str, Path],
    parameters: dict[str, Any],
    device: str,
    repo_root: Path,
) -> dict[str, Any]:
    """Generate a provenance manifest dictionary with sanitized paths."""
    git_info = get_git_info(repo_root)

    lib_versions = {}
    for pkg in [
        "torch",
        "torchvision",
        "transformers",
        "segmentation_models_pytorch",
        "rasterio",
        "geopandas",
        "shapely",
        "numpy",
    ]:
        try:
            mod = __import__(pkg)
            lib_versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass

    output_hashes = {}
    for key, out_file in output_files.items():
        if out_file.is_file():
            output_hashes[key] = {
                "filename": out_file.name,
                "sha256": file_sha256(out_file),
                "size_bytes": out_file.stat().st_size,
            }

    transform_list = (
        [
            metadata.transform.a,
            metadata.transform.b,
            metadata.transform.c,
            metadata.transform.d,
            metadata.transform.e,
            metadata.transform.f,
        ]
        if metadata.transform is not None
        else None
    )

    manifest = {
        "manifest_version": "1.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_info,
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "device": str(device),
            "image_digest": os.getenv("DOCKER_IMAGE_DIGEST", "local-execution"),
        },
        "dependencies": lib_versions,
        "model": {
            "name": model_name,
            "weights_filename": weights_path.name,
            "weights_sha256": file_sha256(weights_path),
        },
        "input_raster": {
            "filename": raster_path.name,
            "sha256": file_sha256(raster_path),
            "width": metadata.width,
            "height": metadata.height,
            "crs": metadata.crs.to_string() if metadata.crs else None,
            "transform": transform_list,
        },
        "parameters": parameters,
        "outputs": output_hashes,
    }
    return manifest


def save_run_manifest(manifest: dict[str, Any], output_path: Path) -> Path:
    """Save manifest dictionary to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return output_path
