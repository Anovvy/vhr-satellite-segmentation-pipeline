"""Environment and runtime diagnostics for the VHR segmentation pipeline."""

from __future__ import annotations

import hashlib
import platform
import sys
from pathlib import Path
from typing import Any

from .models import WEIGHT_FILENAMES

CHECKSUM_FILE = Path(__file__).resolve().parents[1] / "weights" / "checksums.sha256"


def check_python() -> dict[str, Any]:
    """Check Python version and implementation."""
    version = sys.version_info
    healthy = version.major == 3 and version.minor >= 10
    return {
        "check": "python",
        "version": f"{version.major}.{version.minor}.{version.micro}",
        "healthy": healthy,
        "detail": platform.python_implementation(),
    }


def check_core_libraries() -> list[dict[str, Any]]:
    """Check availability and version of essential dependencies."""
    packages = [
        ("torch", "PyTorch", False),
        ("torchvision", "TorchVision", False),
        ("transformers", "Hugging Face Transformers", False),
        ("segmentation_models_pytorch", "Segmentation Models PyTorch", False),
        ("rasterio", "Rasterio", False),
        ("geopandas", "GeoPandas", False),
        ("shapely", "Shapely", False),
        ("numpy", "NumPy", False),
        ("PIL", "Pillow", False),
        ("cv2", "OpenCV (training only)", True),
    ]
    results = []
    for pkg_name, display_name, optional in packages:
        try:
            mod = __import__(pkg_name)
            ver = getattr(mod, "__version__", "installed")
            results.append(
                {
                    "name": display_name,
                    "installed": True,
                    "version": ver,
                    "healthy": True,
                    "optional": optional,
                }
            )
        except ImportError as err:
            results.append(
                {
                    "name": display_name,
                    "installed": False,
                    "version": None,
                    "healthy": optional,
                    "optional": optional,
                    "error": str(err),
                }
            )
    return results


def check_raster_support() -> dict[str, Any]:
    """Check GDAL and Rasterio GeoTIFF and projection engines."""
    try:
        import rasterio
        from rasterio.crs import CRS
        from rasterio.env import Env

        with Env() as env:
            drivers = env.drivers()
            gtiff_ok = "GTiff" in drivers
        crs_ok = CRS.from_epsg(4326).to_string() == "EPSG:4326"
        return {
            "healthy": gtiff_ok and crs_ok,
            "gdal_version": rasterio.__gdal_version__,
            "gtiff_driver": gtiff_ok,
            "crs_engine": crs_ok,
        }
    except Exception as exc:
        return {"healthy": False, "error": str(exc)}


def check_compute_devices() -> dict[str, Any]:
    """Check CPU and CUDA acceleration availability."""
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if cuda_available else 0
        device_names = (
            [torch.cuda.get_device_name(i) for i in range(device_count)]
            if cuda_available
            else []
        )
        return {
            "healthy": True,
            "cuda_available": cuda_available,
            "device_count": device_count,
            "devices": device_names,
            "recommended_device": "cuda" if cuda_available else "cpu",
        }
    except Exception as exc:
        return {"healthy": False, "error": str(exc)}


def load_checksum_manifest(checksum_file: Path | None = None) -> dict[str, str]:
    """Load known SHA256 checksums from manifest file."""
    path = checksum_file or CHECKSUM_FILE
    manifest: dict[str, str] = {}
    if not path.is_file():
        return manifest
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                digest, filename = parts
                manifest[filename.strip()] = digest.strip().lower()
    return manifest


def check_checkpoints(weights_dir: Path | None = None) -> list[dict[str, Any]]:
    """Verify presence and checksums of trained checkpoints."""
    repo_root = Path(__file__).resolve().parents[1]
    wdir = weights_dir or (repo_root / "weights")
    manifest = load_checksum_manifest(wdir / "checksums.sha256")

    results = []
    for model_key, filename in WEIGHT_FILENAMES.items():
        ckpt_path = wdir / filename
        if not ckpt_path.is_file():
            results.append(
                {
                    "model": model_key,
                    "filename": filename,
                    "present": False,
                    "status": "MISSING",
                    "healthy": False,
                }
            )
            continue

        # Compute file hash
        sha256 = hashlib.sha256()
        with ckpt_path.open("rb") as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
        digest = sha256.hexdigest().lower()

        expected = manifest.get(filename)
        matches = (digest == expected) if expected else None
        results.append(
            {
                "model": model_key,
                "filename": filename,
                "present": True,
                "sha256": digest,
                "status": "VERIFIED"
                if matches is True
                else ("UNVERIFIED_HASH" if matches is False else "PRESENT"),
                "healthy": matches is not False,
            }
        )
    return results


def run_doctor(weights_dir: Path | None = None) -> bool:
    """Run all diagnostics and print a formatted summary report."""
    print("=" * 70)
    print(" VHR SATELLITE SEGMENTATION PIPELINE - ENVIRONMENT DOCTOR ".center(70, "="))
    print("=" * 70)

    py = check_python()
    status_py = "[OK]" if py["healthy"] else "[FAIL]"
    print(f"{status_py:<8} Python : {py['version']} ({py['detail']})")

    print("\n[Core Libraries]")
    libs = check_core_libraries()
    all_libs_ok = True
    for lib in libs:
        status = (
            "[OK]" if lib["installed"] else ("[INFO]" if lib["optional"] else "[FAIL]")
        )
        if not lib["optional"]:
            all_libs_ok = all_libs_ok and lib["installed"]
        ver = lib["version"] or lib.get("error", "missing")
        print(f"  {status:<6} {lib['name']:<32} : {ver}")

    print("\n[Raster & GIS Support]")
    raster = check_raster_support()
    raster_status = "[OK]" if raster.get("healthy") else "[FAIL]"
    print(
        f"  {raster_status:<6} GDAL Version                  : {raster.get('gdal_version', 'N/A')}"
    )
    print(
        f"  {raster_status:<6} GTiff Driver & CRS Engine     : {'Supported' if raster.get('healthy') else 'Error'}"
    )

    print("\n[Compute Hardware]")
    compute = check_compute_devices()
    cuda_str = (
        "Available"
        if compute.get("cuda_available")
        else "Not available (CPU inference mode)"
    )
    print(f"  {'[OK]':<6} CUDA Device Acceleration      : {cuda_str}")
    if compute.get("devices"):
        for i, dev_name in enumerate(compute["devices"]):
            print(f"         GPU {i}                         : {dev_name}")

    print("\n[Model Checkpoints]")
    ckpts = check_checkpoints(weights_dir)
    for ckpt in ckpts:
        status = "[OK]" if ckpt.get("present") else "[INFO]"
        tag = ckpt["status"]
        print(f"  {status:<6} {ckpt['filename']:<30} : {tag}")
        if ckpt.get("sha256"):
            print(f"         SHA-256: {ckpt['sha256']}")

    print("=" * 70)
    all_ok = py["healthy"] and all_libs_ok and raster.get("healthy", False)
    if all_ok:
        print("Doctor status: HEALTHY (Environment is ready for execution)")
    else:
        print("Doctor status: ISSUES DETECTED (Please check missing dependencies)")
    print("=" * 70)
    return all_ok
