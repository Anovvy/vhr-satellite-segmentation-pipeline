"""Tests for CLI arguments parsing and input validation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.run_demo import parse_args, resolve_inputs


def test_cli_help():
    """Verify scripts/run_demo.py --help succeeds with code 0."""
    result = subprocess.run(
        [sys.executable, "scripts/run_demo.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Run the VHR land-cover segmentation pipeline" in result.stdout


def test_cli_mutually_exclusive_inputs(monkeypatch):
    """Passing neither or both --demo and --raster must fail argument parsing."""
    # Neither passed:
    monkeypatch.setattr(sys, "argv", ["run_demo.py"])
    with pytest.raises(SystemExit):
        parse_args()

    # Both passed:
    monkeypatch.setattr(sys, "argv", ["run_demo.py", "--demo", "--raster", "test.tif"])
    with pytest.raises(SystemExit):
        parse_args()


def test_cli_skip_evaluation_with_reference_raises(monkeypatch, tmp_path: Path):
    """Combining --skip-evaluation with --reference must raise ValueError."""
    ref_file = tmp_path / "ref.gpkg"
    ref_file.touch()
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_demo.py", "--raster", "test.tif", "--reference", str(ref_file), "--skip-evaluation"],
    )
    args = parse_args()
    with pytest.raises(ValueError, match="--skip-evaluation cannot be combined with --reference"):
        resolve_inputs(args)


def test_cli_missing_checkpoint_actionable_error(tmp_path: Path):
    """Running CLI with missing weights file reports clean actionable error without crash."""
    test_raster = tmp_path / "test.tif"
    test_raster.touch()
    empty_weights_dir = tmp_path / "empty_weights"
    empty_weights_dir.mkdir()

    cmd = [
        sys.executable,
        "scripts/run_demo.py",
        "--model",
        "segformer",
        "--raster",
        str(test_raster),
        "--weights-dir",
        str(empty_weights_dir),
        "--skip-evaluation",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "Checkpoint not found" in result.stderr or "ERROR" in result.stderr
