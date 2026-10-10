#!/usr/bin/env python3
"""Run environment and runtime diagnostic for the VHR segmentation pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.doctor import run_doctor  # noqa: E402


def main() -> int:
    healthy = run_doctor()
    return 0 if healthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
