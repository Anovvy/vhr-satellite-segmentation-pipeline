#!/usr/bin/env bash
# ==============================================================================
# VHR Satellite Segmentation - Docker Runner Script (Linux / macOS / Git Bash)
# Run complete pipeline without any manual Python/GDAL environment setup.
# ==============================================================================
set -e

if [ "$1" = "--doctor" ] || [ "$1" = "doctor" ]; then
    docker compose run --rm doctor
    exit 0
elif [ "$1" = "--test" ] || [ "$1" = "test" ]; then
    docker compose run --rm test
    exit 0
elif [ "$1" = "--gpu" ]; then
    shift
    docker compose run --rm gpu "$@"
    exit 0
fi

if [ $# -eq 0 ]; then
    docker compose run --rm cpu --help
    exit 0
fi

docker compose run --rm cpu "$@"
