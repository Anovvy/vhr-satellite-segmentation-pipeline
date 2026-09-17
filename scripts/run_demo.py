#!/usr/bin/env python3
"""Run inference, vectorization, evaluation, and visualization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the VHR land-cover segmentation pipeline on a georeferenced RGB raster."
        )
    )
    parser.add_argument(
        "--model",
        default="segformer",
        choices=("unet", "deeplabv3plus", "segformer", "all"),
        help="Model backend to run (default: segformer).",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--raster", type=Path, help="Input RGB GeoTIFF.")
    source.add_argument(
        "--demo",
        action="store_true",
        help="Use demo/imagery_demo.tif and demo/reference_demo.gpkg.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="Optional vector reference for evaluation (.gpkg or .shp).",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help=(
            "Run prediction without reference evaluation. With --demo, use "
            "demo imagery but ignore demo/reference_demo.gpkg."
        ),
    )
    parser.add_argument(
        "--reference-field",
        help="Class field in the reference vector; inferred when omitted.",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=REPO_ROOT / "weights",
        help="Directory containing standardized model filenames.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        help="Custom checkpoint path; valid only when one model is selected.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs",
        help="Root output directory (default: outputs/).",
    )
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument(
        "--input-scale",
        type=float,
        default=255.0,
        help="Radiometric divisor used before ImageNet normalization.",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument(
        "--vector-format", choices=("gpkg", "shp", "none"), default="gpkg"
    )
    parser.add_argument(
        "--exclude-unclassified-vector",
        action="store_true",
        help="Do not polygonize class 0.",
    )
    parser.add_argument(
        "--all-touched",
        action="store_true",
        help="Use all_touched=True while rasterizing reference polygons.",
    )
    return parser.parse_args()


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path | None]:
    if args.skip_evaluation and args.reference is not None:
        raise ValueError("--skip-evaluation cannot be combined with --reference.")

    if args.demo:
        raster = REPO_ROOT / "demo" / "imagery_demo.tif"
        reference = REPO_ROOT / "demo" / "reference_demo.gpkg"
        required = (raster,) if args.skip_evaluation else (raster, reference)
        missing = [path for path in required if not path.is_file()]
        if missing:
            names = ", ".join(path.name for path in missing)
            raise FileNotFoundError(
                f"Required demo file(s) are missing: {names}. "
                "Restore the files listed in demo/README.md or run with --raster "
                "and an optional --reference path."
            )
        return raster, None if args.skip_evaluation else reference
    return args.raster, None if args.skip_evaluation else args.reference


def checkpoint_for(args: argparse.Namespace, model_name: str) -> Path:
    from src.models import WEIGHT_FILENAMES

    if args.weights:
        if args.model == "all":
            raise ValueError("--weights cannot be combined with --model all.")
        return args.weights
    return args.weights_dir / WEIGHT_FILENAMES[model_name]


def run_one(
    args: argparse.Namespace,
    model_name: str,
    raster_path: Path,
    reference_path: Path | None,
) -> None:
    import torch

    from src.evaluate import (
        evaluate_prediction,
        print_report,
        rasterize_reference,
        save_metrics,
    )
    from src.inference import predict_raster, save_prediction_raster
    from src.models import MODEL_DISPLAY_NAMES, canonical_model_name, load_model
    from src.vectorize import vectorize_prediction
    from src.visualize import create_evaluation_figure, create_prediction_figure

    name = canonical_model_name(model_name)
    label = MODEL_DISPLAY_NAMES[name]
    checkpoint = checkpoint_for(args, name)
    model_dir = args.output_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{label}] Loading checkpoint: {checkpoint}")
    model, device = load_model(name, checkpoint, device=args.device)
    print(f"[{label}] Device: {device}")
    print(f"[{label}] Running sliding-window inference...")
    prediction, metadata = predict_raster(
        raster_path,
        model,
        name,
        device,
        tile_size=args.tile_size,
        stride=args.stride,
        input_scale=args.input_scale,
    )

    raster_output = save_prediction_raster(
        prediction, metadata, model_dir / "prediction.tif"
    )
    print(f"[{label}] Raster: {raster_output}")

    if args.vector_format != "none":
        vector_output = model_dir / f"prediction.{args.vector_format}"
        vectorize_prediction(
            prediction,
            metadata,
            vector_output,
            include_unclassified=not args.exclude_unclassified_vector,
        )
        print(f"[{label}] Vector: {vector_output}")

    if reference_path:
        ground_truth = rasterize_reference(
            reference_path,
            metadata,
            class_field=args.reference_field,
            all_touched=args.all_touched,
        )
        metrics, compatibility, _ = evaluate_prediction(prediction, ground_truth)
        save_metrics(metrics, model_dir)
        print_report(label, metrics)
        figure = create_evaluation_figure(
            ground_truth,
            prediction,
            compatibility,
            metrics,
            label,
            model_dir / "evaluation.png",
        )
        print(f"[{label}] Evaluation figure: {figure}")
    else:
        figure = create_prediction_figure(
            prediction, label, model_dir / "prediction.png"
        )
        print(f"[{label}] No reference supplied; evaluation skipped.")
        print(f"[{label}] Prediction figure: {figure}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> int:
    args = parse_args()
    try:
        raster_path, reference_path = resolve_inputs(args)
        models = (
            ("unet", "deeplabv3plus", "segformer")
            if args.model == "all"
            else (args.model,)
        )
        for model_name in models:
            run_one(args, model_name, raster_path, reference_path)
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    print(f"\nDone. Outputs are available under: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
