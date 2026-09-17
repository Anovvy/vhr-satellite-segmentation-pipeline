"""Reference rasterization and segmentation metrics."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
from pandas.api.types import is_numeric_dtype
from rasterio.features import rasterize
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from . import CLASS_NAMES, REFERENCE_NODATA
from .inference import RasterMetadata


REFERENCE_CLASS_MAPPING = {
    "Unclassified": 0,
    "Unclassified / Other": 0,
    "Bangunan Permukiman/campuran": 1,
    "Residential / Mixed-use": 1,
    "Residential / Mixed-use Built-up": 1,
    "Residential Built-up": 1,
    "Bangunan bukan permukiman": 2,
    "Non-residential": 2,
    "Non-residential Built-up": 2,
    "Padang Rumput": 3,
    "Grassland": 3,
    "Hutan Kota, jalur hijau dan taman kota": 4,
    "Urban Forest": 4,
    "Urban Forest / Green Belt / Urban Park": 4,
}

DEFAULT_FIELD_CANDIDATES = (
    "class_id",
    "Dataset_Cl",
    "Dataset_Class",
    "keterangan",
    "category",
)


def _resolve_reference_field(frame: gpd.GeoDataFrame, requested: str | None) -> str:
    if requested:
        if requested not in frame.columns:
            raise ValueError(
                f"Reference field '{requested}' not found. Available fields: "
                + ", ".join(map(str, frame.columns))
            )
        return requested
    for candidate in DEFAULT_FIELD_CANDIDATES:
        if candidate in frame.columns:
            return candidate
    raise ValueError(
        "Could not infer the reference class field. Use --reference-field. "
        f"Available fields: {', '.join(map(str, frame.columns))}"
    )


def _class_ids(values) -> np.ndarray:
    if values.isna().any():
        raise ValueError("Reference class field contains null values.")
    if is_numeric_dtype(values):
        numeric = values.to_numpy(dtype=np.float64)
        if not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError("Numeric reference class IDs must be whole numbers.")
        result = numeric.astype(np.int64)
    else:
        cleaned = values.astype(str).str.strip()
        mapped = cleaned.map(REFERENCE_CLASS_MAPPING)
        if mapped.isna().any():
            unknown = sorted(set(cleaned[mapped.isna()]))
            raise ValueError(
                "Unknown reference class values: " + ", ".join(unknown)
            )
        result = mapped.to_numpy(dtype=np.int64)
    if not np.isin(result, list(CLASS_NAMES)).all():
        unknown_ids = sorted(set(result) - set(CLASS_NAMES))
        raise ValueError(f"Reference contains unsupported class IDs: {unknown_ids}")
    return result


def rasterize_reference(
    reference_path: str | Path,
    metadata: RasterMetadata,
    *,
    class_field: str | None = None,
    all_touched: bool = False,
) -> np.ndarray:
    """Rasterize reference labels while preserving class 0 as a valid class.

    Pixels outside every reference geometry receive ``REFERENCE_NODATA`` so
    they can be excluded without conflating them with Unclassified / Other.
    """
    path = Path(reference_path)
    if not path.exists():
        raise FileNotFoundError(f"Reference vector not found: {path}")
    frame = gpd.read_file(path)
    if frame.empty:
        raise ValueError("Reference vector is empty.")
    if frame.crs is None:
        raise ValueError("Reference vector has no CRS.")
    if metadata.crs is None:
        raise ValueError("Input raster has no CRS; reference alignment is undefined.")
    if frame.crs != metadata.crs:
        frame = frame.to_crs(metadata.crs)

    field = _resolve_reference_field(frame, class_field)
    frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty].copy()
    frame["_class_id"] = _class_ids(frame[field])
    pairs = (
        (geom, int(value))
        for geom, value in zip(frame.geometry, frame["_class_id"])
    )
    return rasterize(
        pairs,
        out_shape=(metadata.height, metadata.width),
        transform=metadata.transform,
        fill=int(REFERENCE_NODATA),
        dtype=np.uint8,
        all_touched=all_touched,
    )


def evaluate_prediction(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Evaluate class IDs 0-4 inside the covered reference area."""
    if prediction.shape != ground_truth.shape:
        raise ValueError("Prediction and ground-truth arrays must have equal shapes.")

    class_labels = list(CLASS_NAMES)
    if not np.isin(prediction, class_labels).all():
        unknown = sorted(set(np.unique(prediction)) - set(class_labels))
        raise ValueError(f"Prediction contains unsupported class IDs: {unknown}")

    allowed_reference = [*class_labels, int(REFERENCE_NODATA)]
    if not np.isin(ground_truth, allowed_reference).all():
        unknown = sorted(set(np.unique(ground_truth)) - set(allowed_reference))
        raise ValueError(f"Ground truth contains unsupported class IDs: {unknown}")

    valid = ground_truth != REFERENCE_NODATA
    evaluated_pixels = int(valid.sum())
    if evaluated_pixels == 0:
        raise ValueError("Reference contains no covered pixels with class IDs 0-4.")

    y_true = ground_truth[valid].astype(np.uint8)
    y_pred = prediction[valid].astype(np.uint8)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=class_labels,
        zero_division=0,
    )

    per_class = []
    ious = []
    for index, class_id in enumerate(class_labels):
        intersection = int(np.logical_and(y_true == class_id, y_pred == class_id).sum())
        union = int(np.logical_or(y_true == class_id, y_pred == class_id).sum())
        iou = intersection / union if union else 0.0
        ious.append(iou)
        per_class.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1_score": float(f1[index]),
                "iou": float(iou),
                "support_pixels": int(support[index]),
            }
        )

    overall_accuracy = float((y_true == y_pred).mean())
    compatibility = np.zeros_like(ground_truth, dtype=np.uint8)
    compatibility[valid & (ground_truth == prediction)] = 1
    compatibility[valid & (ground_truth != prediction)] = 2
    matrix = confusion_matrix(y_true, y_pred, labels=class_labels)

    metrics = {
        "evaluation_scope": "reference-covered pixels with ground-truth class_id 0-4",
        "reference_nodata_value": int(REFERENCE_NODATA),
        "evaluated_pixels": evaluated_pixels,
        "overall_accuracy": overall_accuracy,
        "mean_iou": float(np.mean(ious)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1_score": float(np.mean(f1)),
        "per_class": per_class,
        "confusion_matrix_labels": class_labels,
        "confusion_matrix": matrix.tolist(),
    }
    return metrics, compatibility, matrix


def save_metrics(metrics: dict, output_dir: str | Path) -> dict[str, Path]:
    """Save summary JSON, per-class CSV, and confusion-matrix CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "metrics.json"
    class_csv_path = output_dir / "metrics_per_class.csv"
    matrix_csv_path = output_dir / "confusion_matrix.csv"

    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, ensure_ascii=False)

    rows = metrics["per_class"]
    with class_csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels = metrics["confusion_matrix_labels"]
    matrix = metrics["confusion_matrix"]
    with matrix_csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ground_truth\\prediction", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row])

    return {
        "json": json_path,
        "per_class_csv": class_csv_path,
        "confusion_matrix_csv": matrix_csv_path,
    }


def print_report(model_label: str, metrics: dict) -> None:
    """Print a compact human-readable metric report."""
    print("\n" + "=" * 78)
    print(f" {model_label} EVALUATION REPORT ".center(78, "="))
    print("=" * 78)
    print(f"Overall accuracy : {metrics['overall_accuracy'] * 100:.2f}%")
    print(f"Mean IoU         : {metrics['mean_iou'] * 100:.2f}%")
    print(f"Macro precision  : {metrics['macro_precision'] * 100:.2f}%")
    print(f"Macro recall     : {metrics['macro_recall'] * 100:.2f}%")
    print(f"Macro F1-score   : {metrics['macro_f1_score'] * 100:.2f}%")
    print(f"Evaluated pixels : {metrics['evaluated_pixels']:,}")
    print("-" * 78)
    print(f"{'Class':<39} {'Precision':>9} {'Recall':>9} {'F1':>9} {'IoU':>9}")
    for row in metrics["per_class"]:
        print(
            f"{row['class_name']:<39} "
            f"{row['precision']:>9.4f} {row['recall']:>9.4f} "
            f"{row['f1_score']:>9.4f} {row['iou']:>9.4f}"
        )
    print("=" * 78)
