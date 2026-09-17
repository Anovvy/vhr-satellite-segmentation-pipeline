"""Static figures for segmentation predictions and evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from . import CLASS_NAMES, REFERENCE_NODATA


CLASS_COLORS = ("#000000", "#ffff00", "#ff9f1a", "#90ee90", "#006400")
CLASS_CMAP = ListedColormap(CLASS_COLORS)
GROUND_TRUTH_CMAP = CLASS_CMAP.with_extremes(bad="#d9d9d9")
COMPATIBILITY_CMAP = ListedColormap(("#000000", "#00e640", "#ff1f2d"))


def _legend_patches() -> list[mpatches.Patch]:
    return [
        mpatches.Patch(color=CLASS_COLORS[class_id], label=f"{class_id}: {name}")
        for class_id, name in CLASS_NAMES.items()
    ]


def create_evaluation_figure(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    compatibility: np.ndarray,
    metrics: dict,
    model_label: str,
    output_path: str | Path,
    *,
    dpi: int = 150,
) -> Path:
    """Create the refined three-panel evaluation layout."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 3, figsize=(24, 8))
    figure.suptitle(
        f"{model_label} Segmentation Evaluation",
        fontsize=23,
        fontweight="bold",
        y=0.97,
    )

    visible_ground_truth = np.ma.masked_equal(ground_truth, REFERENCE_NODATA)
    axes[0].imshow(visible_ground_truth, cmap=GROUND_TRUTH_CMAP, vmin=0, vmax=4)
    axes[0].set_title("Ground Truth", fontsize=18)
    axes[1].imshow(prediction, cmap=CLASS_CMAP, vmin=0, vmax=4)
    axes[1].set_title(f"{model_label} Prediction", fontsize=18)
    axes[2].imshow(compatibility, cmap=COMPATIBILITY_CMAP, vmin=0, vmax=2)
    axes[2].set_title(
        "Compatibility Map\n"
        f"Green: match · Red: mismatch · Accuracy: {metrics['overall_accuracy'] * 100:.2f}%",
        fontsize=15,
    )
    for axis in axes:
        axis.axis("off")

    figure.legend(
        handles=[
            *_legend_patches(),
            mpatches.Patch(color="#d9d9d9", label="Outside reference"),
        ],
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.03),
        fontsize=12,
        frameon=True,
    )
    figure.text(
        0.5,
        0.095,
        f"mIoU: {metrics['mean_iou'] * 100:.2f}% · "
        f"Macro F1: {metrics['macro_f1_score'] * 100:.2f}% · "
        f"Evaluated pixels: {metrics['evaluated_pixels']:,}",
        ha="center",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0.13, 1, 0.93))
    figure.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(figure)
    return output_path


def create_prediction_figure(
    prediction: np.ndarray,
    model_label: str,
    output_path: str | Path,
    *,
    dpi: int = 150,
) -> Path:
    """Create a prediction-only figure when no reference is supplied."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 10))
    axis.imshow(prediction, cmap=CLASS_CMAP, vmin=0, vmax=4)
    axis.set_title(f"{model_label} Prediction", fontsize=18, fontweight="bold")
    axis.axis("off")
    figure.legend(
        handles=_legend_patches(),
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.01),
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    figure.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(figure)
    return output_path
