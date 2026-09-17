#!/usr/bin/env python3
"""Shared experimental training entry point for all three architectures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import albumentations as A
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import NUM_CLASSES  # noqa: E402
from src.models import (  # noqa: E402
    WEIGHT_FILENAMES,
    build_model,
    canonical_model_name,
    forward_logits,
)


class SegmentationDataset(Dataset):
    def __init__(self, images_dir: Path, masks_dir: Path, filenames, transform):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.filenames = list(filenames)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int):
        filename = self.filenames[index]
        image = cv2.imread(str(self.images_dir / filename), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(self.masks_dir / filename), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(f"Could not read image/mask pair: {filename}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=image, mask=mask)
        return transformed["image"], transformed["mask"].long()


def transforms(training: bool) -> A.Compose:
    operations = []
    if training:
        operations.extend(
            [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5)]
        )
    operations.extend(
        [
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ]
    )
    return A.Compose(operations)


def batch_metrics(logits: torch.Tensor, masks: torch.Tensor) -> tuple[float, float]:
    predictions = torch.argmax(logits, dim=1)
    accuracy = float((predictions == masks).float().mean().item())
    ious = []
    for class_id in range(NUM_CLASSES):
        predicted = predictions == class_id
        target = masks == class_id
        union = int((predicted | target).sum().item())
        if union:
            intersection = int((predicted & target).sum().item())
            ious.append(intersection / union)
    return accuracy, float(np.mean(ious)) if ious else 0.0


def parse_args(default_model: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    if default_model is None:
        parser.add_argument(
            "--model", required=True, choices=("unet", "deeplabv3plus", "segformer")
        )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--min-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if default_model is not None:
        args.model = default_model
    return args


def main(default_model: str | None = None) -> int:
    args = parse_args(default_model)
    model_name = canonical_model_name(args.model)
    defaults = {
        "unet": (16, 1e-4),
        "deeplabv3plus": (8, 1e-4),
        "segformer": (16, 6e-5),
    }
    default_batch, default_lr = defaults[model_name]
    batch_size = args.batch_size or default_batch
    learning_rate = args.learning_rate or default_lr

    if not 0 < args.validation_fraction < 1:
        raise ValueError("validation-fraction must be between 0 and 1.")
    images_dir = args.data_dir / "images"
    masks_dir = args.data_dir / "masks"
    filenames = sorted(
        path.name
        for path in images_dir.glob("*.png")
        if (masks_dir / path.name).is_file()
    )
    if len(filenames) < 2:
        raise ValueError("At least two matching PNG image/mask pairs are required.")

    generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(filenames), generator=generator).tolist()
    validation_count = max(1, round(len(filenames) * args.validation_fraction))
    validation_files = [filenames[index] for index in order[:validation_count]]
    training_files = [filenames[index] for index in order[validation_count:]]

    training_data = SegmentationDataset(
        images_dir, masks_dir, training_files, transforms(training=True)
    )
    validation_data = SegmentationDataset(
        images_dir, masks_dir, validation_files, transforms(training=False)
    )
    training_loader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(training_data) >= batch_size,
        num_workers=args.num_workers,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = build_model(model_name, pretrained_encoder=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / WEIGHT_FILENAMES[model_name]

    history = []
    best_validation_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        training_loss = 0.0
        progress = tqdm(training_loader, desc=f"Epoch {epoch}/{args.epochs} [train]")
        for images, masks in progress:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = forward_logits(model, model_name, images, tuple(masks.shape[-2:]))
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        validation_loss = validation_accuracy = validation_miou = 0.0
        with torch.inference_mode():
            for images, masks in validation_loader:
                images, masks = images.to(device), masks.to(device)
                logits = forward_logits(model, model_name, images, tuple(masks.shape[-2:]))
                validation_loss += criterion(logits, masks).item()
                accuracy, miou = batch_metrics(logits, masks)
                validation_accuracy += accuracy
                validation_miou += miou

        row = {
            "epoch": epoch,
            "train_loss": training_loss / len(training_loader),
            "val_loss": validation_loss / len(validation_loader),
            "val_acc": validation_accuracy / len(validation_loader),
            "val_miou": validation_miou / len(validation_loader),
        }
        history.append(row)
        print(
            f"train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} "
            f"val_acc={row['val_acc']:.4f} val_miou={row['val_miou']:.4f}"
        )

        if row["val_loss"] < best_validation_loss:
            best_validation_loss = row["val_loss"]
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
        if epoch >= args.min_epochs and patience_counter >= args.patience:
            print("Early stopping triggered.")
            break

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(args.output_dir / f"training_history_{model_name}.csv", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history_frame.epoch, history_frame.train_loss, label="Train loss")
    axes[0].plot(history_frame.epoch, history_frame.val_loss, label="Validation loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(history_frame.epoch, history_frame.val_acc, label="Validation accuracy")
    axes[1].plot(history_frame.epoch, history_frame.val_miou, label="Validation mIoU")
    axes[1].set_title("Validation metrics")
    axes[1].legend()
    axes[1].grid(True)
    figure.tight_layout()
    figure.savefig(args.output_dir / f"training_history_{model_name}.png", dpi=150)
    plt.close(figure)
    print(f"Best checkpoint: {checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
