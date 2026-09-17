"""Model construction and checkpoint loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from . import NUM_CLASSES


MODEL_ALIASES = {
    "unet": "unet",
    "u-net": "unet",
    "deeplab": "deeplabv3plus",
    "deeplabv3+": "deeplabv3plus",
    "deeplabv3plus": "deeplabv3plus",
    "segformer": "segformer",
}

MODEL_DISPLAY_NAMES = {
    "unet": "U-Net (ResNet34)",
    "deeplabv3plus": "DeepLabV3+ (ResNet50)",
    "segformer": "SegFormer (MiT-B0)",
}

WEIGHT_FILENAMES = {
    "unet": "unet_resnet34.pth",
    "deeplabv3plus": "deeplabv3plus_resnet50.pth",
    "segformer": "segformer_mit_b0.pth",
}


def canonical_model_name(name: str) -> str:
    """Return the canonical CLI model name."""
    key = name.lower().strip()
    if key not in MODEL_ALIASES:
        choices = ", ".join(sorted(set(MODEL_ALIASES.values())))
        raise ValueError(f"Unknown model '{name}'. Choose one of: {choices}.")
    return MODEL_ALIASES[key]


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve ``auto``, ``cpu``, or ``cuda`` to a PyTorch device."""
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("Device must be one of: auto, cpu, cuda.")
    return torch.device(requested)


def build_model(
    model_name: str,
    *,
    num_classes: int = NUM_CLASSES,
    pretrained_encoder: bool = False,
) -> torch.nn.Module:
    """Build an architecture compatible with the project checkpoints."""
    name = canonical_model_name(model_name)

    if name in {"unet", "deeplabv3plus"}:
        import segmentation_models_pytorch as smp

        encoder_weights = "imagenet" if pretrained_encoder else None
        if name == "unet":
            return smp.Unet(
                encoder_name="resnet34",
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=num_classes,
            )
        return smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
        )

    from transformers import SegformerForSemanticSegmentation

    # The original experiment used the pretrained MiT-B0 configuration. The
    # complete project checkpoint is loaded after construction.
    return SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0",
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a PyTorch state dictionary.")

    for key in ("state_dict", "model_state_dict", "model"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict):
            checkpoint = candidate
            break

    state_dict = {}
    for key, value in checkpoint.items():
        clean_key = key.removeprefix("module.")
        state_dict[clean_key] = value
    return state_dict


def load_model(
    model_name: str,
    weights_path: str | Path,
    *,
    device: str | torch.device = "auto",
    num_classes: int = NUM_CLASSES,
) -> tuple[torch.nn.Module, torch.device]:
    """Construct a model, load its state dictionary, and switch to eval mode."""
    name = canonical_model_name(model_name)
    resolved_device = resolve_device(device) if isinstance(device, str) else device
    path = Path(weights_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    model = build_model(name, num_classes=num_classes, pretrained_encoder=False)
    try:
        checkpoint = torch.load(path, map_location=resolved_device, weights_only=True)
    except TypeError:  # Compatibility with older PyTorch releases.
        checkpoint = torch.load(path, map_location=resolved_device)

    model.load_state_dict(_extract_state_dict(checkpoint), strict=True)
    model.to(resolved_device)
    model.eval()
    return model, resolved_device


def forward_logits(
    model: torch.nn.Module,
    model_name: str,
    batch: torch.Tensor,
    output_size: tuple[int, int],
) -> torch.Tensor:
    """Run one model and return logits at the requested spatial resolution."""
    name = canonical_model_name(model_name)
    if name == "segformer":
        logits = model(pixel_values=batch).logits
    else:
        logits = model(batch)

    if tuple(logits.shape[-2:]) != tuple(output_size):
        logits = F.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
    return logits

