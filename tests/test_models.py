"""Tests for offline model construction and checkpoint validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.models import (
    build_model,
    canonical_model_name,
    forward_logits,
    load_model,
    validate_checkpoint,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SEGFORMER_WEIGHTS = REPO_ROOT / "weights" / "segformer_mit_b0.pth"


@pytest.mark.parametrize("model_name", ["unet", "deeplabv3plus", "segformer"])
def test_build_model_offline(model_name: str):
    """All models must build completely offline without network calls."""
    model = build_model(model_name, num_classes=5, pretrained_encoder=False)
    assert isinstance(model, torch.nn.Module)
    model.eval()

    dummy_input = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
    with torch.inference_mode():
        logits = forward_logits(model, model_name, dummy_input, output_size=(64, 64))
    assert logits.shape == (1, 5, 64, 64)


def test_canonical_model_name():
    """Verify alias mapping and invalid model name handling."""
    assert canonical_model_name("U-Net") == "unet"
    assert canonical_model_name("DeepLabV3+") == "deeplabv3plus"
    assert canonical_model_name("segformer") == "segformer"

    with pytest.raises(ValueError, match="Unknown model 'invalid_arch'"):
        canonical_model_name("invalid_arch")


def test_validate_checkpoint_missing(tmp_path: Path):
    """Missing checkpoint file raises FileNotFoundError."""
    missing = tmp_path / "nonexistent.pth"
    with pytest.raises(FileNotFoundError):
        validate_checkpoint(missing, "segformer")


def test_validate_checkpoint_corrupt(tmp_path: Path):
    """Unreadable/corrupt checkpoint file raises ValueError."""
    corrupt = tmp_path / "corrupt.pth"
    corrupt.write_text("not a valid torch file")
    with pytest.raises(ValueError, match="Corrupt or unreadable"):
        validate_checkpoint(corrupt, "segformer")


def test_validate_checkpoint_class_mismatch(tmp_path: Path):
    """Checkpoint with wrong number of output classes raises ValueError."""
    # Create fake state dict with 10 classes
    fake_ckpt = {"decode_head.classifier.weight": torch.zeros((10, 256, 1, 1))}
    ckpt_path = tmp_path / "wrong_classes.pth"
    torch.save(fake_ckpt, ckpt_path)

    with pytest.raises(ValueError, match="Checkpoint class mismatch"):
        validate_checkpoint(ckpt_path, "segformer", num_classes=5)


@pytest.mark.skipif(not SEGFORMER_WEIGHTS.is_file(), reason="segformer_mit_b0.pth not available in weights/")
def test_load_model_real_segformer():
    """Verify loading real segformer_mit_b0.pth checkpoint offline on CPU."""
    model, device = load_model("segformer", SEGFORMER_WEIGHTS, device="cpu")
    assert str(device) == "cpu"
    assert not model.training

    # Single tile inference test
    x = torch.zeros((1, 3, 128, 128), dtype=torch.float32)
    with torch.inference_mode():
        logits = forward_logits(model, "segformer", x, output_size=(128, 128))
    assert logits.shape == (1, 5, 128, 128)
