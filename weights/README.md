# Model Checkpoints

This directory stores trained deep-learning parameter checkpoints (`.pth`).

## Standard Checkpoint Files

Official release binaries are cataloged in [`checksums.sha256`](checksums.sha256):

| Model Architecture | Target Filename | SHA-256 Digest |
|---|---|---|
| **SegFormer (MiT-B0)** | `segformer_mit_b0.pth` | `b313598c922fc8f53c1a68b1a96dc9ede567488c38b13767191eeeefe71ffea0` |
| **U-Net (ResNet34)** | `unet_resnet34.pth` | *(Download from GitHub Release)* |
| **DeepLabV3+ (ResNet50)** | `deeplabv3plus_resnet50.pth` | *(Download from GitHub Release)* |

## Automatic Verification

Verify the integrity of downloaded checkpoints using the containerized doctor:

```bash
# Windows PowerShell
.\run.ps1 -Doctor

# Linux / macOS
./run.sh --doctor
```
