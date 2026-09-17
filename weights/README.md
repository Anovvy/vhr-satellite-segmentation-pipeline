# Trained Model Files

A weight file stores the learned parameters required for inference. Download the files from the repository's GitHub Release and place them in this folder.

## Standard Filenames

| Model | Required filename |
|---|---|
| U-Net with ResNet34 | `unet_resnet34.pth` |
| DeepLabV3+ with ResNet50 | `deeplabv3plus_resnet50.pth` |
| SegFormer with MiT-B0 | `segformer_mit_b0.pth` |

The checkpoints require three RGB input channels and five output channels in the order listed in the [main README](../README.md#class-definitions). IDs `1–4` are the four target classes; ID `0` is the semantic `Unclassified / Other` catch-all rather than background.

The training domain consists of dense residential areas in Jakarta. Unfamiliar locations, sensors, resolutions, seasons, or image-processing pipelines require local validation and can produce incorrect classifications.

## Checkpoint Compatibility

- Renaming a checkpoint does not change its model architecture.
- A size-mismatch error usually indicates a different architecture, backbone, class count, or library version.
- The loader accepts a plain PyTorch `state_dict` and common wrappers named `state_dict`, `model_state_dict`, or `model`.
- Checkpoints should only come from a trusted source.

## File Verification

Linux:

```bash
sha256sum weights/*.pth
```

macOS:

```bash
shasum -a 256 weights/*.pth
```

Windows PowerShell:

```powershell
Get-FileHash weights\*.pth -Algorithm SHA256
```

Published checksums allow detection of incomplete or modified downloads.
