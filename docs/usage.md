# Demo Commands and Advanced Usage

Start with the [installation and demo guide](../README.md#installation-and-demo-guide). All commands below assume an active Python environment, complete demo files, and matching model weights.

## Run U-Net

```bash
python scripts/run_demo.py --model unet --demo
```

## Run DeepLabV3+

```bash
python scripts/run_demo.py --model deeplabv3plus --demo
```

## Run SegFormer

```bash
python scripts/run_demo.py --model segformer --demo
```

## Run All Models

```bash
python scripts/run_demo.py --model all --demo --output-dir outputs/demo_comparison
```

The models run one after another. Each model receives a separate output folder.

## Run Without a Reference Layer

Reference data is optional. The following command runs SegFormer on the demo image without calculating evaluation metrics:

```bash
python scripts/run_demo.py --model segformer --demo --skip-evaluation
```

For a custom image, omit `--reference`:

```bash
python scripts/run_demo.py --model segformer --raster "/path/to/image.tif"
```

Imagery-only runs create `prediction.tif`, the selected vector output, and `prediction.png`. Metric files, a confusion matrix, and `evaluation.png` are created only when a reference vector is supplied.

## Select CPU or CUDA

Automatic device selection is the default. CUDA is selected when available.

CPU:

```bash
python scripts/run_demo.py --model unet --demo --device cpu
```

CUDA:

```bash
python scripts/run_demo.py --model unet --demo --device cuda
```

The CUDA command requires a working NVIDIA driver and CUDA-enabled PyTorch installation.

## Create Shapefile Output

```bash
python scripts/run_demo.py --model segformer --demo --vector-format shp
```

Shapefile output consists of several files. Keep `.shp`, `.shx`, `.dbf`, and `.prj` sidecars together.

Skip vector output:

```bash
python scripts/run_demo.py --model segformer --demo --vector-format none
```

## Change the Output Folder

```bash
python scripts/run_demo.py --model segformer --demo --output-dir outputs/demo_run_02
```

Use a new folder name to preserve earlier outputs.

## Advanced Settings

The default settings match the refined notebooks.

| Option | Default | Meaning |
|---|---:|---|
| `--tile-size` | 512 | Tile width and height in pixels |
| `--stride` | 256 | Distance between neighboring tiles |
| `--input-scale` | 255 | Divisor before ImageNet normalization |
| `--vector-format` | `gpkg` | Polygon output format |
| `--all-touched` | Off | Rasterize every pixel touched by a reference polygon |
| `--exclude-unclassified-vector` | Off | Remove class `0` from polygon output |
| `--skip-evaluation` | Off | Ignore the demo reference and create predictions only |

Changing tile size, stride, input scaling, or reference rasterization can change model predictions and reported metrics. Use identical settings for model comparison.

## Run a Custom Raster

Custom data is an advanced path outside the one-command demo. A reference vector is optional.

Imagery only:

```bash
python scripts/run_demo.py --model segformer --raster "/path/to/image.tif"
```

Imagery with reference evaluation:

```bash
python scripts/run_demo.py --model segformer --raster "/path/to/image.tif" --reference "/path/to/reference.gpkg" --reference-field class_id
```

Requirements:

1. a georeferenced raster with RGB in bands 1–3;
2. pixel values compatible with the training preprocessing;
3. valid raster and vector coordinate reference systems;
4. for evaluation, reference class IDs from `0` to `4`;
5. for evaluation, spatial overlap between imagery and reference polygons.

The original experiment used 30 cm imagery from dense residential areas in Jakarta. Another resolution changes the ground area visible to the model, while a different urban form, sensor, season, or radiometric profile can cause domain-shift errors. The program preserves the supplied grid and does not automatically resample custom inputs to 30 cm. Local reference evaluation is strongly recommended for unfamiliar locations.

## Reproducibility Record

For each published run, retain:

- the exact command;
- imagery and reference checksums;
- checkpoint checksum;
- `metrics.json`;
- per-class CSV and confusion matrix;
- evaluation PNG;
- installed package versions.

Save the environment after a successful run:

```bash
python -m pip freeze > environment-lock.txt
```

## Metric Scope

See [results.md](results.md) for the distinction between training validation, private-scene-imagery evaluation, and public demo evaluation.
