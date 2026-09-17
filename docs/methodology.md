# Methodology

This document explains the technical details behind the project. For installation and copyable commands, start with the [main user guide](../README.md#installation-and-demo-guide).

### In short: 
prepare labeled image chips, train a model, use its saved weights to predict a new image, then compare that prediction with reference polygons. **The supplied training graphs and final inference scores measure different things.** See [result notes](results.md) for the verified original reports.

## 1. Project scope

This project compares three deep-learning architectures for land-cover segmentation from 30 cm very-high-resolution RGB satellite imagery. Each model has five output channels: four target land-cover classes and one semantic catch-all class. The public software boundary begins with a trained checkpoint and a georeferenced raster. Original commercial imagery and training labels are not distributed.

The models were trained on dense residential areas in Jakarta. Predictions are therefore most credible for imagery with similar urban form, spatial resolution, season, sensor characteristics, radiometry, and land-cover appearance. A geographically or visually different scene can produce systematic misclassification and requires local validation before operational use.

The target classes are:

1. Unclassified / Other;
2. Residential / Mixed-use Built-up;
3. Non-residential Built-up;
4. Grassland;
5. Urban forest, green belts, and urban parks.

The numeric IDs are `0`–`4` in the same order.

Class `0` is not a background label. It represents objects outside the four target classes, including water bodies, plantations, open land, and other land cover not represented by IDs `1–4`. Class `1` represents built-up land used as housing or residential mixed use. Class `2` represents built-up land not used as housing, such as hospitals, government offices, schools, and comparable institutional or service buildings.

## 2. Dataset preparation

The original preparation notebook was converted into `experiments/prepare_dataset.py`.

1. Read the source raster and polygon reference.
2. Reproject polygons to the raster CRS when required.
3. Map source text labels to class IDs `0`–`4`.
4. Traverse the raster in non-overlapping `512 × 512` windows.
5. Skip incomplete border chips and all-zero imagery chips.
6. Read the first three bands as RGB.
7. Rasterize the wall-to-wall class polygons onto each chip grid.
8. Save imagery and masks as lossless PNG pairs with matching filenames.

The original training masks classify every chip pixel as one of IDs `0–4`; no training-background class exists. The rasterizer uses `0` as its initialization value because that value is also the valid semantic `Unclassified / Other` class. Wall-to-wall source labels are therefore required for faithful training-mask preparation. Uncovered source pixels would otherwise be indistinguishable from a genuine class `0` label.

## 3. Training experiments

All models use:

- `512 × 512` RGB image/mask pairs;
- ImageNet mean `(0.485, 0.456, 0.406)`;
- ImageNet standard deviation `(0.229, 0.224, 0.225)`;
- horizontal flip, vertical flip, and random 90° rotation for training augmentation;
- cross-entropy loss;
- AdamW optimization;
- an 80/20 train/validation split;
- early stopping after at least 15 epochs with patience 10;
- a maximum of 100 epochs;
- checkpoint selection by lowest validation loss.

| Model | Backbone | Batch size | Learning rate |
|---|---|---:|---:|
| U-Net | ResNet34 | 16 | `1e-4` |
| DeepLabV3+ | ResNet50 | 8 | `1e-4` |
| SegFormer | MiT-B0 | 16 | `6e-5` |

U-Net and DeepLabV3+ use ImageNet-pretrained encoders. SegFormer starts from `nvidia/mit-b0`; its lower-resolution logits are bilinearly upsampled to the mask dimensions before loss and metric calculation.

## 4. Large-raster inference

The shared inference engine is implemented in `src/inference.py`.

### Window geometry

- Tile size: `512 × 512` px
- Default stride: `256` px
- Default overlap: 50% in each axis
- Border handling: reflect padding, with an edge-padding fallback for one-pixel axes in this implementation

For each window, the first three bands are transformed from band-first to RGB channel-last form, scaled by `255`, normalized with ImageNet statistics, and converted to a PyTorch tensor.

### Probability blending

For class $c$ and pixel $p$, let $P_{w,c,p}$ be the softmax probability predicted by window $w$. The blended probability is:

$$
\bar{P}_{c,p} = \frac{1}{N_p}\sum_{w \ni p} P_{w,c,p}
$$

where $N_p$ is the number of windows contributing to pixel $p$. The final class is:

$$
\hat{y}_p = \operatorname*{arg\,max}_{c} \bar{P}_{c,p}
$$

All-zero imagery windows are skipped and uncovered pixels default to class `0`. This implementation detail reuses the semantic `Unclassified / Other` ID; it does not redefine class `0` as training background.

### Geospatial output

The prediction is written as a single-band `uint8` GeoTIFF using the source CRS, affine transform, dimensions, and spatial profile. Class `0` is a real output value rather than a GeoTIFF nodata flag.

## 5. Raster-to-vector conversion

`src/vectorize.py` uses four-connected polygonization. Each feature stores:

- `class_id`;
- `category`;
- polygon geometry.

GeoPackage is the default because it is a single file and avoids several Shapefile limitations. Shapefile remains available through `--vector-format shp`.

## 6. Reference rasterization

Reference polygons are optional. Without a reference, the pipeline still creates the classified raster, vector layer, and prediction figure. With a reference, polygons are reprojected to the image CRS and rasterized to the prediction grid. By default, a pixel receives a class when its center lies within the polygon. `--all-touched` changes this behavior to include every touched pixel and will therefore change the reported metrics.

The class field can contain integer IDs or supported Indonesian/English labels. If no field is specified, the code checks:

1. `class_id`
2. `Dataset_Cl`
3. `Dataset_Class`
4. `keterangan`
5. `category`

## 7. Evaluation metrics

The valid evaluation mask is:

$$
M_p = [y_p \in \{0,1,2,3,4\}]
$$

Therefore, all reported metrics describe **agreement on reference-covered pixels labeled as one of the five model classes**. Pixels with internal reference value `255` are outside the evaluation area.

### Overall accuracy

$$
\mathrm{OA} = \frac{\sum_{p \in M}[\hat{y}_p = y_p]}{|M|}
$$

### Precision, recall, and F1

For each model class $c \in \{0,1,2,3,4\}$:

$$
\mathrm{Precision}_c = \frac{TP_c}{TP_c + FP_c}
$$

$$
\mathrm{Recall}_c = \frac{TP_c}{TP_c + FN_c}
$$

$$
\mathrm{F1}_c = \frac{2\,\mathrm{Precision}_c\,\mathrm{Recall}_c}{\mathrm{Precision}_c + \mathrm{Recall}_c}
$$

Zero denominators are reported as zero.

### Intersection over Union

$$
\mathrm{IoU}_c = \frac{TP_c}{TP_c + FP_c + FN_c}
$$

Mean IoU is the arithmetic mean of class IoUs for IDs `0`–`4`. A class with zero union receives IoU `0` rather than being dropped from the mean.

### Confusion matrix

Rows represent reference labels and columns represent predictions. Labels `0`–`4` are retained in the matrix, including a complete ground-truth row and prediction column for semantic class `0`. Pixels outside the reference geometries are excluded before the matrix is calculated.

## 8. Compatibility visualization

The evaluation figure contains:

1. rasterized ground truth;
2. model prediction;
3. a compatibility map.

Compatibility values are:

- `0`: outside the labeled evaluation area;
- `1`: correct prediction (green);
- `2`: incorrect prediction (red).

## 9. Reproducibility boundary

| Component | Status | Requirement |
|---|---|---|
| Dataset preparation | Portable reference code supplied | Licensed raster and polygons |
| Training | Portable reference code supplied | Prepared chips, compute, compatible environment |
| Model inference | Original runs completed for all three models; portable implementation supplied | Matching checkpoint and RGB GeoTIFF |
| Vectorization | Original runs completed; portable implementation supplied | Prediction raster and geospatial dependencies |
| Evaluation | Original private-scene reports verified; portable implementation supplied | Prediction and vector reference |
| Public one-command demo | Designed around `--demo` | Redistributable sample, reference, and released weights |

The demo command can also run in imagery-only mode with `--skip-evaluation`. This mode requires the demo raster and checkpoint but not `reference_demo.gpkg`.

The portable package passes Python syntax checks and synthetic metric/JSON/CSV/figure checks. The supplied Colab reports confirm complete original inference and evaluation runs for U-Net, DeepLabV3+, and SegFormer. The portable refactor still requires a clean-environment release test with the final public demo assets and released checkpoint files.

The original private-scene evaluation scores are documented in `docs/results.md` and `results/benchmark.csv`. Demo scores must be generated separately because the public sample is a different scene.

Dependency versions are ranges, not a recovered lockfile for the original experiment. Retain the actual successful environment and input/checkpoint revisions when publishing reproducible results.

## 10. Fair model comparison checklist

Before replacing benchmark placeholders, verify that every model uses:

- the identical raster extent and radiometry;
- the identical reference vector revision;
- the identical rasterization rule (`all_touched` setting);
- the same tile size, stride, and input scale;
- the same evaluation mask and class order;
- the best intended checkpoint;
- the same software environment.

Store the final command, `metrics.json`, checkpoint checksum, and environment information with each published result.

## 11. Known limitations

- Training data represents dense residential areas in Jakarta rather than a geographically diverse sample.
- Domain shift can affect predictions in cities or landscapes with different architecture, climate, vegetation, road patterns, sensors, seasons, resolutions, or image processing.
- Class `0` combines many out-of-scope objects into one heterogeneous catch-all class, so a single IoU value can hide substantial variation among water, plantations, open land, and other objects.
- Operational use requires local reference data, quality review, and preferably local fine-tuning.
