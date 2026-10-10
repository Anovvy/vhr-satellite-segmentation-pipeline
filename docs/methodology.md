# Methodology & Technical Architecture

This document details the mathematical, algorithmic, and geospatial design behind the VHR Satellite Segmentation Pipeline.

---

## 1. Problem Formulation & Taxonomy

The pipeline performs 5-class semantic segmentation on 30 cm very-high-resolution (VHR) RGB satellite imagery. As defined in the [Main Taxonomy](../README.md#class-definitions), target classes cover residential, non-residential built-up, grassland, and urban forest (IDs `1–4`). 

**Class `0` (`Unclassified / Other`)** serves as a valid semantic catch-all for objects outside target boundaries (water bodies, plantations, bare soil, transport corridors) and is explicitly evaluated in all confusion matrices and IoU metrics.

---

## 2. Dataset Preparation & Augmentation

The preparation workflow (`experiments/prepare_dataset.py`) converts source satellite rasters and wall-to-wall cadastral polygons into chip pairs:

1. **Reprojection**: Vector polygons are reprojected on-the-fly to the raster Coordinate Reference System (CRS).
2. **Chip Generation**: Non-overlapping `512 × 512` pixel windows are extracted across the imagery. Incomplete border chips and unexposed (all-zero) areas are discarded.
3. **Rasterization**: Wall-to-wall polygons are rasterized onto the `512 × 512` grid using zero as the initial value (corresponding to semantic Class 0).
4. **Data Augmentation**: Horizontal flipping, vertical flipping, and random 90° rotations are applied during training.
5. **Normalization**: Standard ImageNet mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`.

---

## 3. Deep Learning Architectures

| Model | Encoder / Backbone | Optimization | Loss Function | Input Size |
|---|---|---|---|---|
| **SegFormer** | MiT-B0 (Hierarchical Transformer) | AdamW (`lr=6e-5`) | Cross-Entropy | `512 × 512` |
| **U-Net** | ResNet34 (ImageNet Pretrained) | AdamW (`lr=1e-4`) | Cross-Entropy | `512 × 512` |
| **DeepLabV3+** | ResNet50 (ImageNet Pretrained) | AdamW (`lr=1e-4`) | Cross-Entropy | `512 × 512` |

- **SegFormer**: Produces multi-level feature maps via overlapping patch embedding and efficient self-attention. Low-resolution logits are bilinearly interpolated back to `512 × 512` prior to loss calculation.
- **U-Net / DeepLabV3+**: Standard skip-connection and Atrous Spatial Pyramid Pooling (ASPP) implementations via `segmentation_models_pytorch`.

---

## 4. Large-Raster Sliding-Window Inference

Large georeferenced rasters exceed GPU/CPU memory if ingested whole. The inference engine (`src/inference.py`) applies overlapping tiled windowing:

- **Tile Geometry**: `512 × 512` px windows with `256` px stride (50% overlap along both axes).
- **Border Handling**: Reflect padding ensures border pixels receive proper convolutional receptive fields.

### Probability Blending
For class $c$ and pixel $p$, let $P_{w,c,p}$ represent the softmax probability predicted by window $w$. The blended probability $\bar{P}_{c,p}$ across all $N_p$ overlapping windows covering pixel $p$ is:

$$
\bar{P}_{c,p} = \frac{1}{N_p}\sum_{w \ni p} P_{w,c,p}
$$

The final predicted class label $\hat{y}_p$ is resolved via argmax:

$$
\hat{y}_p = \operatorname*{arg\,max}_{c \in \{0..4\}} \bar{P}_{c,p}
$$

The resulting prediction is exported as an 8-bit single-band GeoTIFF preserving the original source affine transform, CRS, and pixel dimensions.

---

## 5. Raster-to-Vector Polygonization

Vector polygonization (`src/vectorize.py`) converts discrete pixel clusters into vector geometry using 4-connected topology:
- Extracts outer boundaries and inner rings (holes) into valid OGC geometries.
- Attaches thematic attributes: integer `class_id` and string `category`.
- Saves to **GeoPackage** (`.gpkg`, default) or ESRI Shapefile (`.shp`).
- Optional `--exclude-unclassified-vector` strips Class 0 geometries to reduce vector payload.

---

## 6. Spatial Reference Evaluation

When a ground-truth vector is provided, reference polygons are rasterized to the prediction grid. Pixels outside reference coverage are labeled `255` (nodata) and excluded from metric calculation.

The evaluation mask $M$ is defined as:
$$
M = \{p \mid y_p \in \{0, 1, 2, 3, 4\}\}
$$

### Metric Formulations

#### Overall Accuracy (OA)
$$
\mathrm{OA} = \frac{\sum_{p \in M}[\hat{y}_p = y_p]}{|M|}
$$

#### Per-Class Precision, Recall, and F1
For each class $c \in \{0..4\}$:
$$
\mathrm{Precision}_c = \frac{TP_c}{TP_c + FP_c}, \quad \mathrm{Recall}_c = \frac{TP_c}{TP_c + FN_c}
$$

$$
\mathrm{F1}_c = \frac{2 \cdot \mathrm{Precision}_c \cdot \mathrm{Recall}_c}{\mathrm{Precision}_c + \mathrm{Recall}_c}
$$

#### Intersection over Union (IoU)
$$
\mathrm{IoU}_c = \frac{TP_c}{TP_c + FP_c + FN_c}, \quad \mathrm{mIoU} = \frac{1}{5}\sum_{c=0}^{4} \mathrm{IoU}_c
$$

---

## 7. Compatibility Visualization

The diagnostic plot (`src/visualize.py`) renders a 3-panel side-by-side figure:
1. **Ground Truth**: Rasterized reference labels.
2. **Model Prediction**: Final classified raster.
3. **Compatibility Map**:
   - `0`: Unlabeled area (gray / transparent)
   - `1`: Spatial agreement / correct prediction (green)
   - `2`: Spatial disagreement / error (red)
