# Training and Evaluation Results

## Result Types

| Result | Data | Metric scope | Purpose |
|---|---|---|---|
| Training validation | Held-out image chips | Classes 0–4, averaged by validation batch | Monitor training |
| Inference evaluation | One labeled research scene | Classes 1–4 across the full labeled area | Compare final predictions |
| Public demo evaluation | Redistributable demo scene | Classes 0–4 across the demo reference coverage | Reproduce the repository workflow |

Scores from these three groups must not be mixed in one benchmark.

All models were trained on dense residential areas in Jakarta. These results do not establish accuracy in geographically or visually different locations.

## Training Reports

The training figures record the original experiment runs.

| Model | Figure | Recorded epochs | Final validation accuracy | Final validation mIoU |
|---|---|---:|---:|---:|
| U-Net | [grafik_training_unet.png](../results/figures/grafik_training_unet.png) | 22 | 75.8% | 51.6% |
| DeepLabV3+ | [grafik_training_deeplab.png](../results/figures/grafik_training_deeplab.png) | 23 | 76.0% | 51.0% |
| SegFormer | [grafik_training_segformer.png](../results/figures/grafik_training_segformer.png) | 46 | 76.1% | 53.8% |

### Interpretation

- Training loss decreases for all three models.
- Validation loss decreases early, then levels off or fluctuates.
- Validation accuracy settles near 76%.
- SegFormer records the highest approximate final validation mIoU.

### Summary

| Model | Overall accuracy | Mean IoU | Macro precision* | Macro recall* | Macro F1* |
|---|---:|---:|---:|---:|---:|
| U-Net | 65.65% | 50.57% | 71.13% | 63.02% | 66.08% |
| DeepLabV3+ | 60.00% | 42.45% | 63.02% | 56.23% | 58.34% |
| SegFormer | **71.20%** | **56.03%** | **74.88%** | **67.98%** | **71.08%** |

\* Macro precision, recall, and F1 are arithmetic means of the four per-class values printed to four decimal places. Small rounding differences may occur relative to calculations from unrounded arrays.

These private-scene numbers remain historical four-class results. New demo output contains five per-class rows and a five-class mean IoU, so direct numeric comparison is not valid.

### U-Net per-class metrics

| Class | Precision | Recall | F1-score | IoU |
|---|---:|---:|---:|---:|
| Residential / mixed-use built-up | 0.8421 | 0.7907 | 0.8156 | 0.6886 |
| Non-residential built-up | 0.4173 | 0.5432 | 0.4720 | 0.3089 |
| Grassland | 0.7661 | 0.6162 | 0.6830 | 0.5187 |
| Urban forest | 0.8195 | 0.5706 | 0.6727 | 0.5069 |

![U-Net evaluation](../results/figures/unet_evaluation_research.png)

### DeepLabV3+ per-class metrics

| Class | Precision | Recall | F1-score | IoU |
|---|---:|---:|---:|---:|
| Residential / mixed-use built-up | 0.7451 | 0.7915 | 0.7676 | 0.6229 |
| Non-residential built-up | 0.3516 | 0.4382 | 0.3901 | 0.2423 |
| Grassland | 0.6321 | 0.5578 | 0.5926 | 0.4211 |
| Urban forest | 0.7920 | 0.4616 | 0.5833 | 0.4117 |

![DeepLabV3+ evaluation](../results/figures/deeplabv3plus_evaluation_research.png)

### SegFormer per-class metrics

| Class | Precision | Recall | F1-score | IoU |
|---|---:|---:|---:|---:|
| Residential / mixed-use built-up | 0.8548 | 0.8231 | 0.8386 | 0.7221 |
| Non-residential built-up | 0.5719 | 0.5540 | 0.5628 | 0.3916 |
| Grassland | 0.7308 | 0.6991 | 0.7146 | 0.5559 |
| Urban forest | 0.8375 | 0.6428 | 0.7273 | 0.5715 |

![SegFormer evaluation](../results/figures/segformer_evaluation_research.png)

## Comparison

SegFormer produces the strongest result in this evaluation:

- highest overall accuracy: `71.20%`;
- highest mean IoU: `56.03%`;
- highest IoU in all four evaluated classes;
- largest improvement over U-Net on the non-residential class.

The non-residential built-up class is the weakest class for all three models. SegFormer raises its IoU to `0.3916`, compared with `0.3089` for U-Net and `0.2423` for DeepLabV3+.

Legacy notebook figures may retain the shorter labels `Residential / mixed-use` and `Non-residential`. In this repository, both labels mean the built-up definitions documented in the main README.

## Data and Publication Boundary

The original evaluation scene cannot be redistributed. The supplied figures contain class masks and compatibility maps from that scene, not the original RGB image.

The public demo uses a different legal sample and a separately digitized reference. Demo metrics must be reported as a new benchmark rather than presented as a reproduction of the private-scene scores.

## Benchmark File

`results/benchmark.csv` stores the original inference summary as fractions from `0` to `1`. For example, `0.7120` represents `71.20%`.
