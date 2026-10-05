# Demo Data

The demo contains a small legally redistributable RGB satellite image and an independently prepared land-cover reference layer. The reference file is optional for prediction and required only for evaluation metrics.

## Files

| File | Purpose |
|---|---|
| `imagery_demo.tif` | Georeferenced RGB image used for inference |
| `reference_demo.gpkg` | Reference polygons used for evaluation |

The raster must contain red, green, and blue as the first three bands. The vector layer must contain an integer field named `class_id`.

## Class IDs

| `class_id` | Land-cover class |
|---:|---|
| 0 | Unclassified / Other |
| 1 | Residential / Mixed-use Built-up |
| 2 | Non-residential Built-up |
| 3 | Grassland / Yard|
| 4 | Urban forest, green belt, and urban park |

Class `0` is a semantic catch-all class, not background. It includes water bodies, plantations, open land, and other objects outside classes `1–4`. Class `1` covers built-up land used as housing or residential mixed use. Class `2` covers non-housing built-up land such as hospitals, government offices, and schools.
