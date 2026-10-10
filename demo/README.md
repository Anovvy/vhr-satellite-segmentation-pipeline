# Demo Datasets

Redistributable sample assets for testing the containerized inference pipeline:

| File | Format | Spatial CRS | Description |
|---|---|---|---|
| `imagery_demo.tif` | GeoTIFF | EPSG:32648 | 30 cm VHR RGB satellite raster (Bands 1–3: Red, Green, Blue) |
| `reference_demo.gpkg` | GeoPackage | EPSG:32648 | Ground-truth reference polygons with `class_id` (`0–4`) and `category` attributes |

- **Inference Only**: Pass `--demo --skip-evaluation` to predict without loading the reference layer.
- **Evaluation Mode**: Default `--demo` evaluates metrics against `reference_demo.gpkg`.
- The class taxonomy and definitions are documented in the [Main README](../README.md#class-taxonomy).
