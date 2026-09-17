"""Convert a classified raster array to geospatial polygons."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
from rasterio.features import shapes
from shapely.geometry import shape

from . import CLASS_NAMES
from .inference import RasterMetadata


def vectorize_prediction(
    prediction: np.ndarray,
    metadata: RasterMetadata,
    output_path: str | Path,
    *,
    include_unclassified: bool = True,
) -> Path:
    """Polygonize connected class regions and write GPKG or Shapefile output."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    valid_mask = None if include_unclassified else prediction != 0
    records = []
    for geometry, value in shapes(
        prediction.astype(np.uint8),
        mask=valid_mask,
        transform=metadata.transform,
        connectivity=4,
    ):
        class_id = int(value)
        records.append(
            {
                "class_id": class_id,
                "category": CLASS_NAMES.get(class_id, f"Class {class_id}"),
                "geometry": shape(geometry),
            }
        )

    if not records:
        raise ValueError("No vector geometries were produced from the prediction.")

    frame = gpd.GeoDataFrame(records, geometry="geometry", crs=metadata.crs)
    suffix = output_path.suffix.lower()
    if suffix == ".gpkg":
        frame.to_file(output_path, layer="prediction", driver="GPKG")
    elif suffix == ".shp":
        frame.to_file(output_path, driver="ESRI Shapefile")
    else:
        raise ValueError("Vector output must use the .gpkg or .shp extension.")
    return output_path

