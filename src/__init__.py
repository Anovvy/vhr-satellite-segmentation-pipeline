"""Reusable inference and evaluation modules for VHR land-cover segmentation."""

CLASS_NAMES = {
    0: "Unclassified / Other",
    1: "Residential / Mixed-use Built-up",
    2: "Non-residential Built-up",
    3: "Grassland",
    4: "Urban Forest / Green Belt / Urban Park",
}

NUM_CLASSES = len(CLASS_NAMES)

# Internal value used only for pixels outside vector-reference coverage.
# Model predictions and semantic class labels remain limited to IDs 0-4.
REFERENCE_NODATA = 255
