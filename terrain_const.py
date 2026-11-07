"""
terrain_const.py
────────────────
Torch-free shared constants for the 6 terrain classes.
Importable anywhere — including the lean Railway backend, which has no
torch installed (it calls the HF Space for segmentation instead).
"""

import numpy as np

CLASS_NAMES = [
    "Dense_Vegetation", "Dry_Vegetation", "Ground_Objects",
    "Rocks", "Landscape", "Sky"
]
NUM_CLASSES = len(CLASS_NAMES)

# Traversability score per class index (0-1, higher = easier to cross)
TRAVERSABILITY = {
    0: 0.45,   # Dense_Vegetation
    1: 0.75,   # Dry_Vegetation
    2: 0.30,   # Ground_Objects
    3: 0.15,   # Rocks
    4: 0.95,   # Landscape
    5: 0.00,   # Sky
}

# Risk level per class index
RISK = {
    0: "MED",
    1: "LOW",
    2: "MED-HIGH",
    3: "HIGH",
    4: "LOW",
    5: "N/A",
}

# RGB color per class index for masks
PALETTE = np.array([
    [34,  139,  34],   # Dense_Vegetation  → forest green
    [210, 180, 140],   # Dry_Vegetation    → tan
    [139,  90,  43],   # Ground_Objects    → brown
    [128, 128, 128],   # Rocks             → gray
    [205, 133,  63],   # Landscape         → sandy
    [135, 206, 235],   # Sky               → sky blue
], dtype=np.uint8)
