"""
Core configuration for the Plant Disease Classification system.
All paths are resolved relative to this file — no hardcoding needed.
"""

import os
import torch

# ──────────────────────────────────────────────
# BASE PATHS
# ──────────────────────────────────────────────

# Root of the project (parent of Core/)
BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR: str = os.path.join(BASE_DIR, "Model")

STAGE1_MODEL_PATH: str = os.path.join(MODEL_DIR, "Stage 1", "stage1_plant_classifier.pth")

# ──────────────────────────────────────────────
# DEVICE
# ──────────────────────────────────────────────

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# ──────────────────────────────────────────────
# IMAGE SETTINGS
# ──────────────────────────────────────────────

IMAGE_SIZE: int = 224

# ──────────────────────────────────────────────
# STAGE 1 — Plant Recognition
# ──────────────────────────────────────────────

STAGE1_CLASSES: list[str] = [
    "Corn",
    "Other",
    "Rice_leaf",
    "Apple",
    "Grape",
    "Tomato",
]

# ──────────────────────────────────────────────
# STAGE 2 — Disease Classification per Plant
#
# Key   : matches the folder name under Model/ AND Stage1 class name (case-insensitive)
# Value : ordered list of disease class names used during training
#
# To add a new plant: create Model/Mango/ with Mango_classifier.pth,
# then add an entry here. No other code changes needed.
# ──────────────────────────────────────────────

STAGE2_CLASSES: dict[str, list[str]] = {
    "Apple": [
        "Apple_scab",
        "Black_rot",
        "Cedar_apple_rust",
        "Healthy",
    ],
    "Corn": [
        "Blight",
        "Common_Rust",
        "Gray_Leaf_Spot",
        "Healthy",
    ],
    "Grape": [
        "Black_rot",
        "Esca",
        "Leaf_blight",
        "Healthy",
    ],
    "Rice_leaf": [
        "Bacterial_Leaf_Blight",
        "Healthy",
        "Narrow_Brown_Spot",
        "Neck_Blast",
    ],
    "Tomato": [
        "Early_blight",
        "Late_blight",
        "Leaf_Mold",
        "Healthy",
    ],
}

# Plants that are not "Other" in Stage 1
SUPPORTED_PLANTS: set[str] = set(STAGE2_CLASSES.keys())
