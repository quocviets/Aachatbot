"""
Model Manager — lazy loading + in-memory cache for all plant classifiers.

Design decisions:
- Models are loaded on first request, then cached for the lifetime of the process.
- Swapping the model architecture only requires changing _build_model().
- Auto-scan: no need to hardcode plant → path mapping. The manager discovers
  available models by scanning the Model/ directory at construction time.
"""

import os
import glob
from typing import Optional

import torch
import torch.nn as nn

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core.config import MODEL_DIR, STAGE1_MODEL_PATH, STAGE2_CLASSES, DEVICE
from Core.utils import get_logger

logger = get_logger(__name__)


class ModelManager:
    """
    Manages loading and caching of Stage 1 and Stage 2 models.

    - Stage 1: single plant-recognition model.
    - Stage 2: one disease-classification model per plant.

    All models share the same architecture (MobileNetV3-Small).
    To use a different backbone, override _build_model().

    Usage:
        manager = ModelManager()
        stage1_model = manager.get_stage1_model()
        stage2_model = manager.get_stage2_model("Apple")
    """

    def __init__(self, model_dir: str = MODEL_DIR) -> None:
        self._model_dir = model_dir
        self._cache: dict[str, nn.Module] = {}

        # Auto-discover available plant models at startup
        self._plant_model_paths: dict[str, str] = self._scan_plant_models()
        logger.info(
            "ModelManager ready. Available plants: %s",
            list(self._plant_model_paths.keys()),
        )

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def get_stage1_model(self) -> nn.Module:
        """Return the Stage 1 plant-recognition model (lazy load + cache)."""
        return self._load_cached("__stage1__", STAGE1_MODEL_PATH, num_classes=self._stage1_num_classes())

    def get_stage2_model(self, plant: str) -> nn.Module:
        """
        Return the Stage 2 disease model for the given plant (lazy load + cache).

        Args:
            plant: Plant name, e.g. "Apple". Case-insensitive.

        Raises:
            KeyError: If no model is found for the given plant.
        """
        plant_key = self._normalize_plant(plant)

        if plant_key not in self._plant_model_paths:
            available = list(self._plant_model_paths.keys())
            raise KeyError(
                f"No model found for plant '{plant}'. "
                f"Available: {available}"
            )

        model_path = self._plant_model_paths[plant_key]
        num_classes = len(STAGE2_CLASSES[plant_key])
        return self._load_cached(plant_key, model_path, num_classes)

    def available_plants(self) -> list[str]:
        """Return a sorted list of plants with registered models."""
        return sorted(self._plant_model_paths.keys())

    def clear_cache(self) -> None:
        """Free all cached models from memory."""
        self._cache.clear()
        logger.info("Model cache cleared.")

    # ──────────────────────────────────────────────
    # Swap-friendly: override this to use a different backbone
    # ──────────────────────────────────────────────

    @staticmethod
    def _build_model(num_classes: int) -> nn.Module:
        """
        Build the model architecture.
        Override or replace this method to swap the backbone (e.g. EfficientNet, ResNet).

        Args:
            num_classes: Number of output classes.

        Returns:
            An un-loaded nn.Module instance.
        """
        # Import here to avoid circular dependencies and keep the swap point obvious
        import sys, os
        # Go up from Inference/ to project root
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if root not in sys.path:
            sys.path.insert(0, root)

        from model import build_mobilenetv3_small  # noqa: PLC0415
        return build_mobilenetv3_small(num_classes)

    # ──────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────

    def _load_cached(self, cache_key: str, model_path: str, num_classes: int) -> nn.Module:
        """Load model from disk if not already cached; return cached copy otherwise."""
        if cache_key in self._cache:
            logger.debug("Cache hit for '%s'", cache_key)
            return self._cache[cache_key]

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info("Loading model '%s' from %s", cache_key, model_path)

        model = self._build_model(num_classes)
        state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()

        self._cache[cache_key] = model
        logger.info("Model '%s' loaded and cached (device=%s)", cache_key, DEVICE)
        return model

    def _scan_plant_models(self) -> dict[str, str]:
        """
        Auto-scan Model/ to build a plant→path mapping.

        Convention: Model/<PlantName>/<PlantName>_classifier.pth
        If a folder has any .pth file, it is registered.
        """
        mapping: dict[str, str] = {}

        if not os.path.isdir(self._model_dir):
            logger.warning("Model directory not found: %s", self._model_dir)
            return mapping

        for entry in os.scandir(self._model_dir):
            if not entry.is_dir():
                continue

            folder_name = entry.name

            # Skip Stage 1 folder — it is handled separately
            if folder_name.lower().startswith("stage"):
                continue

            # Skip plants not in STAGE2_CLASSES (no class list = can't do inference)
            norm = self._normalize_plant(folder_name)
            if norm not in STAGE2_CLASSES:
                logger.debug("Skipping folder '%s' — not in STAGE2_CLASSES", folder_name)
                continue

            pth_files = glob.glob(os.path.join(entry.path, "*.pth"))
            if not pth_files:
                logger.warning("No .pth file found in %s — skipping", entry.path)
                continue

            # Pick the first .pth found (convention: one model per plant folder)
            mapping[norm] = pth_files[0]
            logger.debug("Registered plant '%s' → %s", norm, pth_files[0])

        return mapping

    @staticmethod
    def _normalize_plant(plant: str) -> str:
        """
        Normalize plant name to match STAGE2_CLASSES keys.
        Matches by title-casing the input against existing keys (case-insensitive).
        """
        target = plant.strip().lower()
        for key in STAGE2_CLASSES:
            if key.lower() == target:
                return key
        # Return title-cased as fallback (will miss if not in STAGE2_CLASSES)
        return plant.strip().title()

    @staticmethod
    def _stage1_num_classes() -> int:
        from Core.config import STAGE1_CLASSES  # noqa: PLC0415
        return len(STAGE1_CLASSES)
