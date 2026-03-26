"""
Pipeline — the single public entry point for inference.

Usage:
    from Inference.pipeline import predict_image

    result = predict_image("path/to/leaf.jpg")
    # or skip Stage 1 if you already know the plant:
    result = predict_image("path/to/leaf.jpg", plant_type="Apple")

This is the only function that a FastAPI / Flask route (or mobile backend) needs to call.
"""

import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core.config import SUPPORTED_PLANTS
from Core.utils import get_logger
from Inference.preprocess import ImagePreprocessor
from Inference.model_manager import ModelManager
from Inference.predictor import Stage1Predictor, Stage2Predictor
from Inference.postprocess import ResultFormatter

logger = get_logger(__name__)

# ──────────────────────────────────────────────
# Module-level singletons (created once, reused across requests)
# ──────────────────────────────────────────────
_preprocessor = ImagePreprocessor()
_model_manager = ModelManager()
_stage1_predictor = Stage1Predictor()
_stage2_predictor = Stage2Predictor()
_formatter = ResultFormatter()


def predict_image(
    image_path: str,
    plant_type: str | None = None,
) -> dict:
    """
    Run the full 2-stage plant disease inference pipeline.

    Args:
        image_path: Path to the input image (JPG, PNG, etc.)
        plant_type: Optional. If provided, skip Stage 1 and use this plant directly.
                    Must be one of the supported plants (e.g. "Apple", "Corn").
                    Useful for a mobile app where the user has already selected the plant.

    Returns:
        dict with keys:
            status           : "success" | "unsupported" | "error"
            plant            : detected or provided plant name
            plant_confidence : Stage 1 confidence (None if plant_type was given)
            disease          : predicted disease name
            disease_confidence: Stage 2 confidence
            inference_time_ms: total wall-clock time in milliseconds

    Example:
        >>> predict_image("images/apple_leaf.jpg")
        {
            "status": "success",
            "plant": "Apple",
            "plant_confidence": 0.978,
            "disease": "Apple_scab",
            "disease_confidence": 0.943,
            "inference_time_ms": 115.4
        }

        >>> predict_image("images/apple_leaf.jpg", plant_type="Apple")
        {
            "status": "success",
            "plant": "Apple",
            "plant_confidence": 1.0,
            "disease": "Apple_scab",
            "disease_confidence": 0.943,
            "inference_time_ms": 88.2
        }
    """
    t_start = time.perf_counter()

    try:
        # ── 1. Preprocess ──────────────────────────────
        tensor = _preprocessor.load_and_transform(image_path)

        # ── 2. Stage 1 — Plant Recognition ────────────
        if plant_type is not None:
            # Caller already knows the plant; skip Stage 1
            plant = plant_type.strip()
            plant_confidence = 1.0
            logger.info("Plant type provided directly: %s", plant)
        else:
            stage1_model = _model_manager.get_stage1_model()
            plant, plant_confidence = _stage1_predictor.predict(stage1_model, tensor)
            logger.info("Stage 1 result: plant=%s conf=%.4f", plant, plant_confidence)

        # ── 3. Check if plant is supported ────────────
        if plant.lower() == "other" or plant not in SUPPORTED_PLANTS:
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            return _formatter.format_unsupported(plant, plant_confidence, elapsed_ms)

        # ── 4. Stage 2 — Disease Classification ───────
        stage2_model = _model_manager.get_stage2_model(plant)
        disease, disease_confidence = _stage2_predictor.predict(stage2_model, tensor, plant)
        logger.info("Stage 2 result: disease=%s conf=%.4f", disease, disease_confidence)

        # ── 5. Format result ───────────────────────────
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        return _formatter.format_success(
            plant=plant,
            plant_confidence=plant_confidence,
            disease=disease,
            disease_confidence=disease_confidence,
            inference_time_ms=elapsed_ms,
        )

    except FileNotFoundError as exc:
        logger.error("Image not found: %s", exc)
        return _formatter.format_error(str(exc))

    except KeyError as exc:
        logger.error("Unknown plant: %s", exc)
        return _formatter.format_error(str(exc))

    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error during inference: %s", exc)
        return _formatter.format_error(f"Inference failed: {exc}")
