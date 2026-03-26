"""
Result formatting for inference output.
Produces a clean, API-ready dict suitable for JSON serialization (mobile app / REST API).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core.utils import get_logger

logger = get_logger(__name__)


class ResultFormatter:
    """
    Formats raw prediction values into a structured response dict.

    The output schema is stable — mobile apps and REST APIs can rely on this format.
    Adding extra fields here does NOT break the callers (additive changes only).
    """

    def format_success(
        self,
        plant: str,
        plant_confidence: float,
        disease: str,
        disease_confidence: float,
        inference_time_ms: float,
    ) -> dict:
        """
        Build a successful prediction result.

        Returns:
            {
                "status": "success",
                "plant": "Apple",
                "plant_confidence": 0.97,
                "disease": "Apple_scab",
                "disease_confidence": 0.94,
                "inference_time_ms": 120.5
            }
        """
        result = {
            "status": "success",
            "plant": plant,
            "plant_confidence": round(plant_confidence, 4),
            "disease": disease,
            "disease_confidence": round(disease_confidence, 4),
            "inference_time_ms": round(inference_time_ms, 2),
        }
        logger.debug("Result formatted: %s", result)
        return result

    def format_unsupported(
        self,
        plant: str,
        plant_confidence: float,
        inference_time_ms: float,
    ) -> dict:
        """
        Build a result for when Stage 1 returns 'Other' (unsupported plant).

        Returns:
            {
                "status": "unsupported",
                "plant": "Other",
                "plant_confidence": 0.88,
                "disease": null,
                "disease_confidence": null,
                "inference_time_ms": 45.2
            }
        """
        return {
            "status": "unsupported",
            "plant": plant,
            "plant_confidence": round(plant_confidence, 4),
            "disease": None,
            "disease_confidence": None,
            "inference_time_ms": round(inference_time_ms, 2),
        }

    def format_error(self, message: str) -> dict:
        """
        Build an error result dict.

        Returns:
            {
                "status": "error",
                "message": "..."
            }
        """
        return {
            "status": "error",
            "message": message,
        }
