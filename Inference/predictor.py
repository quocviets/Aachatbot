"""
Predictors for Stage 1 (plant recognition) and Stage 2 (disease classification).
Each predictor is stateless — it receives a tensor and a loaded model, returns a result.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core.config import STAGE1_CLASSES, STAGE2_CLASSES
from Core.utils import get_logger

logger = get_logger(__name__)


class Stage1Predictor:
    """
    Identifies the plant type from an image tensor.

    Returns (plant_name, confidence) where plant_name is one of STAGE1_CLASSES
    (including "Other" for unsupported images).
    """

    def predict(self, model: nn.Module, tensor: torch.Tensor) -> tuple[str, float]:
        """
        Run Stage 1 inference.

        Args:
            model:  Loaded Stage 1 nn.Module in eval mode.
            tensor: Preprocessed image tensor, shape [1, 3, H, W].

        Returns:
            (plant_name, confidence) — e.g. ("Apple", 0.97)
        """
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()

        plant = STAGE1_CLASSES[pred_idx]
        confidence = float(probs[0][pred_idx].item())

        logger.debug("Stage1 prediction: plant=%s confidence=%.4f", plant, confidence)
        return plant, confidence


class Stage2Predictor:
    """
    Classifies the disease of a specific plant from an image tensor.

    Returns (disease_name, confidence).
    """

    def predict(
        self,
        model: nn.Module,
        tensor: torch.Tensor,
        plant: str,
    ) -> tuple[str, float]:
        """
        Run Stage 2 inference.

        Args:
            model:  Loaded Stage 2 nn.Module for the given plant, in eval mode.
            tensor: Preprocessed image tensor, shape [1, 3, H, W].
            plant:  Plant name matching a key in STAGE2_CLASSES.

        Returns:
            (disease_name, confidence) — e.g. ("Apple_scab", 0.94)

        Raises:
            KeyError: If plant is not in STAGE2_CLASSES.
        """
        if plant not in STAGE2_CLASSES:
            raise KeyError(f"Plant '{plant}' not in STAGE2_CLASSES.")

        class_names = STAGE2_CLASSES[plant]

        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()

        disease = class_names[pred_idx]
        confidence = float(probs[0][pred_idx].item())

        logger.debug(
            "Stage2 prediction: plant=%s disease=%s confidence=%.4f",
            plant, disease, confidence,
        )
        return disease, confidence
