"""
PredictionService — orchestrates the full predict flow.
This is the only class that knows about Storage, AI, and Repository together.
"""

import uuid
import sys
import os
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from backend.repositories.prediction_repository import PredictionRepository
from backend.storage.base_storage import AbstractStorage
from backend.core.exceptions import InferenceError, PlantNotSupportedError
from backend.core.logger import get_logger

logger = get_logger(__name__)

# ── Add project root to path so Inference/ package can be found ─────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from Inference.pipeline import predict_image  # noqa: E402


class PredictionService:

    def __init__(self, storage: AbstractStorage, db: AsyncSession) -> None:
        self._storage = storage
        self._repo = PredictionRepository(db)

    async def predict(
        self,
        file_bytes: bytes,
        original_filename: str,
        plant_type: str | None = None,
    ) -> dict:
        """
        Full prediction flow:
          1. Save image
          2. Run AI inference
          3. Persist result to DB
          4. Return response dict

        Args:
            file_bytes:        Raw bytes of the uploaded image.
            original_filename: Original filename from the client (used for extension).
            plant_type:        Optional plant override — skips Stage 1.

        Returns:
            dict compatible with PredictionResponse schema.

        Raises:
            PlantNotSupportedError: If plant_type is given but not in the supported list.
            InferenceError:         If the AI pipeline raises unexpectedly.
        """
        # ── 1. Generate safe filename (UUID) ────────────────────────────────
        ext = os.path.splitext(original_filename)[-1].lower() or ".jpg"
        safe_filename = f"{uuid.uuid4()}{ext}"

        # ── 2. Save to storage ───────────────────────────────────────────────
        image_path = await self._storage.save(file_bytes, safe_filename)
        image_url = self._storage.get_url(image_path)

        # ── 3. Run AI inference ──────────────────────────────────────────────
        try:
            ai_result: dict = predict_image(image_path, plant_type=plant_type)
        except Exception as exc:
            logger.error("AI inference error: %s", exc)
            await self._storage.delete(image_path)
            raise InferenceError(str(exc)) from exc

        if ai_result.get("status") == "error":
            await self._storage.delete(image_path)
            raise InferenceError(ai_result.get("message", "Unknown inference error"))

        # ── 4. Persist to DB ─────────────────────────────────────────────────
        record_id = str(uuid.uuid4())
        record_data = {
            "id": record_id,
            "image_path": image_path,
            "plant": ai_result.get("plant", "Unknown"),
            "plant_confidence": ai_result.get("plant_confidence"),
            "disease": ai_result.get("disease"),
            "disease_confidence": ai_result.get("disease_confidence"),
            "inference_time_ms": ai_result.get("inference_time_ms", 0.0),
            "status": ai_result.get("status", "success"),
        }
        await self._repo.save(record_data)

        # ── 5. Build response ────────────────────────────────────────────────
        return {
            **record_data,
            "image_url": image_url,
            "timestamp": datetime.utcnow(),
        }
