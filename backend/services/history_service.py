"""
HistoryService — pagination and filtering of prediction history.
"""

from datetime import date

from sqlalchemy.ext.asyncio import AsyncSession

from backend.repositories.prediction_repository import PredictionRepository
from backend.storage.base_storage import AbstractStorage
from backend.core.logger import get_logger

logger = get_logger(__name__)


class HistoryService:

    def __init__(self, storage: AbstractStorage, db: AsyncSession) -> None:
        self._repo = PredictionRepository(db)
        self._storage = storage

    async def get_history(
        self,
        page: int = 1,
        limit: int = 10,
        date_from: date | None = None,
        date_to: date | None = None,
        plant: str | None = None,
    ) -> dict:
        """
        Fetch paginated prediction history.

        Returns:
            dict compatible with HistoryResponse schema.
        """
        limit = min(limit, 50)  # enforce server-side max
        records, total = await self._repo.get_all(
            page=page, limit=limit,
            date_from=date_from, date_to=date_to,
            plant=plant,
        )

        items = []
        for r in records:
            items.append({
                "id": r.id,
                "status": r.status,
                "plant": r.plant,
                "plant_confidence": r.plant_confidence,
                "disease": r.disease,
                "disease_confidence": r.disease_confidence,
                "inference_time_ms": r.inference_time_ms,
                "image_url": self._storage.get_url(r.image_path),
                "timestamp": r.created_at,
            })

        return {
            "total": total,
            "page": page,
            "limit": limit,
            "has_next": (page * limit) < total,
            "items": items,
        }
