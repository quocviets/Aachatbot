"""
Prediction repository — all database access for predictions lives here.
The service layer never touches SQLAlchemy directly.
"""

import uuid
from datetime import datetime, date
from typing import Any

from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.models import PredictionRecord
from backend.core.logger import get_logger

logger = get_logger(__name__)


class PredictionRepository:

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def save(self, data: dict[str, Any]) -> PredictionRecord:
        """Persist a new prediction record and return it."""
        record = PredictionRecord(
            id=data.get("id", str(uuid.uuid4())),
            image_path=data["image_path"],
            plant=data["plant"],
            plant_confidence=data.get("plant_confidence", 0.0),
            disease=data.get("disease"),
            disease_confidence=data.get("disease_confidence"),
            inference_time_ms=data["inference_time_ms"],
            status=data.get("status", "success"),
        )
        self._db.add(record)
        await self._db.flush()
        logger.info("Prediction saved: id=%s plant=%s disease=%s", record.id, record.plant, record.disease)
        return record

    async def get_all(
        self,
        page: int = 1,
        limit: int = 10,
        date_from: date | None = None,
        date_to: date | None = None,
        plant: str | None = None,
    ) -> tuple[list[PredictionRecord], int]:
        """
        Return paginated predictions with optional filters.
        Returns (records, total_count).
        """
        filters = []

        if date_from:
            filters.append(PredictionRecord.created_at >= datetime.combine(date_from, datetime.min.time()))
        if date_to:
            filters.append(PredictionRecord.created_at <= datetime.combine(date_to, datetime.max.time()))
        if plant:
            filters.append(PredictionRecord.plant == plant)

        where_clause = and_(*filters) if filters else True

        # Total count
        count_stmt = select(func.count()).select_from(PredictionRecord).where(where_clause)
        total = (await self._db.execute(count_stmt)).scalar_one()

        # Paginated records
        offset = (page - 1) * limit
        stmt = (
            select(PredictionRecord)
            .where(where_clause)
            .order_by(PredictionRecord.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        result = await self._db.execute(stmt)
        records = list(result.scalars().all())

        return records, total
