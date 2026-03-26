"""
Dependency providers for FastAPI Depends().
All shared instances are created once here — no global state scattered around.
"""

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.connection import get_db
from backend.storage.local_storage import LocalStorage
from backend.storage.base_storage import AbstractStorage
from backend.services.prediction_service import PredictionService
from backend.services.history_service import HistoryService


def get_storage() -> AbstractStorage:
    """
    Storage provider — swap LocalStorage → FirebaseStorage here only.
    """
    return LocalStorage()


async def get_prediction_service(
    storage: AbstractStorage = Depends(get_storage),
    db: AsyncSession = Depends(get_db),
) -> PredictionService:
    return PredictionService(storage=storage, db=db)


async def get_history_service(
    storage: AbstractStorage = Depends(get_storage),
    db: AsyncSession = Depends(get_db),
) -> HistoryService:
    return HistoryService(storage=storage, db=db)
