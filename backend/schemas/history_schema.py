"""
Pydantic schemas for /history endpoint.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class HistoryItem(BaseModel):
    id: str
    status: str
    plant: str
    plant_confidence: float | None = None
    disease: str | None = None
    disease_confidence: float | None = None
    inference_time_ms: float
    image_url: str
    timestamp: datetime

    model_config = {"from_attributes": True}


class HistoryResponse(BaseModel):
    total: int
    page: int
    limit: int
    has_next: bool
    items: list[HistoryItem]
