"""
Pydantic schemas for /predict endpoint — request validation and response serialization.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """Response returned to mobile after a successful/unsupported prediction."""

    id: str = Field(..., description="Unique prediction ID (UUID)")
    status: Literal["success", "unsupported", "error"]
    plant: str
    plant_confidence: float | None = None
    disease: str | None = None
    disease_confidence: float | None = None
    inference_time_ms: float
    image_url: str
    timestamp: datetime

    model_config = {"from_attributes": True}


class ErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    message: str
