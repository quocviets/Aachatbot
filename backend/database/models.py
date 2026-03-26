"""
SQLAlchemy ORM models — database table definitions.
"""

import uuid
from datetime import datetime

from sqlalchemy import String, Float, DateTime, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    image_path: Mapped[str] = mapped_column(String(512), nullable=False)
    plant: Mapped[str] = mapped_column(String(64), nullable=False)
    plant_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    disease: Mapped[str | None] = mapped_column(String(128), nullable=True)
    disease_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    inference_time_ms: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="success")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<PredictionRecord id={self.id} plant={self.plant} disease={self.disease}>"
