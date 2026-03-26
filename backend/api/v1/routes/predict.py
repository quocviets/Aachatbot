"""
POST /api/v1/predict — upload image, run AI, return prediction result.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile

from backend.core.config import get_settings
from backend.core.dependencies import get_prediction_service
from backend.core.exceptions import FileTooLargeError, InvalidFileTypeError, PlantNotSupportedError
from backend.schemas.predict_schema import PredictionResponse
from backend.services.prediction_service import PredictionService

router = APIRouter()
settings = get_settings()

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Magic bytes for JPEG and PNG — replaces imghdr (removed in Python 3.13)
_MAGIC_BYTES: list[tuple[bytes, str]] = [
    (b"\xff\xd8\xff", "jpeg"),
    (b"\x89PNG\r\n\x1a\n", "png"),
]


def _detect_image_type(data: bytes) -> str | None:
    """Return 'jpeg' or 'png' by inspecting file magic bytes. Returns None if unknown."""
    for magic, img_type in _MAGIC_BYTES:
        if data[:len(magic)] == magic:
            return img_type
    return None



@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Upload image and get plant disease prediction",
    tags=["Prediction"],
)
async def predict(
    file: Annotated[UploadFile, File(description="Leaf image (JPG/PNG, max 5MB)")],
    plant_type: Annotated[str | None, Form(description="Optional: skip Stage 1 detection")] = None,
    service: PredictionService = Depends(get_prediction_service),
) -> PredictionResponse:
    """
    Upload a leaf image and receive a plant disease classification result.

    - **file**: JPG or PNG image, max 5 MB
    - **plant_type**: Optional. If provided, skips Stage 1 (plant detection) — useful
      when the mobile user has already selected the crop type.
    """
    # ── 1. Validate file extension ───────────────────────────────────────────
    filename = file.filename or "upload.jpg"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise InvalidFileTypeError()

    # ── 2. Read bytes ────────────────────────────────────────────────────────
    file_bytes = await file.read()

    # ── 3. Validate file size ────────────────────────────────────────────────
    if len(file_bytes) > settings.max_file_size_bytes:
        raise FileTooLargeError(
            f"File size {len(file_bytes) // 1024} KB exceeds limit of {settings.max_file_size_mb} MB"
        )

    # ── 4. Validate actual MIME type (not just extension) ────────────────────
    detected = _detect_image_type(file_bytes)
    if detected is None:
        raise InvalidFileTypeError("File content is not a valid image")

    # ── 5. Validate optional plant_type ─────────────────────────────────────
    if plant_type is not None:
        import sys, os
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sys.path.insert(0, root)
        from Core.config import SUPPORTED_PLANTS  # noqa: PLC0415
        if plant_type not in SUPPORTED_PLANTS:
            raise PlantNotSupportedError(
                f"'{plant_type}' not supported. Available: {sorted(SUPPORTED_PLANTS)}"
            )

    # ── 6. Delegate to service ───────────────────────────────────────────────
    result = await service.predict(file_bytes, filename, plant_type)
    return PredictionResponse(**result)
