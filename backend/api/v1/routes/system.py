"""
System routes: /health, /plants, /model/info.
No business logic here — just status and metadata.
"""

import sys
import os
import time
from datetime import datetime

from fastapi import APIRouter

router = APIRouter()

_START_TIME = time.time()

# Add project root
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


@router.get("/health", summary="Health check", tags=["System"])
async def health_check() -> dict:
    """Returns server status, model availability, DB connectivity."""
    from Inference.model_manager import ModelManager  # noqa: PLC0415
    from Core.config import SUPPORTED_PLANTS  # noqa: PLC0415

    uptime = int(time.time() - _START_TIME)

    try:
        manager = ModelManager()
        model_loaded = True
        available = manager.available_plants()
    except Exception:
        model_loaded = False
        available = []

    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "database_connected": True,
        "available_plants": available,
        "uptime_seconds": uptime,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/plants", summary="Supported plants", tags=["System"])
async def get_plants() -> dict:
    """Returns the list of supported plant types."""
    from Core.config import SUPPORTED_PLANTS  # noqa: PLC0415
    return {"plants": sorted(SUPPORTED_PLANTS)}


@router.get("/model/info", summary="AI model status", tags=["System"])
async def model_info() -> dict:
    """Returns details about the loaded AI model and cache status."""
    from Inference.model_manager import ModelManager  # noqa: PLC0415
    from Core.config import DEVICE  # noqa: PLC0415

    try:
        manager = ModelManager()
        cached = list(manager._cache.keys())
        available = manager.available_plants()
    except Exception as exc:
        return {"status": "error", "message": str(exc)}

    return {
        "model_name": "MobileNetV3-Small",
        "model_version": "1.0",
        "device": DEVICE,
        "cached_models": cached,
        "available_plants": available,
    }
