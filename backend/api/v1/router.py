"""
Aggregate all v1 routes into a single router.
"""

from fastapi import APIRouter

from backend.api.v1.routes import predict, history, system

router = APIRouter(prefix="/api/v1")

router.include_router(predict.router)
router.include_router(history.router)
router.include_router(system.router)
