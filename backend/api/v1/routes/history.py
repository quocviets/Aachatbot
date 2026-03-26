"""
GET /api/v1/history — paginated prediction history with optional filters.
"""

from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from backend.core.dependencies import get_history_service
from backend.schemas.history_schema import HistoryResponse
from backend.services.history_service import HistoryService

router = APIRouter()


@router.get(
    "/history",
    response_model=HistoryResponse,
    summary="Get prediction history",
    tags=["History"],
)
async def get_history(
    page: Annotated[int, Query(ge=1, description="Page number")] = 1,
    limit: Annotated[int, Query(ge=1, le=50, description="Items per page (max 50)")] = 10,
    date_from: Annotated[date | None, Query(description="Filter from date YYYY-MM-DD")] = None,
    date_to: Annotated[date | None, Query(description="Filter to date YYYY-MM-DD")] = None,
    plant: Annotated[str | None, Query(description="Filter by plant name")] = None,
    service: HistoryService = Depends(get_history_service),
) -> HistoryResponse:
    """
    Returns paginated prediction history, newest first.

    - **page**: Page number (default: 1)
    - **limit**: Items per page, max 50 (default: 10)
    - **date_from / date_to**: Optional date range filter
    - **plant**: Optional plant name filter (e.g. `Apple`, `Corn`)
    """
    result = await service.get_history(
        page=page, limit=limit,
        date_from=date_from, date_to=date_to,
        plant=plant,
    )
    return HistoryResponse(**result)
