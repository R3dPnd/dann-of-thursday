"""
Metrics endpoint — aggregated Claude Code call statistics.

GET /api/v1/metrics?period=day|week|all
GET /api/v1/metrics/calls-by-day?days=7
GET /api/v1/metrics/by-project
"""

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from app.services import metrics_service

router = APIRouter()


@router.get("", summary="Aggregated Claude call metrics")
async def get_metrics(
    period: str = Query("all", pattern="^(day|week|all)$"),
    session_id: str | None = Query(None),
) -> JSONResponse:
    data = metrics_service.aggregate(period=period, session_id=session_id)
    return JSONResponse(data)


@router.get("/calls-by-day", summary="Claude calls per day (last N days)")
async def calls_by_day(days: int = Query(7, ge=1, le=90)) -> JSONResponse:
    data = metrics_service.calls_by_day(days=days)
    return JSONResponse(data)


@router.get("/by-project", summary="Per-project call breakdown")
async def by_project() -> JSONResponse:
    data = metrics_service.by_project()
    return JSONResponse(data)
