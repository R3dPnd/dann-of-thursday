"""
Metrics endpoint — aggregated turn/latency statistics.

GET /api/v1/metrics?period=session|day|week|all
GET /api/v1/metrics/sessions-by-day?days=7
GET /api/v1/metrics/latency-by-session
"""

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from app.services import metrics_service

router = APIRouter()


@router.get("", summary="Aggregated metrics")
async def get_metrics(
    period: str = Query("all", pattern="^(session|day|week|all)$"),
    session_id: str | None = Query(None),
) -> JSONResponse:
    data = metrics_service.aggregate(period=period, session_id=session_id)
    return JSONResponse(data)


@router.get("/sessions-by-day", summary="Sessions per day (last N days)")
async def sessions_by_day(days: int = Query(7, ge=1, le=90)) -> JSONResponse:
    data = metrics_service.sessions_by_day(days=days)
    return JSONResponse(data)


@router.get("/latency-by-session", summary="Per-session latency breakdown")
async def latency_by_session() -> JSONResponse:
    data = metrics_service.latency_by_session()
    return JSONResponse(data)
