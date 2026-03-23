"""
Logs endpoint — queryable log entries.

GET /api/v1/logs?level=DEBUG&module=orchestrator&search=text&limit=200&offset=0
"""

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from app.services import log_service

router = APIRouter()

_VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


@router.get("", summary="Paginated log entries")
async def get_logs(
    level: str = Query("DEBUG"),
    module: str | None = Query(None),
    search: str | None = Query(None),
    limit: int = Query(200, ge=1, le=2000),
    offset: int = Query(0, ge=0),
) -> JSONResponse:
    lvl = level.upper()
    if lvl not in _VALID_LEVELS:
        lvl = "DEBUG"
    entries = log_service.get_entries(
        level=lvl,
        module=module,
        search=search,
        limit=limit,
        offset=offset,
    )
    return JSONResponse({
        "total": log_service.entry_count(),
        "offset": offset,
        "limit": limit,
        "entries": entries,
    })
