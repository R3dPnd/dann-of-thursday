"""
Devteam pipeline jobs endpoint — read-only view of background Claude Code
pipelines launched via the devteam MCP module.

GET /api/v1/devteam?project=dann-of-thursday&limit=50&offset=0
"""

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from app.services import devteam_service

router = APIRouter()


@router.get("", summary="Recent devteam pipeline jobs")
async def get_devteam_jobs(
    project: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> JSONResponse:
    jobs, total = devteam_service.list_jobs(project=project, limit=limit, offset=offset)
    return JSONResponse({
        "total": total,
        "offset": offset,
        "limit": limit,
        "jobs": jobs,
    })
