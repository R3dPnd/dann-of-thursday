"""GET /api/v1/state — current orchestrator state snapshot."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("", summary="Current orchestrator state")
async def get_state() -> JSONResponse:
    """Return the live mode, active project, session ID, and uptime."""
    from app.main import orchestrator  # imported here to avoid circular import at startup
    if orchestrator is None:
        return JSONResponse({
            "mode": "idle",
            "project": None,
            "session_id": None,
            "running": False,
            "uptime_s": 0,
        })
    return JSONResponse(orchestrator.snapshot())
