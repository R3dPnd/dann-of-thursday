"""GET /api/v1/history — persisted conversation turn history."""

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("", summary="Conversation history")
async def get_history(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    session_id: str | None = Query(None),
) -> JSONResponse:
    from app.services import history_service
    records = history_service.get_history(limit=limit, offset=offset, session_id=session_id)
    return JSONResponse({"total": len(records), "offset": offset, "limit": limit, "records": records})
