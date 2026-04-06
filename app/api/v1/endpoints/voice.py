"""POST /api/v1/voice/trigger — trigger a voice session from the UI."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter()


@router.post("/trigger", summary="Trigger a voice session")
async def trigger_voice() -> JSONResponse:
    """Simulate a wake word detection to start a full multi-turn voice session."""
    from app.main import orchestrator
    if orchestrator is None or not orchestrator._running:
        raise HTTPException(
            status_code=503,
            detail="Orchestrator not running. Start the server without NO_VOICE=1.",
        )
    orchestrator._on_wake()
    return JSONResponse({"triggered": True})
