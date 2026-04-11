"""Voice control endpoints — trigger, enable, and disable listening."""

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


@router.post("/enable", summary="Enable wake word listening")
async def enable_voice() -> JSONResponse:
    from app.main import orchestrator
    if orchestrator is None or not orchestrator._running:
        raise HTTPException(status_code=503, detail="Orchestrator not running.")
    orchestrator.enable_listening()
    return JSONResponse({"listening": True})


@router.post("/disable", summary="Disable wake word listening")
async def disable_voice() -> JSONResponse:
    from app.main import orchestrator
    if orchestrator is None or not orchestrator._running:
        raise HTTPException(status_code=503, detail="Orchestrator not running.")
    orchestrator.disable_listening()
    return JSONResponse({"listening": False})
