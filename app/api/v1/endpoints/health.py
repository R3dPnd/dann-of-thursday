"""
Health check endpoints
"""
import httpx
from fastapi import APIRouter
from app.models.schemas import HealthResponse
from app.core.config import settings

router = APIRouter()

_OLLAMA_URL = "http://localhost:11434/api/tags"


@router.get("", response_model=HealthResponse, summary="Health check")
async def health_check():
    """
    Health check endpoint to verify API is running

    Returns:
        HealthResponse: Service health status
    """
    return HealthResponse(
        status="healthy",
        service=settings.PROJECT_NAME,
        version=settings.VERSION,
    )


@router.get("/ready", summary="Readiness check")
async def readiness_check():
    """
    Readiness check — verifies that upstream dependencies are reachable.

    Returns:
        dict: Per-service readiness status and overall status
    """
    checks: dict = {}

    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(_OLLAMA_URL, timeout=3.0)
        checks["ollama"] = "ok" if r.status_code == 200 else "error"
    except Exception:
        checks["ollama"] = "unreachable"

    overall = "ready" if all(v == "ok" for v in checks.values()) else "degraded"
    return {"status": overall, "checks": checks}


@router.get("/live", summary="Liveness check")
async def liveness_check():
    """
    Liveness check endpoint
    
    Returns:
        dict: Liveness status
    """
    return {"status": "alive"}

