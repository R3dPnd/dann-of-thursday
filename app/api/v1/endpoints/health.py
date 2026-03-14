"""
Health check endpoints
"""
from fastapi import APIRouter
from app.models.schemas import HealthResponse
from app.core.config import settings

router = APIRouter()


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
    Readiness check endpoint
    
    Returns:
        dict: Readiness status
    """
    # TODO: Add checks for database, external services, etc.
    return {"status": "ready"}


@router.get("/live", summary="Liveness check")
async def liveness_check():
    """
    Liveness check endpoint
    
    Returns:
        dict: Liveness status
    """
    return {"status": "alive"}

