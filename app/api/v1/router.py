"""
Main API router that aggregates all endpoint modules
"""
from fastapi import APIRouter
from app.api.v1.endpoints import health, mcp, tools

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(mcp.router, prefix="/mcp", tags=["mcp"])
api_router.include_router(tools.router, prefix="/tools", tags=["tools"])

