"""
Main API router that aggregates all endpoint modules
"""
from fastapi import APIRouter
from app.api.v1.endpoints import events, health, logs, mcp, metrics, projects, state, terminals, tools, voice

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(mcp.router, prefix="/mcp", tags=["mcp"])
api_router.include_router(tools.router, prefix="/tools", tags=["tools"])
api_router.include_router(state.router, prefix="/state", tags=["state"])
api_router.include_router(projects.router, prefix="/projects", tags=["projects"])
api_router.include_router(events.router, prefix="/events", tags=["events"])
api_router.include_router(metrics.router, prefix="/metrics", tags=["metrics"])
api_router.include_router(logs.router, prefix="/logs", tags=["logs"])
api_router.include_router(terminals.router, prefix="/terminals", tags=["terminals"])
api_router.include_router(voice.router, prefix="/voice", tags=["voice"])

