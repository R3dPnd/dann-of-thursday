"""
Main API router that aggregates all endpoint modules
"""
from fastapi import APIRouter
from app.api.v1.endpoints import chat, devteam, events, focus_areas, health, history, logs, metrics, modules, notes, projects, prompt_builder, runs, state, system, terminals, tools, voice

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(tools.router, prefix="/tools", tags=["tools"])
api_router.include_router(state.router, prefix="/state", tags=["state"])
api_router.include_router(projects.router, prefix="/projects", tags=["projects"])
api_router.include_router(modules.router, prefix="/modules", tags=["modules"])
api_router.include_router(focus_areas.router, prefix="/focus-areas", tags=["focus-areas"])
api_router.include_router(notes.router, prefix="/notes", tags=["notes"])
api_router.include_router(events.router, prefix="/events", tags=["events"])
api_router.include_router(metrics.router, prefix="/metrics", tags=["metrics"])
api_router.include_router(logs.router, prefix="/logs", tags=["logs"])
api_router.include_router(terminals.router, prefix="/terminals", tags=["terminals"])
api_router.include_router(voice.router, prefix="/voice", tags=["voice"])
api_router.include_router(runs.router, prefix="/runs", tags=["runs"])
api_router.include_router(prompt_builder.router, prefix="/prompt-builder", tags=["prompt-builder"])
api_router.include_router(history.router, prefix="/history", tags=["history"])
api_router.include_router(devteam.router, prefix="/devteam", tags=["devteam"])
api_router.include_router(system.router, prefix="/system", tags=["system"])

