"""
Run configuration endpoints.

POST   /api/v1/runs/{project_name}/start   — start the project's run command
POST   /api/v1/runs/{project_name}/stop    — stop the running process
GET    /api/v1/runs/{project_name}/status  — current status
WS     /api/v1/runs/{project_name}/ws      — stream stdout/stderr
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from fastapi.websockets import WebSocket, WebSocketDisconnect

from app.services import run_service

router = APIRouter()


def _load_run_configs() -> dict[str, dict]:
    """Return {project_name: {path, run}} for projects that have a run command."""
    from src.config import load_config
    cfg = load_config()
    result = {}
    for p in cfg.get("projects", []):
        if "run" in p:
            result[p["name"]] = {"path": p["path"], "run": p["run"]}
    return result


def _resolve_project(project_name: str) -> dict:
    configs = _load_run_configs()
    if project_name not in configs:
        raise HTTPException(
            status_code=404,
            detail=f"No run configuration found for project '{project_name}'",
        )
    return configs[project_name]


@router.post("/{project_name}/start", summary="Start project run command")
async def start_run(project_name: str) -> JSONResponse:
    config = _resolve_project(project_name)
    session = run_service.get_or_create(project_name, config["path"], config["run"])
    if session.alive:
        return JSONResponse({"status": "already_running", **session.to_dict()})
    session.start()
    return JSONResponse({"status": "started", **session.to_dict()})


@router.post("/{project_name}/stop", summary="Stop project run command")
async def stop_run(project_name: str) -> JSONResponse:
    session = run_service.get_session(project_name)
    if session is None or not session.alive:
        raise HTTPException(status_code=404, detail="No running session for this project")
    session.stop()
    return JSONResponse({"status": "stopped", "project_name": project_name})


@router.get("/{project_name}/status", summary="Get run status")
async def run_status(project_name: str) -> JSONResponse:
    session = run_service.get_session(project_name)
    if session is None:
        config = _resolve_project(project_name)  # raises 404 if no run config
        return JSONResponse({"project_name": project_name, "alive": False, "exit_code": None, "command": config["run"]})
    return JSONResponse(session.to_dict())


@router.get("", summary="List all run sessions")
async def list_runs() -> JSONResponse:
    return JSONResponse(run_service.list_sessions())


@router.websocket("/{project_name}/ws")
async def run_ws(websocket: WebSocket, project_name: str) -> None:
    """Stream stdout/stderr of the project's run process."""
    await websocket.accept()

    session = run_service.get_session(project_name)
    if session is None:
        await websocket.send_text("[no active run session]\n")
        await websocket.close(code=4004)
        return

    # Send buffered output first
    for line in session.get_buffer():
        try:
            await websocket.send_text(line)
        except Exception:
            return

    q = session.subscribe()
    try:
        while True:
            try:
                line = await asyncio.wait_for(q.get(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send keepalive
                try:
                    await websocket.send_text("")
                except Exception:
                    break
                continue

            if line is None:
                # Process exited
                await websocket.send_text("\n[process exited]\n")
                break
            try:
                await websocket.send_text(line)
            except Exception:
                break
    except WebSocketDisconnect:
        pass
    finally:
        session.unsubscribe(q)
