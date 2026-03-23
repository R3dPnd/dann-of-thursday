"""
Terminal endpoints.

POST /api/v1/terminals              — create a PTY session for a project
GET  /api/v1/terminals              — list active sessions
DELETE /api/v1/terminals/{id}       — close a session
WS   /api/v1/terminals/{id}/ws     — bidirectional PTY ↔ browser stream
"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from fastapi.websockets import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from app.services import terminal_service
from app.services.terminal_service import get_session, remove_session

router = APIRouter()


class CreateTerminalRequest(BaseModel):
    project_name: str
    rows: int = 24
    cols: int = 80


@router.post("", summary="Create a PTY terminal session")
async def create_terminal(body: CreateTerminalRequest) -> JSONResponse:
    # Resolve project path from config
    from src.mcp_servers.claude_code_server import _find_projects
    projects = _find_projects()
    match = next((p for p in projects if p["name"] == body.project_name), None)
    if match is None:
        raise HTTPException(status_code=404, detail=f"Project '{body.project_name}' not found")

    session = terminal_service.create_session(
        project_name=body.project_name,
        project_path=match["path"],
        rows=body.rows,
        cols=body.cols,
    )
    return JSONResponse(session.to_dict(), status_code=201)


@router.get("", summary="List active terminal sessions")
async def list_terminals() -> JSONResponse:
    return JSONResponse(terminal_service.list_sessions())


@router.delete("/{session_id}", summary="Close a terminal session")
async def close_terminal(session_id: str) -> JSONResponse:
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    remove_session(session_id)
    return JSONResponse({"closed": session_id})


@router.websocket("/{session_id}/ws")
async def terminal_ws(websocket: WebSocket, session_id: str) -> None:
    """Proxy bytes between the PTY and the browser.

    Protocol:
    - Binary frames  → raw terminal data (both directions)
    - Text frames    → JSON control messages:
        { "type": "resize", "rows": N, "cols": M }
    """
    await websocket.accept()

    session = get_session(session_id)
    if session is None:
        await websocket.send_text(json.dumps({"type": "error", "message": "Session not found"}))
        await websocket.close(code=4004)
        return

    loop = asyncio.get_event_loop()

    async def _read_pty() -> None:
        """Forward PTY output → WebSocket (runs in thread executor to avoid blocking)."""
        while True:
            try:
                data: bytes = await loop.run_in_executor(None, session.read, 4096)
                await websocket.send_bytes(data)
            except EOFError:
                # Process exited
                try:
                    await websocket.send_text(json.dumps({"type": "exit"}))
                except Exception:
                    pass
                break
            except Exception:
                break

    read_task = asyncio.create_task(_read_pty())

    try:
        while True:
            msg = await websocket.receive()
            if "bytes" in msg and msg["bytes"]:
                session.write(msg["bytes"])
            elif "text" in msg and msg["text"]:
                try:
                    ctrl = json.loads(msg["text"])
                    if ctrl.get("type") == "resize":
                        rows = int(ctrl.get("rows", 24))
                        cols = int(ctrl.get("cols", 80))
                        session.resize(rows, cols)
                except (json.JSONDecodeError, ValueError):
                    session.write(msg["text"].encode())
    except WebSocketDisconnect:
        pass
    finally:
        # Only cancel the read task — do NOT remove the session here.
        # The session stays alive so the browser can reconnect (e.g. React
        # StrictMode mounts effects twice in development). Cleanup happens
        # via DELETE /terminals/{id} when the tab is explicitly closed.
        read_task.cancel()
