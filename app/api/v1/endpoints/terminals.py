"""
Terminal endpoints.

POST /api/v1/terminals              — create a PTY session for a focus area
                                       (origin "user" — the dashboard's own
                                       "Open Terminal" button)
POST /api/v1/terminals/claude-code  — get-or-create Dann's own Claude Code
                                       session for a project (origin "dann"),
                                       optionally waiting for a response
POST /api/v1/terminals/dann         — get-or-create Dann's own terminal for
                                       a focus area with no project involved
                                       (origin "dann"), same wait behaviour
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
    focus_area: str
    command: str | None = None
    rows: int = 24
    cols: int = 80


class ClaudeCodeTerminalRequest(BaseModel):
    project_name: str
    task: str = ""
    focus_area: str = ""


class DannTerminalRequest(BaseModel):
    focus_area: str
    task: str = ""


def _find_dann_session(*, project: str | None, focus_area: str | None) -> dict | None:
    return next(
        (
            s for s in terminal_service.list_sessions()
            if s["origin"] == "dann" and s["project"] == project and s["focus_area"] == focus_area and s["alive"]
        ),
        None,
    )


def _get_or_create_dann_session(*, path: str, focus_area: str | None, project: str | None) -> tuple[dict, bool]:
    """Returns (session_dict, continued). Never bakes `task` into the spawn
    command — _send_and_capture (called separately, off the event loop)
    writes it, so nothing from the response is missed. A freshly created
    session gets a moment to clear its startup banner/MCP connections
    before anything is sent — writing into it mid-boot lands in the input
    box without being processed."""
    existing = _find_dann_session(project=project, focus_area=focus_area)
    if existing:
        return existing, True
    session = terminal_service.create_session(
        path=path, origin="dann", focus_area=focus_area, project=project,
    )
    session.wait_for_quiet()
    return session.to_dict(), False


def _send_and_capture(session_id: str, task: str) -> tuple[str, bool]:
    session = get_session(session_id)
    if session is None or not task:
        return "", True
    return session.send_and_capture(task)


@router.post("", summary="Create a PTY terminal session")
async def create_terminal(body: CreateTerminalRequest) -> JSONResponse:
    from voice.config import load_config
    from shared.focus_areas_store import focus_area_dir

    cfg = load_config()
    known = {a.get("name") for a in (cfg.get("focus_areas") or []) if a.get("name")}
    if body.focus_area not in known:
        raise HTTPException(status_code=404, detail=f"Unknown focus area '{body.focus_area}'")

    session = terminal_service.create_session(
        path=str(focus_area_dir(body.focus_area)),
        origin="user",
        focus_area=body.focus_area,
        rows=body.rows,
        cols=body.cols,
        command=body.command,
    )
    return JSONResponse(session.to_dict(), status_code=201)


@router.post("/claude-code", summary="Get-or-create Dann's Claude Code session for a project")
async def claude_code_terminal(body: ClaudeCodeTerminalRequest) -> JSONResponse:
    """Reuses the same PTY session across calls for a given project *within
    the same focus area*, so a running Claude Code conversation Dann started
    is the exact thing shown live in the dashboard's terminal pane for that
    work stream — not a fresh, disconnected one each time, and not one
    borrowed from an unrelated conversation about the same project. Only
    ever reuses a session with origin "dann": a terminal the user opened by
    hand is never hijacked to hand Claude Code a task.

    When `task` is set, waits briefly for Claude Code's response (or a quiet
    gap suggesting it's done) and returns it as `output`/`settled`, so the
    caller can summarize instead of just confirming the task was sent."""
    from integrations.servers.claude_code_server import resolve_project

    project = resolve_project(body.project_name)
    if not project:
        raise HTTPException(status_code=404, detail=f"Unknown project '{body.project_name}'")

    focus_area = body.focus_area or None
    session_dict, continued = await asyncio.to_thread(
        _get_or_create_dann_session, path=project["path"], focus_area=focus_area, project=project["name"],
    )
    output, settled = await asyncio.to_thread(_send_and_capture, session_dict["session_id"], body.task)
    return JSONResponse(
        {**session_dict, "continued": continued, "output": output, "settled": settled},
        status_code=200 if continued else 201,
    )


@router.post("/dann", summary="Get-or-create Dann's terminal for a focus area with no project")
async def dann_focus_area_terminal(body: DannTerminalRequest) -> JSONResponse:
    """Same reuse and wait-for-response behaviour as /claude-code, for a
    focus area that isn't tied to any git project — cwd is that focus
    area's own directory (shared/focus_areas_store.py) rather than a repo."""
    from voice.config import load_config
    from shared.focus_areas_store import focus_area_dir

    cfg = load_config()
    known = {a.get("name") for a in (cfg.get("focus_areas") or []) if a.get("name")}
    if body.focus_area not in known:
        raise HTTPException(status_code=404, detail=f"Unknown focus area '{body.focus_area}'")

    session_dict, continued = await asyncio.to_thread(
        _get_or_create_dann_session,
        path=str(focus_area_dir(body.focus_area)), focus_area=body.focus_area, project=None,
    )
    output, settled = await asyncio.to_thread(_send_and_capture, session_dict["session_id"], body.task)
    return JSONResponse(
        {**session_dict, "continued": continued, "output": output, "settled": settled},
        status_code=200 if continued else 201,
    )


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

    Subscribes to the session's fan-out (app/services/terminal_service.py)
    rather than reading the PTY directly — a concurrent send_and_capture
    call (or another browser tab) gets its own independent subscription, so
    neither one steals bytes the other needs.
    """
    await websocket.accept()

    session = get_session(session_id)
    if session is None:
        await websocket.send_text(json.dumps({"type": "error", "message": "Session not found"}))
        await websocket.close(code=4004)
        return

    loop = asyncio.get_running_loop()
    outbox = session.subscribe()

    async def _read_pty() -> None:
        """Forward PTY output → WebSocket (runs in thread executor to avoid blocking)."""
        while True:
            try:
                data = await loop.run_in_executor(None, outbox.get)
            except asyncio.CancelledError:
                raise
            except BaseException:
                break
            if data is None:
                # Process exited
                try:
                    await websocket.send_text(json.dumps({"type": "exit"}))
                except Exception:
                    pass
                break
            try:
                await websocket.send_bytes(data)
            except Exception:
                break

    read_task = asyncio.create_task(_read_pty())

    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break
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
        # Only cancel the read task and unsubscribe — do NOT remove the
        # session here. The session stays alive so the browser can
        # reconnect (e.g. React StrictMode mounts effects twice in
        # development). Cleanup happens via DELETE /terminals/{id} when the
        # tab is explicitly closed.
        read_task.cancel()
        session.unsubscribe(outbox)
