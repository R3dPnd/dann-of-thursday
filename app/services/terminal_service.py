"""
PTY terminal session management.

Each session spawns ``claude`` in a project directory inside a pseudo-terminal
and exposes read/write/resize operations. The FastAPI WebSocket endpoint proxies
bytes between the browser (xterm.js) and the PTY.
"""

from __future__ import annotations

import fcntl
import os
import struct
import termios
import threading
import uuid
from typing import Any

import ptyprocess

# session_id → TerminalSession
_sessions: dict[str, "TerminalSession"] = {}
_lock = threading.Lock()


class TerminalSession:
    def __init__(self, session_id: str, project_name: str, project_path: str) -> None:
        self.session_id = session_id
        self.project_name = project_name
        self.project_path = project_path
        self._proc: ptyprocess.PtyProcess | None = None

    def start(self, rows: int = 24, cols: int = 80) -> None:
        env = {**os.environ, "TERM": "xterm-256color", "COLORTERM": "truecolor"}
        self._proc = ptyprocess.PtyProcess.spawn(
            ["claude"],
            cwd=self.project_path,
            dimensions=(rows, cols),
            env=env,
        )

    def read(self, size: int = 4096) -> bytes:
        """Blocking read from the PTY. Raises EOFError when the process exits."""
        if self._proc is None:
            raise EOFError("not started")
        return self._proc.read(size)

    def write(self, data: bytes) -> None:
        if self._proc is not None and self._proc.isalive():
            self._proc.write(data)

    def resize(self, rows: int, cols: int) -> None:
        if self._proc is not None and self._proc.isalive():
            self._proc.setwinsize(rows, cols)

    def close(self) -> None:
        if self._proc is not None and self._proc.isalive():
            self._proc.terminate(force=True)

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.isalive()

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "project_name": self.project_name,
            "project_path": self.project_path,
            "alive": self.alive,
        }


def create_session(project_name: str, project_path: str, rows: int = 24, cols: int = 80) -> TerminalSession:
    session_id = str(uuid.uuid4())
    session = TerminalSession(session_id, project_name, project_path)
    session.start(rows=rows, cols=cols)
    with _lock:
        _sessions[session_id] = session
    return session


def get_session(session_id: str) -> TerminalSession | None:
    with _lock:
        return _sessions.get(session_id)


def remove_session(session_id: str) -> None:
    with _lock:
        session = _sessions.pop(session_id, None)
    if session:
        session.close()


def list_sessions() -> list[dict[str, Any]]:
    with _lock:
        return [s.to_dict() for s in _sessions.values()]
