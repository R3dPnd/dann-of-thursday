"""
Per-project run session management.

Each project may define a `run` command in config.yaml. This service
starts/stops those commands as subprocesses, buffers their output, and
broadcasts new lines to any WebSocket subscribers.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import threading
from typing import Any

# project_name → RunSession
_runs: dict[str, "RunSession"] = {}
_lock = threading.Lock()

_MAX_BUFFER = 500


class RunSession:
    def __init__(self, project_name: str, project_path: str, command: str) -> None:
        self.project_name = project_name
        self.project_path = project_path
        self.command = command
        self._proc: subprocess.Popen | None = None
        self._buffer: list[str] = []
        self._buffer_lock = threading.Lock()
        self._queues: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = []
        self._queues_lock = threading.Lock()
        self._exit_code: int | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self.alive:
            return
        self._exit_code = None
        self._proc = subprocess.Popen(
            self.command,
            cwd=self.project_path,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ},
        )
        threading.Thread(target=self._read, daemon=True, name=f"run:{self.project_name}").start()

    def stop(self) -> None:
        if self._proc and self.alive:
            self._proc.terminate()

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # ── Output pipeline ───────────────────────────────────────────────────────

    def _read(self) -> None:
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            with self._buffer_lock:
                self._buffer.append(line)
                if len(self._buffer) > _MAX_BUFFER:
                    self._buffer = self._buffer[-_MAX_BUFFER:]
            self._broadcast(line)
        self._exit_code = self._proc.wait()
        self._broadcast(None)  # EOF sentinel

    def _broadcast(self, item: str | None) -> None:
        with self._queues_lock:
            dead = []
            for loop, q in self._queues:
                try:
                    loop.call_soon_threadsafe(q.put_nowait, item)
                except Exception:
                    dead.append((loop, q))
            for entry in dead:
                self._queues.remove(entry)

    def subscribe(self) -> asyncio.Queue:
        loop = asyncio.get_event_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=500)
        with self._queues_lock:
            self._queues.append((loop, q))
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._queues_lock:
            self._queues = [(l, qq) for l, qq in self._queues if qq is not q]

    def get_buffer(self) -> list[str]:
        with self._buffer_lock:
            return list(self._buffer)

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_name": self.project_name,
            "command": self.command,
            "alive": self.alive,
            "exit_code": self._exit_code,
        }


# ── Public API ────────────────────────────────────────────────────────────────

def get_or_create(project_name: str, project_path: str, command: str) -> RunSession:
    with _lock:
        if project_name not in _runs:
            _runs[project_name] = RunSession(project_name, project_path, command)
        return _runs[project_name]


def get_session(project_name: str) -> RunSession | None:
    with _lock:
        return _runs.get(project_name)


def list_sessions() -> list[dict]:
    with _lock:
        return [s.to_dict() for s in _runs.values()]
