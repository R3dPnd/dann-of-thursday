"""
PTY terminal session management.

Each session spawns ``claude`` inside a pseudo-terminal. A single background
thread per session does the actual blocking reads from the PTY and fans
each chunk out to every current subscriber — the browser's WebSocket view
(app/api/v1/endpoints/terminals.py's terminal_ws) and, when Dann sends it a
task, send_and_capture's short-lived wait for a summary. Two independent
consumers calling the underlying blocking read directly would race for the
same bytes and silently drop whatever the other one grabbed first; fan-out
is what lets both watch the exact same stream without stealing from each
other.

Two kinds of session, distinguished by `origin`:
  "user" — opened via the dashboard's "Open Terminal" button, cwd is a focus
            area's own directory (shared/focus_areas_store.py). General
            scratch use.
  "dann" — opened by the open_claude_code MCP tool
            (integrations/servers/claude_code_server.py) when Dann's routing
            brain starts/continues a real coding session against a project.
            `project` identifies which one, so a later call for the same
            project can find and continue this exact session rather than
            starting a new one — see app/api/v1/endpoints/terminals.py's
            /claude-code route. `focus_area`, if the call came from a work
            stream, lets the dashboard's terminal pane for that stream find
            this same session, so the chat and the terminal are the same
            live conversation.

Windows: uses pywinpty (ConPTY backend).
Linux/Mac: uses ptyprocess.
"""

from __future__ import annotations

import os
import queue
import re
import sys
import threading
import time
import uuid
from typing import Any

_WINDOWS = sys.platform == "win32"

if _WINDOWS:
    import winpty
else:
    import ptyprocess

# session_id → TerminalSession
_sessions: dict[str, "TerminalSession"] = {}
_lock = threading.Lock()

_MAX_BUFFER = 65_536  # rolling scrollback kept per session, bytes

# Best-effort strip of ANSI escape sequences (CSI, OSC, charset-select) and
# bare carriage returns, so captured output is readable prose for an LLM to
# summarize rather than raw terminal control codes. Box-drawing/Unicode UI
# chrome from Claude Code's TUI will still get through — this just removes
# the control-code noise, not a full terminal emulator.
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07]*(?:\x07|\x1b\\)|\x1b[()][A-B0-2]|\r")
_BLANK_RUN_RE = re.compile(r"\n{3,}")

# Claude Code's own end-of-turn status line, e.g. "Crunched for 1s · done
# 1:31 PM" — a real completion signal, not a timing guess. Checked against
# raw (not ANSI-stripped) accumulated output, so this needs to tolerate
# escape codes/box-drawing between words; it does, since it only anchors on
# "done" followed by a time.
_DONE_MARKER_RE = re.compile(r"done \d{1,2}:\d{2}\s*[AP]M")


def _clean_output(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="replace")
    text = _ANSI_RE.sub("", text)
    return _BLANK_RUN_RE.sub("\n\n", text).strip()


class TerminalSession:
    def __init__(
        self,
        session_id: str,
        path: str,
        *,
        origin: str = "user",
        focus_area: str | None = None,
        project: str | None = None,
    ) -> None:
        self.session_id = session_id
        self.path = path
        self.origin = origin
        self.focus_area = focus_area
        self.project = project
        self._proc: "winpty.PtyProcess | ptyprocess.PtyProcess | None" = None
        self._buffer = bytearray()
        self._subscribers: list["queue.Queue[bytes | None]"] = []
        self._state_lock = threading.Lock()
        self._pump_thread: threading.Thread | None = None

    def start(self, rows: int = 24, cols: int = 80, command: str = "claude") -> None:
        import shlex
        env = {**os.environ, "TERM": "xterm-256color", "COLORTERM": "truecolor"}
        if _WINDOWS:
            self._proc = winpty.PtyProcess.spawn(
                command,
                cwd=self.path,
                dimensions=(rows, cols),
                env=env,
            )
        else:
            self._proc = ptyprocess.PtyProcess.spawn(
                shlex.split(command),
                cwd=self.path,
                dimensions=(rows, cols),
                env=env,
            )
        self._pump_thread = threading.Thread(target=self._pump, daemon=True)
        self._pump_thread.start()

    def _raw_read(self, size: int = 4096) -> bytes:
        """Blocking read directly from the PTY. Raises EOFError on exit.
        Only ever called from the pump thread — everyone else goes through
        subscribe()/send_and_capture()."""
        if self._proc is None:
            raise EOFError("not started")
        if _WINDOWS:
            if self._proc.eof():
                raise EOFError("process exited")
            data = self._proc.read(size)
            if data is None:
                raise EOFError("process exited")
            return data if isinstance(data, bytes) else data.encode("utf-8", errors="replace")
        return self._proc.read(size)

    def _pump(self) -> None:
        """Owns the only direct read of the PTY for this session's lifetime,
        broadcasting each chunk to the rolling buffer and every subscriber."""
        while True:
            try:
                data = self._raw_read(4096)
            except EOFError:
                with self._state_lock:
                    subs = list(self._subscribers)
                for q in subs:
                    q.put(None)
                return
            except Exception:
                return
            if not data:
                continue
            with self._state_lock:
                self._buffer.extend(data)
                if len(self._buffer) > _MAX_BUFFER:
                    del self._buffer[: len(self._buffer) - _MAX_BUFFER]
                subs = list(self._subscribers)
            for q in subs:
                q.put(data)

    def subscribe(self) -> "queue.Queue[bytes | None]":
        """Register for live output going forward. Does not replay the
        buffer — send_and_capture subscribes before writing so it never
        needs to, and a WebSocket reconnect just picks up from here same as
        before this fan-out existed."""
        q: "queue.Queue[bytes | None]" = queue.Queue()
        with self._state_lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: "queue.Queue[bytes | None]") -> None:
        with self._state_lock:
            if q in self._subscribers:
                self._subscribers.remove(q)
        # A reader can be blocked in q.get() right now (e.g. the WebSocket
        # handler's executor thread, between a client disconnect and this
        # call) — once removed above, the pump thread will never put to
        # this queue again, so that .get() would hang forever without this
        # nudge. Harmless when nothing's waiting; the item is just dropped
        # with the abandoned queue.
        q.put(None)

    def write(self, data: bytes) -> None:
        if self._proc is not None and self._proc.isalive():
            if _WINDOWS:
                text = data.decode("utf-8", errors="replace")
                self._proc.write(text)
            else:
                self._proc.write(data)

    def _collect(self, q: "queue.Queue[bytes | None]", *, max_wait: float, quiet_for: float) -> tuple[str, bool]:
        """Shared wait loop for wait_for_quiet/send_and_capture — collect
        from an already-subscribed queue until either output looks done or
        `max_wait` total elapses while it's still arriving (settled=False —
        treat as a long-running task still in progress).

        "Looks done" isn't just "nothing new for `quiet_for` seconds" — a
        brief lull mid-response (Claude Code pausing while it reasons, or
        between tool calls) is easy to mistake for completion, which
        previously meant capturing a tentative/partial fragment instead of
        the real answer and summarizing that as if it were final. Once
        Claude Code's own end-of-turn marker ("... · done H:MM AM/PM")
        shows up in the accumulated text, a short quiet gap is enough to
        confirm it; until then, require a longer one so a mid-thought pause
        doesn't get mistaken for done."""
        chunks: list[bytes] = []
        start = time.monotonic()
        settled = False
        confirmed_done = False
        while True:
            remaining = max_wait - (time.monotonic() - start)
            if remaining <= 0:
                break
            effective_quiet = 1.0 if confirmed_done else quiet_for
            try:
                item = q.get(timeout=min(effective_quiet, remaining))
            except queue.Empty:
                settled = bool(chunks)
                break
            if item is None:  # process exited
                settled = True
                break
            chunks.append(item)
            # Checked against ANSI-stripped text, not raw bytes — an escape
            # code landing between "done" and the time would otherwise
            # break the match.
            if not confirmed_done and _DONE_MARKER_RE.search(_clean_output(b"".join(chunks))):
                confirmed_done = True
        return _clean_output(b"".join(chunks)), settled

    def wait_for_quiet(self, *, max_wait: float = 15.0, quiet_for: float = 2.0) -> tuple[str, bool]:
        """Watch output without sending anything — e.g. to let a freshly
        spawned interactive session finish its startup banner/MCP
        connections before handing it a task; sending input while it's
        still booting can land in the input box without being processed."""
        q = self.subscribe()
        try:
            return self._collect(q, max_wait=max_wait, quiet_for=quiet_for)
        finally:
            self.unsubscribe(q)

    def send_and_capture(
        self, text: str, *, max_wait: float = 25.0, quiet_for: float = 4.5
    ) -> tuple[str, bool]:
        """Write `text` then submit it (a bare carriage return — Claude
        Code's interactive input box treats "\\n" as inserting a newline in
        the message, not sending it), then collect output via _collect.
        Subscribes before writing so the echo and response can never be
        missed, regardless of what else is watching this session.

        The text and the submitting \\r are two separate writes with a
        short gap between them — sent as one immediate burst, the input box
        appears to treat it as a paste (common bracketed-paste handling)
        and the trailing \\r never submits, just sits in the box."""
        q = self.subscribe()
        try:
            self.write(text.encode())
            time.sleep(0.3)
            self.write(b"\r")
            return self._collect(q, max_wait=max_wait, quiet_for=quiet_for)
        finally:
            self.unsubscribe(q)

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
            "origin": self.origin,
            "focus_area": self.focus_area,
            "project": self.project,
            "path": self.path,
            "alive": self.alive,
        }


def create_session(
    path: str,
    *,
    origin: str = "user",
    focus_area: str | None = None,
    project: str | None = None,
    rows: int = 50,
    cols: int = 220,
    command: str | None = None,
) -> TerminalSession:
    session_id = str(uuid.uuid4())
    session = TerminalSession(session_id, path, origin=origin, focus_area=focus_area, project=project)
    session.start(rows=rows, cols=cols, command=command or "claude")
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
