"""
Log persistence service.

Listens for ``error`` and ``warning`` EventBus events and writes them to
``~/.dann/dann.log``.  Also keeps an in-memory ring buffer of the last 2 000
entries for the /api/v1/logs endpoint.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_DANN_DIR = Path.home() / ".dann"
_LOG_FILE = _DANN_DIR / "dann.log"
_MAX_ENTRIES = 2000

# Ordered from most verbose to most severe (matches Python logging levels).
LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
_LEVEL_RANK = {lvl: i for i, lvl in enumerate(LEVELS)}

# Event type → log level mapping.
_EVENT_LEVEL: dict[str, str] = {
    "session.start": "INFO",
    "session.end": "INFO",
    "state.changed": "INFO",
    "turn.start": "DEBUG",
    "turn.stt": "DEBUG",
    "turn.llm": "DEBUG",
    "turn.code": "DEBUG",
    "turn.tts": "DEBUG",
    "metric": "DEBUG",
    "error": "ERROR",
    "warning": "WARNING",
}

_entries: deque[dict[str, Any]] = deque(maxlen=_MAX_ENTRIES)
_lock = threading.Lock()


def _ensure_dir() -> None:
    _DANN_DIR.mkdir(parents=True, exist_ok=True)


def record_event(event_type: str, payload: dict[str, Any]) -> None:
    """Called by the EventBus subscriber for every emitted event."""
    level = _EVENT_LEVEL.get(event_type, "DEBUG")

    entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "module": payload.get("module", event_type.split(".")[0]),
        "event_type": event_type,
        "message": _message(event_type, payload),
        "detail": payload.get("traceback") or payload.get("detail"),
    }

    with _lock:
        _entries.append(entry)

    # Persist errors and warnings to disk.
    if level in ("ERROR", "WARNING", "CRITICAL"):
        _ensure_dir()
        try:
            with open(_LOG_FILE, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except OSError:
            pass


def _message(event_type: str, payload: dict[str, Any]) -> str:
    if event_type == "error":
        return payload.get("message", "Unknown error")
    if event_type == "warning":
        return payload.get("message", "Warning")
    if event_type == "state.changed":
        mode = payload.get("mode", "?")
        proj = payload.get("project")
        return f"Mode → {mode}" + (f" ({proj})" if proj else "")
    if event_type == "session.start":
        return f"Session started: {payload.get('session_id', '')}"
    if event_type == "session.end":
        return f"Session ended: {payload.get('reason', '')}"
    if event_type == "turn.stt":
        text = payload.get("text", "")
        blank = payload.get("blank", False)
        return f"STT: {'(blank)' if blank else repr(text[:80])}"
    if event_type == "turn.llm":
        ms = payload.get("latency_ms")
        text = payload.get("text", "")
        return f"LLM ({ms} ms): {repr(text[:80])}"
    if event_type == "turn.code":
        return (
            f"Code turn [{payload.get('status', '?')}] "
            f"{payload.get('project', '')} — {payload.get('task', '')[:60]}"
        )
    return json.dumps(payload)[:120]


# ── Query helpers ─────────────────────────────────────────────────────────────

def get_entries(
    level: str = "DEBUG",
    module: str | None = None,
    search: str | None = None,
    limit: int = 200,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Return filtered log entries (newest first)."""
    min_rank = _LEVEL_RANK.get(level.upper(), 0)

    with _lock:
        all_entries = list(_entries)

    # Reverse so newest come first.
    all_entries.reverse()

    result: list[dict[str, Any]] = []
    for entry in all_entries:
        entry_rank = _LEVEL_RANK.get(entry.get("level", "DEBUG"), 0)
        if entry_rank < min_rank:
            continue
        if module and entry.get("module") != module:
            continue
        if search and search.lower() not in entry.get("message", "").lower():
            continue
        result.append(entry)

    return result[offset: offset + limit]


def entry_count() -> int:
    with _lock:
        return len(_entries)
