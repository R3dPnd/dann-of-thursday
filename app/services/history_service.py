"""
Conversation history persistence.

Subscribes to EventBus events and assembles complete turn records (user text +
Dann response + pipeline latencies), writing each to ~/.dann/history.jsonl.
Provides a query helper for the /api/v1/history endpoint.
"""
from __future__ import annotations

import json
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_DANN_DIR = Path.home() / ".dann"
_HISTORY_FILE = _DANN_DIR / "history.jsonl"
_MAX_IN_MEMORY = 500

_records: deque[dict[str, Any]] = deque(maxlen=_MAX_IN_MEMORY)
_pending: dict[str, Any] = {}
_lock = threading.Lock()
_disk_loaded = False


def _ensure_dir() -> None:
    _DANN_DIR.mkdir(parents=True, exist_ok=True)


def record_event(event_type: str, payload: dict[str, Any]) -> None:
    """Accumulate per-event data into complete turn records; flush on metric."""
    global _pending

    if event_type == "turn.start":
        with _lock:
            _pending = {
                "session_id": payload.get("session_id"),
                "started_at": datetime.now(timezone.utc).isoformat(),
            }

    elif event_type == "turn.stt":
        text = payload.get("text", "")
        if text:
            with _lock:
                _pending["user_text"] = text
                _pending["stt_ms"] = payload.get("latency_ms")

    elif event_type == "turn.llm":
        with _lock:
            _pending.update({
                "dann_text": payload.get("text", ""),
                "mode": "normal",
                "llm_ms": payload.get("latency_ms"),
            })

    elif event_type == "turn.code":
        with _lock:
            _pending.update({
                "user_text": _pending.get("user_text") or payload.get("task", ""),
                "dann_text": payload.get("response", ""),
                "mode": "code",
                "project": payload.get("project"),
                "code_ms": payload.get("latency_ms"),
                "status": payload.get("status"),
            })

    elif event_type == "metric":
        with _lock:
            if not _pending.get("dann_text"):
                return  # blank/goodbye turns — nothing worth persisting
            record: dict[str, Any] = {
                **_pending,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "stt_ms": _pending.get("stt_ms") or payload.get("stt_ms"),
                "llm_ms": _pending.get("llm_ms") or payload.get("llm_ms"),
                "tts_ms": payload.get("tts_ms"),
                "code_ms": _pending.get("code_ms") or payload.get("code_ms"),
                "status": payload.get("status", _pending.get("status", "ok")),
            }
            _pending = {}

        _ensure_dir()
        try:
            with open(_HISTORY_FILE, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except OSError:
            pass

        with _lock:
            _records.append(record)


def _load_from_disk() -> list[dict[str, Any]]:
    if not _HISTORY_FILE.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with open(_HISTORY_FILE, encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped:
                    try:
                        rows.append(json.loads(stripped))
                    except json.JSONDecodeError:
                        pass
    except OSError:
        pass
    return rows


def get_history(
    limit: int = 100,
    offset: int = 0,
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return persisted turns, newest first."""
    global _disk_loaded

    with _lock:
        if not _disk_loaded:
            disk = _load_from_disk()
            seen: set[str] = set()
            merged: list[dict[str, Any]] = []
            for r in disk:
                key = r.get("started_at", "") + r.get("session_id", "")
                if key not in seen:
                    seen.add(key)
                    merged.append(r)
            for r in _records:
                key = r.get("started_at", "") + r.get("session_id", "")
                if key not in seen:
                    seen.add(key)
                    merged.append(r)
            _records.clear()
            _records.extend(merged[-_MAX_IN_MEMORY:])
            _disk_loaded = True

    with _lock:
        all_records = list(_records)

    if session_id:
        all_records = [r for r in all_records if r.get("session_id") == session_id]

    all_records.reverse()  # newest first
    return all_records[offset: offset + limit]
