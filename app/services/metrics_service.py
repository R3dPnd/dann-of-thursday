"""
Metrics persistence service.

Listens for ``metric`` events on the EventBus and appends each record to
``~/.dann/metrics.jsonl`` (one JSON object per line).  Provides query helpers
used by the /api/v1/metrics endpoint.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_DANN_DIR = Path.home() / ".dann"
_METRICS_FILE = _DANN_DIR / "metrics.jsonl"

# In-memory cache of metric records for the current process lifetime.
# Protected by _lock.
_records: list[dict[str, Any]] = []
_lock = threading.Lock()


def _ensure_dir() -> None:
    _DANN_DIR.mkdir(parents=True, exist_ok=True)


def record_metric(event_type: str, payload: dict[str, Any]) -> None:
    """Called by the EventBus subscriber for every emitted event."""
    if event_type != "metric":
        return

    record = {**payload, "recorded_at": datetime.now(timezone.utc).isoformat()}

    _ensure_dir()
    try:
        with open(_METRICS_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError:
        pass  # non-fatal — still cache in memory

    with _lock:
        _records.append(record)


def _load_from_disk() -> list[dict[str, Any]]:
    """Read the full metrics file from disk (called once on first query)."""
    if not _METRICS_FILE.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with open(_METRICS_FILE, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except OSError:
        pass
    return rows


_disk_loaded = False


def _all_records() -> list[dict[str, Any]]:
    global _disk_loaded
    with _lock:
        if not _disk_loaded:
            disk = _load_from_disk()
            # Merge: disk records come first; in-memory may overlap if process
            # wrote them already (they were appended to disk and to _records).
            # Deduplicate by recorded_at + session_id.
            seen: set[tuple[str, str]] = set()
            merged: list[dict[str, Any]] = []
            for r in disk + _records:
                key = (r.get("recorded_at", ""), r.get("session_id", ""))
                if key not in seen:
                    seen.add(key)
                    merged.append(r)
            _records.clear()
            _records.extend(merged)
            _disk_loaded = True
        return list(_records)


# ── Query helpers ─────────────────────────────────────────────────────────────

def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def _cutoff(period: str) -> datetime | None:
    now = datetime.now(timezone.utc)
    if period == "session":
        return None  # caller filters by session_id
    if period == "day":
        return now.replace(hour=0, minute=0, second=0, microsecond=0)
    if period == "week":
        from datetime import timedelta
        return (now - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0)
    return None  # "all"


def aggregate(period: str = "all", session_id: str | None = None) -> dict[str, Any]:
    """Return aggregated metrics for the requested period."""
    records = _all_records()

    cut = _cutoff(period)
    filtered: list[dict[str, Any]] = []
    for r in records:
        if session_id and r.get("session_id") != session_id:
            continue
        if cut:
            ts = _parse_ts(r.get("recorded_at"))
            if ts and ts < cut:
                continue
        filtered.append(r)

    total = len(filtered)
    if total == 0:
        return {
            "period": period,
            "total_turns": 0,
            "blank_turns": 0,
            "code_turns": 0,
            "normal_turns": 0,
            "error_turns": 0,
            "avg_stt_ms": None,
            "avg_llm_ms": None,
            "avg_tts_ms": None,
            "avg_total_ms": None,
            "sessions": [],
        }

    def _avg(key: str) -> float | None:
        vals = [r[key] for r in filtered if isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 1) if vals else None

    sessions: list[str] = list({r.get("session_id", "") for r in filtered if r.get("session_id")})

    blank = sum(1 for r in filtered if r.get("blank"))
    code = sum(1 for r in filtered if r.get("mode") == "CODE")
    normal = sum(1 for r in filtered if r.get("mode") == "NORMAL")
    errors = sum(1 for r in filtered if r.get("status") == "error")

    return {
        "period": period,
        "total_turns": total,
        "blank_turns": blank,
        "code_turns": code,
        "normal_turns": normal,
        "error_turns": errors,
        "avg_stt_ms": _avg("stt_ms"),
        "avg_llm_ms": _avg("llm_ms"),
        "avg_tts_ms": _avg("tts_ms"),
        "avg_total_ms": _avg("total_ms"),
        "sessions": sorted(sessions),
    }


def sessions_by_day(days: int = 7) -> list[dict[str, Any]]:
    """Return per-day session counts for the last *days* days."""
    from datetime import timedelta

    records = _all_records()
    now = datetime.now(timezone.utc)
    buckets: dict[str, set[str]] = {}
    for i in range(days):
        day = (now - timedelta(days=i)).strftime("%Y-%m-%d")
        buckets[day] = set()

    for r in records:
        ts = _parse_ts(r.get("recorded_at"))
        sid = r.get("session_id", "")
        if ts and sid:
            day = ts.strftime("%Y-%m-%d")
            if day in buckets:
                buckets[day].add(sid)

    return [
        {"date": day, "sessions": len(sids)}
        for day, sids in sorted(buckets.items())
    ]


def latency_by_session() -> list[dict[str, Any]]:
    """Return per-session average latency breakdown."""
    records = _all_records()
    sessions: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        sid = r.get("session_id", "unknown")
        sessions.setdefault(sid, []).append(r)

    result = []
    for sid, rows in sessions.items():
        def _avg(key: str) -> float | None:
            vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
            return round(sum(vals) / len(vals), 1) if vals else None

        result.append({
            "session_id": sid,
            "turns": len(rows),
            "avg_stt_ms": _avg("stt_ms"),
            "avg_llm_ms": _avg("llm_ms"),
            "avg_tts_ms": _avg("tts_ms"),
            "avg_total_ms": _avg("total_ms"),
        })
    return result
