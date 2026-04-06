"""
Metrics persistence service.

Listens for ``turn.code`` events on the EventBus and appends each record to
``~/.dann/code_calls.jsonl`` (one JSON object per line). Provides query helpers
used by the /api/v1/metrics endpoint.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_DANN_DIR = Path.home() / ".dann"
_CALLS_FILE = _DANN_DIR / "code_calls.jsonl"

_records: list[dict[str, Any]] = []
_lock = threading.Lock()


def _ensure_dir() -> None:
    _DANN_DIR.mkdir(parents=True, exist_ok=True)


def record_metric(event_type: str, payload: dict[str, Any]) -> None:
    """Called by the EventBus subscriber for every emitted event."""
    if event_type != "turn.code":
        return

    record = {
        "project": payload.get("project", "unknown"),
        "status": payload.get("status", "ok"),
        "latency_ms": payload.get("latency_ms"),
        "session_id": payload.get("session_id"),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }

    _ensure_dir()
    try:
        with open(_CALLS_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError:
        pass

    with _lock:
        _records.append(record)


def _load_from_disk() -> list[dict[str, Any]]:
    if not _CALLS_FILE.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with open(_CALLS_FILE, encoding="utf-8") as fh:
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


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def aggregate(period: str = "all", session_id: str | None = None) -> dict[str, Any]:
    from datetime import timedelta
    records = _all_records()
    now = datetime.now(timezone.utc)

    cutoffs = {
        "day": now.replace(hour=0, minute=0, second=0, microsecond=0),
        "week": (now - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0),
    }
    cut = cutoffs.get(period)

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
            "total_calls": 0,
            "successful_calls": 0,
            "error_calls": 0,
            "empty_calls": 0,
            "avg_response_ms": None,
            "projects_used": 0,
            "sessions": [],
        }

    latencies = [r["latency_ms"] for r in filtered if isinstance(r.get("latency_ms"), (int, float))]
    avg_ms = round(sum(latencies) / len(latencies), 1) if latencies else None

    return {
        "period": period,
        "total_calls": total,
        "successful_calls": sum(1 for r in filtered if r.get("status") == "ok"),
        "error_calls": sum(1 for r in filtered if r.get("status") == "error"),
        "empty_calls": sum(1 for r in filtered if r.get("status") == "empty"),
        "avg_response_ms": avg_ms,
        "projects_used": len({r.get("project") for r in filtered if r.get("project")}),
        "sessions": sorted({r.get("session_id", "") for r in filtered if r.get("session_id")}),
    }


def calls_by_day(days: int = 7) -> list[dict[str, Any]]:
    from datetime import timedelta
    records = _all_records()
    now = datetime.now(timezone.utc)
    buckets: dict[str, int] = {}
    for i in range(days):
        day = (now - timedelta(days=i)).strftime("%Y-%m-%d")
        buckets[day] = 0

    for r in records:
        ts = _parse_ts(r.get("recorded_at"))
        if ts:
            day = ts.strftime("%Y-%m-%d")
            if day in buckets:
                buckets[day] += 1

    return [{"date": day, "calls": count} for day, count in sorted(buckets.items())]


def by_project() -> list[dict[str, Any]]:
    records = _all_records()
    projects: dict[str, dict[str, Any]] = {}

    for r in records:
        proj = r.get("project", "unknown")
        if proj not in projects:
            projects[proj] = {"project": proj, "total": 0, "ok": 0, "error": 0, "empty": 0, "latencies": []}
        p = projects[proj]
        p["total"] += 1
        status = r.get("status", "ok")
        if status in ("ok", "error", "empty"):
            p[status] += 1
        if isinstance(r.get("latency_ms"), (int, float)):
            p["latencies"].append(r["latency_ms"])

    result = []
    for p in projects.values():
        lats = p.pop("latencies")
        p["avg_response_ms"] = round(sum(lats) / len(lats), 1) if lats else None
        result.append(p)

    return sorted(result, key=lambda x: x["total"], reverse=True)
