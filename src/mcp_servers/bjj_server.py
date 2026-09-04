#!/usr/bin/env python3
"""MCP server for tracking your own BJJ training — sessions, techniques, and notes.

Local-only storage (~/.dann/bjj.json) — no external account needed. Like the
gardening module, this deliberately doesn't try to answer general BJJ
questions (how to escape mount, rules, etc.) — the base LLM already knows
that. What it adds is a record of *your* training: what you drilled, how
sessions went, and technique notes specific to your game.

Exposed tools:
  log_session       — record a training session
  list_sessions     — list recent sessions, optionally filtered by type
  log_technique     — record a technique you drilled or want to remember
  list_techniques   — list recent techniques, optionally filtered by category
  add_training_note — free-form note (injuries, goals, roll observations)
  search_training_log — substring search across sessions, techniques, and notes
"""
import uuid
from datetime import date, datetime

from mcp.server.fastmcp import FastMCP

from src.mcp_servers._store import load_json, save_json

mcp = FastMCP("bjj")

_FILE = "bjj.json"
_DEFAULT = {"sessions": [], "techniques": [], "notes": []}


def _load() -> dict:
    data = load_json(_FILE, _DEFAULT)
    for key in _DEFAULT:
        data.setdefault(key, [])
    return data


def _save(data: dict) -> None:
    save_json(_FILE, data)


@mcp.tool()
def log_session(session_type: str, duration_minutes: int = 0, notes: str = "") -> str:
    """Record a training session.

    Args:
        session_type: e.g. "gi", "no-gi", "open mat", "competition".
        duration_minutes: Optional session length in minutes.
        notes: Optional notes (partners, focus, how it went).
    """
    data = _load()
    entry = {
        "id": uuid.uuid4().hex[:8],
        "session_type": session_type,
        "duration_minutes": duration_minutes,
        "notes": notes,
        "trained_on": date.today().isoformat(),
    }
    data["sessions"].append(entry)
    _save(data)
    return f"Logged {session_type} session" + (f" ({duration_minutes} min)" if duration_minutes else "") + "."


@mcp.tool()
def list_sessions(session_type: str = "", limit: int = 20) -> str:
    """List recent sessions, most recent first.

    Args:
        session_type: Optional type to filter by (e.g. "gi", "no-gi").
        limit: Max sessions to return.
    """
    sessions = _load()["sessions"]
    if session_type:
        sessions = [s for s in sessions if session_type.lower() in s["session_type"].lower()]
    sessions = sorted(sessions, key=lambda s: s["trained_on"], reverse=True)[:limit]
    if not sessions:
        return "No sessions found."
    return "\n".join(
        f"[{s['id']}] {s['session_type']} — {s['duration_minutes'] or '?'} min ({s['trained_on']})"
        for s in sessions
    )


@mcp.tool()
def log_technique(name: str, category: str = "", notes: str = "") -> str:
    """Record a technique you drilled or want to remember.

    Args:
        name: The technique (e.g. "armbar from closed guard").
        category: Optional category (e.g. "submission", "sweep", "guard pass", "takedown").
        notes: Optional notes — key details, what made it click, common mistakes.
    """
    data = _load()
    entry = {
        "id": uuid.uuid4().hex[:8],
        "name": name,
        "category": category,
        "notes": notes,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    data["techniques"].append(entry)
    _save(data)
    return f"Logged technique: {name}" + (f" ({category})" if category else "") + "."


@mcp.tool()
def list_techniques(category: str = "", limit: int = 20) -> str:
    """List recent techniques, most recent first.

    Args:
        category: Optional category to filter by.
        limit: Max techniques to return.
    """
    techniques = _load()["techniques"]
    if category:
        techniques = [t for t in techniques if category.lower() in t.get("category", "").lower()]
    techniques = sorted(techniques, key=lambda t: t["created_at"], reverse=True)[:limit]
    if not techniques:
        return "No techniques found."
    return "\n".join(f"[{t['id']}] {t['name']} — {t['category'] or 'uncategorized'}" for t in techniques)


@mcp.tool()
def add_training_note(text: str, tags: str = "") -> str:
    """Save a free-form training note — injuries, goals, roll observations, etc.

    Args:
        text: The note content.
        tags: Optional comma-separated tags (e.g. "injury,knee").
    """
    data = _load()
    entry = {
        "id": uuid.uuid4().hex[:8],
        "text": text,
        "tags": [t.strip() for t in tags.split(",") if t.strip()],
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    data["notes"].append(entry)
    _save(data)
    return f"Saved training note {entry['id']}."


@mcp.tool()
def search_training_log(query: str) -> str:
    """Search sessions, techniques, and notes for a substring match.

    Args:
        query: Text to search for (matches session types, technique names, note text).
    """
    data = _load()
    q = query.lower()
    hits = []
    for s in data["sessions"]:
        if q in s["session_type"].lower() or q in s.get("notes", "").lower():
            hits.append(f"[session {s['id']}] {s['session_type']} — {s['duration_minutes'] or '?'} min ({s['trained_on']})")
    for t in data["techniques"]:
        if q in t["name"].lower() or q in t.get("category", "").lower() or q in t.get("notes", "").lower():
            hits.append(f"[technique {t['id']}] {t['name']} — {t['category'] or 'uncategorized'}")
    for n in data["notes"]:
        if q in n["text"].lower() or any(q in tag.lower() for tag in n.get("tags", [])):
            hits.append(f"[note {n['id']}] {n['text']}")
    if not hits:
        return f"Nothing in the training log matching '{query}'."
    return "\n".join(hits)


if __name__ == "__main__":
    mcp.run()
