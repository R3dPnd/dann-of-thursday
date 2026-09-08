#!/usr/bin/env python3
"""MCP server for quick voice-captured notes and reminders.

Local-only storage (~/.dann/notes.json) — no external account needed.

Exposed tools:
  add_note        — save a note, optionally tagged
  list_notes      — list recent notes, optionally filtered by tag
  search_notes    — substring search over saved notes
  delete_note     — remove a note by id
  add_reminder    — save a note with a target time (see limitation below)
  list_reminders  — list saved reminders, soonest first

Limitation: reminders are stored and listable, but Dann does not currently
speak them proactively when they come due — there's no background scheduler
wired into the orchestrator yet. list_reminders is how you'd check on them
for now; treat this module as a to-do list, not an alarm clock.
"""
import uuid
from datetime import datetime

from mcp.server.fastmcp import FastMCP

from integrations.servers._store import load_json, save_json

mcp = FastMCP("notes")

_FILE = "notes.json"


def _load() -> list[dict]:
    return load_json(_FILE, [])


def _save(notes: list[dict]) -> None:
    save_json(_FILE, notes)


@mcp.tool()
def add_note(text: str, tags: str = "") -> str:
    """Save a quick note.

    Args:
        text: The note content.
        tags: Optional comma-separated tags for later filtering.
    """
    notes = _load()
    note = {
        "id": uuid.uuid4().hex[:8],
        "text": text,
        "tags": [t.strip() for t in tags.split(",") if t.strip()],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "remind_at": None,
    }
    notes.append(note)
    _save(notes)
    return f"Saved note {note['id']}."


@mcp.tool()
def list_notes(tag: str = "", limit: int = 20) -> str:
    """List recent notes, most recent first.

    Args:
        tag: Optional tag to filter by.
        limit: Max notes to return.
    """
    notes = _load()
    if tag:
        notes = [n for n in notes if tag in n.get("tags", [])]
    notes = sorted(notes, key=lambda n: n["created_at"], reverse=True)[:limit]
    if not notes:
        return "No notes found."
    return "\n".join(f"[{n['id']}] {n['text']}" for n in notes)


@mcp.tool()
def search_notes(query: str) -> str:
    """Search saved notes for a substring match.

    Args:
        query: Text to search for.
    """
    notes = _load()
    q = query.lower()
    hits = [n for n in notes if q in n["text"].lower()]
    if not hits:
        return f"No notes matching '{query}'."
    return "\n".join(f"[{n['id']}] {n['text']}" for n in hits)


@mcp.tool()
def delete_note(note_id: str) -> str:
    """Delete a note by its id.

    Args:
        note_id: The id shown alongside the note (e.g. from list_notes).
    """
    notes = _load()
    remaining = [n for n in notes if n["id"] != note_id]
    if len(remaining) == len(notes):
        return f"No note with id '{note_id}'."
    _save(remaining)
    return f"Deleted note {note_id}."


@mcp.tool()
def add_reminder(text: str, when: str) -> str:
    """Save a reminder for a future time.

    Note: this only stores the reminder for later lookup via list_reminders
    — Dann does not yet speak reminders proactively when they come due.

    Args:
        text: What to be reminded of.
        when: When, as free text or ISO 8601 (e.g. "2026-09-01T09:00").
    """
    notes = _load()
    note = {
        "id": uuid.uuid4().hex[:8],
        "text": text,
        "tags": ["reminder"],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "remind_at": when,
    }
    notes.append(note)
    _save(notes)
    return (
        f"Saved reminder {note['id']} for {when}. "
        "I can't speak this proactively yet — ask me to check reminders."
    )


@mcp.tool()
def list_reminders(upcoming_only: bool = True) -> str:
    """List saved reminders, soonest first.

    Args:
        upcoming_only: If true, only show reminders at or after now (best-effort
            string comparison — works reliably for ISO 8601 remind_at values).
    """
    notes = [n for n in _load() if n.get("remind_at")]
    if upcoming_only:
        now = datetime.now().isoformat(timespec="seconds")
        notes = [n for n in notes if n["remind_at"] >= now]
    notes.sort(key=lambda n: n["remind_at"])
    if not notes:
        return "No reminders found."
    return "\n".join(f"[{n['id']}] {n['remind_at']} — {n['text']}" for n in notes)


if __name__ == "__main__":
    mcp.run()
