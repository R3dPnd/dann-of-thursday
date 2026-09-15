"""Per-focus-area local directories — free-form notes that get folded into
a work stream's/terminal's system prompt whenever that focus area is
active, distinct from focus_areas_config.py's always-on routing hint.

Lives under ~/.dann/focus_areas/<name>/ — the same "personal, non-git data"
home as integrations/servers/_store.py — rather than in config.yaml, since
it's free-form content the user (or Dann) writes to, not structured config.

v1 just concatenates everything in the directory into the prompt; once a
focus area's notes outgrow the context window, swap read_focus_area_context
for real retrieval (chunk + embed + top-k) without touching call sites.

Each note is its own file — `<created_at>-<slug>.md`, oldest-first order
matching filename sort — so the dashboard can list/delete them individually
(app/api/v1/endpoints/focus_areas.py) and integrations/servers/
focus_areas_server.py can let Dann add one mid-conversation
(save_focus_area_note).
"""
import json
import re
import time
from pathlib import Path
from typing import Any

from integrations.servers._store import dann_home

_TEXT_SUFFIXES = {".md", ".markdown", ".txt"}

# Paths already confirmed trusted this process — avoids re-reading
# ~/.claude.json (tens of KB) on every focus_area_dir() call.
_trusted_cache: set[str] = set()


def _mark_trusted(path: Path) -> None:
    """Best-effort: set the same hasTrustDialogAccepted flag Claude Code's
    own interactive trust dialog writes when a human accepts it, so a
    Dann-opened terminal doesn't sit stuck at that one-time prompt in a
    directory Dann itself created and fully controls. Only ever called with
    a focus_area_dir() path — never a real project directory, which keeps
    its normal one-time manual prompt. Silently does nothing if
    ~/.claude.json can't be read/parsed/written; this is a convenience, not
    something anything else depends on."""
    key = str(path)
    if key in _trusted_cache:
        return
    claude_json = Path.home() / ".claude.json"
    try:
        data = json.loads(claude_json.read_text()) if claude_json.exists() else {}
    except (json.JSONDecodeError, OSError):
        return
    projects = data.setdefault("projects", {})
    entry = projects.get(key)
    if isinstance(entry, dict) and entry.get("hasTrustDialogAccepted"):
        _trusted_cache.add(key)
        return
    projects[key] = {**(entry or {}), "hasTrustDialogAccepted": True}
    try:
        claude_json.write_text(json.dumps(data, indent=2))
    except OSError:
        return
    _trusted_cache.add(key)


def focus_area_dir(name: str) -> Path:
    """The focus area's own directory — its terminal's cwd, and where its
    notes live. Auto-created on first use, and marked trusted for Claude
    Code (see _mark_trusted)."""
    d = dann_home() / "focus_areas" / name
    d.mkdir(parents=True, exist_ok=True)
    _mark_trusted(d)
    return d


def read_focus_area_context(name: str) -> str:
    """Concatenate every text/markdown file directly in the focus area's
    directory into one context block. Returns "" if the directory has no
    notes yet, so callers can always safely append the result."""
    d = focus_area_dir(name)
    parts = []
    for path in sorted(d.iterdir()):
        if path.is_file() and path.suffix.lower() in _TEXT_SUFFIXES:
            try:
                text = path.read_text().strip()
            except OSError:
                continue
            if text:
                parts.append(f"--- {path.name} ---\n{text}")
    return "\n\n".join(parts)


def _slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len] or "note"


def _note_from_file(path: Path) -> dict[str, Any] | None:
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    lines = text.splitlines()
    if lines and lines[0].startswith("#"):
        title = lines[0].lstrip("#").strip()
        content = "\n".join(lines[1:]).strip()
    else:
        title = (lines[0][:60] if lines else path.stem)
        content = text
    return {
        "filename": path.name,
        "title": title or path.stem,
        "content": content,
        "modified_at": path.stat().st_mtime,
    }


def save_note(name: str, content: str, title: str = "") -> dict[str, Any]:
    """Save a new note to a focus area's directory as its own markdown
    file. Returns the note dict (see list_notes)."""
    d = focus_area_dir(name)
    ts = int(time.time())
    slug = _slugify(title or content)
    path = d / f"{ts}-{slug}.md"
    while path.exists():  # same-second collision — make the filename unique
        ts += 1
        path = d / f"{ts}-{slug}.md"
    body = f"# {title}\n\n{content}" if title else content
    path.write_text(body.strip() + "\n")
    note = _note_from_file(path)
    assert note is not None
    return note


def list_notes(name: str) -> list[dict[str, Any]]:
    """All notes for a focus area, newest first."""
    d = focus_area_dir(name)
    notes = []
    for path in d.iterdir():
        if path.is_file() and path.suffix.lower() in _TEXT_SUFFIXES:
            note = _note_from_file(path)
            if note is not None:
                notes.append(note)
    notes.sort(key=lambda n: n["modified_at"], reverse=True)
    return notes


def delete_note(name: str, filename: str) -> bool:
    """Delete one note by filename. Returns False if it doesn't exist, or
    if `filename` doesn't resolve to a plain file directly inside this
    focus area's own directory (guards against path traversal)."""
    d = focus_area_dir(name)
    path = (d / filename).resolve()
    if path.parent != d.resolve() or not path.is_file():
        return False
    path.unlink()
    return True
