"""
Text-chat work streams.

Same routing brain as voice — the local Ollama model decides per-message
whether to answer directly, delegate to Claude, delegate to Claude Code, or
use an MCP module — just driven by typed text instead of a mic, for the
dashboard's silent/no-voice use case (NO_VOICE=1). Each work stream is
bound to a focus area (config.yaml's focus_areas: — see
shared/focus_areas_config.py and shared/focus_areas_store.py) and keeps its
own conversation history; many can run in parallel.

Persisted to ~/.dann/work_streams.json. Also emits the same EventBus event
types voice turns emit (turn.start/turn.llm/metric) so the existing
metrics/logs dashboard panels pick up text-chat activity for free.
"""
import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from shared.agents_config import build_routing_prompt
from shared.focus_areas_config import build_focus_areas_prompt
from shared.focus_areas_store import read_focus_area_context
from voice.config import load_config
from voice.event_bus import bus
from voice.llm.ollama import generate_response
from integrations.client import get_shared_manager

_DANN_DIR = Path.home() / ".dann"
_STREAMS_FILE = _DANN_DIR / "work_streams.json"
_MAX_HISTORY = 200  # messages kept per stream; oldest trimmed beyond this
_DEFAULT_CHAT_MAX_TOKENS = 400  # text doesn't need voice's "1-2 sentences" brevity

_lock = threading.Lock()
_streams: dict[str, dict[str, Any]] = {}
_loaded = False


def _ensure_loaded() -> None:
    global _loaded
    if _loaded:
        return
    _DANN_DIR.mkdir(parents=True, exist_ok=True)
    if _STREAMS_FILE.exists():
        try:
            data = json.loads(_STREAMS_FILE.read_text())
            for s in data:
                _streams[s["id"]] = s
        except (json.JSONDecodeError, OSError):
            pass
    _loaded = True


def _persist() -> None:
    _STREAMS_FILE.write_text(json.dumps(list(_streams.values()), indent=2))


class UnknownStreamError(KeyError):
    pass


def list_focus_areas() -> list[dict[str, Any]]:
    """Focus areas available for binding a new work stream to."""
    cfg = load_config()
    return [a for a in (cfg.get("focus_areas") or []) if a.get("enabled", True) and a.get("name")]


def _resolve_focus_area(name: str) -> dict[str, Any] | None:
    return next((a for a in list_focus_areas() if a["name"] == name), None)


def list_streams() -> list[dict[str, Any]]:
    _ensure_loaded()
    with _lock:
        return sorted(_streams.values(), key=lambda s: s["updated_at"], reverse=True)


def get_stream(stream_id: str) -> dict[str, Any] | None:
    _ensure_loaded()
    with _lock:
        return _streams.get(stream_id)


def create_stream(focus_area: str, title: str = "") -> dict[str, Any]:
    """Create a work stream bound to a known focus area.

    Raises UnknownStreamError if `focus_area` doesn't match any entry
    list_focus_areas() returns.
    """
    _ensure_loaded()
    resolved = _resolve_focus_area(focus_area)
    if not resolved:
        raise UnknownStreamError(focus_area)

    now = time.time()
    stream = {
        "id": uuid.uuid4().hex[:12],
        "focus_area": resolved["name"],
        "title": title or resolved["name"],
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }
    with _lock:
        _streams[stream["id"]] = stream
        _persist()
    return stream


def delete_stream(stream_id: str) -> bool:
    _ensure_loaded()
    with _lock:
        if stream_id not in _streams:
            return False
        del _streams[stream_id]
        _persist()
        return True


def clear_stream(stream_id: str) -> dict[str, Any]:
    """Wipe a stream's message history in place — same id/title/focus_area,
    so its terminal (looked up by focus area, not stream) is unaffected,
    but the model starts fresh instead of conditioning on old turns. Useful
    when a conversation gets stuck imitating its own repeated mistakes.

    Raises UnknownStreamError if stream_id doesn't exist."""
    _ensure_loaded()
    with _lock:
        stream = _streams.get(stream_id)
        if stream is None:
            raise UnknownStreamError(stream_id)
        stream["messages"] = []
        stream["updated_at"] = time.time()
        _persist()
        return stream


def send_message(stream_id: str, text: str) -> dict[str, Any]:
    """Route a text message through the same Ollama + MCP brain as voice,
    scoped to this stream's focus area and conversation history. Blocking —
    call via asyncio.to_thread from the API layer."""
    _ensure_loaded()
    with _lock:
        stream = _streams.get(stream_id)
    if stream is None:
        raise UnknownStreamError(stream_id)

    cfg = load_config()
    ollama_cfg = cfg.get("ollama") or {}
    focus_area = stream["focus_area"]

    system_prompt = ollama_cfg.get("system_prompt", "")
    routing_section = build_routing_prompt(cfg.get("agents"))
    if routing_section:
        system_prompt += f"\n\n{routing_section}"
    focus_section = build_focus_areas_prompt(cfg.get("focus_areas"))
    if focus_section:
        system_prompt += f"\n\n{focus_section}"
    if focus_area:
        system_prompt += (
            f"\n\nThis conversation is a text work stream focused on "
            f"'{focus_area}' — not a git project. If the user asks to use "
            f"\"the terminal\" or \"Claude Code\" for this conversation "
            f"itself, call open_terminal (project_name '{focus_area}' isn't "
            f"a real project, but open_claude_code/ask_claude_code work too "
            f"— they fall back to the same thing automatically)."
        )
        notes = read_focus_area_context(focus_area)
        if notes:
            system_prompt += f"\n\nNotes for '{focus_area}':\n{notes}"

    mcp = get_shared_manager()
    history = [{"role": m["role"], "content": m["content"]} for m in stream["messages"]]

    t0 = time.monotonic()
    response = generate_response(
        text,
        base_url=ollama_cfg.get("base_url", "http://localhost:11434"),
        model=ollama_cfg.get("model", "llama3.2"),
        system_prompt=system_prompt,
        temperature=ollama_cfg.get("temperature", 0.7),
        max_tokens=ollama_cfg.get("chat_max_tokens", _DEFAULT_CHAT_MAX_TOKENS),
        tools=mcp.tools,
        mcp=mcp,
        history=history,
        keep_alive=ollama_cfg.get("keep_alive"),
        session_context={"focus_area": focus_area} if focus_area else None,
    )
    latency_ms = round((time.monotonic() - t0) * 1000)
    response = response or ""

    now = time.time()
    with _lock:
        stream["messages"].append({"role": "user", "content": text, "ts": now})
        stream["messages"].append({"role": "assistant", "content": response, "ts": time.time()})
        stream["messages"] = stream["messages"][-_MAX_HISTORY:]
        stream["updated_at"] = time.time()
        _persist()

    # A single self-contained event, not the turn.start/turn.llm/metric triplet
    # voice uses — that triplet accumulates into a single *global* _pending
    # dict in history_service (not per-session), which concurrent work streams
    # would corrupt, and turn.llm also feeds the voice-only conversation view
    # (voiceTurns) with no way to tell it apart from a real voice turn.
    bus.emit("chat.turn", {
        "session_id": stream_id, "focus_area": focus_area, "text": text,
        "response": response, "latency_ms": latency_ms,
    })

    return {"response": response, "latency_ms": latency_ms}
