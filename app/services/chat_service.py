"""
Text-chat work streams.

Same routing brain as voice — the local Ollama model decides per-message
whether to answer directly, delegate to Claude, delegate to Claude Code, or
use an MCP module — just driven by typed text instead of a mic, for the
dashboard's silent/no-voice use case (NO_VOICE=1). Each work stream is
bound to a project (mirrors voice's "code mode," but many can run in
parallel instead of one at a time) and keeps its own conversation history.

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

from src.agents_config import build_routing_prompt
from src.config import load_config
from src.event_bus import bus
from src.llm.ollama import generate_response
from src.mcp_client import get_shared_manager
from src.mcp_servers.claude_code_server import find_projects, resolve_project

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


def list_projects() -> list[dict[str, Any]]:
    """Projects available for binding a new work stream to."""
    return find_projects()


def list_streams() -> list[dict[str, Any]]:
    _ensure_loaded()
    with _lock:
        return sorted(_streams.values(), key=lambda s: s["updated_at"], reverse=True)


def get_stream(stream_id: str) -> dict[str, Any] | None:
    _ensure_loaded()
    with _lock:
        return _streams.get(stream_id)


def create_stream(project: str, title: str = "") -> dict[str, Any]:
    """Create a work stream bound to a known project.

    Raises UnknownStreamError if `project` doesn't match any project
    list_projects() returns.
    """
    _ensure_loaded()
    resolved = resolve_project(project)
    if not resolved:
        raise UnknownStreamError(project)

    now = time.time()
    stream = {
        "id": uuid.uuid4().hex[:12],
        "project": resolved["name"],
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


def send_message(stream_id: str, text: str) -> dict[str, Any]:
    """Route a text message through the same Ollama + MCP brain as voice,
    scoped to this stream's project and conversation history. Blocking —
    call via asyncio.to_thread from the API layer."""
    _ensure_loaded()
    with _lock:
        stream = _streams.get(stream_id)
    if stream is None:
        raise UnknownStreamError(stream_id)

    cfg = load_config()
    ollama_cfg = cfg.get("ollama") or {}
    project = stream["project"]

    system_prompt = ollama_cfg.get("system_prompt", "")
    routing_section = build_routing_prompt(cfg.get("agents"))
    if routing_section:
        system_prompt += f"\n\n{routing_section}"
    if project:
        system_prompt += (
            f"\n\nThis conversation is a text work stream focused on the "
            f"project '{project}'. When calling ask_claude_code or "
            f"open_claude_code, use this project unless the user clearly "
            f"means a different one."
        )

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
        "session_id": stream_id, "project": project, "text": text,
        "response": response, "latency_ms": latency_ms,
    })

    return {"response": response, "latency_ms": latency_ms}
