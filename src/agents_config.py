"""Compiles the structured `agents:` list in config.yaml into the routing
section appended to the Ollama system prompt (see src/orchestrator.py and
app/services/chat_service.py, both of which use this so voice and text chat
route identically).

Adding a new routing target — a new model, provider, or MCP module — means
adding an entry to config.yaml's `agents:` list, not hand-editing prompt
prose.
"""
from typing import Any


def build_routing_prompt(agents_cfg: list[dict[str, Any]] | None) -> str:
    """Render config.yaml's `agents:` list into an "AGENTS" section for the
    system prompt. Entries with enabled: false, or missing a name/description,
    are skipped. Returns "" if agents_cfg is empty/missing so callers can
    always safely append the result."""
    if not agents_cfg:
        return ""

    lines = [
        'AGENTS — route each request to the single best-fit one below. '
        '"local" means answer yourself, no tool call.',
    ]
    for agent in agents_cfg:
        if not agent.get("enabled", True):
            continue
        name = agent.get("name")
        desc = (agent.get("description") or "").strip()
        if not name or not desc:
            continue
        tools = agent.get("tools")
        tool_hint = f" [{', '.join(tools)}]" if tools else ""
        lines.append(f"- {name}{tool_hint}: {desc}")

    return "\n".join(lines)
