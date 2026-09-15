"""Compiles the structured `focus_areas:` list in config.yaml into a "FOCUS
AREAS" section appended to the Ollama system prompt (see voice/orchestrator.py
and app/services/chat_service.py, both of which use this so voice and text
chat behave identically).

A focus area is distinct from an MCP module (integrations/client.py,
`mcp.servers:` — a technical bundle of tools + data for one area of
expertise) and from a routing agent (shared/agents_config.py, `agents:` — a
tool-call target). It's the set of topics the user most wants Dann to engage
on: something worth leaning into conversationally, whether or not it happens
to be backed by a module's tools.
"""
from typing import Any


def build_focus_areas_prompt(focus_areas_cfg: list[dict[str, Any]] | None) -> str:
    """Render config.yaml's `focus_areas:` list into a "FOCUS AREAS" section
    for the system prompt. Entries with enabled: false, or missing a
    name/description, are skipped. Returns "" if focus_areas_cfg is
    empty/missing so callers can always safely append the result."""
    if not focus_areas_cfg:
        return ""

    lines = [
        "FOCUS AREAS — topics the user especially wants to engage on. Lean "
        "in: ask follow-up questions, offer relevant detail, and "
        "proactively connect new information to these interests when it's "
        "a natural fit. Don't force it into unrelated turns.",
    ]
    for area in focus_areas_cfg:
        if not area.get("enabled", True):
            continue
        name = area.get("name")
        desc = (area.get("description") or "").strip()
        if not name or not desc:
            continue
        module = area.get("module")
        module_hint = f" [module: {module}]" if module else ""
        lines.append(f"- {name}{module_hint}: {desc}")

    return "\n".join(lines)
