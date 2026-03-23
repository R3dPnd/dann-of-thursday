"""Ollama API client for local LLM inference with optional MCP tool calling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import requests

if TYPE_CHECKING:
    from src.mcp_client import MCPManager

_MAX_TOOL_ROUNDS = 5


def generate_response(
    prompt: str,
    *,
    base_url: str = "http://localhost:11434",
    model: str = "llama3.2",
    system_prompt: str = "You are a concise voice assistant. Keep responses brief.",
    temperature: float = 0.7,
    max_tokens: int = 150,
    tools: list[dict[str, Any]] | None = None,
    mcp: MCPManager | None = None,
    history: list[dict[str, Any]] | None = None,
) -> str:
    """Send prompt to Ollama and return generated text.

    When *tools* and *mcp* are provided the model may make tool calls;
    results are fed back automatically until the model emits a final
    text response (up to ``_MAX_TOOL_ROUNDS`` iterations).

    *history* is a list of prior ``{"role": ..., "content": ...}`` messages
    from the current session, prepended before the current user turn.
    """
    url = f"{base_url.rstrip('/')}/api/chat"

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    for _ in range(_MAX_TOOL_ROUNDS):
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if tools:
            payload["tools"] = tools

        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        msg = resp.json().get("message", {})

        tool_calls = msg.get("tool_calls")
        if not tool_calls or not mcp:
            return msg.get("content", "").strip()

        messages.append(msg)
        for tc in tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "")
            args = func.get("arguments", {})
            print(f"[mcp] Calling tool '{name}'...", flush=True)
            try:
                result = mcp.call_tool(name, args)
            except Exception as exc:
                result = f"Error calling {name}: {exc}"
            messages.append({"role": "tool", "content": str(result)})

    return msg.get("content", "").strip()
