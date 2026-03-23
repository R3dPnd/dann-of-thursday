"""MCP client manager — bridges async MCP SDK to sync orchestrator."""

import asyncio
import logging
import threading
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

_log = logging.getLogger(__name__)


class MCPManager:
    """Maintains persistent MCP server connections in a background event loop.

    All public methods are synchronous so the rest of the (sync) codebase
    can call them directly.  Internally, coroutines are dispatched to a
    dedicated asyncio loop running on a daemon thread.
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._exit_stack: AsyncExitStack | None = None
        self._sessions: dict[str, ClientSession] = {}
        self._tool_map: dict[str, str] = {}  # tool_name -> server_name
        self._tools: list[dict[str, Any]] = []  # Ollama-formatted tool defs

    # ------------------------------------------------------------------
    # Sync public API
    # ------------------------------------------------------------------

    def start(self, server_configs: list[dict[str, Any]]) -> None:
        """Start background loop and connect to all configured MCP servers."""
        self._thread.start()
        self._run(self._connect_all(server_configs))

    def stop(self) -> None:
        """Disconnect all servers and tear down the event loop."""
        if self._exit_stack:
            try:
                self._run(self._disconnect_all())
            except Exception:
                # anyio cancel-scope errors on cross-thread teardown are expected
                pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)

    @property
    def tools(self) -> list[dict[str, Any]]:
        """Ollama-formatted tool definitions from all connected servers."""
        return self._tools

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool on the appropriate MCP server (blocking)."""
        return self._run(self._async_call_tool(tool_name, arguments))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run(self, coro: Any, timeout: float = 30) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    async def _connect_all(self, server_configs: list[dict[str, Any]]) -> None:
        self._exit_stack = AsyncExitStack()
        for cfg in server_configs:
            name = cfg["name"]
            try:
                params = StdioServerParameters(
                    command=cfg["command"],
                    args=cfg.get("args", []),
                    env=cfg.get("env"),
                )
                read_stream, write_stream = await self._exit_stack.enter_async_context(
                    stdio_client(params)
                )
                session = await self._exit_stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )
                await session.initialize()
                self._sessions[name] = session

                tools_result = await session.list_tools()
                for tool in tools_result.tools:
                    self._tool_map[tool.name] = name
                    self._tools.append({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": tool.inputSchema,
                        },
                    })
                print(
                    f"[mcp] Connected to '{name}' — "
                    f"{len(tools_result.tools)} tool(s) available",
                    flush=True,
                )
            except Exception as exc:
                print(f"[mcp] Failed to connect to '{name}': {exc}", flush=True)

    async def _disconnect_all(self) -> None:
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._exit_stack = None
        self._sessions.clear()
        self._tool_map.clear()
        self._tools.clear()

    async def _async_call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        server_name = self._tool_map.get(tool_name)
        if not server_name:
            raise ValueError(f"Unknown MCP tool: {tool_name}")
        session = self._sessions[server_name]
        result = await session.call_tool(tool_name, arguments)
        parts = []
        for content in result.content:
            if hasattr(content, "text"):
                parts.append(content.text)
            else:
                parts.append(str(content))
        return "\n".join(parts) if parts else ""
