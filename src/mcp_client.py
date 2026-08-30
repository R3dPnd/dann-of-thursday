"""MCP client manager — bridges async MCP SDK to sync orchestrator.

Each entry in config.yaml's mcp.servers list is a module with one of two
lifecycles:

  always_on: true    Connected immediately in start() and stays up for the
                      whole session — for modules used on nearly every turn
                      (e.g. the "projects"/Claude Code bridge). This is the
                      "easy switch": flip the flag, restart Dann.
  always_on: false    (default) Registered but NOT started. Three synthetic
                      meta-tools (list_modules / enable_module /
                      disable_module) are exposed instead, so the LLM
                      starts a module's process only when a turn actually
                      needs it, and can stop it again afterwards. This is
                      what keeps both the tool list sent to Ollama and the
                      process count small as more modules get added.
"""

import asyncio
import logging
import threading
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

_log = logging.getLogger(__name__)

_META_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_modules",
            "description": (
                "List optional Dann modules and whether each is active or "
                "just available. Call this when the user asks what you can "
                "do, or if you're unsure whether a module is already on."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "enable_module",
            "description": (
                "Start an optional module so its tools become available. "
                "Call this the first time a turn needs a tool from a module "
                "that isn't active yet — its real tools appear right after."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "module": {
                        "type": "string",
                        "description": "Module name, e.g. 'schedule', 'notes', 'system'.",
                    }
                },
                "required": ["module"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "disable_module",
            "description": "Stop an active optional module to free resources once it's no longer needed this session.",
            "parameters": {
                "type": "object",
                "properties": {"module": {"type": "string"}},
                "required": ["module"],
            },
        },
    },
]
_META_TOOL_NAMES = {t["function"]["name"] for t in _META_TOOLS}


class MCPManager:
    """Maintains MCP server connections in a background event loop.

    All public methods are synchronous so the rest of the (sync) codebase
    can call them directly.  Internally, coroutines are dispatched to a
    dedicated asyncio loop running on a daemon thread.
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._configs: dict[str, dict[str, Any]] = {}   # name -> full config
        # name -> (owning task, its stop event). stdio_client's cancel scope
        # (anyio) must be entered and exited in the *same* asyncio Task, so
        # each connected server gets one long-lived task that opens its own
        # stdio_client/ClientSession and only tears them down when its own
        # stop event fires — connect and disconnect are just signals to it,
        # not separate enter/exit calls from different tasks.
        self._server_tasks: dict[str, tuple[asyncio.Task, asyncio.Event]] = {}
        self._sessions: dict[str, ClientSession] = {}
        self._tool_map: dict[str, str] = {}              # tool_name -> server_name
        self._tools: list[dict[str, Any]] = []            # Ollama-formatted tool defs, mutated in place

    # ------------------------------------------------------------------
    # Sync public API
    # ------------------------------------------------------------------

    def start(self, server_configs: list[dict[str, Any]]) -> None:
        """Start background loop, connect always_on servers, register the rest."""
        self._thread.start()
        for cfg in server_configs:
            self._configs[cfg["name"]] = cfg

        always_on = [cfg for cfg in server_configs if cfg.get("always_on")]
        on_demand = [cfg for cfg in server_configs if not cfg.get("always_on")]

        for cfg in always_on:
            try:
                self._run(self._connect_one(cfg))
            except Exception:
                pass  # already logged in _connect_one; one bad always_on server shouldn't block startup

        if on_demand:
            self._tools.extend(_META_TOOLS)

    def stop(self) -> None:
        """Disconnect all servers and tear down the event loop."""
        for name in list(self._server_tasks):
            try:
                self._run(self._disconnect_one(name))
            except Exception:
                pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)

    @property
    def tools(self) -> list[dict[str, Any]]:
        """Ollama-formatted tool definitions — always_on tools, meta-tools (if
        any on-demand modules are configured), and whatever's currently
        enabled on demand. Returns the live list; enabling a module mid-turn
        is visible to the caller immediately since this isn't copied."""
        return self._tools

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool — routed to the owning MCP server, or handled
        locally if it's one of the module-management meta-tools."""
        if tool_name in _META_TOOL_NAMES:
            return self._run(self._handle_meta_tool(tool_name, arguments))
        return self._run(self._async_call_tool(tool_name, arguments))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run(self, coro: Any, timeout: float = 30) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    async def _server_owner_task(
        self, cfg: dict[str, Any], ready: asyncio.Future, stop: asyncio.Event
    ) -> None:
        """Owns one server's stdio_client/ClientSession for their whole
        lifetime — opened and closed within this single task, as anyio's
        cancel scopes require. Resolves *ready* once connected (or with the
        connect exception), then just waits for *stop* before letting the
        `async with` blocks close themselves."""
        name = cfg["name"]
        try:
            params = StdioServerParameters(
                command=cfg["command"],
                args=cfg.get("args", []),
                env=cfg.get("env"),
            )
            async with stdio_client(params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    tools_result = await session.list_tools()
                    if not ready.done():
                        ready.set_result((session, tools_result.tools))
                    await stop.wait()
        except Exception as exc:
            if not ready.done():
                ready.set_exception(exc)
            else:
                print(f"[mcp] '{name}' server task error: {exc}", flush=True)
        finally:
            self._server_tasks.pop(name, None)

    async def _connect_one(self, cfg: dict[str, Any]) -> None:
        name = cfg["name"]
        if name in self._sessions:
            return

        ready: asyncio.Future = self._loop.create_future()
        stop = asyncio.Event()
        task = self._loop.create_task(self._server_owner_task(cfg, ready, stop))
        self._server_tasks[name] = (task, stop)

        try:
            session, tools = await ready
        except Exception as exc:
            print(f"[mcp] Failed to connect to '{name}': {exc}", flush=True)
            raise

        for tool in tools:
            self._tool_map[tool.name] = name
            self._tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema,
                },
            })
        self._sessions[name] = session
        print(f"[mcp] Connected to '{name}' — {len(tools)} tool(s) available", flush=True)

    async def _disconnect_one(self, name: str) -> None:
        self._sessions.pop(name, None)
        removed = {tn for tn, sn in self._tool_map.items() if sn == name}
        for tn in removed:
            del self._tool_map[tn]
        self._tools[:] = [t for t in self._tools if t["function"]["name"] not in removed]

        entry = self._server_tasks.get(name)
        if entry:
            task, stop = entry
            stop.set()
            try:
                await asyncio.wait_for(task, timeout=5)
            except Exception:
                pass
        print(f"[mcp] Disconnected '{name}'", flush=True)

    async def _handle_meta_tool(self, name: str, args: dict[str, Any]) -> str:
        if name == "list_modules":
            lines = []
            for mod_name, cfg in self._configs.items():
                if cfg.get("always_on"):
                    continue  # not toggleable at runtime, don't clutter the list
                status = "active" if mod_name in self._sessions else "available"
                lines.append(f"- {mod_name} ({status}): {cfg.get('description', '')}")
            return "Optional modules:\n" + "\n".join(lines) if lines else "No optional modules configured."

        module = args.get("module", "")
        cfg = self._configs.get(module)
        if not cfg:
            known = ", ".join(n for n, c in self._configs.items() if not c.get("always_on"))
            return f"Unknown module '{module}'. Available: {known or 'none'}"
        if cfg.get("always_on"):
            return f"'{module}' is always on, nothing to do."

        if name == "enable_module":
            if module in self._sessions:
                return f"'{module}' is already active."
            try:
                await self._connect_one(cfg)
            except Exception as exc:
                return f"Failed to start '{module}': {exc}"
            new_tools = sorted(tn for tn, sn in self._tool_map.items() if sn == module)
            return f"'{module}' is now active. New tools available: {', '.join(new_tools)}"

        if name == "disable_module":
            if module not in self._sessions:
                return f"'{module}' isn't active."
            await self._disconnect_one(module)
            return f"'{module}' stopped."

        return f"Unknown meta tool '{name}'"

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
