#!/usr/bin/env python3
"""MCP server that discovers local Git projects and opens Claude Code sessions.

Everything here runs through the `claude` CLI — your existing Claude Code
subscription — never the billed Anthropic API.

Exposed tools:
  list_projects      — return configured (or auto-discovered) git repos
  open_claude_code   — open a new Terminal running `claude` in a project directory
  ask_claude_code    — run Claude Code non-interactively and return its response
  ask_claude         — route a query to Claude for deep reasoning/analysis
  web_search         — current/up-to-date info via Claude's built-in web search
"""

import os
import re
import subprocess
from pathlib import Path

import requests
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("claude-code")

# open_claude_code targets the FastAPI dashboard's own Claude Code terminal
# pool (POST /api/v1/terminals/claude-code) when it's reachable, so a session
# it starts/continues shows up live in the dashboard's terminal pane —
# specifically the same one the work stream that triggered it is watching,
# via the focus_area passed alongside. Falls back to a native Terminal.app
# window when the dashboard isn't running, e.g. plain `python -m voice.main`
# with no API.
_API_BASE_URL = os.environ.get("DANN_API_BASE_URL", "http://localhost:8000")

# Directories that are never git repos and may be very large — skip them entirely
_SKIP_DIRS = {
    ".venv", "venv", "node_modules", "__pycache__",
    "dist", "build", "target", ".gradle", ".terraform",
    "vendor", ".tox", ".eggs", "*.egg-info",
}

# Cache so repeated tool calls in the same session don't re-scan the filesystem
_projects_cache: list[dict] | None = None

# Where auto-discovery falls back to when config.yaml has no `projects:` list.
# A module-level constant (rather than inlined in find_projects()) so tests
# can patch it without touching the real filesystem.
_SEARCH_ROOTS: list[Path] = [Path.home() / "Git"]


def _load_configured_projects() -> list[dict] | None:
    """Return the explicit project list from config.yaml, or None if not defined."""
    try:
        from voice.config import load_config
        cfg = load_config()
        entries = cfg.get("projects")
        if not entries or not isinstance(entries, list):
            return None
        projects = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            path = Path(str(entry.get("path", ""))).expanduser()
            name = entry.get("name") or path.name
            if path.exists():
                project: dict = {"name": str(name), "path": str(path)}
                if "run" in entry:
                    project["run"] = str(entry["run"])
                projects.append(project)
        return sorted(projects, key=lambda p: p["name"].lower()) if projects else None
    except Exception:
        return None


def _discover_projects(search_roots: list[Path]) -> list[dict]:
    """Walk search_roots and return all git repos found."""
    projects = []
    for root in search_roots:
        if not root.exists():
            continue
        for dirpath, dirnames, _ in os.walk(root):
            dirnames[:] = [
                d for d in dirnames
                if d not in _SKIP_DIRS and not d.startswith(".")
            ]
            if ".git" in os.listdir(dirpath):
                projects.append({"name": Path(dirpath).name, "path": dirpath})
                dirnames.clear()
    return sorted(projects, key=lambda p: p["name"].lower())


def find_projects() -> list[dict]:
    """Return the project list.

    Priority:
    1. ``projects`` list in config.yaml — explicit, config-driven
    2. Auto-discovery under ~/Git — fallback when no config list is present

    Public (no leading underscore) because this is the canonical source of
    Dann's project list — other MCP modules needing project resolution
    (e.g. devteam_server.py) import it directly rather than duplicating
    discovery logic.
    """
    global _projects_cache
    if _projects_cache is not None:
        return _projects_cache

    configured = _load_configured_projects()
    if configured is not None:
        _projects_cache = configured
    else:
        _projects_cache = _discover_projects(_SEARCH_ROOTS)

    return _projects_cache


def _normalise(name: str) -> str:
    """Lowercase and replace hyphens/underscores with spaces for fuzzy matching."""
    return re.sub(r"[-_]", " ", name.lower()).strip()


def resolve_project(name: str) -> dict | None:
    """Find a project by normalised exact name, then partial match.

    Public alongside find_projects() — see that docstring."""
    projects = find_projects()
    needle = _normalise(name)
    for p in projects:
        if _normalise(p["name"]) == needle:
            return p
    for p in projects:
        if needle in _normalise(p["name"]):
            return p
    return None


@mcp.tool()
def list_projects() -> str:
    """List all Git projects available on this machine."""
    projects = find_projects()
    if not projects:
        return "No projects found."
    lines = [f"- {p['name']}  ({p['path']})" for p in projects]
    return "Available projects:\n" + "\n".join(lines)


def _open_native_terminal(project: dict, task: str) -> str:
    """Fallback when the dashboard API isn't reachable: open a real
    Terminal.app window, same as this tool's original behaviour.

    Not an MCP tool itself — only called directly by open_claude_code below.
    Its `project: dict` signature isn't something an LLM should construct."""
    proj_path = project["path"]
    if task:
        safe_task = task.replace("'", "'\\''")
        shell_cmd = f"cd '{proj_path}' && claude '{safe_task}'"
    else:
        shell_cmd = f"cd '{proj_path}' && claude"

    apple_script = f'tell application "Terminal" to do script "{shell_cmd}"'
    result = subprocess.run(["osascript", "-e", apple_script], capture_output=True, text=True)
    if result.returncode != 0:
        return f"Failed to open terminal: {result.stderr.strip()}"

    msg = f"Opened Claude Code in '{project['name']}' at {proj_path}"
    if task:
        msg += f' — starting with task: "{task}"'
    return msg


# Server-side, a fresh session waits up to ~15s to clear its startup banner
# (wait_for_quiet) before a task is even sent, then up to ~25s more for a
# response or a confirmed-done quiet gap (send_and_capture) — the client
# timeout has to comfortably exceed the combined worst case.
_TERMINAL_REQUEST_TIMEOUT = 60


# Terminal output is raw, ANSI-stripped-but-still-noisy TUI text — capping
# it keeps the router model's next call fast (a long tail here means a
# bigger prompt, which on a small local model means real added latency, not
# just a bigger response) and keeps the excerpt focused on the actual
# answer, which tends to land at the end of the capture rather than the
# start (spinners/banners/thinking chrome come first).
_MAX_OUTPUT_CHARS = 1200


def _format_terminal_result(data: dict, label: str, task: str) -> str:
    """Build the tool's return message from a /terminals/claude-code or
    /terminals/dann response. When Claude Code answered (or made visible
    progress) within the wait window, that becomes the result to summarize;
    otherwise it's just confirmation of where to watch."""
    verb = "Resumed" if data.get("continued") else "Opened"
    output = (data.get("output") or "").strip()
    settled = data.get("settled", True)

    if output and len(output) > _MAX_OUTPUT_CHARS:
        output = "…" + output[-_MAX_OUTPUT_CHARS:]

    if output and settled:
        return f"Claude Code ({label}): {output}"
    if output:
        return (
            f"Claude Code is still working on this in '{label}' — here's what's "
            f"come through so far, check the terminal for the rest: {output}"
        )
    msg = f"{verb} Claude Code for '{label}' — see the dashboard's terminal pane."
    if task:
        msg += f' Task: "{task}"'
    return msg


@mcp.tool()
def open_claude_code(project_name: str, task: str = "", focus_area: str = "") -> str:
    """Open an interactive Claude Code session for a project, or hand it a
    task — the response (or a note that it's still working) is sent to the
    dashboard's watchable terminal and summarized back to you. Calling this
    again for the same project resumes that same session rather than
    starting a fresh one — hand it follow-up tasks the same way you started
    it.

    Args:
        project_name: Name (or partial name) of the project to open.
        task: Optional task description passed to Claude as the initial prompt.
        focus_area: Do not set this — the caller supplies it automatically.
    """
    project = resolve_project(project_name)
    if not project:
        if focus_area:
            # project_name didn't match a real git repo, but we know which
            # focus area this conversation belongs to — that's not a
            # codebase, so open a plain terminal for it instead of just
            # failing and hoping the model retries with the right tool.
            return open_terminal(task=task, focus_area=focus_area)
        available = ", ".join(p["name"] for p in find_projects())
        return f"Project '{project_name}' not found. Available projects: {available or 'none'}."

    try:
        resp = requests.post(
            f"{_API_BASE_URL}/api/v1/terminals/claude-code",
            json={"project_name": project["name"], "task": task, "focus_area": focus_area},
            timeout=_TERMINAL_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return _format_terminal_result(resp.json(), project["name"], task)
    except requests.RequestException:
        # Dashboard API not running (e.g. standalone voice mode) — fall back
        # to a real Terminal.app window so this still works either way.
        return _open_native_terminal(project, task)


@mcp.tool()
def open_terminal(task: str = "", focus_area: str = "") -> str:
    """Open or continue an interactive Claude Code terminal for the current
    conversation, when the user asks to use "the terminal" or "Claude Code"
    but there's no specific git project involved — e.g. a focus area that's
    a hobby or interest (BJJ, gardening), not a codebase. The response (or a
    note that it's still working) is sent to the dashboard's watchable
    terminal and summarized back to you. Calling this again in the same
    conversation resumes that same terminal rather than starting a fresh
    one. For work on an actual project, use open_claude_code instead.

    Args:
        task: What to ask/tell Claude Code once the terminal is open.
        focus_area: Do not set this — the caller supplies it automatically.
    """
    if not focus_area:
        return "No terminal available — this conversation isn't tied to a focus area."

    try:
        resp = requests.post(
            f"{_API_BASE_URL}/api/v1/terminals/dann",
            json={"focus_area": focus_area, "task": task},
            timeout=_TERMINAL_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return _format_terminal_result(resp.json(), focus_area, task)
    except requests.RequestException:
        return "Could not open a terminal — the dashboard isn't reachable."


@mcp.tool()
def ask_claude_code(project_name: str, task: str, focus_area: str = "") -> str:
    """Ask Claude Code something about a project — sent to the same live
    terminal open_claude_code uses, so the user can watch it answer, rather
    than a hidden one-shot call. Use this for questions like "summarise the
    project", "what does this function do", or "what tests are missing".
    Same underlying terminal as open_claude_code — calling either for the
    same project/focus area continues the same conversation. The answer is
    not returned directly here; tell the user to check the terminal pane.

    Args:
        project_name: Name (or partial name) of the project to ask about.
        task: The question or instruction for Claude Code.
        focus_area: Do not set this — the caller supplies it automatically.
    """
    return open_claude_code(project_name, task, focus_area)


@mcp.tool()
def ask_claude(question: str, context: str = "", depth: str = "deep") -> str:
    """Route a question to Claude for deep reasoning or analysis.

    Use this when the query needs broad knowledge, nuanced reasoning, detailed
    explanation, or creative thinking beyond what the local model handles well.
    Runs through the `claude` CLI (your existing Claude Code subscription) —
    not the billed Anthropic API. Returns a spoken-friendly response (no
    markdown, 1-3 sentences).

    Args:
        question: The question or task for Claude.
        context: Optional extra context to help Claude give a better answer.
        depth: "fast" for a quick/cheap answer (haiku), "deep" (default) for
            the CLI's normal model — better for hard reasoning or analysis.
    """
    prompt = f"{context.strip()}\n\n{question.strip()}" if context.strip() else question.strip()

    # Append TTS formatting instruction so the response is speakable
    full_prompt = (
        f"{prompt}\n\n"
        "Answer in 1-3 concise spoken sentences. "
        "No markdown, bullet points, or formatting — your response will be read aloud."
    )

    cmd = ["claude"]
    if depth == "fast":
        cmd += ["--model", "haiku"]
    cmd += ["-p", full_prompt]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode != 0:
        err = result.stderr.strip()
        return f"Claude returned an error: {err or 'unknown error'}"

    output = result.stdout.strip()
    return output if output else "Claude returned an empty response."


@mcp.tool()
def web_search(query: str) -> str:
    """Search the web for current information — news, prices, recent
    releases, anything that might be after the local model's knowledge
    cutoff. Runs through the `claude` CLI's built-in web search (your
    existing subscription), not a separate billed API.

    Args:
        query: What to search for / the question to answer using the web.
    """
    prompt = (
        f"Search the web for: {query.strip()}\n\n"
        "Answer in 1-3 concise spoken sentences based on what you find. "
        "No markdown, bullet points, or formatting — your response may be read aloud."
    )

    try:
        # --allowedTools is variadic (commander.js) — it greedily consumes
        # every following arg as another tool name, including the prompt,
        # unless the prompt comes first.
        result = subprocess.run(
            ["claude", "-p", prompt, "--allowedTools", "WebSearch"],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return "Web search timed out."

    if result.returncode != 0:
        err = result.stderr.strip()
        return f"Web search error: {err or 'unknown error'}"

    output = result.stdout.strip()
    return output if output else "No answer found."


if __name__ == "__main__":
    mcp.run()
