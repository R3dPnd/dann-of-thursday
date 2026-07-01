#!/usr/bin/env python3
"""MCP server that discovers local Git projects and opens Claude Code sessions.

Exposed tools:
  list_projects      — return configured (or auto-discovered) git repos
  open_claude_code   — open a new Terminal running `claude` in a project directory
  ask_claude_code    — run Claude Code non-interactively and return its response
  ask_claude         — route a query to Claude API for deep reasoning/analysis
"""

import os
import re
import subprocess
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("claude-code")

# Directories that are never git repos and may be very large — skip them entirely
_SKIP_DIRS = {
    ".venv", "venv", "node_modules", "__pycache__",
    "dist", "build", "target", ".gradle", ".terraform",
    "vendor", ".tox", ".eggs", "*.egg-info",
}

# Cache so repeated tool calls in the same session don't re-scan the filesystem
_projects_cache: list[dict] | None = None


def _load_configured_projects() -> list[dict] | None:
    """Return the explicit project list from config.yaml, or None if not defined."""
    try:
        from src.config import load_config
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


def _find_projects() -> list[dict]:
    """Return the project list.

    Priority:
    1. ``projects`` list in config.yaml — explicit, config-driven
    2. Auto-discovery under ~/Git — fallback when no config list is present
    """
    global _projects_cache
    if _projects_cache is not None:
        return _projects_cache

    configured = _load_configured_projects()
    if configured is not None:
        _projects_cache = configured
    else:
        _projects_cache = _discover_projects([Path.home() / "Git"])

    return _projects_cache


def _normalise(name: str) -> str:
    """Lowercase and replace hyphens/underscores with spaces for fuzzy matching."""
    return re.sub(r"[-_]", " ", name.lower()).strip()


def _resolve_project(name: str) -> dict | None:
    """Find a project by normalised exact name, then partial match."""
    projects = _find_projects()
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
    projects = _find_projects()
    if not projects:
        return "No projects found."
    lines = [f"- {p['name']}  ({p['path']})" for p in projects]
    return "Available projects:\n" + "\n".join(lines)


@mcp.tool()
def open_claude_code(project_name: str, task: str = "") -> str:
    """Open an interactive Claude Code session in a new terminal window.

    Use this when the user wants to start a coding session and interact with
    Claude Code themselves. For getting an answer back immediately, use
    ask_claude_code instead.

    Args:
        project_name: Name (or partial name) of the project to open.
        task: Optional task description passed to Claude as the initial prompt.
    """
    project = _resolve_project(project_name)
    if not project:
        available = ", ".join(p["name"] for p in _find_projects())
        return (
            f"Project '{project_name}' not found. "
            f"Available projects: {available or 'none'}"
        )

    proj_path = project["path"]

    if task:
        safe_task = task.replace("'", "'\\''")
        shell_cmd = f"cd '{proj_path}' && claude '{safe_task}'"
    else:
        shell_cmd = f"cd '{proj_path}' && claude"

    apple_script = f'tell application "Terminal" to do script "{shell_cmd}"'
    result = subprocess.run(
        ["osascript", "-e", apple_script],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return f"Failed to open terminal: {result.stderr.strip()}"

    msg = f"Opened Claude Code in '{project['name']}' at {proj_path}"
    if task:
        msg += f' — starting with task: "{task}"'
    return msg


@mcp.tool()
def ask_claude_code(project_name: str, task: str) -> str:
    """Ask Claude Code a question about a project and get the answer back.

    Runs Claude Code non-interactively and returns its response so it can
    be spoken aloud. Use this for questions like "summarise the project",
    "what does this function do", or "what tests are missing".

    For starting an interactive session where the user drives the conversation,
    use open_claude_code instead.

    Args:
        project_name: Name (or partial name) of the project to query.
        task: The question or instruction for Claude Code.
    """
    project = _resolve_project(project_name)
    if not project:
        available = ", ".join(p["name"] for p in _find_projects())
        return (
            f"Project '{project_name}' not found. "
            f"Available projects: {available or 'none'}"
        )

    proj_path = project["path"]

    result = subprocess.run(
        ["claude", "-p", task],
        cwd=proj_path,
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        err = result.stderr.strip()
        return f"Claude Code returned an error: {err or 'unknown error'}"

    output = result.stdout.strip()
    return output if output else "Claude Code returned an empty response."


@mcp.tool()
def ask_claude(question: str, context: str = "") -> str:
    """Route a question to Claude for deep reasoning or analysis.

    Use this when the query needs broad knowledge, nuanced reasoning, detailed
    explanation, or creative thinking beyond what the local model handles well.
    Returns a spoken-friendly response (no markdown, 1-3 sentences).

    Args:
        question: The question or task for Claude.
        context: Optional extra context to help Claude give a better answer.
    """
    prompt = f"{context.strip()}\n\n{question.strip()}" if context.strip() else question.strip()

    # Append TTS formatting instruction so the response is speakable
    full_prompt = (
        f"{prompt}\n\n"
        "Answer in 1-3 concise spoken sentences. "
        "No markdown, bullet points, or formatting — your response will be read aloud."
    )

    result = subprocess.run(
        ["claude", "-p", full_prompt],
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode != 0:
        err = result.stderr.strip()
        return f"Claude returned an error: {err or 'unknown error'}"

    output = result.stdout.strip()
    return output if output else "Claude returned an empty response."


if __name__ == "__main__":
    mcp.run()
