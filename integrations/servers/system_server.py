#!/usr/bin/env python3
"""MCP server for local macOS system control.

Deliberately conservative tool surface — volume, opening apps, locking the
screen. No shutdown/restart/quit-arbitrary-app tools: those are destructive
or disruptive enough that they shouldn't be one voice command away.

Exposed tools:
  get_volume   — read current output volume
  set_volume   — set output volume (0-100)
  open_app     — activate/launch an application by name
  lock_screen  — lock the screen (does not log out or close anything)
"""
import subprocess

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("system")


def _osascript(script: str) -> str:
    result = subprocess.run(
        ["osascript", "-e", script], capture_output=True, text=True, timeout=10
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "osascript failed")
    return result.stdout.strip()


@mcp.tool()
def get_volume() -> str:
    """Get the current system output volume (0-100)."""
    try:
        out = _osascript("output volume of (get volume settings)")
        return f"Volume is {out}."
    except RuntimeError as exc:
        return f"Couldn't read volume: {exc}"


@mcp.tool()
def set_volume(level: int) -> str:
    """Set the system output volume.

    Args:
        level: Volume level from 0 (mute) to 100 (max).
    """
    level = max(0, min(100, int(level)))
    try:
        _osascript(f"set volume output volume {level}")
        return f"Volume set to {level}."
    except RuntimeError as exc:
        return f"Couldn't set volume: {exc}"


@mcp.tool()
def open_app(app_name: str) -> str:
    """Open (or bring to front) an application by name.

    Args:
        app_name: The application's name as it appears in Finder, e.g. "Safari".
    """
    safe_name = app_name.replace('"', '\\"')
    try:
        _osascript(f'tell application "{safe_name}" to activate')
        return f"Opened {app_name}."
    except RuntimeError as exc:
        return f"Couldn't open '{app_name}': {exc}"


@mcp.tool()
def lock_screen() -> str:
    """Lock the screen. Does not log out, quit apps, or shut down."""
    try:
        subprocess.run(
            [
                "/System/Library/CoreServices/Menu Extras/User.menu/Contents/Resources/CGSession",
                "-suspend",
            ],
            check=True,
            capture_output=True,
            timeout=10,
        )
        return "Screen locked."
    except Exception as exc:
        return f"Couldn't lock the screen: {exc}"


if __name__ == "__main__":
    mcp.run()
