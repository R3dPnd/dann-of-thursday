"""Self-restart via process-image replacement (os.execv) — re-runs the
current process from scratch with the same argv/env, reloading all Python
modules. Callers MUST stop their own audio streams / subprocess children
first: execv does not run atexit/shutdown handlers, and leaked file
descriptors (an open mic stream, an MCP server subprocess) would survive
into a state where nothing owns them — the exact class of orphaned-process
bug that cost a long manual cleanup earlier in this project's history.
"""
import os
import sys


def restart_process() -> None:
    os.execv(sys.executable, [sys.executable] + sys.argv)
