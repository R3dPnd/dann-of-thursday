#!/usr/bin/env python3
"""
dann CLI — developer tooling for Dann of Thursday.

Usage:
    dann dev          Start API + UI dev servers (NO_VOICE=1 by default)
    dann dev --voice  Start API + UI dev servers with voice pipeline enabled
    dann start        Start API only, with voice pipeline enabled
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parent
VENV_PYTHON = REPO / ".venv" / "Scripts" / "python.exe"
VENV_UVICORN = REPO / ".venv" / "Scripts" / "uvicorn.exe"
UI_DIR = REPO / "ui"


def _stream(stream, prefix: str) -> None:
    """Forward a stream to stdout with a label prefix."""
    try:
        for line in stream:
            print(f"{prefix} {line}", end="", flush=True)
    except ValueError:
        pass  # pipe closed


def cmd_dev(voice: bool = False) -> None:
    env_api = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if not voice:
        env_api["NO_VOICE"] = "1"

    api_proc = subprocess.Popen(
        [
            str(VENV_UVICORN),
            "app.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
        ],
        cwd=str(REPO),
        env=env_api,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
    ui_proc = subprocess.Popen(
        [npm_cmd, "run", "dev"],
        cwd=str(UI_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    procs = [api_proc, ui_proc]

    threading.Thread(target=_stream, args=(api_proc.stdout, "\033[36m[api]\033[0m "), daemon=True).start()
    threading.Thread(target=_stream, args=(api_proc.stderr, "\033[36m[api]\033[0m "), daemon=True).start()
    threading.Thread(target=_stream, args=(ui_proc.stdout,  "\033[35m[ui] \033[0m "), daemon=True).start()

    def _shutdown(sig=None, frame=None):
        print("\n[dann] Shutting down…", flush=True)
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _shutdown)

    voice_label = "with voice" if voice else "NO_VOICE=1"
    print(f"\033[32m[dann]\033[0m API  → http://localhost:8000  ({voice_label})", flush=True)
    print(f"\033[32m[dann]\033[0m UI   → http://localhost:5173", flush=True)
    print(f"\033[32m[dann]\033[0m Ctrl+C to stop both\n", flush=True)

    # Wait — exit when either process dies
    api_proc.wait()
    ui_proc.terminate()


def cmd_start() -> None:
    """Start API with voice pipeline."""
    api_proc = subprocess.Popen(
        [
            str(VENV_UVICORN),
            "app.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
        ],
        cwd=str(REPO),
    )

    def _shutdown(sig=None, frame=None):
        api_proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _shutdown)

    api_proc.wait()


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)

    command = args[0]

    if command == "dev":
        voice = "--voice" in args
        cmd_dev(voice=voice)
    elif command == "start":
        cmd_start()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
