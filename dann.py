#!/usr/bin/env python3
"""
dann CLI — developer tooling for Dann of Thursday.

Usage:
    dann code             Start API + UI dashboard, no voice (alias for `dev`)
    dann dev               Start API + UI dev servers (NO_VOICE=1 by default),
                          opens a browser tab
    dann dev --voice      Start API + UI dev servers with voice pipeline enabled,
                          opens a fullscreen Electron window (not a browser tab)
    dann electron         Start API + Electron dev window (NO_VOICE=1 by default)
    dann electron --voice Start API + Electron dev window with voice pipeline enabled
    dann start            Start API only, with voice pipeline enabled
    dann stop             Stop any running dann dev/electron/start processes —
                          including orphans left behind by a closed terminal or
                          killed shell (Ctrl+C in the launching terminal also
                          works when that terminal is still open; saying
                          "goodbye"/"thanks Dann" to the voice pipeline only
                          ends the current conversation, not the process —
                          use this or Ctrl+C to actually stop Dann)
    dann restart          Stop whatever's currently running and relaunch it with
                          the same mode/flags — also available as the dashboard
                          restart button, or by saying "restart yourself"/
                          "restart Dann" (all manual/human-triggered — Dann never
                          restarts himself on his own after editing his own code)
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error
import webbrowser
from pathlib import Path

REPO = Path(__file__).resolve().parent
_WINDOWS = sys.platform == "win32"
_VENV_BIN = REPO / ".venv" / ("Scripts" if _WINDOWS else "bin")
VENV_PYTHON = _VENV_BIN / ("python.exe" if _WINDOWS else "python")
VENV_UVICORN = _VENV_BIN / ("uvicorn.exe" if _WINDOWS else "uvicorn")
UI_DIR = REPO / "ui"

API_PORT = 8000
UI_PORT  = 3000


def _stream(stream, prefix: str) -> None:
    """Forward a stream to stdout with a label prefix."""
    try:
        for line in stream:
            print(f"{prefix} {line}", end="", flush=True)
    except ValueError:
        pass  # pipe closed


def _wait_for_api(port: int, timeout: float = 30.0, interval: float = 0.15) -> bool:
    """Poll http://127.0.0.1:<port>/health until it returns 200 or timeout expires.

    Returns True if the API came up, False if it timed out.
    """
    url = f"http://127.0.0.1:{port}/api/v1/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(interval)
    return False


def _open_browser_when_ready(port: int, timeout: float = 30.0) -> None:
    """Poll the UI dev server and open it in the default browser once it's
    actually serving — otherwise `dann dev` starts everything with nothing
    visibly appearing on screen unless you know to navigate there yourself."""
    url = f"http://localhost:{port}"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    webbrowser.open(url)
                    return
        except Exception:
            pass
        time.sleep(0.3)


def _launch_electron_when_ready(port: int, timeout: float = 30.0) -> None:
    """Same wait-for-ready pattern as _open_browser_when_ready, but launches
    a fullscreen Electron window instead of a browser tab — used for
    `dann dev --voice`, since running Dann for real should feel like an app,
    not a dev-server tab. Electron's own dev-mode code path (ui/electron/
    main.cjs) already just points at this same Vite dev server, so this
    reuses it rather than going through the electron:dev npm script (which
    would start a second, redundant Vite instance via `concurrently`)."""
    url = f"http://localhost:{port}"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    npx_cmd = "npx.cmd" if sys.platform == "win32" else "npx"
                    env = {**os.environ, "NODE_ENV": "development", "DANN_FULLSCREEN": "1"}
                    subprocess.Popen([npx_cmd, "electron", "."], cwd=str(UI_DIR), env=env)
                    return
        except Exception:
            pass
        time.sleep(0.3)


def cmd_dev(voice: bool = False) -> None:
    env_api = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if not voice:
        env_api["NO_VOICE"] = "1"

    # ── Start the API first ───────────────────────────────────────────────────
    api_proc = subprocess.Popen(
        [
            str(VENV_UVICORN),
            "app.main:app",
            "--host", "0.0.0.0",
            "--port", str(API_PORT),
        ],
        cwd=str(REPO),
        env=env_api,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    threading.Thread(target=_stream, args=(api_proc.stdout, "\033[36m[api]\033[0m "), daemon=True).start()
    threading.Thread(target=_stream, args=(api_proc.stderr, "\033[36m[api]\033[0m "), daemon=True).start()

    # ── Wait until the API is accepting connections ───────────────────────────
    print(f"\033[36m[dann]\033[0m Waiting for API on :{API_PORT}…", flush=True)
    ready = _wait_for_api(API_PORT, timeout=30.0)
    if not ready:
        print("\033[31m[dann]\033[0m API did not start within 30s — aborting.", flush=True)
        api_proc.terminate()
        sys.exit(1)

    # ── Now start the UI ──────────────────────────────────────────────────────
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
    threading.Thread(target=_stream, args=(ui_proc.stdout, "\033[35m[ui] \033[0m "), daemon=True).start()
    if voice:
        threading.Thread(target=_launch_electron_when_ready, args=(UI_PORT,), daemon=True).start()
    else:
        threading.Thread(target=_open_browser_when_ready, args=(UI_PORT,), daemon=True).start()

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
    print(f"\033[32m[dann]\033[0m API  → http://localhost:{API_PORT}  ({voice_label})", flush=True)
    print(f"\033[32m[dann]\033[0m UI   → http://localhost:{UI_PORT}" + ("  (fullscreen Electron window)" if voice else ""), flush=True)
    print(f"\033[32m[dann]\033[0m Ctrl+C to stop both\n", flush=True)

    # Wait — exit when either process dies
    api_proc.wait()
    ui_proc.terminate()


def cmd_electron(voice: bool = False) -> None:
    """Start API + Electron dev window (Vite hot reload inside Electron)."""
    env_api = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if not voice:
        env_api["NO_VOICE"] = "1"

    # ── Start the API first ───────────────────────────────────────────────────
    api_proc = subprocess.Popen(
        [
            str(VENV_UVICORN),
            "app.main:app",
            "--host", "0.0.0.0",
            "--port", str(API_PORT),
        ],
        cwd=str(REPO),
        env=env_api,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    threading.Thread(target=_stream, args=(api_proc.stdout, "\033[36m[api]\033[0m "), daemon=True).start()
    threading.Thread(target=_stream, args=(api_proc.stderr, "\033[36m[api]\033[0m "), daemon=True).start()

    # ── Wait until the API is accepting connections ───────────────────────────
    print(f"\033[36m[dann]\033[0m Waiting for API on :{API_PORT}…", flush=True)
    ready = _wait_for_api(API_PORT, timeout=30.0)
    if not ready:
        print("\033[31m[dann]\033[0m API did not start within 30s — aborting.", flush=True)
        api_proc.terminate()
        sys.exit(1)

    # ── Now start Electron (concurrently runs Vite + Electron internally) ─────
    npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
    electron_proc = subprocess.Popen(
        [npm_cmd, "run", "electron:dev"],
        cwd=str(UI_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    procs = [api_proc, electron_proc]
    threading.Thread(target=_stream, args=(electron_proc.stdout, "\033[35m[ui] \033[0m "), daemon=True).start()

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
    print(f"\033[32m[dann]\033[0m API      → http://localhost:{API_PORT}  ({voice_label})", flush=True)
    print(f"\033[32m[dann]\033[0m Electron → opening window…", flush=True)
    print(f"\033[32m[dann]\033[0m Ctrl+C to stop all\n", flush=True)

    api_proc.wait()
    electron_proc.terminate()


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


def cmd_stop() -> None:
    """Find and stop every dann-related process on this machine: the
    dann.py dev/electron/start launcher, its API (uvicorn) and UI (vite)
    children, and — critically — anything matching those that's still
    running with no live parent (e.g. the terminal that launched it was
    closed, or a previous kill only got part of the process tree). This is
    exactly the kind of cleanup that was previously done by hand, PID by
    PID, after multiple overlapping instances ended up fighting over the
    same microphone.
    """
    import psutil

    def _matches(proc: psutil.Process) -> bool:
        try:
            cmdline = " ".join(proc.cmdline())
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return False
        # Any dann.py invocation, any subcommand — not enumerated, so this
        # doesn't silently miss a future one (e.g. `restart`). Killing
        # children (below) already indirectly unblocks and exits a blocked
        # parent in most cases, but matching the launcher directly here
        # means stop doesn't depend on that happening in the right order.
        if "dann.py" in cmdline:
            return True
        try:
            cwd = proc.cwd()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            cwd = None
        if "app.main:app" in cmdline and cwd == str(REPO):
            return True
        if "vite" in cmdline.lower() and cwd == str(UI_DIR):
            return True
        if "electron" in cmdline.lower() and cwd == str(UI_DIR):
            return True
        return False

    me = os.getpid()
    targets = [
        p for p in psutil.process_iter(["pid"])
        if p.pid != me and _matches(p)
    ]

    if not targets:
        print("[dann] Nothing running.")
        return

    print(f"[dann] Stopping {len(targets)} process(es): {[p.pid for p in targets]}")
    for p in targets:
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass

    _, alive = psutil.wait_procs(targets, timeout=5)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    if alive:
        print(f"[dann] Force-killed {len(alive)} process(es) that didn't exit cleanly: {[p.pid for p in alive]}")

    print("[dann] Stopped.")


def cmd_restart() -> None:
    """Stop whatever's currently running and relaunch it with the same
    mode/flags — the manual, human-triggered restart mechanism (also
    reachable via the dashboard restart button, or by saying "restart
    yourself"/"restart Dann" to the voice pipeline). Finds the live
    dann.py launcher's own argv via psutil rather than requiring you to
    retype the mode/flags."""
    import psutil

    launcher_args: list[str] | None = None
    for p in psutil.process_iter(["pid"]):
        if p.pid == os.getpid():
            continue
        try:
            cmdline = p.cmdline()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        for i, part in enumerate(cmdline):
            if part.endswith("dann.py"):
                launcher_args = cmdline[i + 1:]
                break
        if launcher_args is not None:
            break

    if not launcher_args:
        print("[dann] Nothing running to restart — use `dann dev`/`dann electron`/`dann start` instead.")
        return

    print(f"[dann] Restarting with args: {launcher_args}")
    cmd_stop()
    time.sleep(1)
    _dispatch(launcher_args)


def _dispatch(args: list[str]) -> None:
    if not args:
        print(__doc__)
        sys.exit(0)

    command = args[0]

    if command in ("dev", "code"):
        cmd_dev(voice="--voice" in args)
    elif command == "electron":
        cmd_electron(voice="--voice" in args)
    elif command == "start":
        cmd_start()
    elif command == "stop":
        cmd_stop()
    elif command == "restart":
        cmd_restart()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


def main() -> None:
    _dispatch(sys.argv[1:])


if __name__ == "__main__":
    main()
