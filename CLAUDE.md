# Dann of Thursday — Claude Code Context

## What this is

A local voice AI agent: say **"ok Dann"** → STT (faster-whisper) → Ollama LLM → Piper TTS response.
Also has a React dashboard for live state/debug and a FastAPI backend that bridges the two.

## Directory layout

```
src/          Voice pipeline (wake word → STT → LLM → TTS)
app/          FastAPI backend + API (serves dashboard state, REST, WebSocket)
ui/           React + Tailwind dashboard (Vite dev server)
models/       Wake word models, Piper voice, openwakeword submodule
config.yaml   Runtime config (audio, STT, TTS, wake word, projects)
```

## Key entry points

| What | Command |
|------|---------|
| Voice agent (full) | `.venv/bin/python -m src.main` |
| Backend only (no mic) | `NO_VOICE=1 .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000` |
| UI dev server | `cd ui && npm run dev` (port 3000) |

## Architecture

```
wake word (Picovoice / openwakeword)
  → src/orchestrator.py  ← core pipeline, emits events via src/event_bus.py
  → src/stt/whisper.py   ← faster-whisper transcription
  → src/llm/ollama.py    ← Ollama LLM (local models)
  → src/tts/piper.py     ← Piper TTS synthesis
  → src/audio/playback.py

app/main.py              FastAPI — port 8000
  app/api/v1/endpoints/  state, events (WS), logs, metrics, projects,
                         terminals, runs, notes, prompt_builder, voice, mcp, tools
  app/services/          terminal_service, run_service, mcp_service, etc.

ui/src/                  React dashboard (port 3000 dev / served from FastAPI prod)
```

## Python venv

Always use `.venv/bin/python` / `.venv/bin/pip` (macOS). `requirements.txt` needs Python 3.10+ —
`scripts/setup.sh` checks for this and installs a newer interpreter via Homebrew if the system
`python3` is too old. `mcp` is pinned to `1.29.1` — 2.x removed `mcp.server.fastmcp.FastMCP`, which
every MCP server module here uses.

## MCP integration

`src/mcp_client.py`'s `MCPManager` connects to the MCP servers listed under `mcp.servers` in
`config.yaml` — each one a separate module (own process, own dependencies):

- `src/mcp_servers/claude_code_server.py` (`projects`, `always_on: true`) — project discovery
  and the Claude Code bridge. Sessions with `SessionMode.CODE` bypass Ollama and route to
  Claude Code directly.
- `src/mcp_servers/schedule_server.py` (`schedule`) — Google Calendar (needs one-time OAuth
  setup, see `schedule-setup.md`).
- `src/mcp_servers/notes_server.py` (`notes`) — local notes/reminders, `~/.dann/notes.json`.
- `src/mcp_servers/system_server.py` (`system`) — volume/open-app/lock-screen via `osascript`.

Modules default to `always_on: false` — registered but not started. `MCPManager` exposes three
meta-tools (`list_modules`, `enable_module`, `disable_module`) so the LLM starts a module's
process only when a turn actually needs it, keeping both the tool schema sent to Ollama and the
process count small as more modules get added. `always_on: true` is the escape hatch for a
module that should just always be connected (edit config, restart).

Personal per-machine module data (OAuth tokens, notes) lives under `~/.dann/`, not in the repo —
see `src/mcp_servers/_store.py`.

## Config

`config.yaml` (from `config.example.yaml`) controls wake word model path, Picovoice access key,
Ollama model name, Piper voice model path, audio device, and project list.
