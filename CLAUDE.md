# Dann of Thursday — Claude Code Context

## What this is

A local voice AI agent: say **"ok Dann"** → STT (faster-whisper) → Ollama LLM → Piper TTS response.
Also has a React dashboard for live state/debug and a FastAPI backend that bridges the two.

## Directory layout

```
voice/          Voice pipeline (wake word → STT → LLM → TTS)
integrations/   MCP client + tool-server modules (projects, schedule, notes, gardening, bjj, devteam, system)
shared/         Cross-cutting code used by both voice/ and app/ (agent routing config, restart)
app/            FastAPI backend + API (serves dashboard state, REST, WebSocket)
ui/             React + Tailwind dashboard (Vite dev server)
models/         Wake word models, Piper voice, openwakeword submodule
docs/           Design/tech-spec/UI-spec docs and setup notes
config.yaml     Runtime config (audio, STT, TTS, wake word, agents, MCP servers)
```

## Key entry points

| What | Command |
|------|---------|
| Voice agent (full) | `.venv/bin/python -m voice.main` |
| Backend only (no mic) | `NO_VOICE=1 .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000` |
| UI dev server | `cd ui && npm run dev` (port 3000) |

## Architecture

```
wake word (Picovoice / openwakeword)
  → voice/orchestrator.py  ← core pipeline, emits events via voice/event_bus.py
  → voice/stt/whisper.py   ← faster-whisper transcription
  → voice/llm/ollama.py    ← Ollama LLM (local models)
  → voice/tts/piper.py     ← Piper TTS synthesis
  → voice/audio/playback.py

integrations/client.py    MCPManager — shared by voice/orchestrator.py and
                           app/services/chat_service.py (one set of MCP server
                           processes, not one per consumer; see MCP integration below)

app/main.py              FastAPI — port 8000
  app/api/v1/endpoints/  state, events (WS), logs, metrics, projects,
                         terminals, runs, notes, prompt_builder, voice, chat,
                         devteam, system, tools
  app/services/          terminal_service, run_service, chat_service, etc.

ui/src/                  React dashboard (port 3000 dev / served from FastAPI prod)
```

## Python venv

Always use `.venv/bin/python` / `.venv/bin/pip` (macOS). `requirements.txt` needs Python 3.10+ —
`scripts/setup.sh` checks for this and installs a newer interpreter via Homebrew if the system
`python3` is too old. `mcp` is pinned to `1.29.1` — 2.x removed `mcp.server.fastmcp.FastMCP`, which
every MCP server module here uses. (The `integrations/` package is named to avoid shadowing that
`mcp` PyPI import.)

## MCP integration

`integrations/client.py`'s `MCPManager` connects to the MCP servers listed under `mcp.servers` in
`config.yaml` — each one a separate module (own process, own dependencies), implemented under
`integrations/servers/`:

- `claude_code_server.py` (`projects`, `always_on: true`) — project discovery and the Claude Code
  bridge. Sessions with `SessionMode.CODE` bypass Ollama and route to Claude Code directly.
- `schedule_server.py` (`schedule`) — Google Calendar (needs one-time OAuth setup, see
  `docs/schedule-setup.md`).
- `notes_server.py` (`notes`) — local notes/reminders, `~/.dann/notes.json`.
- `gardening_server.py` (`gardening`) — garden log (plantings, harvests, care notes).
- `bjj_server.py` (`bjj`) — BJJ training log.
- `devteam_server.py` (`devteam`) — background multi-phase Claude Code pipelines.
- `system_server.py` (`system`) — volume/open-app/lock-screen via `osascript`.

Modules default to `always_on: false` — registered but not started. `MCPManager` exposes three
meta-tools (`list_modules`, `enable_module`, `disable_module`) so the LLM starts a module's
process only when a turn actually needs it, keeping both the tool schema sent to Ollama and the
process count small as more modules get added. `always_on: true` is the escape hatch for a
module that should just always be connected (edit config, restart).

`MCPManager` is a process-wide shared instance (`get_shared_manager()`) — both the voice
orchestrator and text chat (`app/services/chat_service.py`) use the same server processes and
tool state rather than starting their own.

Personal per-machine module data (OAuth tokens, notes) lives under `~/.dann/`, not in the repo —
see `integrations/servers/_store.py`.

## Config

`config.yaml` (from `config.example.yaml`) controls wake word model path, Picovoice access key,
Ollama model name, Piper voice model path, audio device, the `agents:` routing list, and the
`mcp.servers` list.
