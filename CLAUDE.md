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
| Voice agent (full) | `.venv/Scripts/python.exe -m src.main` |
| Backend only (no mic) | `NO_VOICE=1 .venv/Scripts/uvicorn.exe app.main:app --host 0.0.0.0 --port 8000` |
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

Always use `.venv/Scripts/python.exe` / `.venv/Scripts/pip.exe` on Windows.

## MCP integration

`src/mcp_client.py` + `src/mcp_servers/claude_code_server.py` — Dann can invoke Claude Code via MCP.
Sessions with `SessionMode.CODE` bypass Ollama and route to Claude Code directly.

## Config

`config.yaml` (from `config.example.yaml`) controls wake word model path, Picovoice access key,
Ollama model name, Piper voice model path, audio device, and project list.
