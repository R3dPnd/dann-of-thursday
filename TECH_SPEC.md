# Dann of Thursday — Technical Specification

## 1. Vision

Dann is a local-first voice AI agent: say **"ok Dann"**, ask a question, get
a spoken answer — with every stage of the pipeline (wake word, speech-to-text,
reasoning, text-to-speech) running on this machine, not in the cloud. The
only thing that ever leaves the machine is a request explicitly routed to
Claude Code (via the `ask_claude_code` / `open_claude_code` tools) or, in the
optional remote-access setup, a request from a browser we've authenticated.

Two things are being built on top of that core pipeline:

1. **A dashboard** (FastAPI + React) for watching the pipeline live, browsing
   and running coding projects, and giving Dann a terminal/MCP bridge into
   Claude Code — effectively turning "ok Dann, work on X" into a voice
   front-end for an agentic coding session.
2. **A local AI workstation** — this Mac reachable from a browser on another
   machine (Cloudflare Tunnel + Access, see README.md), turning it into a
   remotely usable dev/AI box, not just a voice assistant appliance.

## 2. System Architecture

```
"ok Dann" ─▶ wake word ─▶ record ─▶ STT ─▶ LLM (router) ─▶ TTS ─▶ speak
             Porcupine/     mic     faster-  Ollama          Piper  speaker
             openWakeWord           whisper  (local)          local
                                       │
                                       ├─ ROUTE local: answer inline
                                       ├─ ROUTE ask_claude: Anthropic API
                                       ├─ ROUTE ask_claude_code /
                                       │  open_claude_code: MCP → Claude Code
                                       └─ ROUTE list_projects: config.yaml projects
```

```
src/            Voice pipeline — wake word → STT → LLM → TTS, orchestrated
                by src/orchestrator.py, event-driven via src/event_bus.py
app/            FastAPI backend — serves dashboard state/REST/WebSocket,
                shares the orchestrator's EventBus so the UI sees every
                pipeline event live
ui/             React + Tailwind dashboard (Vite dev / Electron optional)
models/         Wake word models, Piper voice models, openWakeWord submodule
config.yaml     Runtime config: audio, STT, TTS, wake word, Ollama, projects
```

Backend service breakdown (`app/services/`): `terminal_service` (PTY-backed
shell sessions), `run_service` (start/stop/track project runs), `mcp_service`
(MCP server lifecycle), `tool_service`, `history_service`, `log_service`,
`metrics_service`. These are what make Dann's `terminals`/`runs`/`mcp`/`tools`
API endpoints powerful — and why they're treated as a remote-code-execution
surface in the security section below.

## 3. Languages

| Language | Where | Why |
|---|---|---|
| Python 3.10+ | `src/`, `app/`, `scripts/` | Voice/ML ecosystem (Whisper, Piper, ONNX, PyTorch) is Python-first; FastAPI for the backend. **3.10+ is a hard requirement** — the `mcp` package won't install on older Python, and macOS's built-in `python3` (3.9.x) is too old. `scripts/setup.sh` checks for this and installs a newer interpreter via Homebrew if needed. |
| TypeScript / JavaScript | `ui/` | React dashboard; TypeScript for the type safety a WebSocket-event-driven UI benefits from (event payloads, store shape). |
| Bash | `scripts/`, `deploy/` | Setup/bootstrap automation — guided local setup, workstation provisioning, launchd service templates. |
| YAML | `config.yaml`, `.github/workflows/` | Runtime config and CI. |

## 4. Libraries & Frameworks

### Voice pipeline (`src/`)

| Library | Why |
|---|---|
| `pvporcupine` | Wake word engine (Picovoice Porcupine). High accuracy, low false-positive rate, and lets you train a custom "ok Dann" acoustic model via a web console with no local ML work. Needs a free Picovoice AccessKey. |
| `openwakeword` | Alternative wake word engine — fully offline, no account, ONNX-based. `src/wakeword/openwakeword_detector.py` supports it as a drop-in swap via `wake_word.engine` in config. Currently the trained custom model artifact for this engine is missing locally (see §8); the pipeline runs on Porcupine instead. |
| `sounddevice` | Cross-platform mic capture / speaker playback via PortAudio — needed by every stage that touches raw audio (wake word streaming, STT recording, TTS playback). |
| `numpy` | Audio buffer math (float↔int16 conversion, RMS silence detection). |
| `soundfile` | WAV read/write for recorded utterances and synthesized speech. |
| `faster-whisper` | STT. CTranslate2-reimplemented Whisper — several times faster than the reference implementation on CPU, which matters because this all needs to run locally without a dedicated GPU. |
| `piper-tts` | TTS. Small, fast neural voice models that run well on Apple Silicon (the Python API, not the piper CLI binary, is used — `piper_path` in config is unused). |
| `PyYAML` | Parses `config.yaml`. |
| `httpx` | Async HTTP client — calls Ollama's local API and (in `app/core/cf_access.py`) Cloudflare's JWKS endpoint. |

### Backend API (`app/`)

| Library | Why |
|---|---|
| `fastapi` | REST + WebSocket API framework — async, automatic OpenAPI docs (`/docs`), used to expose orchestrator state to the dashboard in real time. |
| `uvicorn[standard]` | ASGI server running FastAPI. |
| `pydantic-settings` | Typed, env-var-aware settings (`app/core/config.py`) — CORS origins, Cloudflare Access config, etc. all validated at startup rather than read ad hoc. |
| `mcp` | Model Context Protocol SDK — lets Dann's orchestrator act as an MCP client and drive Claude Code as a tool-using agent (`src/mcp_client.py`, `src/mcp_servers/claude_code_server.py`). This is the mechanism behind `SessionMode.CODE` sessions bypassing Ollama entirely. |
| `anthropic` | Claude API SDK — used for `ask_claude` routing (general reasoning delegated to Claude rather than the local Ollama model) and by the Claude Code MCP bridge. |
| `ptyprocess` | Real pseudo-terminal sessions for `terminal_service` — lets the dashboard (and Claude Code sessions) drive an actual shell, not just subprocess calls. |
| `aiofiles` | Async file I/O in the backend so log/history writes don't block the event loop. |
| `PyJWT[crypto]` | Verifies Cloudflare Access JWTs (`app/core/cf_access.py`) as a defense-in-depth check when the workstation is exposed remotely — see §7. |

### Frontend (`ui/`)

| Library | Why |
|---|---|
| React 18 + Vite | Dashboard SPA; Vite for a fast dev server against the FastAPI backend on :8000. |
| TypeScript | Type safety across WebSocket event payloads and the global store. |
| Zustand | Lightweight global state store for live orchestrator/event-bus state — no Redux boilerplate needed for what's essentially "mirror the backend's event stream." |
| Tailwind CSS | Utility-first styling for the dashboard. |
| Recharts | Metrics charts (latency, turn counts) on the metrics page. |
| `@xterm/xterm` + `@xterm/addon-fit`/`addon-web-links` | Embeds a real terminal emulator in the browser for `terminal_service`-backed sessions. |
| Electron | Optional desktop-app wrapper (`electron:dev`/`electron:start`) so the dashboard can run as a native window instead of only in-browser. |

### Deployment / ops (`deploy/`, `scripts/`)

| Tool | Why |
|---|---|
| `cloudflared` (Cloudflare Tunnel) | Exposes services to the internet via an outbound-only connection — no port forwarding, no static IP, free TLS. See README "Local AI Workstation." |
| Cloudflare Access | OAuth-gated auth in front of every tunneled hostname; the real auth boundary given how powerful the `terminals`/`runs`/`mcp` endpoints are. |
| launchd | macOS service supervision — keeps the backend and `cloudflared` running across reboots/logout (`deploy/launchd/*.plist`). |
| Homebrew | Package manager for `ollama`, `cloudflared`, `code-server`, `python@3.12`, etc. |

## 5. Data & Model Resources

| Resource | Location | Notes |
|---|---|---|
| Wake word model (Porcupine) | `models/ok_dann.ppn` | Present, working. Trained via Picovoice Console. |
| Wake word model (openWakeWord) | `models/ok_dann.onnx` | **Missing** — gitignored (`models/*.onnx`) and never committed. Regenerate with `scripts/train_wakeword.py` using the existing `training/positive/` (3002 samples) and `training/adversarial/` (1502 samples) clips, ~10–20 min on CPU. |
| Whisper STT model | Auto-downloaded by `faster-whisper` on first use | Size controlled by `stt.model_size` (tiny→large-v3-turbo); `small` is the current default — a CPU/accuracy tradeoff. |
| Ollama LLM | Pulled via `ollama pull <model>` | `llama3.2` is the current default (`ollama.model` in config), ~2GB. Ollama serves it locally over HTTP on :11434. |
| Piper TTS voice | `models/pieper/en_GB-northern_english_male-medium.onnx` (+`.json`) | ~60MB neural voice model; note the directory is named `pieper` (typo) not `piper` on this machine — config points at the real path so it works, but it's inconsistent with `config.example.yaml`'s `models/piper/...` convention. |
| openWakeWord training data | `training/positive/`, `training/adversarial/` | Already generated (thousands of synthetic clips via `piper-tts`); only needed if regenerating the custom `.onnx` model. |

## 6. External Services & Accounts Required

| Service | Required for | Cost |
|---|---|---|
| Picovoice Console | Porcupine wake word AccessKey + custom `.ppn` training | Free tier |
| Ollama (local, no account) | Local LLM inference | Free, runs on-device |
| Anthropic API | `ask_claude` routing, Claude Code MCP bridge | Pay-per-use API key |
| Cloudflare account | Tunnel + Access, for the remote "AI workstation" | Free for personal/small-team use |
| Domain (added to Cloudflare) | Public hostnames for the tunnel | Cost of the domain only |
| GitHub | Repo hosting, CI (`.github/workflows/ci.yml`) | Free (public repo) |

**Known issue:** the Picovoice AccessKey that was in `config.yaml` was committed
to git history at some point and is exposed on this public repo. Treat it as
compromised — rotate it at console.picovoice.ai and paste the new key into
`config.yaml` (which itself should be gitignored going forward; only
`config.example.yaml` belongs in git).

## 7. Hardware Requirements

- Microphone + speakers/output device (any built-in or USB audio device
  `sounddevice`/PortAudio can see).
- CPU capable of running `faster-whisper` (small model) and Piper TTS in
  real time — Apple Silicon handles this comfortably; no GPU required.
  `stt.device: auto` in config will use CUDA if present, otherwise CPU.
- Enough disk for models: Whisper (tens to low-hundreds of MB depending on
  size), Ollama models (multi-GB per model), Piper voices (tens of MB each).
- For the workstation vision specifically: enough always-on uptime for
  `launchd` services + `cloudflared` to be useful as a remote box.

## 8. Current State (as of this session)

- ✅ Backend, dashboard, and voice pipeline all import/run under Python 3.12
  in a fresh `.venv` (`scripts/setup.sh`, fixed to require 3.10+ after
  discovering the stock macOS `python3` is 3.9.6).
- ✅ Ollama installed, `llama3.2` pulled and running.
- ✅ Piper voice model present and configured.
- ✅ Wake word switched to Porcupine (`models/ok_dann.ppn`, present) after
  discovering the configured openWakeWord model (`models/ok_dann.onnx`) was
  never committed — gitignored, only a stray `.onnx.data` sidecar survived.
- ⏳ Porcupine AccessKey needs to be regenerated (old one leaked, see §6)
  before the wake word can actually run end-to-end.
- 📋 Remote workstation access (Cloudflare Tunnel + Access, code-server,
  JupyterLab, filebrowser) is documented and scaffolded (`deploy/`,
  `scripts/setup_workstation.sh`) but not yet provisioned against a real
  Cloudflare account/domain.

## 9. Planned Work

Pulled from `IMPROVEMENTS.md` (Priorities 1–8 — the Priority 9 section in
that file is unrelated resume feedback, not part of this project) and
`UI_SPEC.md`'s phased plan:

- **Security hardening**: shell-injection risk in subprocess calls, CORS
  wildcard, no auth on API/WebSocket endpoints (partially addressed for the
  remote-access case by Cloudflare Access + `cf_access.py`, but the
  localhost/LAN case is still open), no rate limiting, no HTTPS on the raw
  backend.
- **Code quality**: input validation, consistent error handling in API
  routes, type hints, a logging framework, consolidating fragmented config.
- **Testing**: current coverage is minimal; CI (`ci.yml`) runs `pytest` and
  a UI build but there's no meaningful test suite depth yet.
- **Dashboard build-out**: per `UI_SPEC.md`'s phases — EventBus/orchestrator
  instrumentation (done), status/project panels, metrics + logging views,
  task tracking, theming/polish.
- **Voice pipeline quality**: several concrete bugs logged in
  `IMPROVEMENTS.md` §8 (goodbye-detection punctuation bug, no conversation
  history between turns, STT mishearing "Claude Code," etc.) — worth
  triaging before/alongside the workstation work.

## 10. Open Questions

- Which wake word engine is the long-term default — Porcupine (account +
  leaked-key cleanup, but currently working) or openWakeWord (offline, but
  needs the custom model retrained)?
- Is `config.yaml` staying in git going forward, or does it move to
  gitignored-with-example-template (recommended, given the leak)?
- Scope of the "AI workstation" vision beyond Dann — is code-server /
  JupyterLab / filebrowser actually getting provisioned, or staying
  documented-but-unbuilt for now?
