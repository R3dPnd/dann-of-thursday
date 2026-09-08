# Consolidated Improvement Recommendations

This document consolidates improvement suggestions derived from analysis of two related edge AI projects:
- **[pocket-ai](https://github.com/nazirlouis/pocket-ai)** — FastAPI + Electron local AI assistant for Raspberry Pi 5
- **[be-more-hailo](https://github.com/moorew/be-more-hailo)** — BMO character agent on Raspberry Pi 5 with Hailo NPU

Both projects share the same core architecture patterns (FastAPI backend, local LLM inference, Piper TTS, Whisper STT, wake word detection) and the same classes of issues. The recommendations below apply broadly to any project in this family.

---

## Priority 1: Security (Critical — Fix First)

### 1.1 Shell Injection in Subprocess Calls
Both repos build shell commands by concatenating variables directly into strings.

**Vulnerable pattern:**
```python
os.system(f"ffmpeg -i {audio_file} -acodec pcm_s16le ...")
os.system(f"aplay -D {device} {filename}")
```

**Fix:** Always use `subprocess.run()` with a list of arguments — never a string:
```python
subprocess.run(
    ["ffmpeg", "-i", audio_file, "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000", output_file],
    check=True, capture_output=True
)
subprocess.run(["aplay", "-D", device, filename], check=True)
```

### 1.2 CORS Wildcard
APIs should not accept requests from any origin.

**Fix:** Replace `allow_origins=["*"]` with an explicit allowlist read from environment variables:
```python
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, ...)
```

### 1.3 No Authentication on API/WebSocket Endpoints
Tool execution endpoints (GPIO control, network scan, security mode) and WebSocket connections accept traffic without authentication.

**Fix:** Add a simple token-based auth guard. Read the token from an environment variable:
```python
API_TOKEN = os.getenv("API_TOKEN", "")

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(HTTPBearer())):
    if API_TOKEN and credentials.credentials != API_TOKEN:
        raise HTTPException(status_code=403)
```

### 1.4 No Rate Limiting
Endpoints accept unlimited concurrent requests which can exhaust LLM inference capacity.

**Fix:** Use `slowapi`:
```python
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

@app.post("/api/chat")
@limiter.limit("10/minute")
async def chat(request: Request, body: ChatRequest): ...
```

### 1.5 No HTTPS
Both services run plain HTTP. In production, terminate TLS at the reverse proxy (nginx/Caddy) or pass `--ssl-keyfile` / `--ssl-certfile` to uvicorn.

---

## Priority 2: Code Quality (Critical)

### 2.1 No Input Validation
API endpoints accept raw user input without schema enforcement.

**Fix:** Use Pydantic models for every request and response:
```python
class ChatRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    image: Optional[str] = None

    @validator("image")
    def validate_image(cls, v):
        if v and not v.startswith("data:image/"):
            raise ValueError("Invalid image format")
        return v
```

### 2.2 No Error Handling in API Routes
WebSocket and REST endpoints crash silently or return unstructured errors.

**Fix:** Wrap all route handlers in try/except with structured error responses:
```python
@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        response = await asyncio.to_thread(brain.think, request.text)
        return {"response": response}
    except ModelError as e:
        logger.error("Model inference failed", exc_info=True)
        raise HTTPException(status_code=503, detail="Model unavailable")
    except Exception as e:
        logger.error("Unexpected error", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error")
```

### 2.3 Synchronous I/O Blocking the Event Loop
Model inference (CPU/NPU-bound) is called directly inside `async def` handlers, blocking FastAPI's event loop.

**Fix:** Offload to a thread pool:
```python
response = await asyncio.to_thread(brain.think, request.text)
```

### 2.4 Missing Type Hints
~40% of functions lack type annotations, reducing IDE support and making bugs harder to catch.

**Fix:** Add type hints to all public function signatures:
```python
def think(self, user_text: str) -> str: ...
def stream_think(self, user_text: str) -> Generator[str, None, None]: ...
def transcribe_audio(filename: str) -> Optional[str]: ...
```

### 2.5 Hardcoded Magic Numbers and Strings
Sample rates, ports, thresholds, and UI colors are scattered inline throughout the code.

**Fix:** Centralize in a `constants.py`:
```python
class AudioConfig:
    INPUT_RATE = 48000
    WHISPER_RATE = 16000

class ModelConfig:
    WAKEWORD_THRESHOLD = 0.35
    MAX_HISTORY = 20

class ServerConfig:
    PORT = 8080
    OLLAMA_HOST = "http://127.0.0.1:8000"
```

### 2.6 No Logging Framework
Both repos rely on `print()` for diagnostics with no log levels, rotation, or filtering.

**Fix:** Replace all `print()` calls with structured logging:
```python
import logging
logger = logging.getLogger(__name__)
logger.info("Model loaded in %.2fs", elapsed)
logger.error("Transcription failed", exc_info=True)
```

### 2.7 Fragmented Configuration
Config values are split across hardcoded defaults, JSON files, and environment variables with no single source of truth and no validation.

**Fix:** Use Pydantic `BaseSettings` for a unified, validated config that reads from environment variables with JSON fallback:
```python
from pydantic import BaseSettings

class Settings(BaseSettings):
    ollama_host: str = "http://localhost:8000"
    ollama_model: str = "qwen2.5-instruct:1.5b"
    wakeword_threshold: float = 0.35
    audio_input_device: int = 0

    class Config:
        env_file = ".env"

settings = Settings()
```

### 2.8 Model Management Without Singleton
Multiple modules each initialize their own model instances, causing redundant loads and excessive memory use.

**Fix:** Implement a `ModelManager` singleton with lazy loading:
```python
class ModelManager:
    _instance = None
    _brain = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def brain(self):
        if self._brain is None:
            self._brain = Brain()
        return self._brain
```

---

## Priority 3: Testing (Critical)

### 3.1 Minimal Test Coverage
Existing test files are manual run scripts, not automated tests. Coverage is estimated below 20%.

**Fix:** Build a proper `pytest` suite:
```
tests/
├── conftest.py          # Fixtures, mock factories
├── unit/
│   ├── test_config.py
│   ├── test_llm.py
│   ├── test_tts.py
│   └── test_stt.py
└── integration/
    ├── test_api_endpoints.py
    └── test_websocket.py
```

### 3.2 No CI/CD Pipeline
No automated test execution on push or pull request.

**Fix:** Add a GitHub Actions workflow:
```yaml
# .github/workflows/test.yml
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with: { python-version: "3.11" }
      - run: pip install -r requirements.txt pytest pytest-cov
      - run: pytest tests/ --cov=. --cov-report=xml
```

### 3.3 No Mocking of External Services
Tests call real Ollama instances, real audio devices, and real external APIs (weather, geocoding).

**Fix:** Mock all external boundaries:
```python
@pytest.fixture
def mock_ollama(requests_mock):
    requests_mock.post(
        "http://localhost:8000/api/chat",
        json={"message": {"content": "Hello!"}, "done": True}
    )

@patch("subprocess.run")
def test_tts_does_not_call_piper_on_empty_string(mock_run):
    generate_audio("")
    mock_run.assert_not_called()
```

---

## Priority 4: Performance

### 4.1 No Response Caching
Repeated identical queries (e.g., "what time is it?", geocoding lookups) recompute from scratch every time.

**Fix:** Add LRU cache for pure functions and TTL cache for external API results:
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def get_route(text: str) -> str: ...

# For time-sensitive data (weather), cache with TTL
```

### 4.2 Camera/MJPEG Stream Inefficiency
MJPEG frames are encoded and transmitted at full resolution without negotiation.

**Fix:** Add a `?resolution=480p` query parameter and skip frames if the client queue is full.

### 4.3 Audio Queue Spawns Subprocess Per Sentence
TTS creates a new subprocess for every sentence fragment, adding latency.

**Fix:** Batch short sentences or stream audio chunks from a single Piper process kept alive as a persistent subprocess.

### 4.4 No Resource Cleanup Policies
`conversations.json` grows unbounded; captured images are never pruned.

**Fix:** Add configurable retention limits and a scheduled cleanup task:
```python
MAX_CAPTURES = 1000
CONVERSATION_RETENTION_DAYS = 90
```

---

## Priority 5: Architecture

### 5.1 No Service/Abstraction Layer
Tool execution and LLM calls are made directly in route handlers, making components hard to test or swap.

**Fix:** Extract a service layer:
```python
class ChatService:
    def __init__(self, brain: Brain, config: Settings):
        self.brain = brain
        self.config = config

    def process(self, text: str, image: Optional[str] = None) -> str:
        if image:
            return self.brain.think_with_image(text, image)
        return self.brain.think(text)
```

### 5.2 Undefined WebSocket Message Protocol
WebSocket message formats are inferred from reading the code — no schema, no versioning.

**Fix:** Define a typed protocol on both sides:
```typescript
// Frontend
interface ChatMessage {
  version: "1.0";
  type: "user" | "assistant" | "error";
  conversationId: string;
  content: string;
  timestamp: number;
}
```
```python
# Backend
class WsMessage(BaseModel):
    version: str = "1.0"
    type: Literal["user", "assistant", "error"]
    conversation_id: str
    content: str
    timestamp: float
```

### 5.3 Dual UI Implementations Out of Sync
Both repos have parallel on-device (Tkinter) and web (FastAPI) interfaces that share core logic but manage UI state independently, causing drift.

**Fix:** Formalize a shared `StateMachine` in `core/` that both interfaces subscribe to:
```python
class BmoState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ERROR = "error"
```

---

## Priority 6: Documentation

### 6.1 No API Documentation
WebSocket message formats, REST endpoints, tool call schemas, and query parameters are undocumented.

**Fix:** Enable FastAPI's built-in OpenAPI docs (`/docs`) and create `docs/API.md` covering WebSocket schemas.

### 6.2 No Architecture Documentation
There is no document explaining data flow, component interaction, or design decisions (e.g., why STT runs on CPU rather than the NPU).

**Fix:** Create `docs/ARCHITECTURE.md` with a data flow diagram and `docs/adr/` for architecture decision records.

### 6.3 No Contributor Guide
Neither repo documents how to set up a local dev environment, run tests, or submit changes.

**Fix:** Create `CONTRIBUTING.md` covering:
- Virtual environment setup
- Running tests (`pytest tests/`)
- Code style (`black`, `isort`, `pylint`)
- Branch and PR conventions

---

## Priority 7: Developer Experience

### 7.1 No Unified Launcher
Services require multiple manual commands across directories (backend, frontend, services).

**Fix:** Add a root-level `Makefile`:
```makefile
run:        ## Start all services
	./run.sh

test:       ## Run test suite
	pytest tests/ -v --cov=.

lint:       ## Lint and type-check
	black --check . && pylint src/ && mypy src/

setup:      ## Install dependencies
	pip install -r requirements.txt
```

### 7.2 Audio Device Configuration Is Manual
Default audio device indices are hardcoded and must be edited per-machine.

**Fix:** Add an auto-discovery script:
```python
# scripts/detect_audio.py
import sounddevice as sd
for i, dev in enumerate(sd.query_devices()):
    marker = " ← default" if i == sd.default.device[0] else ""
    print(f"[{i}] {dev['name']} ({dev['max_input_channels']}ch in){marker}")
```

### 7.3 No Pre-commit Hooks
Formatting and linting checks are not enforced before commits.

**Fix:**
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks: [{ id: black }]
  - repo: https://github.com/PyCQA/isort
    rev: 5.12.0
    hooks: [{ id: isort }]
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: check-added-large-files
```

### 7.4 No VS Code / IDE Configuration
Neither repo includes workspace settings, debug launch configs, or recommended extensions.

**Fix:** Add `.vscode/settings.json` and `.vscode/launch.json` with formatter, linter, and test runner configuration pointing to the local virtualenv.

---

---

## Priority 8: Voice Pipeline Quality (Observed in Testing)

The following issues were discovered during live testing of the dann-of-thursday voice pipeline.

### 8.1 Goodbye Detection Broken by Punctuation

STT transcribes "Goodbye, Dan." with punctuation. A bare `in` substring check against phrases like `"goodbye dan"` fails because the comma is present.

**Fix:** Strip punctuation before matching:
```python
normalized = re.sub(r"[^\w\s]", " ", text.lower()).strip()
return any(phrase in normalized for phrase in _GOODBYE_PHRASES)
```
**Status:** Fixed in `src/orchestrator.py`.

### 8.2 LLM Returns Raw JSON When Tool Call Fails

When the model attempts a tool call but produces malformed output (e.g. `{"name": "summarize", "parameters": {...}}`), the JSON string is passed directly to TTS and spoken aloud verbatim.

**Fix:** Detect JSON-shaped responses before TTS and substitute a graceful fallback message. **Status:** Fixed in `src/orchestrator.py`.

### 8.3 MCP Shutdown Crash on Ctrl+C

Pressing Ctrl+C triggers a `RuntimeError: Attempted to exit cancel scope in a different task than it was entered in` from the anyio library during MCP teardown. This is caused by the async exit stack being closed from a different OS thread than it was created in.

**Fix:** Wrap `_disconnect_all()` in a bare `except Exception: pass` in `MCPManager.stop()`. **Status:** Fixed in `src/mcp_client.py`.

### 8.4 Silent Audio Sent to Whisper

Between turns, `record_until_silence` exits due to the silence timeout and returns essentially silent PCM. Passing this to Whisper triggers numpy `divide by zero` / `overflow` warnings and wastes ~1-2s per turn.

**Fix:** Compute RMS energy on the PCM buffer and skip Whisper when below a minimum threshold. **Status:** Fixed in `src/orchestrator.py`.

### 8.5 No Conversation History Between Turns

Each turn was a stateless request to Ollama. A follow-up like "No." had no context, so the model guessed intent and called irrelevant tools.

**Fix:** Maintain a `_history` list per session and pass it to `generate_response`. Clear on session start/end. **Status:** Fixed in `src/orchestrator.py` and `src/llm/ollama.py`.

### 8.6 STT Consistently Mishears "Claude Code" as "Cloud Code"

Whisper transcribes "Claude Code" as "Cloud Code" reliably, so the LLM never maps the phrase to the `open_claude_code` tool.

**Fix:** Post-process STT output with a known substitutions dictionary before sending to the LLM. **Status:** Fixed in `src/orchestrator.py`.

### 8.7 Claude Code Tool Is One-Way Only

The `open_claude_code` MCP tool opens a new Terminal window with `claude` running. It cannot receive output back from that session. Requests like "summarize the project and read out the summary" cannot work — there is no mechanism to capture Claude Code's response and pipe it back to Dann's TTS.

**Fix:** Added a second tool `ask_claude_code` that runs `claude -p "task"` non-interactively, captures stdout, and returns it so Dann can speak the answer. `open_claude_code` is kept for interactive sessions. **Status:** Fixed in `src/mcp_servers/claude_code_server.py`.

### 8.8 Tool Service Never Executed Real Subprocesses

`ToolService._run_tool` was a stub that slept for 0.1s and returned a fake success message regardless of what tool was called.

**Fix:** Implemented real `asyncio.create_subprocess_exec` execution with proper argument building for nmap and sqlmap, timeout enforcement, and stdout/stderr capture. **Status:** Fixed in `app/services/tool_service.py`.

### 8.9 MCP Service Tool List Was Hardcoded

`MCPService._handle_tools_list` returned a static list of two tools, disconnected from the actual `ToolService` registry.

**Fix:** Connected to `ToolService().get_available_tools()`. **Status:** Fixed in `app/services/mcp_service.py`.

### 8.10 Readiness Endpoint Always Returned "ready"

`GET /health/ready` returned `{"status": "ready"}` unconditionally, making it useless for detecting when Ollama is down.

**Fix:** Added an async httpx probe to `http://localhost:11434/api/tags`. Returns `{"status": "degraded", "checks": {"ollama": "unreachable"}}` when Ollama is not running. **Status:** Fixed in `app/api/v1/endpoints/health.py`.

### 8.11 API Key Verification Always Passed

`verify_api_key` in `dependencies.py` returned `True` in all code paths, making it a no-op in production.

**Fix:** Reads `SECRET_KEY` from settings and checks the `X-API-Key` header. Skips enforcement only when `DEBUG=True` or the key is still the placeholder default. **Status:** Fixed in `app/core/dependencies.py`.

### 8.12 Wake Word Model Never Recognized Real Speech

`models/ok_dann.onnx` (openWakeWord) was trained entirely on 3000 clips synthesized from one Piper TTS voice, with no real human recordings. Live scoring against actual speech came back at 0.0005 (essentially zero) against a 0.7 threshold — not a marginal miss, the model had no signal for a real speaker at all.

**Fix:** Added `scripts/record_wakeword_samples.py` to record real "ok Dann" clips (interactive `input()`-gated mode, plus a fixed-cadence `--auto` mode for hands-free batch recording). `scripts/train_wakeword.py` now loads `training/positive_real/` if present and oversamples each real clip (`REAL_OVERSAMPLE`, tuned from 20x to 50x) against the synthetic set. 40 real clips at 50x oversampling brought the live score from 0.0005 to 0.9969, confirmed working end-to-end (wake detected, correct STT transcription, LLM routing, spoken response) in a live run. **Status:** Fixed.

### 8.13 Wake Word Training Pipeline Was Broken by Environment Drift (Multiple Causes)

Running `scripts/train_wakeword.py` failed three separate ways before producing a model, none related to the training logic itself:
- `OWW_MODELS_DIR` pointed at `models/openwakeword/openwakeword/resources/models` — a git submodule reference (`160000` gitlink) with no corresponding `.gitmodules` entry, so it was never actually populated. `AudioFeatures()` defaults to the *installed* `openwakeword` package's own resources dir, which didn't match.
- `torch.onnx.export`'s new default (`dynamo=True` as of the installed torch 2.13) requires `onnxscript`, which isn't installed; explicitly passing `dynamo=False` to use the legacy exporter avoids that dependency but needs the separate `onnx` package, also not installed.
- Installing `onnx` pulled in `numpy==2.5.2`, upgrading past the pinned `numpy==1.26.4` in `requirements.txt`.

**Fix:** `OWW_MODELS_DIR` now resolves dynamically via `Path(openwakeword.__file__).resolve().parent / "resources" / "models"`. `torch.onnx.export(..., dynamo=False)` plus `onnx` added to `requirements.txt`. `numpy` pin bumped to `2.5.2` after verifying `faster-whisper`/`sounddevice`/`soundfile`/`onnxruntime`/`openwakeword`/`piper`/`pvporcupine` all still import correctly under it. **Status:** Fixed.

### 8.14 MCP `command: python` Silently Broken (No Bare `python` on PATH)

`config.yaml`'s `mcp.servers` entries used `command: python`, but this machine's PATH has no bare `python` — only `python3` and `.venv/bin/python`. This predates the module-system work in this session; the original single-server (`claude-code`) setup was already non-functional before any of today's changes.

**Fix:** All `mcp.servers` entries in `config.yaml` and `config.example.yaml` now use `command: .venv/bin/python`. **Status:** Fixed.

### 8.15 `mcp` Package 2.x Removed `FastMCP`, Breaking Every MCP Server

`requirements.txt` pinned `mcp>=1.0.0` with no ceiling. A routine `pip install -r requirements.txt` resolved to `mcp==2.1.0`, which restructured the package and dropped `mcp.server.fastmcp.FastMCP` entirely — breaking `claude_code_server.py` (pre-existing) and every new module server built on the same base.

**Fix:** Pinned `mcp==1.29.1` in `requirements.txt`, confirmed to have both the server-side `FastMCP` API and the client-side `ClientSession`/`stdio_client` API this codebase uses. **Status:** Fixed.

### 8.16 anyio Cancel-Scope Crash on Cross-Task MCP Server Disconnect

Related to but distinct from 8.3 (Ctrl+C shutdown): disconnecting a single on-demand MCP module mid-session (`disable_module`) crashed with the same `RuntimeError: Attempted to exit cancel scope in a different task than it was entered in`. `stdio_client`'s cancel scope (anyio) must be entered and exited in the same asyncio Task; the original `MCPManager` opened each server's `stdio_client`/`ClientSession` in one `_run()` call (one Task) and closed it in a later, separate `_run()` call (a different Task on the same loop/thread), which anyio forbids regardless of thread.

**Fix:** Each connected server now gets one long-lived owner task (`_server_owner_task`) that opens its own `stdio_client`/`ClientSession` and only tears them down when its own `asyncio.Event` is set from outside — connect/disconnect are signals to that task, not separate enter/exit calls from different tasks. Verified clean connect → disconnect → reconnect cycles with no exceptions. **Status:** Fixed in `src/mcp_client.py`. (8.3's `except Exception: pass` band-aid around final shutdown teardown is superseded by this for the on-demand path, but is still in place as a defensive fallback in `stop()`.)

### 8.17 Porcupine `.ppn` Model Was the Wrong Platform's File

`config.yaml` pointed Porcupine at `models/ok_dann.ppn`, which failed with `PorcupineInvalidArgumentError: Keyword file (.ppn) file has incorrect format or belongs to a different platform`. A second file, `models/mac_ok_dann.ppn`, was already present — `ok_dann.ppn` was Windows-trained (this repo has Windows path history throughout, e.g. `C:/Git/...` project paths), `mac_ok_dann.ppn` is the macOS one.

**Fix:** N/A — the project moved to openWakeWord as primary (8.12) rather than continuing with Porcupine, since Porcupine also requires a Picovoice AccessKey (the previous one leaked in git history — see `schedule-setup.md`/README security notes) and has recurring platform-file confusion. `config.example.yaml` documents the correct `.ppn` file selection as a comment for anyone who wants Porcupine instead. **Status:** Open / not applicable (superseded by 8.12's fix).

### 8.18 No Voice Command Actually Stops the Process

`src/orchestrator.py`'s `run()` loop (`while self._running: ...`) only exits on `KeyboardInterrupt` (Ctrl+C). Saying "goodbye Dann" (or any goodbye phrase) only ends the current conversation session and returns to idle wake-word listening — there's no spoken command that sets `self._running = False` and actually stops the program. Observed live: a user tried "goodbye Dann" expecting a full shutdown and it just kept listening.

**Fix:** Not yet fixed. Would need a distinct phrase (e.g. "shut down Dann" / "stop listening") routed to actually break the `run()` loop, separate from the existing per-session goodbye handling. **Status:** Open.

### 8.19 Router Over-Calls `list_modules` / `enable_module`

Since adding the optional-module meta-tools (8.12 workstream), the local model (llama3.2) called `list_modules` for turns where it wasn't relevant — a bare "Okay, Dan" with no real request, and a self-referential question about `max_tokens` config. Neither needed a module lookup.

**Fix:** Not yet fixed. Likely needs tighter guidance in `ollama.system_prompt`'s "OPTIONAL MODULES" section about when *not* to call `enable_module`/`list_modules` (e.g. only when the request clearly needs calendar/notes/system-control tools). **Status:** Open.

### 8.20 Tool-Result Responses Truncated by `max_tokens: 80`

`ollama.max_tokens: 80` is tuned for short spoken answers, but a response that needs to enumerate a tool result (e.g. naming 3 module names + descriptions after `list_modules`) can need more than that, and gets cut off mid-sentence with no indication to the user that it was truncated.

**Fix:** Not yet fixed. Consider a higher `max_tokens` specifically for turns that follow a tool call, or trimming what tool results return so they fit in ~80 tokens regardless. **Status:** Open.

### 8.21 §8.2's JSON-Artifact Fix Doesn't Catch Prose-About-JSON

`_is_json_artifact()` (src/orchestrator.py) only flags responses that are themselves valid, parseable JSON (`json.loads(stripped)` succeeds). Observed live: pushed on an incorrect answer, llama3.2 responded "Here is the correct JSON for a function call with its proper arguments that best answers the given prompt:" and stopped there (likely `max_tokens` truncation, see 8.20) — prose *narrating* an intent to produce JSON, not JSON itself, so it passed straight through to TTS unfiltered. §8.2's fix and status ("Fixed") should be read as covering only the pure-JSON-response case, not this narrower prose variant.

**Fix:** Not yet fixed. Could extend `_is_json_artifact` (or add a second check) to flag responses containing tool-calling scaffold language (e.g. "here is the JSON", "function call") even when they don't parse as JSON outright. **Status:** Open.

---

## Summary Table

| # | Category | Issue | Priority | Effort |
|---|----------|-------|----------|--------|
| 1.1 | Security | Shell injection in subprocess calls | Critical | Low |
| 1.2 | Security | CORS wildcard | Critical | Low |
| 1.3 | Security | No API authentication | Critical | Medium |
| 1.4 | Security | No rate limiting | High | Low |
| 1.5 | Security | No HTTPS | High | Low |
| 2.1 | Code Quality | No input validation (Pydantic) | Critical | Medium |
| 2.2 | Code Quality | No error handling in routes | Critical | Medium |
| 2.3 | Code Quality | Blocking I/O in async handlers | High | Low |
| 2.4 | Code Quality | Missing type hints | Medium | Medium |
| 2.5 | Code Quality | Hardcoded magic values | Medium | Low |
| 2.6 | Code Quality | No logging framework | Medium | Low |
| 2.7 | Code Quality | Fragmented configuration | Medium | Medium |
| 2.8 | Code Quality | No model singleton | Medium | Low |
| 3.1 | Testing | <20% test coverage | Critical | High |
| 3.2 | Testing | No CI/CD pipeline | High | Low |
| 3.3 | Testing | No mocking of external services | High | Medium |
| 4.1 | Performance | No response caching | Medium | Low |
| 4.2 | Performance | Unoptimized camera stream | Low | Medium |
| 4.3 | Performance | Per-sentence TTS subprocess | Medium | Medium |
| 4.4 | Performance | No resource cleanup policy | Medium | Low |
| 5.1 | Architecture | No service layer | Medium | High |
| 5.2 | Architecture | Undefined WebSocket protocol | Medium | Low |
| 5.3 | Architecture | Dual UI state out of sync | Low | High |
| 6.1 | Documentation | No API docs | High | Low |
| 6.2 | Documentation | No architecture docs | Medium | Medium |
| 6.3 | Documentation | No contributor guide | Medium | Low |
| 7.1 | DX | No unified launcher (Makefile) | Low | Low |
| 7.2 | DX | Manual audio device config | Low | Low |
| 7.3 | DX | No pre-commit hooks | Low | Low |
| 7.4 | DX | No IDE configuration | Low | Low |
| 8.1 | Voice Pipeline | Goodbye detection broken by punctuation | Critical | Low — **Fixed** |
| 8.2 | Voice Pipeline | LLM JSON artifacts spoken aloud | Critical | Low — **Fixed** |
| 8.3 | Voice Pipeline | MCP shutdown crash on Ctrl+C | High | Low — **Fixed** |
| 8.4 | Voice Pipeline | Silent audio sent to Whisper | High | Low — **Fixed** |
| 8.5 | Voice Pipeline | No conversation history between turns | High | Low — **Fixed** |
| 8.6 | Voice Pipeline | STT mishears "Claude Code" as "Cloud Code" | Medium | Low — **Fixed** |
| 8.7 | Voice Pipeline | Claude Code tool is one-way only | Medium | High — **Fixed** |
| 8.8 | Tool Service | Subprocess never executed — stub only | Critical | Medium — **Fixed** |
| 8.9 | MCP Service | Tool list hardcoded, not from ToolService | Medium | Low — **Fixed** |
| 8.10 | Health | /ready always returned "ready" | Medium | Low — **Fixed** |
| 8.11 | Security | API key check always passed | High | Low — **Fixed** |
| 8.12 | Voice Pipeline | Wake word never recognized real speech (synthetic-only training) | Critical | Medium — **Fixed** |
| 8.13 | Voice Pipeline | Wake word training pipeline broken by env drift (3 causes) | High | Medium — **Fixed** |
| 8.14 | MCP | `command: python` not on PATH, MCP silently non-functional | Critical | Low — **Fixed** |
| 8.15 | MCP | `mcp` 2.x removed FastMCP, broke every MCP server | Critical | Low — **Fixed** |
| 8.16 | MCP | anyio cancel-scope crash on on-demand module disconnect | High | Medium — **Fixed** |
| 8.17 | Voice Pipeline | Porcupine `.ppn` was wrong platform's file | Medium | N/A — superseded |
| 8.18 | Voice Pipeline | No voice command actually stops the process | Medium | Low — Open |
| 8.19 | Voice Pipeline | Router over-calls list_modules/enable_module | Low | Low — Open |
| 8.20 | Voice Pipeline | Tool-result responses truncated by max_tokens: 80 | Low | Low — Open |
| 8.21 | Voice Pipeline | JSON-artifact filter misses prose-about-JSON | Medium | Low — Open |

---

## Priority 9: Resume / ATS Compatibility

Issues identified in the current resume that will reduce ATS match scores or cause parsing failures.

### 9.1 Company Names and Dates Concatenated Without Spaces
ATS parsers will read 'AxonSeptember 2025' and 'MicrosoftMay 2022' as a single token, mangling job titles, employers, and tenure calculations.

**Fix:** Ensure a tab or consistent spacing separates company name from date range in the DOCX template.

### 9.2 Education Section Omits Graduation Year
ATS systems that require date fields may flag or reject an entry with no graduation year.

**Fix:** Add the graduation year to the education entry.

### 9.3 Inconsistent Title — Summary vs. Job Titles
Summary header reads 'SOFTWARE DEVELOPER' but all job titles are 'Software Engineer II/I'. Inconsistent title signals may reduce match score on 'Software Engineer' searches.

**Fix:** Align the summary title with actual job titles ('Software Engineer').

### 9.4 'DotNet' Should Be '.NET'
ATS keyword matching is literal. '.NET' is the canonical form in job postings; 'DotNet' will not match.

**Fix:** Replace all instances of 'DotNet' with '.NET'.

### 9.5 'PCF' Is Unexpanded
'Pivotal Cloud Foundry' is not universally recognized as 'PCF'. ATS parsers may miss the keyword.

**Fix:** Write 'Pivotal Cloud Foundry (PCF)' on first use so both forms are searchable.

### 9.6 Infosys (Boeing) Parenthetical May Confuse Employer Parsing
ATS employer-name extraction may read '(Boeing)' as a separate entity or mangle the company name.

**Fix:** Use a standard contractor format, e.g., 'Infosys — client: Boeing' or a dedicated 'Client' line, rather than a parenthetical.

### 9.7 Agile/Scrum Missing from Skills Section
Agile and Scrum are mentioned in role bullets but not listed in the Skills block. ATS scanners often search Skills sections specifically.

**Fix:** Add 'Agile / Scrum' explicitly to the Technical Skills section.

### 9.8 No Certifications Section
Significant AWS and Azure experience is present but no certifications are listed. Cloud certs are high-weight ATS keywords in most engineering JDs.

**Fix:** Add a Certifications section; list any current certs or note in-progress ones.

### 9.9 LinkedIn URL Missing HTTPS Prefix
'linkedin.com/in/...' format may fail ATS link extraction. Full URL with protocol is safer.

**Fix:** Use 'https://www.linkedin.com/in/...' in the contact header.

### 9.10 Technical Skills Section Mixes Abstraction Levels Inconsistently
JSON and Unix appear alongside frameworks and cloud platforms, which can confuse skill-extraction parsers that bucket by category.

**Fix:** Reorganize into named subcategories (e.g., Languages, Frameworks, Cloud & DevOps, Tools) so parsers can assign each keyword to the correct domain.

---

## Recommended Execution Order

1. **Immediate (< 1 day):** Items 1.1, 1.2, 2.3, 2.5, 2.6 — low-effort, high-impact security and quality fixes
2. **Week 1:** Items 1.3, 2.1, 2.2, 2.7 — authentication, input validation, error handling, unified config
3. **Week 2:** Items 3.1, 3.2, 3.3 — establish test foundation and CI pipeline
4. **Week 3:** Items 2.4, 2.8, 5.1, 5.2 — type hints, model singleton, service layer, protocol schema
5. **Ongoing:** Documentation (6.x) and DX improvements (7.x) as changes are made
