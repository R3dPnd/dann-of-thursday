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

## Recommended Execution Order

1. **Immediate (< 1 day):** Items 1.1, 1.2, 2.3, 2.5, 2.6 — low-effort, high-impact security and quality fixes
2. **Week 1:** Items 1.3, 2.1, 2.2, 2.7 — authentication, input validation, error handling, unified config
3. **Week 2:** Items 3.1, 3.2, 3.3 — establish test foundation and CI pipeline
4. **Week 3:** Items 2.4, 2.8, 5.1, 5.2 — type hints, model singleton, service layer, protocol schema
5. **Ongoing:** Documentation (6.x) and DX improvements (7.x) as changes are made
