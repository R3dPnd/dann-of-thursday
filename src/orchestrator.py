"""Orchestrates wake word -> session (multi-turn) -> STT -> LLM -> TTS -> playback."""

import json
import re
import tempfile
import threading
import time
import traceback
import uuid
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from src.audio import play_wav, record_until_silence
from src.audio.capture import save_wav
from src.config import load_config
from src.event_bus import bus
from src.llm import generate_response
from src.mcp_client import MCPManager
from src.stt import transcribe_audio
from src.stt import warmup as warmup_stt
from src.tts import synthesize_speech
from src.tts import warmup as warmup_tts
from src.wakeword import WakeWordDetector


class SessionMode(Enum):
    NORMAL = "normal"
    CODE = "code"  # Bypasses Ollama — routes directly to ask_claude_code


# ── Goodbye ──────────────────────────────────────────────────────────────────
# Phrases that always end the session (with or without the name)
_GOODBYE_WITH_NAME = frozenset({
    "thanks dann", "thank you dann", "thanks dan", "thank you dan",
    "bye dann", "goodbye dann", "bye dan", "goodbye dan",
    "thats all dann", "thats all dan", "that is all dann", "that is all dan",
    "cheers dann", "cheers dan",
})
# Short standalone phrases (only matched when utterance is ≤4 words)
_GOODBYE_STANDALONE = frozenset({
    "goodbye", "bye bye", "thats all", "that is all",
    "cheers", "all done", "were done", "we are done",
})

# ── Code mode triggers ────────────────────────────────────────────────────────
_CODE_ENTRY_RE = re.compile(
    r"\bcode\s+mode\b|\bstart\s+cod(?:e|ing)\b|\benter\s+code\b|\bcoding\s+mode\b",
    re.IGNORECASE,
)
_CODE_PROJECT_RE = re.compile(
    r"\b(?:in|for|on)\s+([a-z0-9][a-z0-9\s\-_]+?)(?:\s*$|\s+project\b)",
    re.IGNORECASE,
)
_CODE_EXIT_RE = re.compile(
    r"\b(?:exit|leave|end|stop|quit)\s+code\s+mode\b"
    r"|\bback\s+to\s+normal\b"
    r"|\bexit\s+coding\b"
    r"|\bstop\s+coding\b",
    re.IGNORECASE,
)

# ── STT corrections ───────────────────────────────────────────────────────────
_STT_SUBSTITUTIONS: dict[str, str] = {
    "cloud code": "claude code",
    "clod code": "claude code",
    "claud code": "claude code",
}

_MIN_SPEECH_RMS = 0.005


class Orchestrator:
    """State machine: idle -> session (normal or code mode) -> idle."""

    def __init__(self, config_path: Path | None = None):
        self.config = load_config(config_path)
        self._audio_cfg = self.config.get("audio", {})
        self._wake_cfg = self.config.get("wake_word", {})
        self._stt_cfg = self.config.get("stt", {})
        self._ollama_cfg = self.config.get("ollama", {})
        self._tts_cfg = self.config.get("tts", {})
        self._ux_cfg = self.config.get("ux", {})
        self._mcp_cfg = self.config.get("mcp", {})

        self._detector: WakeWordDetector | None = None
        self._mcp: MCPManager | None = None
        self._running = False
        self._wake_event = threading.Event()
        self._history: list[dict[str, Any]] = []
        self._mode = SessionMode.NORMAL
        self._code_project: str | None = None
        self._session_id: str | None = None
        self._started_at: float = time.time()

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_wake(self) -> None:
        self._wake_event.set()

    # ── Text helpers ──────────────────────────────────────────────────────────

    def _normalise(self, text: str) -> str:
        """Lowercase, strip punctuation, collapse 'good bye' → 'goodbye'."""
        n = text.lower()
        n = re.sub(r"'", "", n)           # strip apostrophes without spacing
        n = re.sub(r"[^\w\s]", " ", n)   # replace other punctuation with space
        n = re.sub(r"\s+", " ", n).strip()
        n = re.sub(r"\bgood\s+bye\b", "goodbye", n)
        return n

    def _is_goodbye(self, text: str) -> bool:
        n = self._normalise(text)
        if any(phrase in n for phrase in _GOODBYE_WITH_NAME):
            return True
        # Only match standalone phrases on short utterances to avoid false positives
        if len(n.split()) <= 4 and any(phrase in n for phrase in _GOODBYE_STANDALONE):
            return True
        return False

    def _fix_stt(self, text: str) -> str:
        lower = text.lower()
        for wrong, right in _STT_SUBSTITUTIONS.items():
            lower = lower.replace(wrong, right)
        return lower

    def _is_json_artifact(self, text: str) -> bool:
        stripped = text.strip()
        if not (stripped.startswith("{") or stripped.startswith("[")):
            return False
        try:
            json.loads(stripped)
            return True
        except (json.JSONDecodeError, ValueError):
            return False

    def _detect_code_entry(self, text: str) -> str | None:
        """Return project name if text is a code-mode entry command, else None."""
        if not _CODE_ENTRY_RE.search(text):
            return None
        m = _CODE_PROJECT_RE.search(text)
        return m.group(1).strip() if m else ""

    # ── State helpers ─────────────────────────────────────────────────────────

    def _set_mode(self, mode: SessionMode, project: str | None = None) -> None:
        """Update mode and project, then emit state.changed."""
        self._mode = mode
        self._code_project = project
        bus.emit("state.changed", {
            "mode": mode.value,
            "project": project,
            "session_id": self._session_id,
        })

    def snapshot(self) -> dict[str, Any]:
        """Return a serialisable snapshot of current orchestrator state."""
        return {
            "mode": self._mode.value,
            "project": self._code_project,
            "session_id": self._session_id,
            "running": self._running,
            "uptime_s": round(time.time() - self._started_at, 1),
        }

    # ── Audio ─────────────────────────────────────────────────────────────────

    def _speak(self, text: str) -> float:
        """Synthesize and play text. Returns TTS latency in ms."""
        t0 = time.monotonic()
        tts_path = synthesize_speech(
            text,
            piper_path=self._tts_cfg.get("piper_path", "piper"),
            voice_model=self._tts_cfg.get("voice_model", "models/piper/en_US-lessac-medium"),
            speed=self._tts_cfg.get("speed", 1.0),
        )
        play_wav(tts_path, device=self._audio_cfg.get("output_device"))
        return round((time.monotonic() - t0) * 1000)

    def _record(self) -> bytes:
        return record_until_silence(
            sample_rate=self._audio_cfg.get("sample_rate", 16000),
            channels=self._audio_cfg.get("channels", 1),
            silence_timeout_ms=self._audio_cfg.get("silence_timeout_ms", 1500),
            max_record_ms=self._audio_cfg.get("max_record_ms", 15000),
            silence_threshold=self._audio_cfg.get("silence_threshold", 0.01),
            device=self._audio_cfg.get("input_device"),
        )

    def _transcribe(self, pcm: bytes) -> str | None:
        """Save PCM to temp WAV, run Whisper, return text or None."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = Path(f.name)
        save_wav(pcm, wav_path, self._audio_cfg.get("sample_rate", 16000))
        try:
            return transcribe_audio(
                wav_path,
                model_size=self._stt_cfg.get("model_size", "base"),
                language=self._stt_cfg.get("language", "en"),
                device=self._stt_cfg.get("device", "cpu"),
                compute_type=self._stt_cfg.get("compute_type", "int8"),
            )
        finally:
            wav_path.unlink(missing_ok=True)

    # ── Turn handlers ─────────────────────────────────────────────────────────

    def _run_turn(self) -> bool:
        """Single listen -> STT -> route -> TTS cycle. Returns False to end session."""
        stt_ms = llm_ms = tts_ms = code_ms = 0

        bus.emit("turn.start", {"session_id": self._session_id})

        pcm = self._record()

        if not pcm:
            return True

        audio_arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32767
        if float(np.sqrt(np.mean(audio_arr ** 2))) < _MIN_SPEECH_RMS:
            return True

        print("[dann] Transcribing...", flush=True)
        t0 = time.monotonic()
        text = self._transcribe(pcm)
        stt_ms = round((time.monotonic() - t0) * 1000)

        bus.emit("turn.stt", {
            "session_id": self._session_id,
            "text": text or "",
            "blank": not bool(text),
            "latency_ms": stt_ms,
        })

        if not text:
            print("[dann] Could not understand.", flush=True)
            tts_ms = self._speak("Sorry, I didn't catch that.")
            bus.emit("metric", {
                "session_id": self._session_id,
                "mode": self._mode.value,
                "stt_ms": stt_ms, "llm_ms": 0, "tts_ms": tts_ms, "code_ms": 0,
                "blank": True, "status": "blank",
            })
            return True

        text = self._fix_stt(text)
        print(f"[dann] You said: {text}", flush=True)

        # ── Goodbye ───────────────────────────────────────────────────────────
        if self._is_goodbye(text):
            print("[dann] Session ended.", flush=True)
            tts_ms = self._speak("No problem, chat soon!")
            bus.emit("metric", {
                "session_id": self._session_id,
                "mode": self._mode.value,
                "stt_ms": stt_ms, "llm_ms": 0, "tts_ms": tts_ms, "code_ms": 0,
                "blank": False, "status": "goodbye",
            })
            return False

        # ── Code mode exit ────────────────────────────────────────────────────
        if self._mode == SessionMode.CODE and _CODE_EXIT_RE.search(text):
            self._set_mode(SessionMode.NORMAL, None)
            self._history.clear()
            print("[dann] Exiting code mode.", flush=True)
            tts_ms = self._speak("Exiting code mode, back to normal.")
            bus.emit("metric", {
                "session_id": self._session_id,
                "mode": "code", "stt_ms": stt_ms, "llm_ms": 0,
                "tts_ms": tts_ms, "code_ms": 0, "blank": False, "status": "mode_exit",
            })
            return True

        # ── Code mode entry ───────────────────────────────────────────────────
        if self._mode == SessionMode.NORMAL:
            project = self._detect_code_entry(text)
            if project is not None:
                if not project:
                    tts_ms = self._speak("Which project should I enter code mode for?")
                    return True
                self._set_mode(SessionMode.CODE, project)
                self._history.clear()
                print(f"[dann] Entering code mode for: {project}", flush=True)
                tts_ms = self._speak(f"Entering code mode for {project}.")
                if self._mcp:
                    try:
                        result = self._mcp.call_tool(
                            "open_claude_code", {"project_name": project}
                        )
                        print(f"[dann] {result}", flush=True)
                        tts_ms += self._speak("Claude Code is open. Ask me anything about the project.")
                    except Exception as e:
                        bus.emit("error", {
                            "module": "orchestrator",
                            "message": f"Could not open Claude Code: {e}",
                            "traceback": traceback.format_exc(),
                        })
                        print(f"[dann] Could not open Claude Code: {e}", flush=True)
                        tts_ms += self._speak("I couldn't open Claude Code, but I can still answer questions.")
                bus.emit("metric", {
                    "session_id": self._session_id,
                    "mode": "normal", "stt_ms": stt_ms, "llm_ms": 0,
                    "tts_ms": tts_ms, "code_ms": 0, "blank": False, "status": "mode_entry",
                })
                return True

        # ── Code mode query (bypass Ollama entirely) ──────────────────────────
        if self._mode == SessionMode.CODE:
            if not self._mcp:
                self._speak("No MCP connection available.")
                return True
            print("[dann] Asking Claude Code...", flush=True)
            self._speak("Asking Claude Code, one moment.")
            t0 = time.monotonic()
            try:
                response = self._mcp.call_tool(
                    "ask_claude_code",
                    {"project_name": self._code_project, "task": text},
                )
                code_ms = round((time.monotonic() - t0) * 1000)
                status = "empty" if not response else "ok"
            except Exception as e:
                code_ms = round((time.monotonic() - t0) * 1000)
                status = "error"
                bus.emit("error", {
                    "module": "orchestrator",
                    "message": f"Claude Code error: {e}",
                    "traceback": traceback.format_exc(),
                })
                print(f"[dann] Claude Code error: {e}", flush=True)
                tts_ms = self._speak("Claude Code returned an error.")
                bus.emit("turn.code", {
                    "session_id": self._session_id,
                    "project": self._code_project,
                    "task": text,
                    "response": "",
                    "status": "error",
                    "latency_ms": code_ms,
                })
                bus.emit("metric", {
                    "session_id": self._session_id,
                    "mode": "code", "stt_ms": stt_ms, "llm_ms": 0,
                    "tts_ms": tts_ms, "code_ms": code_ms, "blank": False, "status": "error",
                })
                return True

            bus.emit("turn.code", {
                "session_id": self._session_id,
                "project": self._code_project,
                "task": text,
                "response": response or "",
                "status": status,
                "latency_ms": code_ms,
            })

            if not response:
                tts_ms = self._speak("Claude Code had no response.")
                bus.emit("metric", {
                    "session_id": self._session_id,
                    "mode": "code", "stt_ms": stt_ms, "llm_ms": 0,
                    "tts_ms": tts_ms, "code_ms": code_ms, "blank": False, "status": "empty",
                })
                return True

            print(f"[dann] {response}", flush=True)
            tts_ms = self._speak(response)
            bus.emit("metric", {
                "session_id": self._session_id,
                "mode": "code", "stt_ms": stt_ms, "llm_ms": 0,
                "tts_ms": tts_ms, "code_ms": code_ms, "blank": False, "status": "ok",
            })
            return True

        # ── Normal mode: Ollama ───────────────────────────────────────────────
        print("[dann] Thinking...", flush=True)
        self._speak("Hmm, one moment.")
        mcp_tools = self._mcp.tools if self._mcp else None
        t0 = time.monotonic()
        response = generate_response(
            text,
            base_url=self._ollama_cfg.get("base_url", "http://localhost:11434"),
            model=self._ollama_cfg.get("model", "llama3.2"),
            system_prompt=self._ollama_cfg.get("system_prompt", ""),
            temperature=self._ollama_cfg.get("temperature", 0.7),
            max_tokens=self._ollama_cfg.get("max_tokens", 80),
            tools=mcp_tools or None,
            mcp=self._mcp,
            history=self._history,
        )
        llm_ms = round((time.monotonic() - t0) * 1000)

        bus.emit("turn.llm", {
            "session_id": self._session_id,
            "text": response or "",
            "latency_ms": llm_ms,
            "model": self._ollama_cfg.get("model", "llama3.2"),
        })

        if not response:
            print("[dann] No response from Ollama.", flush=True)
            bus.emit("metric", {
                "session_id": self._session_id,
                "mode": "normal", "stt_ms": stt_ms, "llm_ms": llm_ms,
                "tts_ms": 0, "code_ms": 0, "blank": False, "status": "empty",
            })
            return True

        if self._is_json_artifact(response):
            print(f"[dann] (suppressed JSON artifact): {response}", flush=True)
            bus.emit("error", {
                "module": "orchestrator",
                "message": "LLM returned a raw JSON artifact; suppressed from TTS.",
                "traceback": None,
            })
            tts_ms = self._speak("Sorry, I couldn't complete that. Could you rephrase?")
            bus.emit("metric", {
                "session_id": self._session_id,
                "mode": "normal", "stt_ms": stt_ms, "llm_ms": llm_ms,
                "tts_ms": tts_ms, "code_ms": 0, "blank": False, "status": "json_artifact",
            })
            return True

        print(f"[dann] {response}", flush=True)
        self._history.append({"role": "user", "content": text})
        self._history.append({"role": "assistant", "content": response})
        tts_ms = self._speak(response)
        bus.emit("metric", {
            "session_id": self._session_id,
            "mode": "normal", "stt_ms": stt_ms, "llm_ms": llm_ms,
            "tts_ms": tts_ms, "code_ms": 0, "blank": False, "status": "ok",
        })
        return True

    def _run_session(self) -> None:
        """Multi-turn session until goodbye."""
        if self._detector:
            self._detector.pause()
        self._history.clear()
        self._session_id = str(uuid.uuid4())
        self._set_mode(SessionMode.NORMAL, None)

        mode_hint = " Say 'code mode for <project>' to switch to Claude Code mode."
        print(f"[dann] Session started. Say 'thanks Dann' to stop.{mode_hint}", flush=True)

        bus.emit("session.start", {"session_id": self._session_id})
        self._speak("I'm listening.")

        try:
            while self._run_turn():
                pass
            bus.emit("session.end", {"session_id": self._session_id, "reason": "goodbye"})
        except Exception as e:
            bus.emit("error", {
                "module": "orchestrator",
                "message": f"Session error: {e}",
                "traceback": traceback.format_exc(),
            })
            bus.emit("session.end", {"session_id": self._session_id, "reason": "error"})
            print(f"[dann] Error: {e}", flush=True)
            self._speak("Something went wrong. Starting over.")
        finally:
            self._history.clear()
            self._set_mode(SessionMode.NORMAL, None)
            self._session_id = None
            if self._detector:
                self._detector.resume()

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Start wake word listener and run until interrupted."""
        # MCP servers
        mcp_servers = self._mcp_cfg.get("servers") or []
        if mcp_servers:
            self._mcp = MCPManager()
            self._mcp.start(mcp_servers)
            print("[dann] MCP server ready.", flush=True)

        # Wake word detector
        wake_engine = self._wake_cfg.get("engine", "porcupine")

        if wake_engine == "openwakeword":
            from src.wakeword.openwakeword_detector import OpenWakeWordDetector
            wake_model = self._wake_cfg.get("model", "hey_jarvis")
            wake_phrase = str(wake_model)
            self._detector = OpenWakeWordDetector(
                model_name=wake_model,
                on_wake=self._on_wake,
                threshold=self._wake_cfg.get("threshold", 0.5),
                debounce=self._wake_cfg.get("debounce", 3),
                cooldown_s=self._wake_cfg.get("cooldown_ms", 2000) / 1000,
                sample_rate=self._audio_cfg.get("sample_rate", 16000),
                device=self._audio_cfg.get("input_device"),
            )
        else:
            # Porcupine
            model_path = Path(self._wake_cfg.get("model_path", "models/ok_dann.ppn"))
            builtin_keyword = self._wake_cfg.get("builtin_keyword")
            wake_phrase = builtin_keyword or "ok Dann"

            if not builtin_keyword and not model_path.exists():
                raise FileNotFoundError(
                    f"Wake word model not found: {model_path}. "
                    "Set builtin_keyword: porcupine in config to test with a built-in keyword."
                )

            access_key = self._wake_cfg.get("access_key")
            if not access_key:
                raise ValueError(
                    "Porcupine access_key required. Get one from https://console.picovoice.ai/"
                )

            self._detector = WakeWordDetector(
                model_path=model_path,
                on_wake=self._on_wake,
                access_key=access_key,
                builtin_keyword=builtin_keyword,
                sensitivity=self._wake_cfg.get("sensitivity", 0.5),
                debounce=self._wake_cfg.get("debounce", 2),
                cooldown_s=self._wake_cfg.get("cooldown_ms", 2000) / 1000,
                sample_rate=self._audio_cfg.get("sample_rate", 16000),
                block_size=512,
                device=self._audio_cfg.get("input_device"),
            )

        # Pre-load models
        print("[dann] Loading models...", flush=True)
        warmup_stt(
            model_size=self._stt_cfg.get("model_size", "base"),
            device=self._stt_cfg.get("device", "cpu"),
            compute_type=self._stt_cfg.get("compute_type", "int8"),
        )
        warmup_tts(voice_model=self._tts_cfg.get("voice_model", "models/piper/en_US-lessac-medium"))
        self._speak("Ready.")
        print("[dann] Models loaded.", flush=True)

        self._running = True
        bus.emit("state.changed", {
            "mode": self._mode.value,
            "project": None,
            "session_id": None,
        })

        print(f"[dann] Listening for '{wake_phrase}'... (Ctrl+C to stop)", flush=True)
        self._detector.start()

        try:
            while self._running:
                if self._wake_event.wait(timeout=0.5):
                    self._wake_event.clear()
                    print("[dann] Wake word detected.", flush=True)
                    self._run_session()
        except KeyboardInterrupt:
            print("\n[dann] Stopping...", flush=True)
        finally:
            self._detector.stop()
            if self._mcp:
                self._mcp.stop()
