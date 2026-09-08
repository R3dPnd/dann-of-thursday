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

from shared.agents_config import build_routing_prompt
from voice.audio import play_wav, record_until_silence
from voice.audio.capture import save_wav
from voice.config import load_config
from voice.event_bus import bus
from shared.restart import restart_process
from voice.llm import generate_response, generate_response_streaming
from integrations.client import MCPManager, get_shared_manager
from voice.stt import transcribe_audio
from voice.stt import warmup as warmup_stt
from voice.tts import synthesize_speech
from voice.tts import warmup as warmup_tts
from voice.wakeword import WakeWordDetector


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

# Manual, human-spoken restart trigger — deliberately not routed through the
# LLM/AGENTS system (deterministic regex, like goodbye/code-mode below),
# both for reliability and because this codebase's own testing found the
# router model unreliable at literal tool-calling under noisy input.
_RESTART_RE = re.compile(r"\brestart\s+(?:yourself|dann)\b", re.IGNORECASE)

# The router model occasionally hallucinates a fake tool call as literal text
# (e.g. "The weather is [web_search(query=\"...\")].") instead of actually
# invoking the tool via Ollama's structured tool-calling — non-deterministic,
# more likely with a noisy/rambling transcript. Same class of problem as
# _is_json_artifact below; same suppress-and-ask-again treatment.
_FAKE_TOOL_CALL_RE = re.compile(r"\[[a-zA-Z_][a-zA-Z0-9_]*\([^()]*\)\]")

# Default only — override via audio.min_speech_rms in config.yaml. This is
# genuinely environment-dependent: a sensitive mic in a noisy room can have
# an ambient noise floor close to this value, causing every ambient sound to
# be treated as a speech attempt (full STT + an apologetic TTS response)
# instead of being silently discarded. Measure your room's ambient RMS and
# tune this above it if "Could not understand" fires on silence.
_MIN_SPEECH_RMS = 0.005
_MAX_HISTORY_TURNS = 10      # normal-mode sliding window (user+assistant pairs)
_MAX_CODE_HISTORY_TURNS = 5  # code-mode context turns passed to ask_claude_code

# A real "ok Dann" from a human this soon after the previous session ended is
# very unlikely — flag it as a probable false wake trigger (see run()).
_RAPID_REWAKE_THRESHOLD_S = 3.0

# Give up and end the session after this many consecutive blank/unintelligible
# turns — otherwise a noisy room can keep Dann "listening" indefinitely,
# repeatedly recording, transcribing, and apologizing for ambient noise with
# no way for the user to escape it except an actual "goodbye".
_MAX_CONSECUTIVE_BLANK_TURNS = 4


class Orchestrator:
    """State machine: idle -> session (normal or code mode) -> idle."""

    def __init__(self, config_path: Path | None = None):
        self.config = load_config(config_path)
        self._audio_cfg = self.config.get("audio", {})
        self._wake_cfg = self.config.get("wake_word", {})
        self._stt_cfg = self.config.get("stt", {})
        self._ollama_cfg = dict(self.config.get("ollama", {}))
        routing_section = build_routing_prompt(self.config.get("agents"))
        if routing_section:
            base_prompt = self._ollama_cfg.get("system_prompt", "")
            self._ollama_cfg["system_prompt"] = f"{base_prompt}\n\n{routing_section}"
        self._tts_cfg = self.config.get("tts", {})
        self._ux_cfg = self.config.get("ux", {})
        self._mcp_cfg = self.config.get("mcp", {})

        self._detector: WakeWordDetector | None = None
        self._mcp: MCPManager | None = None
        self._running = False
        self._user_paused = False
        self._wake_event = threading.Event()
        self._wake_score = 0.0
        self._last_tts_finished_at: float | None = None
        self._last_session_ended_at: float | None = None
        self._consecutive_blank_turns = 0
        self._end_reason = "goodbye"
        self._history: list[dict[str, Any]] = []
        self._code_history: list[dict[str, Any]] = []
        self._mode = SessionMode.NORMAL
        self._code_project: str | None = None
        self._session_id: str | None = None
        self._started_at: float = time.time()

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_wake(self, score: float = 1.0) -> None:
        self._wake_score = score
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
        for wrong, right in _STT_SUBSTITUTIONS.items():
            text = re.sub(re.escape(wrong), right, text, flags=re.IGNORECASE)
        return text

    def _is_json_artifact(self, text: str) -> bool:
        stripped = text.strip()
        if not (stripped.startswith("{") or stripped.startswith("[")):
            return False
        try:
            json.loads(stripped)
            return True
        except (json.JSONDecodeError, ValueError):
            return False

    def _is_fake_tool_call(self, text: str) -> bool:
        return bool(_FAKE_TOOL_CALL_RE.search(text))

    def _is_rapid_rewake(self, since_session_end: float | None) -> bool:
        return since_session_end is not None and since_session_end < _RAPID_REWAKE_THRESHOLD_S

    def _should_give_up(self) -> bool:
        return self._consecutive_blank_turns >= _MAX_CONSECUTIVE_BLANK_TURNS

    def _format_code_task(self, task: str) -> str:
        """Prepend recent code-mode conversation context to the current task."""
        if not self._code_history:
            return task
        turns = self._code_history[-_MAX_CODE_HISTORY_TURNS * 2:]
        ctx = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Dann'}: {m['content']}"
            for m in turns
        )
        return f"Conversation so far:\n{ctx}\n\nCurrent question: {task}"

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
            "listening": not self._user_paused,
            "uptime_s": round(time.time() - self._started_at, 1),
        }

    def enable_listening(self) -> None:
        """Re-enable wake word detection (undo a user-initiated pause)."""
        self._user_paused = False
        if self._detector and self._session_id is None:
            self._detector.resume()
        bus.emit("voice.listening_changed", {"listening": True})

    def disable_listening(self) -> None:
        """Pause wake word detection without stopping the orchestrator."""
        self._user_paused = True
        if self._detector and self._session_id is None:
            self._detector.pause()
        bus.emit("voice.listening_changed", {"listening": False})

    def stop(self) -> None:
        """Graceful shutdown: speak goodbye if idle, stop detector, release MCP."""
        self._running = False
        self._wake_event.set()  # unblock the wait() in run()

        # Speak goodbye only when not mid-session (avoids interrupting a turn)
        if self._session_id is None:
            try:
                self._speak("Shutting down. Goodbye.")
            except Exception:
                pass

        if self._detector:
            try:
                self._detector.stop()
            except Exception:
                pass

        if self._mcp:
            try:
                self._mcp.stop()
            except Exception:
                pass

        bus.emit("state.changed", {
            "mode": "idle",
            "project": None,
            "session_id": None,
            "running": False,
        })

    # ── Audio ─────────────────────────────────────────────────────────────────

    def _speak(self, text: str) -> float:
        """Synthesize and play text. Returns TTS latency in ms."""
        t0 = time.monotonic()
        tts_path = synthesize_speech(
            text,
            voice_model=self._tts_cfg.get("voice_model", "models/piper/en_US-lessac-medium"),
            speed=self._tts_cfg.get("speed", 1.0),
        )
        play_wav(tts_path, device=self._audio_cfg.get("output_device"))
        self._last_tts_finished_at = time.monotonic()
        tts_ms = round((time.monotonic() - t0) * 1000)
        bus.emit("turn.tts.done", {"session_id": self._session_id})
        return tts_ms

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
        min_speech_rms = self._audio_cfg.get("min_speech_rms", _MIN_SPEECH_RMS)
        if float(np.sqrt(np.mean(audio_arr ** 2))) < min_speech_rms:
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
            self._consecutive_blank_turns += 1
            print(f"[dann] Could not understand ({self._consecutive_blank_turns}/{_MAX_CONSECUTIVE_BLANK_TURNS}).", flush=True)
            bus.emit("warning", {
                "module": "stt",
                "message": f"Could not understand audio ({self._consecutive_blank_turns}/{_MAX_CONSECUTIVE_BLANK_TURNS} consecutive)",
            })

            if self._should_give_up():
                print("[dann] Giving up after repeated blank turns — ending session.", flush=True)
                tts_ms = self._speak("I'm having trouble hearing you clearly — I'll stop listening for now.")
                self._end_reason = "no_response"
                bus.emit("metric", {
                    "session_id": self._session_id,
                    "mode": self._mode.value,
                    "stt_ms": stt_ms, "llm_ms": 0, "tts_ms": tts_ms, "code_ms": 0,
                    "blank": True, "status": "no_response",
                })
                return False

            tts_ms = self._speak("Sorry, I didn't catch that.")
            bus.emit("metric", {
                "session_id": self._session_id,
                "mode": self._mode.value,
                "stt_ms": stt_ms, "llm_ms": 0, "tts_ms": tts_ms, "code_ms": 0,
                "blank": True, "status": "blank",
            })
            return True

        self._consecutive_blank_turns = 0
        text = self._fix_stt(text)
        print(f"[dann] You said: {text}", flush=True)

        # ── Goodbye ───────────────────────────────────────────────────────────
        if self._is_goodbye(text):
            print("[dann] Session ended.", flush=True)
            tts_ms = self._speak("No problem, chat soon!")
            self._end_reason = "goodbye"
            bus.emit("metric", {
                "session_id": self._session_id,
                "mode": self._mode.value,
                "stt_ms": stt_ms, "llm_ms": 0, "tts_ms": tts_ms, "code_ms": 0,
                "blank": False, "status": "goodbye",
            })
            return False

        # ── Restart ──────────────────────────────────────────────────────────
        # Manual only — never triggered autonomously by the LLM/any tool, only
        # by this exact spoken phrase from a human.
        if _RESTART_RE.search(text):
            print("[dann] Restarting.", flush=True)
            self._speak("Okay, restarting now.")
            self.stop()  # releases the wake-word detector + stops the MCP manager
            restart_process()

        # ── Code mode exit ────────────────────────────────────────────────────
        if self._mode == SessionMode.CODE and _CODE_EXIT_RE.search(text):
            self._set_mode(SessionMode.NORMAL, None)
            self._history.clear()
            self._code_history.clear()
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
                self._code_history.clear()
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
                    {"project_name": self._code_project, "task": self._format_code_task(text)},
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
            self._code_history.append({"role": "user", "content": text})
            self._code_history.append({"role": "assistant", "content": response})
            tts_ms = self._speak(response)
            bus.emit("metric", {
                "session_id": self._session_id,
                "mode": "code", "stt_ms": stt_ms, "llm_ms": 0,
                "tts_ms": tts_ms, "code_ms": code_ms, "blank": False, "status": "ok",
            })
            return True

        # ── Normal mode: Ollama ───────────────────────────────────────────────
        print("[dann] Thinking...", flush=True)
        mcp_tools = self._mcp.tools if self._mcp else None
        model = self._ollama_cfg.get("model", "llama3.2")
        t0 = time.monotonic()

        if mcp_tools:
            # Tool calls require the non-streaming path (responses interleave with tool round-trips)
            self._speak("Hmm, one moment.")
            response = generate_response(
                text,
                base_url=self._ollama_cfg.get("base_url", "http://localhost:11434"),
                model=model,
                system_prompt=self._ollama_cfg.get("system_prompt", ""),
                temperature=self._ollama_cfg.get("temperature", 0.7),
                max_tokens=self._ollama_cfg.get("max_tokens", 80),
                tools=mcp_tools,
                mcp=self._mcp,
                history=self._history,
            )
            llm_ms = round((time.monotonic() - t0) * 1000)
            bus.emit("turn.llm", {
                "session_id": self._session_id,
                "text": response or "",
                "latency_ms": llm_ms,
                "model": model,
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

            if self._is_fake_tool_call(response):
                print(f"[dann] (suppressed hallucinated tool call): {response}", flush=True)
                bus.emit("warning", {
                    "module": "orchestrator",
                    "message": "LLM described a tool call as text instead of calling it; suppressed from TTS.",
                })
                tts_ms = self._speak("Sorry, I couldn't complete that. Could you ask again?")
                bus.emit("metric", {
                    "session_id": self._session_id,
                    "mode": "normal", "stt_ms": stt_ms, "llm_ms": llm_ms,
                    "tts_ms": tts_ms, "code_ms": 0, "blank": False, "status": "fake_tool_call",
                })
                return True

            print(f"[dann] {response}", flush=True)
            self._history.append({"role": "user", "content": text})
            self._history.append({"role": "assistant", "content": response})
            if len(self._history) > _MAX_HISTORY_TURNS * 2:
                self._history = self._history[-_MAX_HISTORY_TURNS * 2:]
            tts_ms = self._speak(response)
            bus.emit("metric", {
                "session_id": self._session_id,
                "mode": "normal", "stt_ms": stt_ms, "llm_ms": llm_ms,
                "tts_ms": tts_ms, "code_ms": 0, "blank": False, "status": "ok",
            })
            return True

        # Streaming path — speak each sentence chunk as it arrives
        gen = generate_response_streaming(
            text,
            base_url=self._ollama_cfg.get("base_url", "http://localhost:11434"),
            model=model,
            system_prompt=self._ollama_cfg.get("system_prompt", ""),
            temperature=self._ollama_cfg.get("temperature", 0.7),
            max_tokens=self._ollama_cfg.get("max_tokens", 80),
            history=self._history,
        )

        response_parts: list[str] = []
        tts_ms = 0
        try:
            first_chunk = next(gen)
            llm_ms = round((time.monotonic() - t0) * 1000)  # time-to-first-token
        except StopIteration:
            llm_ms = round((time.monotonic() - t0) * 1000)
            print("[dann] No response from Ollama.", flush=True)
            bus.emit("turn.llm", {
                "session_id": self._session_id, "text": "",
                "latency_ms": llm_ms, "model": model,
            })
            bus.emit("metric", {
                "session_id": self._session_id,
                "mode": "normal", "stt_ms": stt_ms, "llm_ms": llm_ms,
                "tts_ms": 0, "code_ms": 0, "blank": False, "status": "empty",
            })
            return True

        if first_chunk.strip().startswith(("{", "[")):
            # Potential JSON artifact — buffer everything before speaking
            response_parts.append(first_chunk)
            for chunk in gen:
                response_parts.append(chunk)
            response = " ".join(response_parts).strip()
            if self._is_json_artifact(response):
                print(f"[dann] (suppressed JSON artifact): {response}", flush=True)
                bus.emit("error", {
                    "module": "orchestrator",
                    "message": "LLM returned a raw JSON artifact; suppressed from TTS.",
                    "traceback": None,
                })
                tts_ms = self._speak("Sorry, I couldn't complete that. Could you rephrase?")
                bus.emit("turn.llm", {
                    "session_id": self._session_id, "text": response,
                    "latency_ms": llm_ms, "model": model,
                })
                bus.emit("metric", {
                    "session_id": self._session_id,
                    "mode": "normal", "stt_ms": stt_ms, "llm_ms": llm_ms,
                    "tts_ms": tts_ms, "code_ms": 0, "blank": False, "status": "json_artifact",
                })
                return True
            tts_ms = self._speak(response)
        else:
            # Normal streaming — speak each chunk as it arrives
            response_parts.append(first_chunk)
            bus.emit("turn.llm.chunk", {"session_id": self._session_id, "chunk": first_chunk, "index": 0})
            tts_ms += self._speak(first_chunk)
            for i, chunk in enumerate(gen, 1):
                response_parts.append(chunk)
                bus.emit("turn.llm.chunk", {"session_id": self._session_id, "chunk": chunk, "index": i})
                tts_ms += self._speak(chunk)
            response = " ".join(response_parts).strip()

        bus.emit("turn.llm", {
            "session_id": self._session_id,
            "text": response,
            "latency_ms": llm_ms,
            "model": model,
        })
        print(f"[dann] {response}", flush=True)
        self._history.append({"role": "user", "content": text})
        self._history.append({"role": "assistant", "content": response})
        if len(self._history) > _MAX_HISTORY_TURNS * 2:
            self._history = self._history[-_MAX_HISTORY_TURNS * 2:]
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
        self._code_history.clear()
        self._session_id = str(uuid.uuid4())
        self._set_mode(SessionMode.NORMAL, None)
        self._consecutive_blank_turns = 0
        self._end_reason = "goodbye"

        mode_hint = " Say 'code mode for <project>' to switch to Claude Code mode."
        print(f"[dann] Session started. Say 'thanks Dann' to stop.{mode_hint}", flush=True)

        bus.emit("session.start", {"session_id": self._session_id})
        self._speak("I'm listening.")

        try:
            while self._run_turn():
                pass
            bus.emit("session.end", {"session_id": self._session_id, "reason": self._end_reason})
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
            self._last_session_ended_at = time.monotonic()
            if self._detector and not self._user_paused:
                self._detector.resume()

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Start wake word listener and run until interrupted."""
        # MCP servers
        mcp_servers = self._mcp_cfg.get("servers") or []
        if mcp_servers:
            self._mcp = get_shared_manager()
            self._mcp.start(mcp_servers)  # idempotent — no-op if the FastAPI app already started it
            print("[dann] MCP server ready.", flush=True)

        # Wake word detector
        wake_engine = self._wake_cfg.get("engine", "porcupine")

        if wake_engine == "openwakeword":
            from voice.wakeword.openwakeword_detector import OpenWakeWordDetector
            wake_model = self._wake_cfg.get("model", "hey_jarvis")
            wake_phrase = Path(wake_model).stem if str(wake_model).endswith(".onnx") else str(wake_model)
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
            "running": True,
        })

        print(f"[dann] Listening for '{wake_phrase}'... (Ctrl+C to stop)", flush=True)
        self._detector.start()

        try:
            while self._running:
                if self._wake_event.wait(timeout=0.5):
                    self._wake_event.clear()
                    now = time.monotonic()
                    since_tts = (now - self._last_tts_finished_at) if self._last_tts_finished_at is not None else None
                    since_session_end = (now - self._last_session_ended_at) if self._last_session_ended_at is not None else None
                    print(f"[dann] Wake word detected (score={self._wake_score:.2f}).", flush=True)
                    bus.emit("wake.detected", {
                        "score": self._wake_score,
                        "since_tts_s": since_tts,
                        "since_session_end_s": since_session_end,
                    })
                    # A session ending and a fresh wake firing seconds later is
                    # exactly the shape of the mic picking up Dann's own TTS
                    # output as a false "ok Dann" — genuinely happened live,
                    # repeatedly, right after a goodbye response. Real user
                    # wake-ups are almost never this fast after a session ends.
                    if self._is_rapid_rewake(since_session_end):
                        bus.emit("warning", {
                            "module": "wakeword",
                            "message": (
                                f"Re-triggered {since_session_end:.1f}s after the previous "
                                f"session ended (score={self._wake_score:.2f}) — possible TTS "
                                "echo picked up as a wake word rather than a real request."
                            ),
                        })
                    self._run_session()
        except KeyboardInterrupt:
            print("\n[dann] Stopping...", flush=True)
        finally:
            self._detector.stop()
            if self._mcp:
                self._mcp.stop()
