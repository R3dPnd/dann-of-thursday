"""Orchestrates wake word -> record -> STT -> LLM -> TTS -> playback."""

import tempfile
from pathlib import Path

from src.audio import play_wav, record_until_silence
from src.audio.capture import save_wav
from src.config import load_config
from src.llm import generate_response
from src.stt import transcribe_audio
from src.tts import synthesize_speech
from src.wakeword import WakeWordDetector


class Orchestrator:
    """State machine: idle -> listening -> transcribing -> thinking -> speaking -> idle."""

    def __init__(self, config_path: Path | None = None):
        self.config = load_config(config_path)
        self._audio_cfg = self.config.get("audio", {})
        self._wake_cfg = self.config.get("wake_word", {})
        self._stt_cfg = self.config.get("stt", {})
        self._ollama_cfg = self.config.get("ollama", {})
        self._tts_cfg = self.config.get("tts", {})
        self._ux_cfg = self.config.get("ux", {})

        self._detector: WakeWordDetector | None = None
        self._running = False

    def _on_wake(self) -> None:
        """Called when wake word detected. Run full pipeline in main thread."""
        print("[dann] Wake word detected. Listening...", flush=True)
        self._run_pipeline()

    def _run_pipeline(self) -> None:
        """Record -> STT -> Ollama -> TTS -> playback."""
        # Pause wake word during processing to avoid echo
        if self._detector:
            self._detector.pause()

        try:
            # 1. Record
            pcm = record_until_silence(
                sample_rate=self._audio_cfg.get("sample_rate", 16000),
                channels=self._audio_cfg.get("channels", 1),
                silence_timeout_ms=self._audio_cfg.get("silence_timeout_ms", 1500),
                max_record_ms=self._audio_cfg.get("max_record_ms", 15000),
                silence_threshold=self._audio_cfg.get("silence_threshold", 0.01),
                device=self._audio_cfg.get("input_device"),
            )

            if not pcm:
                print("[dann] No audio captured.", flush=True)
                return

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wav_path = Path(f.name)
            save_wav(pcm, wav_path, self._audio_cfg.get("sample_rate", 16000))

            try:
                # 2. STT
                print("[dann] Transcribing...", flush=True)
                text = transcribe_audio(
                    wav_path,
                    model_size=self._stt_cfg.get("model_size", "base"),
                    language=self._stt_cfg.get("language", "en"),
                    device=self._stt_cfg.get("device", "cpu"),
                    compute_type=self._stt_cfg.get("compute_type", "int8"),
                )

                if not text:
                    print("[dann] Could not understand. Please try again.", flush=True)
                    return

                print(f"[dann] You said: {text}", flush=True)

                # 3. LLM
                print("[dann] Thinking...", flush=True)
                response = generate_response(
                    text,
                    base_url=self._ollama_cfg.get("base_url", "http://localhost:11434"),
                    model=self._ollama_cfg.get("model", "llama3.2"),
                    system_prompt=self._ollama_cfg.get("system_prompt", ""),
                    temperature=self._ollama_cfg.get("temperature", 0.7),
                    max_tokens=self._ollama_cfg.get("max_tokens", 150),
                )

                if not response:
                    print("[dann] No response from Ollama.", flush=True)
                    return

                print(f"[dann] {response}", flush=True)

                # 4. TTS
                print("[dann] Speaking...", flush=True)
                tts_path = synthesize_speech(
                    response,
                    piper_path=self._tts_cfg.get("piper_path", "piper"),
                    voice_model=self._tts_cfg.get("voice_model", "models/piper/en_US-lessac-medium"),
                    speed=self._tts_cfg.get("speed", 1.0),
                )

                # 5. Playback
                play_wav(tts_path, device=self._audio_cfg.get("output_device"))

            finally:
                wav_path.unlink(missing_ok=True)

        except Exception as e:
            print(f"[dann] Error: {e}", flush=True)
        finally:
            if self._detector:
                self._detector.resume()

    def run(self) -> None:
        """Start wake word listener and run until interrupted."""
        model_path = Path(self._wake_cfg.get("model_path", "models/ok_dann.ppn"))
        builtin_keyword = self._wake_cfg.get("builtin_keyword")

        # Require either custom .ppn or built-in keyword
        if not builtin_keyword and not model_path.exists():
            raise FileNotFoundError(
                f"Wake word model not found: {model_path}. "
                "Either download correct .ppn (Windows x86_64) from Picovoice Console, "
                "or set builtin_keyword: porcupine in config to test with built-in."
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

        self._running = True
        wake_phrase = builtin_keyword or "ok Dann"
        print(f"[dann] Listening for '{wake_phrase}'... (Ctrl+C to stop)", flush=True)
        self._detector.start()

        try:
            import time
            while self._running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[dann] Stopping...", flush=True)
        finally:
            self._detector.stop()
