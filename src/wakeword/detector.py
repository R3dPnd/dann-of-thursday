"""Wake word detector using Picovoice Porcupine."""

import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pvporcupine
import sounddevice as sd


class WakeWordDetector:
    """Listens for wake word and invokes callback on detection."""

    def __init__(
        self,
        model_path: Path | str,
        on_wake: Callable[[], None],
        *,
        access_key: str,
        sensitivity: float = 0.5,
        debounce: int = 2,
        cooldown_s: float = 2.0,
        sample_rate: int = 16000,
        block_size: int = 512,
        device: int | None = None,
    ):
        self.model_path = Path(model_path)
        self.on_wake = on_wake
        self.access_key = access_key
        self.sensitivity = sensitivity
        self.debounce = debounce
        self.cooldown_s = cooldown_s
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.device = device

        # Porcupine requires 16-bit PCM audio (int16), not float32
        # Porcupine expects exactly 512 samples per frame for 16kHz
        if self.block_size != 512:
            raise ValueError("Porcupine requires block_size=512 for 16kHz audio")

        # Initialize Porcupine with custom keyword model
        self._porcupine = pvporcupine.create(
            access_key=self.access_key,
            keyword_paths=[str(self.model_path)],
            sensitivities=[self.sensitivity],
        )

        self._last_trigger = 0.0
        self._consecutive = 0
        self._running = False
        self._paused = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start listening in background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop listening."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._porcupine:
            self._porcupine.delete()
            self._porcupine = None

    def pause(self) -> None:
        """Pause detection (e.g. during TTS playback)."""
        self._paused = True

    def resume(self) -> None:
        """Resume detection."""
        self._paused = False

    def _run(self) -> None:
        def callback(indata: np.ndarray, frames: int, time_info: object, status: object) -> None:
            nonlocal self
            if status:
                print(f"[wakeword] {status}", flush=True)
            if self._paused:
                return

            # Convert float32 audio to int16 PCM for Porcupine
            audio_float = np.frombuffer(indata, dtype=np.float32)
            # Clip to [-1.0, 1.0] range before conversion
            audio_float = np.clip(audio_float, -1.0, 1.0)
            # Convert to int16 PCM
            audio_int16 = (audio_float * 32767).astype(np.int16)

            # Porcupine.process() returns keyword index (0 for first keyword, -1 if no match)
            keyword_index = self._porcupine.process(audio_int16)
            hit = keyword_index >= 0

            self._consecutive = self._consecutive + 1 if hit else 0

            now = time.monotonic()
            if (
                self._consecutive >= self.debounce
                and (now - self._last_trigger) >= self.cooldown_s
            ):
                self._last_trigger = now
                self._consecutive = 0
                try:
                    self.on_wake()
                except Exception as e:
                    print(f"[wakeword] callback error: {e}", flush=True)

        with sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            dtype="float32",
            device=self.device,
            callback=callback,
        ):
            while self._running:
                time.sleep(0.1)
