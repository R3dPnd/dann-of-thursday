"""Wake word detector using openWakeWord."""

import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np
import sounddevice as sd
from openwakeword.model import Model


class WakeWordDetector:
    """Listens for wake word and invokes callback on detection."""

    def __init__(
        self,
        model_path: Path | str,
        on_wake: Callable[[], None],
        *,
        sensitivity: float = 0.5,
        debounce: int = 2,
        cooldown_s: float = 2.0,
        sample_rate: int = 16000,
        block_size: int = 512,
        device: int | None = None,
    ):
        self.model_path = Path(model_path)
        self.on_wake = on_wake
        self.sensitivity = sensitivity
        self.debounce = debounce
        self.cooldown_s = cooldown_s
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.device = device

        self._model = Model(
            wakeword_paths=[str(self.model_path)],
            sensitivity=[sensitivity],
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

            audio = np.frombuffer(indata, dtype=np.float32)
            scores = self._model.predict(audio)
            hit = any(s >= self.sensitivity for s in scores)
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
