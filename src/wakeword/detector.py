"""Wake word detector using Picovoice Porcupine."""

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
        model_path: Path | str | None,
        on_wake: Callable[[], None],
        *,
        access_key: str,
        builtin_keyword: str | None = None,
        sensitivity: float = 0.5,
        debounce: int = 2,
        cooldown_s: float = 2.0,
        sample_rate: int = 16000,
        block_size: int = 512,
        device: int | None = None,
    ):
        self.model_path = Path(model_path) if model_path else None
        self.on_wake = on_wake
        self.access_key = access_key
        self.builtin_keyword = builtin_keyword
        self.sensitivity = sensitivity
        self.debounce = debounce
        self.cooldown_s = cooldown_s
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.device = device

        if self.block_size != 512:
            raise ValueError("Porcupine requires block_size=512 for 16kHz audio")

        if builtin_keyword:
            self._porcupine = pvporcupine.create(
                access_key=self.access_key,
                keywords=[builtin_keyword],
                sensitivities=[self.sensitivity],
            )
        elif self.model_path and self.model_path.exists():
            self._porcupine = pvporcupine.create(
                access_key=self.access_key,
                keyword_paths=[str(self.model_path)],
                sensitivities=[self.sensitivity],
            )
        else:
            raise FileNotFoundError(
                f"Wake word model not found: {self.model_path}. "
                "Use builtin_keyword (e.g. 'porcupine') to test, or download correct .ppn from Picovoice Console."
            )

        self._last_trigger = 0.0
        self._consecutive = 0
        self._running = False
        self._paused = False
        self._stream: sd.InputStream | None = None

    def start(self) -> None:
        """Start listening."""
        if self._running:
            return
        self._running = True
        self._paused = False
        self._open_stream()

    def stop(self) -> None:
        """Stop listening and release all resources."""
        self._running = False
        self._close_stream()
        if self._porcupine:
            self._porcupine.delete()
            self._porcupine = None

    def pause(self) -> None:
        """Pause detection and release the audio device for other streams."""
        self._paused = True
        self._close_stream()

    def resume(self) -> None:
        """Resume detection and reopen the audio device."""
        self._paused = False
        self._consecutive = 0
        if self._running:
            self._open_stream()

    def _open_stream(self) -> None:
        if self._stream is not None:
            return
        self._stream = sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            dtype="float32",
            device=self.device,
            callback=self._audio_callback,
        )
        self._stream.start()

    def _close_stream(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info: object, status: object
    ) -> None:
        if status:
            print(f"[wakeword] {status}", flush=True)
        if self._paused or not self._running:
            return

        audio_float = np.clip(indata[:, 0], -1.0, 1.0)
        audio_int16 = (audio_float * 32767).astype(np.int16)

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
