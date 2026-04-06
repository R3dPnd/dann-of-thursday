"""Wake word detector using openWakeWord (onnxruntime, no API key required)."""

import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import sounddevice as sd

# Ensure openWakeWord local install is importable
_OWW_ROOT = Path(__file__).resolve().parents[3] / "models" / "openwakeword"
if str(_OWW_ROOT) not in sys.path:
    sys.path.insert(0, str(_OWW_ROOT))

import openwakeword
import openwakeword.utils as oww_utils
from openwakeword.model import Model

_BLOCK_SIZE = 1280  # 80 ms at 16 kHz — required by openWakeWord


def _ensure_onnx_model(model_name: str) -> str:
    """
    Resolve a builtin model name (e.g. 'hey_jarvis') to its local .onnx path,
    downloading it (plus required feature/VAD models) on first use.
    """
    entry = openwakeword.MODELS.get(model_name)
    if not entry:
        raise ValueError(
            f"Unknown openWakeWord model '{model_name}'. "
            f"Available: {list(openwakeword.MODELS.keys())}"
        )
    onnx_path = Path(entry["model_path"]).with_suffix(".onnx")
    if not onnx_path.exists():
        print(f"[wakeword] Downloading '{model_name}' and feature models...", flush=True)
        oww_utils.download_models(
            model_names=[model_name],
            target_directory=str(onnx_path.parent),
        )
    return str(onnx_path)


class OpenWakeWordDetector:
    """
    Wake word detector backed by openWakeWord (ONNX backend, no API key).

    model_name can be a builtin name ('hey_jarvis', 'alexa', 'hey_mycroft', …)
    or a path to a custom .onnx file (e.g. 'models/ok_dann.onnx').

    Interface is identical to WakeWordDetector (start/stop/pause/resume).
    """

    def __init__(
        self,
        model_name: str | Path,
        on_wake: Callable[[], None],
        *,
        threshold: float = 0.5,
        debounce: int = 3,
        cooldown_s: float = 2.0,
        sample_rate: int = 16000,
        device: int | None = None,
    ):
        self.on_wake = on_wake
        self.threshold = threshold
        self.debounce = debounce
        self.cooldown_s = cooldown_s
        self.sample_rate = sample_rate
        self.device = device

        # Resolve model to an absolute .onnx path
        p = Path(model_name)
        if p.suffix in (".onnx", ".tflite"):
            if not p.exists():
                raise FileNotFoundError(f"Wake word model not found: {p}")
            self._model_path = str(p)
        else:
            self._model_path = _ensure_onnx_model(str(model_name))

        # Prediction dict key = filename stem, e.g. "hey_jarvis_v0.1"
        self._model_key = Path(self._model_path).stem

        self._oww = Model(
            wakeword_models=[self._model_path],
            inference_framework="onnx",
        )

        self._last_trigger = 0.0
        self._consecutive = 0
        self._running = False
        self._paused = False
        self._stream: sd.InputStream | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._paused = False
        self._open_stream()

    def stop(self) -> None:
        self._running = False
        self._close_stream()

    def pause(self) -> None:
        self._paused = True
        self._close_stream()

    def resume(self) -> None:
        self._paused = False
        self._consecutive = 0
        if self._running:
            self._open_stream()

    # ── Audio stream ──────────────────────────────────────────────────────────

    def _open_stream(self) -> None:
        if self._stream is not None:
            return
        self._stream = sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            blocksize=_BLOCK_SIZE,
            dtype="int16",
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

        audio = indata[:, 0]  # shape (1280,) int16
        preds = self._oww.predict(audio)
        score = float(preds.get(self._model_key, 0.0))

        hit = score >= self.threshold
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
