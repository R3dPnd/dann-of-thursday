"""Speech-to-text using faster-whisper."""

import warnings
from pathlib import Path

from faster_whisper import WhisperModel

# Cache loaded models so they are not re-initialised on every turn
_model_cache: dict[tuple, WhisperModel] = {}


def _get_model(model_size: str, device: str, compute_type: str) -> WhisperModel:
    key = (model_size, device, compute_type)
    if key not in _model_cache:
        _model_cache[key] = WhisperModel(model_size, device=device, compute_type=compute_type)
    return _model_cache[key]


def warmup(
    model_size: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
) -> None:
    """Pre-load the Whisper model so the first transcription is not slow."""
    _get_model(model_size, device, compute_type)


def transcribe_audio(
    audio_path: Path | str,
    *,
    model_size: str = "base",
    language: str = "en",
    device: str = "cpu",
    compute_type: str = "int8",
) -> str:
    """Transcribe WAV file to text. Returns empty string if nothing detected."""
    model = _get_model(model_size, device, compute_type)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        segments, _ = model.transcribe(str(audio_path), language=language)
    text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
    return text.strip()
