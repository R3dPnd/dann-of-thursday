"""Speech-to-text using faster-whisper."""

import warnings
from pathlib import Path

from faster_whisper import WhisperModel

_model_cache: dict[tuple, WhisperModel] = {}


def _resolve_device(device: str) -> str:
    """Resolve 'auto' to 'cuda' if available, else 'cpu'."""
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _get_model(model_size: str, device: str, compute_type: str) -> WhisperModel:
    resolved = _resolve_device(device)
    key = (model_size, resolved, compute_type)
    if key not in _model_cache:
        _model_cache[key] = WhisperModel(model_size, device=resolved, compute_type=compute_type)
    return _model_cache[key]


def warmup(
    model_size: str = "small",
    device: str = "auto",
    compute_type: str = "int8",
) -> None:
    """Pre-load the Whisper model so the first transcription is not slow."""
    _get_model(model_size, device, compute_type)


def transcribe_audio(
    audio_path: Path | str,
    *,
    model_size: str = "small",
    language: str = "en",
    device: str = "auto",
    compute_type: str = "int8",
) -> str:
    """Transcribe WAV file to text. Returns empty string if nothing detected."""
    model = _get_model(model_size, device, compute_type)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        segments, _ = model.transcribe(
            str(audio_path),
            language=language,
            vad_filter=True,
        )
    text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
    return text.strip()
