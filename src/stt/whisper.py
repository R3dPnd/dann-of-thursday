"""Speech-to-text using faster-whisper."""

from pathlib import Path

from faster_whisper import WhisperModel


def transcribe_audio(
    audio_path: Path | str,
    *,
    model_size: str = "base",
    language: str = "en",
    device: str = "cpu",
    compute_type: str = "int8",
) -> str:
    """Transcribe WAV file to text. Returns empty string if nothing detected."""
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, info = model.transcribe(str(audio_path), language=language)
    text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
    return text.strip()
