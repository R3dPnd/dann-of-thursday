"""Speech-to-text."""

from .whisper import transcribe_audio, warmup

__all__ = ["transcribe_audio", "warmup"]
