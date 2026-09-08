"""Play WAV audio through default output device."""

from pathlib import Path

import sounddevice as sd
import soundfile as sf


def play_wav(path: Path | str, device: int | None = None) -> None:
    """Play a WAV file and block until finished."""
    data, samplerate = sf.read(str(path), dtype="float32")
    sd.play(data, samplerate, device=device)
    sd.wait()
