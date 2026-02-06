"""Record audio from microphone until silence or timeout."""

import time
from pathlib import Path

import numpy as np
import sounddevice as sd


def record_until_silence(
    *,
    sample_rate: int = 16000,
    channels: int = 1,
    silence_timeout_ms: int = 1500,
    max_record_ms: int = 15000,
    silence_threshold: float = 0.01,
    device: int | None = None,
) -> bytes:
    """
    Record from mic until `silence_timeout_ms` of silence or `max_record_ms` reached.
    Returns raw PCM bytes (int16, mono).
    """
    block_ms = 100
    block_samples = int(sample_rate * block_ms / 1000)
    silence_blocks = int(silence_timeout_ms / block_ms)
    max_blocks = int(max_record_ms / block_ms)

    chunks: list[np.ndarray] = []
    silent_count = 0
    block_count = 0

    def rms(arr: np.ndarray) -> float:
        return float(np.sqrt(np.mean(arr.astype(np.float64) ** 2)))

    def callback(indata: np.ndarray, frames: int, time_info: object, status: object) -> None:
        nonlocal silent_count
        if status:
            print(f"[audio] {status}", flush=True)
        chunk = indata.copy()
        chunks.append(chunk)
        if rms(chunk) < silence_threshold:
            silent_count += 1
        else:
            silent_count = 0

    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=channels,
        blocksize=block_samples,
        dtype="float32",
        device=device,
        callback=callback,
    )

    with stream:
        while block_count < max_blocks and silent_count < silence_blocks:
            time.sleep(block_ms / 1000)
            block_count += 1

    if not chunks:
        return b""

    audio = np.concatenate(chunks, axis=0)
    # Convert float32 [-1,1] to int16
    audio_int16 = (audio * 32767).astype(np.int16)
    return audio_int16.tobytes()


def save_wav(pcm_bytes: bytes, path: Path, sample_rate: int = 16000) -> None:
    """Save raw PCM (int16 mono) to WAV file."""
    import wave

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
