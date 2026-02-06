"""Text-to-speech using Piper (piper-tts Python API)."""

import tempfile
import wave
from pathlib import Path

try:
    from piper import PiperVoice, SynthesisConfig
    _HAS_PIPER_API = True
except ImportError:
    _HAS_PIPER_API = False


def _resolve_onnx_path(voice_model: str | Path) -> Path:
    """Resolve voice model path to .onnx file."""
    voice = Path(voice_model)
    if voice.is_dir():
        candidates = [voice / "model.onnx", voice.with_suffix(".onnx")]
        for p in candidates:
            if p.exists():
                return p
        onnx_files = list(voice.glob("*.onnx"))
        if onnx_files:
            return onnx_files[0]
    elif voice.suffix != ".onnx":
        voice = voice.with_suffix(".onnx")
    if not voice.exists():
        raise FileNotFoundError(f"Piper voice model not found: {voice}")
    return voice


def synthesize_speech(
    text: str,
    *,
    piper_path: str | None = None,
    voice_model: str | Path = "models/piper/en_US-lessac-medium",
    speed: float = 1.0,
    output_path: Path | None = None,
) -> Path:
    """
    Synthesize text to WAV using Piper. Returns path to WAV file.
    Uses piper-tts Python API (works on M1 Mac). Install: pip install piper-tts
    """
    if not _HAS_PIPER_API:
        raise ImportError(
            "piper-tts is required. Install with: pip install piper-tts"
        )

    onnx_path = _resolve_onnx_path(voice_model)
    out = output_path or Path(tempfile.gettempdir()) / "dann_tts_output.wav"

    voice = PiperVoice.load(onnx_path, use_cuda=False)
    syn_config = SynthesisConfig(
        length_scale=1.0 / speed if speed != 1.0 else None,
    )

    with wave.open(str(out), "wb") as wav_file:
        wav_params_set = False
        for i, chunk in enumerate(voice.synthesize(text.strip(), syn_config)):
            if not wav_params_set:
                wav_file.setframerate(chunk.sample_rate)
                wav_file.setsampwidth(chunk.sample_width)
                wav_file.setnchannels(chunk.sample_channels)
                wav_params_set = True
            wav_file.writeframes(chunk.audio_int16_bytes)

    return out
