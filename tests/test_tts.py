"""Unit tests for the TTS (Piper) module."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call


# ── _resolve_onnx_path ────────────────────────────────────────────────────────

class TestResolveOnnxPath:
    def test_adds_onnx_suffix(self, tmp_path):
        model_file = tmp_path / "voice.onnx"
        model_file.touch()

        from src.tts.piper import _resolve_onnx_path
        result = _resolve_onnx_path(str(tmp_path / "voice"))
        assert result == model_file

    def test_returns_path_directly_if_already_onnx(self, tmp_path):
        model_file = tmp_path / "voice.onnx"
        model_file.touch()

        from src.tts.piper import _resolve_onnx_path
        result = _resolve_onnx_path(model_file)
        assert result == model_file

    def test_directory_with_model_onnx(self, tmp_path):
        onnx_file = tmp_path / "model.onnx"
        onnx_file.touch()

        from src.tts.piper import _resolve_onnx_path
        result = _resolve_onnx_path(tmp_path)
        assert result == onnx_file

    def test_directory_globs_any_onnx(self, tmp_path):
        onnx_file = tmp_path / "custom_voice.onnx"
        onnx_file.touch()

        from src.tts.piper import _resolve_onnx_path
        result = _resolve_onnx_path(tmp_path)
        assert result == onnx_file

    def test_missing_file_raises_file_not_found(self, tmp_path):
        from src.tts.piper import _resolve_onnx_path
        with pytest.raises(FileNotFoundError):
            _resolve_onnx_path(tmp_path / "nonexistent.onnx")


# ── _get_voice / caching ──────────────────────────────────────────────────────

class TestGetVoice:
    def test_loads_voice_on_first_call(self, tmp_path):
        onnx_path = tmp_path / "voice.onnx"
        onnx_path.touch()
        mock_voice = MagicMock()

        with patch("src.tts.piper._voice_cache", {}), \
             patch("src.tts.piper.PiperVoice") as MockPV:
            MockPV.load.return_value = mock_voice
            from src.tts.piper import _get_voice
            result = _get_voice(onnx_path)

        MockPV.load.assert_called_once_with(onnx_path, use_cuda=False)
        assert result is mock_voice

    def test_returns_cached_voice_on_second_call(self, tmp_path):
        onnx_path = tmp_path / "voice.onnx"
        onnx_path.touch()
        mock_voice = MagicMock()
        cache = {}

        with patch("src.tts.piper._voice_cache", cache), \
             patch("src.tts.piper.PiperVoice") as MockPV:
            MockPV.load.return_value = mock_voice
            from src.tts.piper import _get_voice
            first = _get_voice(onnx_path)
            second = _get_voice(onnx_path)

        assert MockPV.load.call_count == 1
        assert first is second


# ── warmup ────────────────────────────────────────────────────────────────────

class TestWarmup:
    def test_warmup_silences_file_not_found(self):
        """warmup() must not raise even if the voice model is absent."""
        with patch("src.tts.piper._resolve_onnx_path",
                   side_effect=FileNotFoundError("no model")):
            from src.tts.piper import warmup
            warmup()  # should not raise

    def test_warmup_loads_voice_when_model_present(self, tmp_path):
        onnx_path = tmp_path / "voice.onnx"
        onnx_path.touch()
        mock_voice = MagicMock()

        with patch("src.tts.piper._resolve_onnx_path", return_value=onnx_path), \
             patch("src.tts.piper._get_voice", return_value=mock_voice) as mock_get:
            from src.tts.piper import warmup
            warmup("some/model/path")

        mock_get.assert_called_once_with(onnx_path)


# ── synthesize_speech ─────────────────────────────────────────────────────────

class TestSynthesizeSpeech:
    def _make_chunk(self):
        chunk = MagicMock()
        chunk.sample_rate = 22050
        chunk.sample_width = 2
        chunk.sample_channels = 1
        chunk.audio_int16_bytes = b"\x00\x01" * 100
        return chunk

    def test_writes_wav_file(self, tmp_path):
        out_path = tmp_path / "output.wav"
        mock_voice = MagicMock()
        mock_voice.synthesize.return_value = [self._make_chunk()]

        with patch("src.tts.piper._HAS_PIPER_API", True), \
             patch("src.tts.piper._resolve_onnx_path", return_value=tmp_path / "v.onnx"), \
             patch("src.tts.piper._get_voice", return_value=mock_voice):
            from src.tts.piper import synthesize_speech
            result = synthesize_speech("Hello world", output_path=out_path)

        assert result == out_path
        assert out_path.exists()

    def test_raises_when_piper_not_installed(self):
        with patch("src.tts.piper._HAS_PIPER_API", False):
            from src.tts.piper import synthesize_speech
            with pytest.raises(ImportError, match="piper-tts"):
                synthesize_speech("Hello")

    def test_speed_sets_length_scale(self, tmp_path):
        out_path = tmp_path / "output.wav"
        mock_voice = MagicMock()
        mock_voice.synthesize.return_value = [self._make_chunk()]

        with patch("src.tts.piper._HAS_PIPER_API", True), \
             patch("src.tts.piper._resolve_onnx_path", return_value=tmp_path / "v.onnx"), \
             patch("src.tts.piper._get_voice", return_value=mock_voice), \
             patch("src.tts.piper.SynthesisConfig") as MockSC:
            from src.tts.piper import synthesize_speech
            synthesize_speech("Hello", speed=2.0, output_path=out_path)

        MockSC.assert_called_once_with(length_scale=0.5)

    def test_speed_1_passes_none_length_scale(self, tmp_path):
        out_path = tmp_path / "output.wav"
        mock_voice = MagicMock()
        mock_voice.synthesize.return_value = [self._make_chunk()]

        with patch("src.tts.piper._HAS_PIPER_API", True), \
             patch("src.tts.piper._resolve_onnx_path", return_value=tmp_path / "v.onnx"), \
             patch("src.tts.piper._get_voice", return_value=mock_voice), \
             patch("src.tts.piper.SynthesisConfig") as MockSC:
            from src.tts.piper import synthesize_speech
            synthesize_speech("Hello", speed=1.0, output_path=out_path)

        MockSC.assert_called_once_with(length_scale=None)
