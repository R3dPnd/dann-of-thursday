"""Unit tests for the STT (Whisper) module."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call


# ── _get_model / caching ──────────────────────────────────────────────────────

class TestGetModel:
    def test_creates_model_on_first_call(self):
        mock_model = MagicMock()
        with patch("src.stt.whisper._model_cache", {}), \
             patch("src.stt.whisper.WhisperModel", return_value=mock_model) as MockWM:
            from src.stt.whisper import _get_model
            result = _get_model("base", "cpu", "int8")
        MockWM.assert_called_once_with("base", device="cpu", compute_type="int8")
        assert result is mock_model

    def test_returns_cached_model_on_second_call(self):
        mock_model = MagicMock()
        cache = {}
        with patch("src.stt.whisper._model_cache", cache), \
             patch("src.stt.whisper.WhisperModel", return_value=mock_model) as MockWM:
            from src.stt.whisper import _get_model
            first = _get_model("base", "cpu", "int8")
            second = _get_model("base", "cpu", "int8")
        assert MockWM.call_count == 1
        assert first is second

    def test_different_params_create_separate_models(self):
        models = [MagicMock(), MagicMock()]
        cache = {}
        with patch("src.stt.whisper._model_cache", cache), \
             patch("src.stt.whisper.WhisperModel", side_effect=models):
            from src.stt.whisper import _get_model
            m1 = _get_model("base", "cpu", "int8")
            m2 = _get_model("large", "cpu", "int8")
        assert m1 is not m2
        assert len(cache) == 2


# ── warmup ────────────────────────────────────────────────────────────────────

class TestWarmup:
    def test_warmup_calls_get_model(self):
        with patch("src.stt.whisper._get_model") as mock_get:
            from src.stt.whisper import warmup
            warmup(model_size="tiny", device="cpu", compute_type="float32")
        mock_get.assert_called_once_with("tiny", "cpu", "float32")

    def test_warmup_uses_defaults(self):
        with patch("src.stt.whisper._get_model") as mock_get:
            from src.stt.whisper import warmup
            warmup()
        mock_get.assert_called_once_with("base", "cpu", "int8")


# ── transcribe_audio ──────────────────────────────────────────────────────────

class TestTranscribeAudio:
    def _make_segment(self, text):
        seg = MagicMock()
        seg.text = text
        return seg

    def test_joins_segments(self, tmp_path):
        audio_file = tmp_path / "test.wav"
        audio_file.touch()

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (
            [self._make_segment("Hello "), self._make_segment(" world")],
            MagicMock(),
        )

        with patch("src.stt.whisper._get_model", return_value=mock_model):
            from src.stt.whisper import transcribe_audio
            result = transcribe_audio(audio_file, model_size="base", language="en",
                                      device="cpu", compute_type="int8")

        assert result == "Hello world"

    def test_empty_segments_returns_empty_string(self, tmp_path):
        audio_file = tmp_path / "test.wav"
        audio_file.touch()

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], MagicMock())

        with patch("src.stt.whisper._get_model", return_value=mock_model):
            from src.stt.whisper import transcribe_audio
            result = transcribe_audio(audio_file)

        assert result == ""

    def test_whitespace_only_segments_filtered(self, tmp_path):
        audio_file = tmp_path / "test.wav"
        audio_file.touch()

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (
            [self._make_segment("  "), self._make_segment("   ")],
            MagicMock(),
        )

        with patch("src.stt.whisper._get_model", return_value=mock_model):
            from src.stt.whisper import transcribe_audio
            result = transcribe_audio(audio_file)

        assert result == ""

    def test_passes_language_to_model(self, tmp_path):
        audio_file = tmp_path / "test.wav"
        audio_file.touch()

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], MagicMock())

        with patch("src.stt.whisper._get_model", return_value=mock_model):
            from src.stt.whisper import transcribe_audio
            transcribe_audio(audio_file, language="fr")

        mock_model.transcribe.assert_called_once_with(str(audio_file), language="fr")
