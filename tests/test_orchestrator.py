"""Unit tests for Orchestrator pure-logic methods (no hardware required)."""

import pytest
from unittest.mock import patch

from src.orchestrator import Orchestrator, SessionMode


@pytest.fixture()
def orch(minimal_config):
    """Orchestrator instance with config injected, no real hardware initialised."""
    with patch("src.orchestrator.load_config", return_value=minimal_config):
        o = Orchestrator()
    return o


# ── _is_goodbye ───────────────────────────────────────────────────────────────

class TestIsGoodbye:
    @pytest.mark.parametrize("text", [
        "thanks Dann",
        "thank you dann",
        "thanks Dan",
        "goodbye dann",
        "bye Dann",
        "goodbye Dan.",
        "good bye dann",        # two-word variant
        "Goodbye, Dan.",        # punctuation
        "that's all dann",
        "that is all dan",
        "cheers dann",
        "goodbye",              # standalone ≤4 words
        "cheers",
        "bye bye",
        "that is all",
        "all done",
        "we're done",
    ])
    def test_recognised_as_goodbye(self, orch, text):
        assert orch._is_goodbye(text) is True

    @pytest.mark.parametrize("text", [
        "what is the weather today",
        "could you rephrase that please",
        "open dev diary",
        "code mode for dev diary",
        "bye",           # single word — intentionally excluded (too short to be certain)
        "thanks for explaining that to me",   # >4 words and no name
        "no",
    ])
    def test_not_goodbye(self, orch, text):
        assert orch._is_goodbye(text) is False


# ── _detect_code_entry ────────────────────────────────────────────────────────

class TestDetectCodeEntry:
    @pytest.mark.parametrize("text, expected_project", [
        ("code mode for dev diary", "dev diary"),
        ("enter code mode for fieldwatch api", "fieldwatch api"),
        ("start coding in dann of thursday", "dann of thursday"),
        ("coding mode for dev-diary", "dev-diary"),
        ("enter code mode for agency service", "agency service"),
    ])
    def test_entry_with_project(self, orch, text, expected_project):
        result = orch._detect_code_entry(text)
        assert result == expected_project

    def test_entry_without_project_returns_empty_string(self, orch):
        result = orch._detect_code_entry("enter code mode")
        assert result == ""

    @pytest.mark.parametrize("text", [
        "what is the weather",
        "open dev diary",
        "goodbye",
        "ask claude code to summarise",
    ])
    def test_non_entry_returns_none(self, orch, text):
        assert orch._detect_code_entry(text) is None


# ── _fix_stt ──────────────────────────────────────────────────────────────────

class TestFixStt:
    @pytest.mark.parametrize("raw, expected", [
        ("open cloud code in dev diary", "open claude code in dev diary"),
        ("clod code session",            "claude code session"),
        ("claud code please",            "claude code please"),
        ("no substitution needed",       "no substitution needed"),
        ("CLOUD CODE uppercase",         "claude code uppercase"),
    ])
    def test_substitutions(self, orch, raw, expected):
        assert orch._fix_stt(raw) == expected


# ── _is_json_artifact ────────────────────────────────────────────────────────

class TestIsJsonArtifact:
    @pytest.mark.parametrize("text", [
        '{"name": "open_claude_code", "arguments": {}}',
        '[{"tool": "list_projects"}]',
        '{}',
        '[]',
    ])
    def test_detects_json(self, orch, text):
        assert orch._is_json_artifact(text) is True

    @pytest.mark.parametrize("text", [
        "Claude Code is now open in dev-diary.",
        "The weather is sunny today.",
        "{not valid json",
        "",
        "  ",
    ])
    def test_passes_normal_text(self, orch, text):
        assert orch._is_json_artifact(text) is False


# ── _normalise ────────────────────────────────────────────────────────────────

class TestNormalise:
    def test_strips_punctuation(self, orch):
        assert orch._normalise("Hello, world!") == "hello world"

    def test_collapses_multiple_spaces(self, orch):
        assert orch._normalise("hello,  world") == "hello world"

    def test_collapses_good_bye(self, orch):
        assert orch._normalise("good bye dann") == "goodbye dann"

    def test_lowercases(self, orch):
        assert orch._normalise("THANKS DANN") == "thanks dann"


# ── SessionMode default ───────────────────────────────────────────────────────

def test_session_mode_starts_normal(orch):
    assert orch._mode == SessionMode.NORMAL
    assert orch._code_project is None
