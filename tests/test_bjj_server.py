"""Unit tests for the BJJ MCP server tools."""

from src.mcp_servers.bjj_server import (
    add_training_note,
    list_sessions,
    list_techniques,
    log_session,
    log_technique,
    search_training_log,
)


class TestLogSession:
    def test_log_and_list(self, tmp_dann_home):
        log_session("gi", duration_minutes=90, notes="worked takedowns")
        result = list_sessions()
        assert "gi" in result
        assert "90" in result

    def test_filters_by_type(self, tmp_dann_home):
        log_session("gi", duration_minutes=90)
        log_session("no-gi", duration_minutes=60)

        result = list_sessions(session_type="no-gi")
        assert "no-gi" in result
        assert "90" not in result

    def test_empty_list_message(self, tmp_dann_home):
        assert list_sessions() == "No sessions found."


class TestLogTechnique:
    def test_log_and_list(self, tmp_dann_home):
        log_technique("armbar from closed guard", category="submission", notes="keep hips tight")
        result = list_techniques()
        assert "armbar from closed guard" in result
        assert "submission" in result

    def test_filters_by_category(self, tmp_dann_home):
        log_technique("armbar", category="submission")
        log_technique("scissor sweep", category="sweep")

        result = list_techniques(category="sweep")
        assert "scissor sweep" in result
        assert "armbar" not in result


class TestSearchTrainingLog:
    def test_matches_across_sessions_techniques_notes(self, tmp_dann_home):
        log_session("gi", duration_minutes=90, notes="focused on armbar setups")
        log_technique("armbar from closed guard", category="submission")
        add_training_note("tweaked knee during armbar drilling", tags="injury")

        result = search_training_log("armbar")
        assert "session" in result
        assert "technique" in result
        assert "note" in result

    def test_no_match_returns_message(self, tmp_dann_home):
        log_session("gi", duration_minutes=90)
        result = search_training_log("nonexistent-technique")
        assert "Nothing in the training log" in result
