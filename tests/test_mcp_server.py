"""Unit tests for the Claude Code MCP server tools."""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# ── _normalise ────────────────────────────────────────────────────────────────

class TestNormalise:
    @pytest.mark.parametrize("name, expected", [
        ("dev-diary",       "dev diary"),
        ("dev_diary",       "dev diary"),
        ("dann-of-thursday", "dann of thursday"),
        ("MyProject",       "myproject"),
        ("already fine",    "already fine"),
    ])
    def test_normalises_correctly(self, name, expected):
        from src.mcp_servers.claude_code_server import _normalise
        assert _normalise(name) == expected


# ── _find_projects ────────────────────────────────────────────────────────────

class TestFindProjects:
    def test_discovers_git_repo(self, tmp_path):
        repo = tmp_path / "my-project"
        repo.mkdir()
        (repo / ".git").mkdir()

        with patch("src.mcp_servers.claude_code_server._SEARCH_ROOTS", [tmp_path]), \
             patch("src.mcp_servers.claude_code_server._projects_cache", None):
            import src.mcp_servers.claude_code_server as srv
            srv._projects_cache = None
            projects = srv._find_projects()

        names = [p["name"] for p in projects]
        assert "my-project" in names

    def test_skips_venv_directories(self, tmp_path):
        repo = tmp_path / "real-project"
        repo.mkdir()
        (repo / ".git").mkdir()

        venv = tmp_path / ".venv"
        venv.mkdir()
        fake_repo = venv / "fake-project"
        fake_repo.mkdir()
        (fake_repo / ".git").mkdir()

        with patch("src.mcp_servers.claude_code_server._SEARCH_ROOTS", [tmp_path]), \
             patch("src.mcp_servers.claude_code_server._projects_cache", None):
            import src.mcp_servers.claude_code_server as srv
            srv._projects_cache = None
            projects = srv._find_projects()

        names = [p["name"] for p in projects]
        assert "fake-project" not in names
        assert "real-project" in names

    def test_returns_sorted_by_name(self, tmp_path):
        for name in ["zebra-project", "alpha-project", "middle-project"]:
            repo = tmp_path / name
            repo.mkdir()
            (repo / ".git").mkdir()

        with patch("src.mcp_servers.claude_code_server._SEARCH_ROOTS", [tmp_path]), \
             patch("src.mcp_servers.claude_code_server._projects_cache", None):
            import src.mcp_servers.claude_code_server as srv
            srv._projects_cache = None
            projects = srv._find_projects()

        names = [p["name"] for p in projects]
        assert names == sorted(names, key=str.lower)

    def test_uses_cache_on_second_call(self, tmp_path):
        cached = [{"name": "cached-project", "path": str(tmp_path)}]

        with patch("src.mcp_servers.claude_code_server._projects_cache", cached):
            import src.mcp_servers.claude_code_server as srv
            result = srv._find_projects()

        assert result is cached

    def test_missing_root_directory_returns_empty(self, tmp_path):
        missing = tmp_path / "nonexistent"

        with patch("src.mcp_servers.claude_code_server._SEARCH_ROOTS", [missing]), \
             patch("src.mcp_servers.claude_code_server._projects_cache", None):
            import src.mcp_servers.claude_code_server as srv
            srv._projects_cache = None
            projects = srv._find_projects()

        assert projects == []


# ── _resolve_project ──────────────────────────────────────────────────────────

class TestResolveProject:
    def _make_projects(self):
        return [
            {"name": "dev-diary",        "path": "/repos/dev-diary"},
            {"name": "dann-of-thursday", "path": "/repos/dann-of-thursday"},
            {"name": "fieldwatch-api",   "path": "/repos/fieldwatch-api"},
        ]

    def test_exact_normalised_match(self):
        projects = self._make_projects()
        with patch("src.mcp_servers.claude_code_server._find_projects",
                   return_value=projects):
            from src.mcp_servers.claude_code_server import _resolve_project
            result = _resolve_project("dev diary")
        assert result["name"] == "dev-diary"

    def test_partial_match_fallback(self):
        projects = self._make_projects()
        with patch("src.mcp_servers.claude_code_server._find_projects",
                   return_value=projects):
            from src.mcp_servers.claude_code_server import _resolve_project
            result = _resolve_project("fieldwatch")
        assert result["name"] == "fieldwatch-api"

    def test_unknown_project_returns_none(self):
        projects = self._make_projects()
        with patch("src.mcp_servers.claude_code_server._find_projects",
                   return_value=projects):
            from src.mcp_servers.claude_code_server import _resolve_project
            result = _resolve_project("nonexistent-repo")
        assert result is None

    def test_hyphens_in_query_normalised(self):
        projects = self._make_projects()
        with patch("src.mcp_servers.claude_code_server._find_projects",
                   return_value=projects):
            from src.mcp_servers.claude_code_server import _resolve_project
            result = _resolve_project("dann-of-thursday")
        assert result["name"] == "dann-of-thursday"


# ── list_projects ─────────────────────────────────────────────────────────────

class TestListProjects:
    def test_lists_all_projects(self):
        projects = [
            {"name": "alpha", "path": "/repos/alpha"},
            {"name": "beta",  "path": "/repos/beta"},
        ]
        with patch("src.mcp_servers.claude_code_server._find_projects",
                   return_value=projects):
            from src.mcp_servers.claude_code_server import list_projects
            result = list_projects()
        assert "alpha" in result
        assert "beta" in result

    def test_empty_returns_no_projects_message(self):
        with patch("src.mcp_servers.claude_code_server._find_projects",
                   return_value=[]):
            from src.mcp_servers.claude_code_server import list_projects
            result = list_projects()
        assert "No projects found" in result


# ── ask_claude_code ───────────────────────────────────────────────────────────

class TestAskClaudeCode:
    def test_returns_stdout_on_success(self):
        project = {"name": "dev-diary", "path": "/repos/dev-diary"}
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "  The project is a diary app.  "
        mock_result.stderr = ""

        with patch("src.mcp_servers.claude_code_server._resolve_project",
                   return_value=project), \
             patch("src.mcp_servers.claude_code_server.subprocess.run",
                   return_value=mock_result):
            from src.mcp_servers.claude_code_server import ask_claude_code
            result = ask_claude_code("dev diary", "Summarise this project")

        assert result == "The project is a diary app."

    def test_returns_error_message_on_non_zero_exit(self):
        project = {"name": "dev-diary", "path": "/repos/dev-diary"}
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "claude: command not found"

        with patch("src.mcp_servers.claude_code_server._resolve_project",
                   return_value=project), \
             patch("src.mcp_servers.claude_code_server.subprocess.run",
                   return_value=mock_result):
            from src.mcp_servers.claude_code_server import ask_claude_code
            result = ask_claude_code("dev diary", "Summarise")

        assert "error" in result.lower()

    def test_unknown_project_returns_not_found(self):
        with patch("src.mcp_servers.claude_code_server._resolve_project",
                   return_value=None), \
             patch("src.mcp_servers.claude_code_server._find_projects",
                   return_value=[]):
            from src.mcp_servers.claude_code_server import ask_claude_code
            result = ask_claude_code("ghost-project", "Do something")

        assert "not found" in result.lower()

    def test_empty_stdout_returns_empty_response_message(self):
        project = {"name": "dev-diary", "path": "/repos/dev-diary"}
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "   "
        mock_result.stderr = ""

        with patch("src.mcp_servers.claude_code_server._resolve_project",
                   return_value=project), \
             patch("src.mcp_servers.claude_code_server.subprocess.run",
                   return_value=mock_result):
            from src.mcp_servers.claude_code_server import ask_claude_code
            result = ask_claude_code("dev diary", "Summarise")

        assert "empty" in result.lower()
