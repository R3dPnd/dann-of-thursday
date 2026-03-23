"""Unit tests for the Ollama LLM client."""

import pytest
from unittest.mock import MagicMock, patch, call


def _mock_response(content="Hello!", tool_calls=None):
    """Build a mock requests.Response for the Ollama /api/chat endpoint."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    msg = {"content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    resp.json.return_value = {"message": msg}
    return resp


# ── Basic response ────────────────────────────────────────────────────────────

class TestGenerateResponse:
    def test_returns_content(self):
        with patch("src.llm.ollama.requests.post", return_value=_mock_response("Hi there!")):
            from src.llm.ollama import generate_response
            result = generate_response("Hello")
        assert result == "Hi there!"

    def test_strips_whitespace(self):
        with patch("src.llm.ollama.requests.post",
                   return_value=_mock_response("  Hi!  \n")):
            from src.llm.ollama import generate_response
            result = generate_response("Hello")
        assert result == "Hi!"

    def test_empty_message_returns_empty_string(self):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"message": {}}
        with patch("src.llm.ollama.requests.post", return_value=resp):
            from src.llm.ollama import generate_response
            result = generate_response("Hello")
        assert result == ""

    def test_posts_to_correct_url(self):
        with patch("src.llm.ollama.requests.post",
                   return_value=_mock_response()) as mock_post:
            from src.llm.ollama import generate_response
            generate_response("Hello", base_url="http://localhost:11434")
        mock_post.assert_called_once()
        url = mock_post.call_args[0][0]
        assert url == "http://localhost:11434/api/chat"

    def test_system_prompt_included_in_messages(self):
        with patch("src.llm.ollama.requests.post",
                   return_value=_mock_response()) as mock_post:
            from src.llm.ollama import generate_response
            generate_response("What's up", system_prompt="You are a pirate.")
        payload = mock_post.call_args[1]["json"]
        messages = payload["messages"]
        assert messages[0] == {"role": "system", "content": "You are a pirate."}
        assert messages[-1] == {"role": "user", "content": "What's up"}

    def test_no_system_prompt_omitted(self):
        with patch("src.llm.ollama.requests.post",
                   return_value=_mock_response()) as mock_post:
            from src.llm.ollama import generate_response
            generate_response("Hi", system_prompt="")
        payload = mock_post.call_args[1]["json"]
        messages = payload["messages"]
        roles = [m["role"] for m in messages]
        assert "system" not in roles


# ── History ───────────────────────────────────────────────────────────────────

class TestHistory:
    def test_history_inserted_between_system_and_user(self):
        history = [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
        ]
        with patch("src.llm.ollama.requests.post",
                   return_value=_mock_response()) as mock_post:
            from src.llm.ollama import generate_response
            generate_response("second question",
                              system_prompt="You are helpful.",
                              history=history)
        payload = mock_post.call_args[1]["json"]
        messages = payload["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1] == history[0]
        assert messages[2] == history[1]
        assert messages[-1] == {"role": "user", "content": "second question"}

    def test_none_history_excluded(self):
        with patch("src.llm.ollama.requests.post",
                   return_value=_mock_response()) as mock_post:
            from src.llm.ollama import generate_response
            generate_response("Hi", system_prompt="Sys.", history=None)
        payload = mock_post.call_args[1]["json"]
        messages = payload["messages"]
        assert len(messages) == 2  # system + user only


# ── Tool calling ──────────────────────────────────────────────────────────────

class TestToolCalling:
    def test_tool_call_result_fed_back(self):
        tool_call = {
            "function": {"name": "list_projects", "arguments": {}}
        }
        responses = [
            _mock_response(content="", tool_calls=[tool_call]),
            _mock_response(content="Here are the projects."),
        ]
        mock_mcp = MagicMock()
        mock_mcp.call_tool.return_value = "project-a, project-b"

        with patch("src.llm.ollama.requests.post", side_effect=responses):
            from src.llm.ollama import generate_response
            result = generate_response("List projects",
                                       tools=[{"name": "list_projects"}],
                                       mcp=mock_mcp)

        mock_mcp.call_tool.assert_called_once_with("list_projects", {})
        assert result == "Here are the projects."

    def test_tool_error_handled_gracefully(self):
        tool_call = {"function": {"name": "bad_tool", "arguments": {}}}
        responses = [
            _mock_response(content="", tool_calls=[tool_call]),
            _mock_response(content="Something went wrong."),
        ]
        mock_mcp = MagicMock()
        mock_mcp.call_tool.side_effect = RuntimeError("tool failed")

        with patch("src.llm.ollama.requests.post", side_effect=responses):
            from src.llm.ollama import generate_response
            result = generate_response("Do something", tools=[{}], mcp=mock_mcp)

        assert result == "Something went wrong."

    def test_no_mcp_skips_tool_execution(self):
        """If mcp is None, tool_calls in response should be treated as final."""
        tool_call = {"function": {"name": "whatever", "arguments": {}}}
        resp = _mock_response(content="fallback text", tool_calls=[tool_call])

        with patch("src.llm.ollama.requests.post", return_value=resp):
            from src.llm.ollama import generate_response
            result = generate_response("Hi", tools=[{}], mcp=None)

        assert result == "fallback text"

    def test_max_tool_rounds_respected(self):
        """Loop exits after _MAX_TOOL_ROUNDS even if model keeps requesting tools."""
        tool_call = {"function": {"name": "loop_tool", "arguments": {}}}
        # Every response includes a tool call — should bail out after 5 rounds
        responses = [_mock_response(content="loop", tool_calls=[tool_call])] * 10
        mock_mcp = MagicMock()
        mock_mcp.call_tool.return_value = "result"

        with patch("src.llm.ollama.requests.post", side_effect=responses):
            from src.llm.ollama import generate_response, _MAX_TOOL_ROUNDS
            result = generate_response("Loop", tools=[{}], mcp=mock_mcp)

        assert mock_mcp.call_tool.call_count == _MAX_TOOL_ROUNDS
