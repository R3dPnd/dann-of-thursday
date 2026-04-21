"""POST /api/v1/prompt-builder — consolidate thoughts/goals/notes via Claude.

Uses the locally-authenticated `claude` CLI (Claude Code) so no separate
ANTHROPIC_API_KEY is required.  Falls back to the anthropic SDK if the CLI
is not available but the SDK + key are present.
"""

import json
import shutil
import subprocess

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class CustomSection(BaseModel):
    title: str = "Section"
    content: str = ""


class PromptBuilderRequest(BaseModel):
    thoughts: str = ""
    goals: str = ""
    notes: str = ""
    custom_sections: list[CustomSection] = []


class PromptBuilderResponse(BaseModel):
    recommended_model: str
    prompt: str
    reasoning: str


_META_PROMPT = """\
You are a prompt engineering expert. A user has provided raw thoughts, goals, and notes. Your task is to:

1. Recommend the best Claude model for this task from the list below
2. Craft a clear, effective, ready-to-use prompt that consolidates their inputs

Available models:
- claude-opus-4-6: Best for complex reasoning, deep analysis, research, nuanced multi-step tasks
- claude-sonnet-4-6: Best for balanced tasks — smart, fast, coding, writing, general use
- claude-haiku-4-5-20251001: Best for quick, simple tasks — classification, extraction, short answers

User's raw input:
{consolidated}

Respond with ONLY valid JSON in this exact format, no markdown fences:
{{
  "recommended_model": "<model_id>",
  "reasoning": "<1-2 sentences explaining why this model fits the task>",
  "prompt": "<the crafted, optimized, ready-to-use prompt>"
}}"""


def _call_via_cli(prompt: str) -> str:
    """Call the locally-installed `claude` CLI in print mode."""
    claude_bin = shutil.which("claude")
    if not claude_bin:
        raise RuntimeError("claude CLI not found on PATH")
    result = subprocess.run(
        [claude_bin, "-p", prompt],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude CLI exited {result.returncode}: {result.stderr.strip()[:400]}")
    return result.stdout.strip()


def _call_via_sdk(prompt: str) -> str:
    """Call Claude via the anthropic SDK (requires ANTHROPIC_API_KEY)."""
    import anthropic
    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


def _parse_response(text: str) -> dict:
    """Strip optional markdown fences and parse JSON."""
    if text.startswith("```"):
        parts = text.split("```")
        for part in parts:
            stripped = part.strip()
            if stripped.startswith("json"):
                stripped = stripped[4:].strip()
            if stripped.startswith("{"):
                text = stripped
                break
    return json.loads(text)


@router.post("", response_model=PromptBuilderResponse, summary="Build optimised prompt from consolidated thoughts")
async def build_prompt(req: PromptBuilderRequest) -> PromptBuilderResponse:
    sections: list[str] = []
    if req.thoughts.strip():
        sections.append(f"## Thoughts\n{req.thoughts.strip()}")
    if req.goals.strip():
        sections.append(f"## Goals\n{req.goals.strip()}")
    if req.notes.strip():
        sections.append(f"## Notes\n{req.notes.strip()}")
    for sec in req.custom_sections:
        if sec.content.strip():
            sections.append(f"## {sec.title}\n{sec.content.strip()}")

    if not sections:
        raise HTTPException(status_code=422, detail="At least one non-empty section is required.")

    consolidated = "\n\n".join(sections)
    prompt = _META_PROMPT.format(consolidated=consolidated)

    # Try the local Claude Code CLI first, then fall back to the SDK
    text: str | None = None
    last_error: str = ""

    if shutil.which("claude"):
        try:
            text = _call_via_cli(prompt)
        except Exception as exc:
            last_error = str(exc)

    if text is None:
        try:
            import anthropic  # noqa: F401
            text = _call_via_sdk(prompt)
        except ImportError:
            pass
        except Exception as exc:
            last_error = str(exc)

    if text is None:
        raise HTTPException(
            status_code=503,
            detail=f"No Claude backend available. CLI error: {last_error}. "
                   "Ensure `claude` CLI is on PATH (Claude Code) or set ANTHROPIC_API_KEY.",
        )

    try:
        data = _parse_response(text)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to parse Claude response as JSON: {exc}\n\nRaw: {text[:500]}",
        )

    return PromptBuilderResponse(
        recommended_model=data.get("recommended_model", "claude-sonnet-4-6"),
        prompt=data.get("prompt", ""),
        reasoning=data.get("reasoning", ""),
    )
