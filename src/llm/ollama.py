"""Ollama API client for local LLM inference."""

import json
from typing import Any

import requests


def generate_response(
    prompt: str,
    *,
    base_url: str = "http://localhost:11434",
    model: str = "llama3.2",
    system_prompt: str = "You are a concise voice assistant. Keep responses brief.",
    temperature: float = 0.7,
    max_tokens: int = 150,
) -> str:
    """Send prompt to Ollama and return generated text."""
    url = f"{base_url.rstrip('/')}/api/generate"
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    if system_prompt:
        payload["system"] = system_prompt

    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "").strip()
