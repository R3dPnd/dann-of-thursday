"""LLM integration (Ollama)."""

from .ollama import generate_response, generate_response_streaming

__all__ = ["generate_response", "generate_response_streaming"]
