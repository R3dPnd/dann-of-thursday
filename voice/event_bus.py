"""Lightweight in-process pub/sub for orchestrator → API event flow."""

from __future__ import annotations

import threading
from typing import Any, Callable


class EventBus:
    """Thread-safe pub/sub bus. Subscribers are called synchronously on the
    emitting thread, so callbacks must be fast (e.g. append to a queue)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: list[Callable[[str, dict[str, Any]], None]] = []

    def subscribe(self, callback: Callable[[str, dict[str, Any]], None]) -> None:
        """Register a callback(event_type, payload). Idempotent."""
        with self._lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[str, dict[str, Any]], None]) -> None:
        """Remove a previously registered callback. No-op if not present."""
        with self._lock:
            self._subscribers = [s for s in self._subscribers if s is not callback]

    def emit(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        """Broadcast an event to all current subscribers."""
        data = payload or {}
        with self._lock:
            targets = list(self._subscribers)
        for cb in targets:
            try:
                cb(event_type, data)
            except Exception:
                pass  # never let a bad subscriber crash the orchestrator


# Process-level singleton — import this everywhere
bus = EventBus()
