"""WS /api/v1/events — real-time orchestrator event stream."""

import asyncio
import json
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.event_bus import bus

router = APIRouter()


@router.websocket("")
async def events_ws(websocket: WebSocket) -> None:
    """Stream orchestrator events to the connected browser client.

    Each message is a JSON object:  {"type": "<event_type>", "payload": {...}}
    """
    await websocket.accept()
    queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def _on_event(event_type: str, payload: dict[str, Any]) -> None:
        # Called from the orchestrator thread — schedule onto the event loop.
        loop.call_soon_threadsafe(queue.put_nowait, (event_type, payload))

    bus.subscribe(_on_event)
    try:
        while True:
            event_type, payload = await queue.get()
            await websocket.send_text(
                json.dumps({"type": event_type, "payload": payload})
            )
    except WebSocketDisconnect:
        pass
    finally:
        bus.unsubscribe(_on_event)
