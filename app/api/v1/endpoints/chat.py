"""
Text-chat work streams — one per focus area, same Ollama+MCP routing brain
as voice, driven by typed text. See app/services/chat_service.py.

POST   /api/v1/chat/streams              — create a work stream
GET    /api/v1/chat/streams              — list work streams
GET    /api/v1/chat/streams/{id}         — stream detail + message history
POST   /api/v1/chat/streams/{id}/messages — send a message, get the response
POST   /api/v1/chat/streams/{id}/clear   — wipe message history, keep the stream
DELETE /api/v1/chat/streams/{id}         — delete a work stream
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.services import chat_service

router = APIRouter()


class CreateStreamRequest(BaseModel):
    focus_area: str
    title: str | None = None


class SendMessageRequest(BaseModel):
    text: str


@router.post("/streams", summary="Create a work stream")
async def create_stream(body: CreateStreamRequest) -> JSONResponse:
    try:
        stream = chat_service.create_stream(body.focus_area, body.title or "")
    except chat_service.UnknownStreamError:
        available = ", ".join(a["name"] for a in chat_service.list_focus_areas())
        raise HTTPException(
            status_code=404,
            detail=f"Unknown focus area '{body.focus_area}'. Available: {available or 'none'}",
        )
    return JSONResponse(stream)


@router.get("/streams", summary="List work streams")
async def list_streams() -> JSONResponse:
    return JSONResponse({"streams": chat_service.list_streams()})


@router.get("/streams/{stream_id}", summary="Get a work stream and its messages")
async def get_stream(stream_id: str) -> JSONResponse:
    stream = chat_service.get_stream(stream_id)
    if stream is None:
        raise HTTPException(status_code=404, detail=f"Unknown work stream: {stream_id}")
    return JSONResponse(stream)


@router.post("/streams/{stream_id}/messages", summary="Send a message in a work stream")
async def send_message(stream_id: str, body: SendMessageRequest) -> JSONResponse:
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    try:
        # generate_response/MCP tool calls are blocking — run off the event loop
        result = await asyncio.to_thread(chat_service.send_message, stream_id, body.text)
    except chat_service.UnknownStreamError:
        raise HTTPException(status_code=404, detail=f"Unknown work stream: {stream_id}")
    return JSONResponse(result)


@router.post("/streams/{stream_id}/clear", summary="Clear a work stream's message history")
async def clear_stream(stream_id: str) -> JSONResponse:
    try:
        stream = chat_service.clear_stream(stream_id)
    except chat_service.UnknownStreamError:
        raise HTTPException(status_code=404, detail=f"Unknown work stream: {stream_id}")
    return JSONResponse(stream)


@router.delete("/streams/{stream_id}", summary="Delete a work stream")
async def delete_stream(stream_id: str) -> JSONResponse:
    deleted = chat_service.delete_stream(stream_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Unknown work stream: {stream_id}")
    return JSONResponse({"deleted": True})
