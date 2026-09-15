"""GET /api/v1/focus-areas — topics the user wants Dann to engage on.

Distinct from /modules (technical tool+data bundles): a focus area is a
conversational interest, optionally backed by one of those modules for real
data. See config.yaml's `focus_areas:` and shared/focus_areas_config.py.

GET/POST/DELETE .../notes — the notes saved to a focus area's own context
directory (shared/focus_areas_store.py), automatically folded into every
conversation about it. Dann can add one via the save_focus_area_note tool;
these endpoints are the same thing for the dashboard.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter()


class CreateNoteRequest(BaseModel):
    content: str
    title: str = ""


@router.get("", summary="List focus areas")
async def list_focus_areas() -> JSONResponse:
    from voice.config import load_config
    from integrations.client import get_shared_manager

    cfg = load_config()
    module_status = {m["name"]: m["enabled"] for m in get_shared_manager().list_modules()}

    areas = []
    for area in cfg.get("focus_areas") or []:
        if not area.get("enabled", True):
            continue
        name = area.get("name")
        description = area.get("description", "")
        if not name:
            continue
        module = area.get("module")
        areas.append({
            "name": name,
            "description": description,
            "module": module,
            "module_enabled": module_status.get(module) if module else None,
        })

    return JSONResponse({"focus_areas": areas})


def _known_focus_areas() -> set[str]:
    from voice.config import load_config
    cfg = load_config()
    return {a.get("name") for a in (cfg.get("focus_areas") or []) if a.get("name")}


@router.get("/{name}/notes", summary="List a focus area's notes")
async def list_notes(name: str) -> JSONResponse:
    from shared.focus_areas_store import list_notes as _list_notes

    if name not in _known_focus_areas():
        raise HTTPException(status_code=404, detail=f"Unknown focus area '{name}'")
    return JSONResponse({"notes": _list_notes(name)})


@router.post("/{name}/notes", summary="Add a note to a focus area", status_code=201)
async def create_note(name: str, body: CreateNoteRequest) -> JSONResponse:
    from shared.focus_areas_store import save_note

    if name not in _known_focus_areas():
        raise HTTPException(status_code=404, detail=f"Unknown focus area '{name}'")
    if not body.content.strip():
        raise HTTPException(status_code=400, detail="content must not be empty")
    return JSONResponse(save_note(name, body.content, body.title))


@router.delete("/{name}/notes/{filename}", summary="Delete a focus area's note")
async def delete_note(name: str, filename: str) -> JSONResponse:
    from shared.focus_areas_store import delete_note as _delete_note

    if name not in _known_focus_areas():
        raise HTTPException(status_code=404, detail=f"Unknown focus area '{name}'")
    if not _delete_note(name, filename):
        raise HTTPException(status_code=404, detail=f"Note '{filename}' not found")
    return JSONResponse({"deleted": filename})
