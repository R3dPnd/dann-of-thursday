"""GET /api/v1/notes — list of note/reference repositories."""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("", summary="List note repositories")
async def list_notes() -> JSONResponse:
    from src.config import load_config
    cfg = load_config()
    notes = []
    for entry in cfg.get("notes", []):
        path = Path(str(entry.get("path", ""))).expanduser()
        name = entry.get("name") or path.name
        if path.exists():
            notes.append({"name": str(name), "path": str(path)})
    notes.sort(key=lambda n: n["name"].lower())
    return JSONResponse({"notes": notes, "count": len(notes)})
