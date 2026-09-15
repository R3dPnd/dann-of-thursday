"""GET /api/v1/modules — MCP module status; POST to enable/disable one.

Mirrors the list_modules / enable_module / disable_module LLM tools
(integrations/client.py) as REST endpoints so the dashboard can show and
control Dann's optional modules directly.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("", summary="List MCP modules and their status")
async def list_modules() -> JSONResponse:
    from integrations.client import get_shared_manager
    mgr = get_shared_manager()
    return JSONResponse({"modules": mgr.list_modules()})


@router.post("/{module_name}/enable", summary="Start an optional module")
async def enable_module(module_name: str) -> JSONResponse:
    from integrations.client import get_shared_manager
    mgr = get_shared_manager()
    try:
        mgr.enable_module(module_name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown module '{module_name}'")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to start '{module_name}': {exc}")
    return JSONResponse({"module": module_name, "enabled": True})


@router.post("/{module_name}/disable", summary="Stop an optional module")
async def disable_module(module_name: str) -> JSONResponse:
    from integrations.client import get_shared_manager
    mgr = get_shared_manager()
    try:
        mgr.disable_module(module_name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown module '{module_name}'")
    return JSONResponse({"module": module_name, "enabled": False})
