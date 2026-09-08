"""
System control endpoint — dashboard-triggered restart.

POST /api/v1/system/restart
"""
import threading
import time

from fastapi import APIRouter

from shared.restart import restart_process

router = APIRouter()


def _do_restart() -> None:
    time.sleep(0.5)  # let the HTTP response flush before the process image is replaced

    from app.main import orchestrator
    if orchestrator is not None:
        orchestrator.stop()  # releases the wake-word detector + stops the MCP manager
    else:
        from integrations.client import get_shared_manager
        mgr = get_shared_manager()
        if mgr.started:
            mgr.stop()

    restart_process()


@router.post("/restart", summary="Restart the Dann process")
async def restart() -> dict:
    threading.Thread(target=_do_restart, daemon=False).start()
    return {"status": "restarting"}
