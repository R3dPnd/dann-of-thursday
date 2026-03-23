"""
FastAPI application entry point with OpenAPI documentation.

Startup behaviour
-----------------
By default the Dann orchestrator is started in a daemon thread so the voice
pipeline and the API share the same process and the same EventBus instance.

Set the environment variable NO_VOICE=1 (or pass --no-voice via the dann CLI)
to skip launching the orchestrator. This is useful when developing the UI
without audio hardware attached.
"""

import os
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import settings

# ── Orchestrator (optional) ───────────────────────────────────────────────────
# Module-level reference so endpoint handlers can call orchestrator.snapshot()
orchestrator = None


def _start_orchestrator() -> None:
    """Initialise and run the voice orchestrator. Runs in a daemon thread."""
    global orchestrator
    try:
        from src.orchestrator import Orchestrator
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"
        orchestrator = Orchestrator(config_path)
        orchestrator.run()
    except Exception as exc:
        from src.event_bus import bus
        import traceback
        bus.emit("error", {
            "module": "orchestrator",
            "message": f"Orchestrator failed to start: {exc}",
            "traceback": traceback.format_exc(),
        })


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Dann of Thursday — voice AI agent + dashboard API",
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "health",    "description": "Health check endpoints"},
        {"name": "state",     "description": "Live orchestrator state"},
        {"name": "projects",  "description": "Discovered Git projects"},
        {"name": "events",    "description": "Real-time WebSocket event stream"},
        {"name": "mcp",       "description": "Model Context Protocol endpoints"},
        {"name": "tools",     "description": "Tool execution endpoints"},
        {"name": "metrics",   "description": "Aggregated turn/latency metrics"},
        {"name": "logs",      "description": "Filterable log entries"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_STR)


@app.on_event("startup")
async def startup_event() -> None:
    # Wire persistence services to the EventBus (always, regardless of voice mode).
    from src.event_bus import bus
    from app.services import metrics_service, log_service
    bus.subscribe(metrics_service.record_metric)
    bus.subscribe(log_service.record_event)

    no_voice = os.environ.get("NO_VOICE", "0").strip() not in ("", "0", "false", "no")
    if not no_voice:
        t = threading.Thread(target=_start_orchestrator, daemon=True, name="orchestrator")
        t.start()


@app.get("/", tags=["health"])
async def root():
    return {
        "message": "Dann of Thursday API",
        "version": settings.VERSION,
        "docs": "/docs",
    }


@app.get("/health", tags=["health"])
async def health_check():
    return {"status": "healthy", "service": settings.PROJECT_NAME}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,  # reload=True is incompatible with the orchestrator thread
        log_level="info" if not settings.DEBUG else "debug",
    )
