#!/usr/bin/env python3
"""
Simple script to run the FastAPI application
"""
import uvicorn
from app.core.config import settings

if __name__ == "__main__":
    # reload=True spawns a subprocess via multiprocessing which does not inherit
    # the virtualenv sys.path, and is incompatible with the orchestrator thread.
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
        log_level="info" if not settings.DEBUG else "debug",
    )

