"""Load and validate configuration."""

import json
from pathlib import Path
from typing import Any


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load config from JSON file. Uses config.json in repo root if path not given."""
    if path is None:
        path = Path(__file__).resolve().parent.parent / "config.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Config not found: {path}. Copy config.example.json to config.json and edit."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)
