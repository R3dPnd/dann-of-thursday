"""Load and validate configuration."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load config from YAML file. Uses config.yaml in repo root if path not given."""
    if path is None:
        path = Path(__file__).resolve().parent.parent / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Config not found: {path}. Copy config.example.yaml to config.yaml and edit."
        )
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
