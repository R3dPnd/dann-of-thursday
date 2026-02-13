"""
Dann of Thursday - Voice AI Agent

Wake word "ok Dann" -> listen -> STT -> Ollama -> Piper TTS -> speak.

Prereqs (macOS):
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Setup:
    cp config.example.yaml config.yaml
    # Edit config.yaml: 
    #   - Set wake_word.access_key (get from https://console.picovoice.ai/)
    #   - Set wake_word.model_path to models/ok_dann.ppn
    #   - Ollama model, Piper voice path
    # Install Piper: https://github.com/rhasspy/piper/releases
    # Create wake word model via Picovoice Console (see wake-word.md)

Run:
    python -m src.main
"""

import sys
from pathlib import Path

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.orchestrator import Orchestrator


def main() -> None:
    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    orch = Orchestrator(config_path)
    orch.run()


if __name__ == "__main__":
    main()
