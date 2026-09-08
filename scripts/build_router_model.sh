#!/usr/bin/env bash
# Builds (or rebuilds) Dann's custom routing model in Ollama.
# Run this after editing models/dann-router.Modelfile, or any time you pull
# a newer qwen2.5:7b-instruct base.
set -euo pipefail
cd "$(dirname "$0")/.."

ollama create dann-router -f models/dann-router.Modelfile
echo "Built 'dann-router'. Set ollama.model: dann-router in config.yaml to use it."
