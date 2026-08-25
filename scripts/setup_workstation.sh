#!/usr/bin/env bash
# Bootstraps the extra tools needed to run this machine as a remotely
# accessible AI workstation (see README.md "Local AI Workstation").
#
# This only installs software and prints next steps — it does NOT create
# a Cloudflare tunnel, does NOT touch launchd, and does NOT start anything.
# Review it before running: bash scripts/setup_workstation.sh
set -euo pipefail

command -v brew >/dev/null || { echo "Homebrew is required: https://brew.sh"; exit 1; }

echo "==> Installing cloudflared (tunnel client)"
brew install cloudflared

echo "==> Installing code-server (VS Code in the browser)"
brew install code-server

echo "==> Installing filebrowser (web file manager)"
brew install filebrowser

echo "==> Installing JupyterLab into the project venv"
if [ ! -d .venv ]; then
  echo "No .venv found — create one first: python3 -m venv .venv && pip install -r requirements.txt"
  exit 1
fi
.venv/bin/pip install jupyterlab

cat <<'EOF'

Installed: cloudflared, code-server, filebrowser, jupyterlab.

Next steps (manual — see README.md "Local AI Workstation" for detail):
  1. cloudflared tunnel login
  2. cloudflared tunnel create dann-workstation
  3. cp deploy/cloudflared/config.yml.example deploy/cloudflared/config.yml
     and fill in your tunnel id + hostnames
  4. cloudflared tunnel route dns dann-workstation <each hostname>
  5. In the Cloudflare Zero Trust dashboard, create an Access application
     + OAuth login policy for each hostname
  6. Install the launchd services in deploy/launchd/ so the backend and
     the tunnel survive reboots
  7. Start code-server / jupyter lab / filebrowser on the ports referenced
     in deploy/cloudflared/config.yml, e.g.:
       code-server --bind-addr 127.0.0.1:8080 --auth none
       .venv/bin/jupyter lab --ip=127.0.0.1 --port=8888 --no-browser
       filebrowser -a 127.0.0.1 -p 8081 -r ~
     (bind to 127.0.0.1, not 0.0.0.0 — Access is the only auth these tools
     get, so they must not be reachable except through the tunnel)
EOF
