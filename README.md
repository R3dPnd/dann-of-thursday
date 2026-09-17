# Dann of Thursday

![img](https://images6.fanpop.com/image/photos/43800000/Dann-of-Thursday-gun-x-sword-43866941-720-480.jpg)

Voice AI agent: say **"ok Dann"** to ask questions. Being repositioned from a general
personal assistant into a **coding partner**, and eventually a **Godot game-dev
assistant**.

## What this is

This repo holds Dann's *persona*: the routing/system prompt, which tools are enabled, and
(going forward) any persona-specific MCP tool-servers — a Godot server, for instance.

The actual runtime — wake word → STT → LLM routing → TTS, the MCP client and its library of
tool-servers (including project discovery and launching Claude Code sessions), and the
FastAPI + React dashboard — lives in
[pnd-mcp](https://github.com/R3dPnd/pnd-mcp), pulled in here as the `runtime/` git
submodule. That split happened so the runtime can be reused by other assistant personas
without forking this repo. For the full architecture (module map, sequence diagrams, code
tour), see [runtime's README](https://github.com/R3dPnd/pnd-mcp#readme) and
`runtime/docs/tech-spec.md`.

## Setup (macOS)

```bash
git clone --recurse-submodules https://github.com/R3dPnd/dann-of-thursday.git
cd dann-of-thursday
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # pulls in runtime/requirements.txt too
cp config.example.yaml config.yaml
```

(Cloned without `--recurse-submodules`? Run `git submodule update --init` to fetch
`runtime/`.)

Then follow `runtime/scripts/setup.sh` for the wake word model, Ollama, and Piper TTS setup
— it's the same guided setup pnd-mcp documents, just run from inside `runtime/`.

Edit `config.yaml` for your paths, model choices, and which MCP modules/focus areas are
enabled.

## Run

```bash
source .venv/bin/activate
python dann.py dev          # API + UI, no mic (NO_VOICE=1), opens a browser tab
python dann.py dev --voice  # same, with the voice pipeline, opens a fullscreen Electron window
python dann.py start        # API only, with voice pipeline enabled
python dann.py stop         # stop everything dann.py started (including orphans)
python dann.py restart      # stop + relaunch with the same mode/flags
```

`dann.py` runs the API/UI out of `runtime/` (the submodule) rather than this repo's root —
see `CLAUDE.md` for exactly how that's wired.

## Updating the runtime

```bash
cd runtime && git pull origin main && cd ..
git add runtime && git commit -m "Bump runtime (pnd-mcp) to latest"
```

This repo pins a specific commit of `pnd-mcp` (like any git submodule) — bump it
deliberately, not automatically, since a runtime change could affect every persona built on
it.
