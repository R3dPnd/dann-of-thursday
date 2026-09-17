# Dann of Thursday — Claude Code Context

## What this is

Dann's persona repo: a local voice AI agent (say **"ok Dann"**) being repositioned as a
coding partner and (eventually) a Godot game-dev assistant, rather than a general-purpose
personal assistant.

The actual runtime — voice pipeline, MCP client, generic MCP tool-servers, and the
FastAPI/React dashboard — was extracted into
[pnd-mcp](https://github.com/R3dPnd/pnd-mcp) so it can be reused by other personas.
It lives here as the `runtime/` git submodule. This repo owns only:

- `config.yaml` / `config.example.yaml` — persona identity: the routing/system prompt,
  which MCP modules and focus areas are enabled, model/voice choices.
- `dann.py` — the CLI launcher (`dann dev`/`electron`/`start`/`stop`/`restart`), which now
  points at `runtime/` for the API and UI instead of running them from the repo root.
- `training/` — the "ok Dann" wake-word training clips (gitignored; regenerate via
  `runtime/scripts/wakeword/generate_wakeword_samples.py`). Persona-specific, not moved.
- Whatever persona-specific MCP tool-servers get added going forward (e.g. a future Godot
  server) — see "Adding a persona-specific MCP server" below.

## Directory layout

```
runtime/        Git submodule → pnd-mcp: voice/, integrations/ (MCP client + tool-server
                library, including claude_code_server and devteam_server), shared/, app/,
                ui/, models/, scripts/, deploy/, docs/. See runtime/CLAUDE.md or
                runtime/docs/tech-spec.md for how those pieces fit together.
docs/           design-doc.md, improvements.md — persona-repo-only notes (not moved)
training/       "ok Dann" wake-word training clips (gitignored, machine-local)
dann.py         CLI launcher
config.yaml     Persona config: routing prompt, enabled MCP modules/focus areas, models
```

## Key entry points

| What | Command |
|------|---------|
| Voice agent (full) | `.venv/bin/python -m voice.main` (cwd: `runtime/`) |
| Backend only (no mic) | `NO_VOICE=1 .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000` (cwd: `runtime/`) |
| UI dev server | `cd runtime/ui && npm run dev` (port 3000) |
| All of the above, wired up | `.venv/bin/python dann.py dev` / `dann.py dev --voice` / `dann.py start` |

`dann.py` is the source of truth for exact invocation — it launches uvicorn/npm with
`cwd=runtime/` so `app.main:app` and `npm run dev` resolve inside the submodule.

## Python venv

One venv at the repo root (`.venv/`) covers both this repo and `runtime/` —
`requirements.txt` here is just `-r runtime/requirements.txt` plus `psutil` (used by
`dann stop`/`dann restart`). Always use `.venv/bin/python` / `.venv/bin/pip`.

`config.yaml`'s `mcp.servers[].command` is `../.venv/bin/python` (not `.venv/bin/python`)
— the API/voice process runs with `cwd=runtime/`, so MCP tool-server subprocesses need to
reach back up one level to find this repo's shared venv.

## MCP integration

`runtime/integrations/client.py`'s `MCPManager` connects to whatever's listed under
`mcp.servers` in `config.yaml`. All the current tool-servers (`claude_code_server`,
`devteam_server`, `schedule_server`, `notes_server`, `gardening_server`, `bjj_server`,
`system_server`, `focus_areas_server`) live in `runtime/integrations/servers/` — see
`runtime/CLAUDE.md` for what each one does.

### Adding a persona-specific MCP server

A tool-server specific to this persona (e.g. a Godot MCP server) should NOT go in
`runtime/integrations/servers/` — that package belongs to the shared repo. Instead:

1. Add it under this repo's own `integrations/servers/` (new directory — none exists here
   yet; everything general-purpose moved to `runtime/`).
2. Register it in `config.yaml`'s `mcp.servers` list. Because `MCPManager` subprocesses
   inherit `cwd=runtime/` from the parent API/voice process (see `dann.py`), the module
   path won't resolve as `-m integrations.servers.godot_server` the way the runtime's own
   servers do — either invoke it with an absolute script path, or set `PYTHONPATH` to
   include this repo's root in the server's `env:` block. Not yet solved generically since
   there's no persona-specific server yet; solve it for real when the first one is added.

## Config

`config.yaml` (from `config.example.yaml`) controls wake word model path, Picovoice access
key, Ollama model name, Piper voice model path, audio device, the `agents:` routing list,
`focus_areas:`, and the `mcp.servers` list. Model/asset paths (e.g. `models/ok_dann.onnx`)
are relative to `runtime/` since that's the process's cwd — they resolve into
`runtime/models/`.
