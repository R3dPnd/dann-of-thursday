# Godot integration roadmap

Goal: Dann gets a `godot` MCP module — direct access to a running Godot 4 editor to create
scenes/nodes/scripts, read errors, and eventually drive playtesting — the same way he
already has `notes`, `gardening`, `bjj`, etc.

## Starting point (checked 2026-09-17)

- No Godot install, no Godot project, on this machine yet. Node.js is already present
  (v26.8.1) — needed either way, since `ui/` already depends on it.
- This does **not** need to be built from scratch. There's a mature open-source ecosystem
  of "Godot MCP" servers as of 2026. Surveyed two:

  | | [mkdevkit/godot-mcp](https://github.com/mkdevkit/godot-mcp) | [3ddelano/gdai-mcp-plugin-godot](https://github.com/3ddelano/gdai-mcp-plugin-godot) |
  |---|---|---|
  | Architecture | Godot EditorPlugin (GDScript) ↔ WebSocket (JSON-RPC 2.0, port 6505) ↔ Node.js MCP server ↔ stdio ↔ MCP client | EditorPlugin + separate MCP server (exact transport undocumented in README — needs hands-on check) |
  | Godot version | 4.4+ | 4.1+ |
  | Server language | Node.js (TypeScript, compiled) | unclear from README |
  | Tool count | 173 tools / 26 categories (scenes, nodes w/ undo-redo, scripts, screenshots, input simulation, runtime inspection, animation/tilemap/shader/physics/3D/audio/nav, Android export) | scene/node/script/property/asset tools, debugger output+errors, screenshots, GDScript context |
  | Launch | `node build/index.js` (stdio) | two-step: install plugin in project + point MCP client at server command |

  **mkdevkit/godot-mcp is the clearer pick to start with** — stdio + a plain `node` command
  is a drop-in fit for how `MCPManager` already launches every other module (see
  `runtime/integrations/client.py`'s `StdioServerParameters`). Worth a hands-on
  side-by-side with GDAI once something's running, since GDAI's tool set (debugger
  output/errors, screenshots) overlaps and either could end up being the better fit —
  don't commit past Phase 1 before comparing.

  Also worth a look once Phase 1 is running: **Godot MCP Pro** (paid, 163 tools,
  Asset Library) — only if the free options are missing something specific.

## Why this is easy to slot in

Because the server is a plain Node process talking stdio, it's just another
`mcp.servers` entry:

```yaml
mcp:
  servers:
    - name: godot
      command: node
      args: ["<path-to-vendored-godot-mcp>/server/build/index.js"]
      description: "Godot editor — scenes, nodes, scripts, errors, screenshots"
      always_on: false
```

No Python wrapper needed, and no `-m integrations.servers.X` module-path problem — that
open question in `CLAUDE.md` ("Adding a persona-specific MCP server") turns out not to
apply here, since this isn't a `-m`-launched Python module at all.

## Phases

### Phase 0 — Prerequisites — done (2026-09-17)
- [x] Install Godot 4.4+. Ended up with **the standard (non-mono) 4.7.2 build**, not the
      Mono/.NET build — the Mono build segfaults on startup (`ERROR: .NET: Assemblies not
      found`) without a .NET SDK installed, and this integration is pure GDScript, so
      Mono buys nothing. `godot` is symlinked at `~/.local/bin/godot` →
      `~/Downloads/Godot.app/Contents/MacOS/Godot` (the mono `.app` and its zips are still
      in `~/Downloads` but unused).
- [x] Test project: `~/Git/r3dpnd/godot-mcp-sandbox/` (hand-written minimal
      `project.godot`, not the real game).

### Phase 1 — Stand up godot-mcp standalone (no Dann yet) — done (2026-09-17)
- [x] Cloned `mkdevkit/godot-mcp` to `~/Git/r3dpnd/godot-mcp/`. Copied
      `addons/godot_mcp/` into the sandbox project and enabled it via `project.godot`'s
      `[editor_plugins]` (no need to click through the Project Settings UI — a plain text
      edit works).
- [x] `cd server && npm install && npm run build` — clean build, `server/build/index.js`
      is the MCP entry point. (`npm install` reported 6 vulnerabilities in transitive
      deps, upstream's problem to fix, not touched here.)
- [x] Full round-trip verified *without Dann in the loop*: ran `node build/index.js`
      directly, opened the sandbox in the Godot editor, confirmed the plugin connected
      over the WebSocket (`[Godot MCP] Connected to MCP server`), then hand-sent raw
      MCP JSON-RPC over the server's stdin —
      - `initialize` → correct handshake response.
      - `tools/call get_project_info` → real project data back (`godot_version`,
        `project_path`, etc.)
      - `tools/call create_script` (`script_path` param — **not** `path**, despite what
        you'd guess) → `{"created": true, ...}`, and the `.gd` file actually appeared on
        disk with the exact content sent.
      This proves the whole chain (AI client → stdio → Node server → WebSocket → editor
      plugin → real file on disk) works, independent of anything Dann-specific.
  - **Reconnect-backoff gotcha**: the plugin's reconnect uses exponential backoff
    (1s → 60s per the README). Killing/restarting the Node server repeatedly in quick
    succession (as happens while iterating on this by hand) pushes the *next* reconnect
    attempt further and further out, making it look like the bridge is broken when it's
    just waiting out backoff. Fix: **restart the Godot editor itself** (resets the
    plugin's backoff state to 1s) rather than repeatedly restarting only the Node server.
    Whatever eventually launches this for real (Phase 2/3) should start the Node server
    *once* per editor session, not on a retry loop.
- [ ] Side-by-side against GDAI's plugin — **skipped for now**. godot-mcp's stdio + plain
      `node` launch is such a clean fit for `MCPManager`, and the standalone test above
      passed cleanly, that there's no concrete reason to compare yet. Revisit only if a
      specific capability turns out to be missing once real Godot work starts (Phase 3+).

### Phase 2 — Wire it into Dann — done (2026-09-17)
- [x] Vendored `mkdevkit/godot-mcp` as a submodule at `godot-mcp/` (repo root, alongside
      `runtime/`). Built it (`cd godot-mcp/server && npm install && npm run build`) — like
      `runtime/`, `node_modules/`/`build/` are gitignored upstream, so this is a per-machine
      setup step, not something committed.
- [x] Added the `godot` entry to `mcp.servers` in **both** `config.yaml` and
      `config.example.yaml` — `command: node`, `args: ["../godot-mcp/server/build/index.js"]`
      (the `../` matters: MCP subprocess cwd is `runtime/`, same reason `command:
      ../.venv/bin/python` needed it for the Python servers).
- [x] Added `godot` to `agents:` and `focus_areas:`, and extended the existing `coding`
      focus area's description to point at `godot` for Godot-specific work — see both
      config files for the actual wording.
- [x] Smoke-tested for real, not just "should work": `dann dev`-equivalent (uvicorn with
      `cwd=runtime/`, `NO_VOICE=1`) + the Phase-1 sandbox open in Godot + Ollama already
      running locally with the `dann-router-qwen3` model. Created a chat work stream on
      the `godot` focus area and sent a plain-English request — Dann's own routing model
      picked the right tool with no hinting, `create_script` ran through the real chain
      (chat → Ollama tool-call → MCPManager → godot-mcp → editor plugin), and
      `player.gd` (`extends CharacterBody2D`, empty `_ready()`) appeared on disk exactly
      as asked, with a correct natural-language reply back in the stream. Cleaned up the
      test file/stream afterward.

**Two real problems found and fixed while wiring this up — not hypothetical, both would
have bitten on first real use:**

1. **173 tools would have flooded the router model's context.** `MCPManager` had no
   concept of a partial tool set — `enable_module` always exposed *everything* the server
   reports. Fine for a 5-tool module, not for a 173-tool one, and the "dann-router" model
   is small/fast specifically because it only ever sees a handful of tool schemas. Added
   an optional `tools:` allowlist to `mcp.servers` entries generically (`runtime/
   integrations/client.py`'s `_connect_one`) — a server config can now list exactly which
   tool names to expose; omitting it keeps every other module's behavior unchanged. Curated
   a 20-tool starter set for `godot` (project/scene/node/script CRUD + error/log reading —
   see `config.example.yaml`) covering "script creation etc." without the other 153
   (input simulation, physics, particles, Android export, stress tests, ...). Confirmed via
   the actual connect log: `[mcp] Connected to 'godot' — 20/173 tool(s) exposed`.
2. **`load_config()`'s default path broke when `voice/`/`app/` moved into `runtime/`.**
   Every call site that omits an explicit path (most of them — `chat_service.py`,
   `claude_code_server.py`, several endpoints) relied on `Path(__file__).resolve()
   .parent.parent / "config.yaml"`, which was correct when this code lived directly in
   `dann-of-thursday` but now resolves to `runtime/config.yaml` — one level too shallow,
   since config.yaml lives in the *persona* repo's root, above `runtime/`. This was a
   silent regression from the Phase-1 extraction that nothing had caught yet (nobody had
   run `dann dev` since the split). Fixed centrally in `voice/config.py`: try a
   `DANN_CONFIG_PATH` env var first, then fall back to one more directory level up.
   Removed the two other places (`app/main.py` ×2, `voice/main.py`) that duplicated (and
   now would have kept getting wrong) the same path computation — they just call
   `load_config()`/`Orchestrator()` with no args now. `dann.py` sets `DANN_CONFIG_PATH`
   explicitly on every subprocess it launches (`cmd_dev`, `cmd_electron`, `cmd_start`)
   rather than leaning on the fallback math. Both fixes landed in `pnd-mcp` (shared runtime,
   since the bug and the fix are both runtime-side), then the `runtime/` submodule pointer
   in `dann-of-thursday` was bumped to pick them up.

### Phase 3 — Point it at a real project — done (2026-09-17)
- [x] No real Godot game existed anywhere on this machine going in — the "sandbox" from
      Phase 1 was always throwaway. Picked `github.com/R3dPnd/pnd-games` (already in the
      `pnd-` naming schema) as the home, cloned it, and found it holds two separate
      "Intro to Godot" tutorial projects, not one game: `intro-to-godot/` (more developed —
      Scripts/Materials/3D/2D folders) and `IntroToGodot/` (an earlier, simpler attempt).
      Installed the plugin in `intro-to-godot/` only.
- [x] Copied `addons/godot_mcp/` into `pnd-games/intro-to-godot/` and enabled it in its
      `project.godot`. Opening the project in this machine's Godot 4.7 (it was authored
      against 4.2) auto-bumped `config/features`, added an `[animation]` compatibility
      section, and generated `.uid` sidecar files for scripts — normal engine-upgrade
      side effects, unrelated to the plugin. There's a pre-existing, unrelated parse error
      in `Scripts/intro-to-scripting.gd` (`Cannot find member "x" in base "float"`) — left
      alone, it's the tutorial's own code, not something this integration touched.
      Enabling the plugin also auto-injected its 3 required autoloads
      (`MCPRuntimeBridge`/`MCPInputBridge`/`MCPScreenshotBridge`) into `project.godot`,
      same as it did in the Phase-1 sandbox.
  - Committed to `pnd-games` (not yet pushed as of writing — check `git log`/`git status`
    there before assuming GitHub has it).
- [x] Re-verified the standalone bridge against this *real* project (not the sandbox):
      opened it in the editor, started the vendored server, confirmed
      `[Godot MCP] Connected to MCP server (ws://127.0.0.1:6505)` again. Didn't re-run the
      full Dann chat-routing test here — Phase 2's test already proved that path works;
      re-running it per-project isn't needed unless something about a specific project
      breaks it.
- [x] **always_on vs. on-demand**: staying on-demand (`always_on: false`, unchanged).
      Matches every other personal module, costs nothing to leave as-is, trivially
      flippable in `config.yaml` later if "turn on Godot" every session gets annoying.
- [x] **Confirm-first policy for destructive tools**: not adding one. The 20-tool
      allowlist from Phase 2 already excludes `delete_scene`, batch operations, and
      export/deploy — the only deletion tool included, `delete_node`, goes through the
      editor's own undo stack (godot-mcp's "UndoRedo integration" feature), so it's
      already reversible with Ctrl+Z. Revisit only if real use finds a tool in the
      allowlist that turns out to be riskier than it looked from the README.

### Phase 4 — Workflow polish
- [ ] Close the edit → error → fix loop: after Dann creates/edits a script, have it pull
      the debugger output/error-reading tool automatically rather than waiting to be told
      something broke.
- [ ] Dashboard: confirm the `godot` module shows up for free via the existing
      `list_modules`/`enable_module` meta-tools and module list UI — no dashboard code
      should need to change for this.

### Phase 5 — Stretch
- [ ] Runtime input simulation / automated playtesting (godot-mcp's tool list already
      covers this) — semi-autonomous "try the change, report what happened" loops.
- [ ] Export/deployment tooling (Android export is already in godot-mcp's tool set).

## Open questions to resolve, not guess at, when reached

- Whether `always_on: true` for `godot` is worth the extra editor-must-be-running
  dependency at Dann startup, once real usage patterns are known (Phase 3).
- Whether GDAI's tool set ends up covering something godot-mcp misses (debugger
  output specifically) — resolve with the Phase 1 side-by-side, not by re-reading READMEs.
