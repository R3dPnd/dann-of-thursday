# Dann of Thursday — UI Specification

**Status:** Draft
**Version:** 0.1
**Last Updated:** 2026-03-20

---

## 1. Purpose

A local web dashboard that surfaces the live state of the Dann voice agent,
making it easier to understand what it is doing, debug problems, and track
work across code sessions. The UI is a read-mostly observer — it does not
drive the voice pipeline, it watches it.

---

## 2. High-Level Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Browser  (React + Tailwind)                             │
│  ┌──────────────┬─────────────────┬────────────────────┐ │
│  │ Status Bar   │ Project Panel   │  Session Panel     │ │
│  ├──────────────┴─────────────────┴────────────────────┤ │
│  │ Metrics Bar                                         │ │
│  ├──────────────────────────────────────────────────────┤ │
│  │ Logs / Error Panel                                  │ │
│  └──────────────────────────────────────────────────────┘ │
└────────────────────────┬─────────────────────────────────┘
                         │ WebSocket  /  REST
┌────────────────────────▼─────────────────────────────────┐
│  FastAPI  (app/)  — existing backend, extended           │
│  New:  /api/v1/state  /api/v1/events (WS)                │
│        /api/v1/metrics  /api/v1/logs  /api/v1/tasks      │
└────────────────────────┬─────────────────────────────────┘
                         │  in-process / IPC
┌────────────────────────▼─────────────────────────────────┐
│  src/orchestrator.py  (Dann voice loop)                  │
│  Emits structured events via EventBus                    │
└──────────────────────────────────────────────────────────┘
```

The UI is served at `http://localhost:3000` (Vite dev server) or as a static
build served from the FastAPI root. The FastAPI server runs at
`http://localhost:8000` — already configured for CORS on port 3000.

---

## 3. Feature Specifications

### 3.1 Status Bar — Mode Indicator

A persistent top bar showing the current `SessionMode` of the orchestrator.

**States and colours:**

| State | Label | Visual |
|---|---|---|
| Idle (waiting for wake word) | **Idle** | Grey pill |
| Session active — NORMAL mode | **Listening** | Green pulsing dot |
| Session active — CODE mode | **Code: \<project\>** | Blue pill with project name |
| Error / crashed | **Error** | Red pill |

**Behaviour:**
- Updates within 500 ms of a state change via WebSocket.
- Clicking the state pill in CODE mode opens the Project Panel for that project.
- The current wake phrase ("ok Dann") is shown in small text next to the state.

---

### 3.2 Project Panel

A scrollable list of all Git projects discovered by the MCP server
(`list_projects`), displayed as cards.

**Each project card shows:**
- Project name and path (truncated, full path on hover)
- Last analysed timestamp (derived from session history)
- Active indicator: highlighted border when Dann is currently in CODE mode
  for that project
- "Open in Terminal" button — calls `open_claude_code` MCP tool
- Expand toggle to show the Terminal Output panel for this project

**Behaviour:**
- Project list is fetched at startup via `GET /api/v1/projects` and refreshed
  every 60 s.
- The active project card scrolls into view automatically when CODE mode
  starts.

---

### 3.3 Terminal Output Panel

For each active CODE-mode session, a scrollable terminal-style pane showing
the prompts sent to `ask_claude_code` and the responses returned.

**Each entry contains:**
- Timestamp
- The user's voice input (the task string fed to `ask_claude_code`)
- Claude Code's response (full, not truncated)
- Status badge: `ok` / `error` / `empty`

**Behaviour:**
- Entries are streamed in real-time via WebSocket.
- The pane auto-scrolls to the latest entry.
- A "Copy" button copies the full response to the clipboard.
- A persistent history for each project is kept for the current browser
  session; older sessions are shown if the orchestrator log file is available.
- NORMAL-mode Ollama turns are also shown in a separate "Conversation" panel
  with the same format, labelled with the model name.

---

### 3.4 Usage Metrics

A horizontal metrics bar below the project panel and a dedicated Metrics page.

**Counters (current session and all-time):**

| Metric | Description |
|---|---|
| Wake events | Number of times the wake word was detected |
| Session duration | Wall time of each session, average across sessions |
| Turns per session | Number of voice turns per session |
| STT calls | Transcriptions performed, blank-vs-recognised breakdown |
| LLM calls | Ollama requests, avg latency, token count where available |
| Code mode entries | How many times CODE mode was activated |
| `ask_claude_code` calls | Count, success/error/empty breakdown, avg latency |
| TTS renders | Count and avg synthesis time |

**Charts (Metrics page):**
- Timeline of sessions over the past 7 days (bar chart)
- Latency breakdown per turn: record → STT → LLM/MCP → TTS → speak (stacked bar)
- Mode distribution pie: % of turns in NORMAL vs CODE

**Persistence:** Metrics are written to `~/.dann/metrics.jsonl` (one record per
turn) by the orchestrator. The API reads this file; no external database is
required initially.

---

### 3.5 Error Tracking and Logging

A filterable log panel at the bottom of the dashboard.

**Log levels captured:**
- `DEBUG` — verbose pipeline steps (gated behind a toggle)
- `INFO` — normal flow events (wake word, session start/end, mode changes)
- `WARNING` — recoverable issues (blank STT result, JSON artifact suppressed)
- `ERROR` — failures that produced a fallback response
- `CRITICAL` — crashes that ended the session unexpectedly

**Each log entry shows:**
- Timestamp (relative "2s ago" with absolute on hover)
- Level badge (colour-coded)
- Module (orchestrator / stt / tts / llm / mcp)
- Message
- Optional expandable detail block (stack trace for errors)

**Filters:**
- Level minimum slider (DEBUG → CRITICAL)
- Module checkboxes
- Free-text search
- "Errors only" quick filter

**Behaviour:**
- Log entries stream via WebSocket from the orchestrator.
- Up to 2 000 entries are kept in memory; older entries are discarded unless
  written to `~/.dann/dann.log`.
- A "Download log" button exports the current filtered view as plain text.
- Error entries trigger a brief badge count on the browser tab favicon.

---

### 3.6 Task Tracking

A lightweight task board scoped to CODE-mode sessions.

**Anatomy of a task:**
- Created automatically when `ask_claude_code` returns a response that
  contains actionable text (heuristic: response includes keywords like
  "you should", "next step", "TODO", "implement", "fix").
- Can also be created manually by the user in the UI.

**Columns:** `Suggested` → `In Progress` → `Done`

**Each task card:**
- Project name
- Extracted or typed task text
- Source turn (timestamp + voice transcript)
- Status (can be dragged between columns)
- "Ask Dann" button that pre-populates a prompt the user can speak

**Persistence:** Tasks are stored in `~/.dann/tasks.json` and loaded on
start-up. The API exposes `GET/POST/PATCH /api/v1/tasks`.

---

## 4. Backend Extensions Required

The existing `app/` FastAPI backend needs the following additions.

### 4.1 EventBus

A lightweight in-process pub/sub that the orchestrator writes to and the API
reads from.

```python
# src/event_bus.py
class EventBus:
    def emit(self, event_type: str, payload: dict) -> None: ...
    def subscribe(self, callback: Callable) -> None: ...
```

The orchestrator calls `bus.emit(...)` at each significant state transition
(see §4.2). The FastAPI WebSocket endpoint subscribes and forwards events to
connected browser clients.

### 4.2 Orchestrator Events

| Event type | When emitted | Key payload fields |
|---|---|---|
| `state.changed` | Mode or project changes | `mode`, `project` |
| `session.start` | Wake word detected | `session_id` |
| `session.end` | Goodbye or error | `session_id`, `reason` |
| `turn.start` | Recording begins | `session_id` |
| `turn.stt` | Transcription complete | `text`, `blank` |
| `turn.llm` | Ollama response ready | `text`, `latency_ms` |
| `turn.code` | `ask_claude_code` response | `project`, `task`, `response`, `status`, `latency_ms` |
| `turn.tts` | Speech synthesis complete | `latency_ms` |
| `error` | Any caught exception | `module`, `message`, `traceback` |
| `metric` | End of each turn | full latency breakdown dict |

### 4.3 New API Endpoints

```
GET  /api/v1/state          Current orchestrator state snapshot
GET  /api/v1/projects       List of discovered git projects
GET  /api/v1/metrics        Aggregated metrics (period=session|day|week|all)
GET  /api/v1/logs           Paginated log entries (level, module, search filters)
GET  /api/v1/tasks          All tasks
POST /api/v1/tasks          Create a task
PATCH /api/v1/tasks/{id}    Update status or text
WS   /api/v1/events         Real-time event stream
```

---

## 5. Frontend Technology Choices

| Concern | Choice | Rationale |
|---|---|---|
| Framework | React 18 | Familiarity; good WebSocket / streaming support |
| Build tool | Vite | Fast HMR, minimal config |
| Styling | Tailwind CSS | Utility-first, dark theme easy to configure |
| Component library | shadcn/ui (Radix primitives) | Accessible, unstyled base, composable |
| State management | Zustand | Lightweight, no boilerplate |
| WebSocket | native browser `WebSocket` via custom hook | No extra library needed |
| Charts | Recharts | React-native, easy to theme |
| Terminal pane | `xterm.js` or custom `<pre>` scroll | xterm if interactive terminal ever needed |

The UI lives in `ui/` at the repo root. `npm run build` outputs to `ui/dist/`,
which FastAPI can serve as static files from `/`.

---

## 6. Data Flow — Sequence Diagram

```
Browser          FastAPI           EventBus         Orchestrator
  |                 |                  |                  |
  |-- GET /state -->|                  |                  |
  |<-- snapshot ----|                  |                  |
  |                 |                  |                  |
  |-- WS /events -->|                  |                  |
  |                 |<-- subscribe() --|                  |
  |                 |                  |                  |
  |                 |                  |<-- bus.emit() ---|  wake word
  |                 |<-- event --------|                  |
  |<-- WS msg ------|                  |                  |
  |  (state.changed: LISTENING)        |                  |
  |                 |                  |                  |
  |                 |                  |<-- bus.emit() ---|  turn.code
  |                 |<-- event --------|                  |
  |<-- WS msg ------|                  |                  |
  |  (turn.code: project, response)    |                  |
```

---

## 7. File / Directory Layout

```
dann-of-thursday/
├── src/
│   ├── event_bus.py            # new — EventBus singleton
│   └── orchestrator.py         # extended to call bus.emit()
│
├── app/
│   ├── api/v1/endpoints/
│   │   ├── state.py            # new
│   │   ├── projects.py         # new
│   │   ├── metrics.py          # new
│   │   ├── logs.py             # new
│   │   ├── tasks.py            # new
│   │   └── events.py           # new — WebSocket endpoint
│   └── services/
│       ├── metrics_service.py  # new — reads ~/.dann/metrics.jsonl
│       ├── log_service.py      # new — reads ~/.dann/dann.log
│       └── task_service.py     # new — reads/writes ~/.dann/tasks.json
│
└── ui/                         # new
    ├── index.html
    ├── vite.config.ts
    ├── tailwind.config.ts
    ├── src/
    │   ├── main.tsx
    │   ├── App.tsx
    │   ├── components/
    │   │   ├── StatusBar.tsx
    │   │   ├── ProjectPanel.tsx
    │   │   ├── TerminalOutput.tsx
    │   │   ├── MetricsBar.tsx
    │   │   ├── MetricsPage.tsx
    │   │   ├── LogPanel.tsx
    │   │   └── TaskBoard.tsx
    │   ├── hooks/
    │   │   ├── useDannEvents.ts    # WebSocket hook
    │   │   └── useDannState.ts     # Zustand store + REST fetch
    │   └── lib/
    │       └── api.ts              # fetch wrappers
    └── package.json
```

---

## 8. Implementation Phases

---

### Phase 1 — Backend foundation

**Goal:** Orchestrator emits structured events; API can stream them to a client.
Nothing visual yet — this phase is complete when `websocat` or a browser console
can connect to `ws://localhost:8000/api/v1/events` and see live events while
Dann is running.

#### 1.1 EventBus

- [x] Create `src/event_bus.py` — `EventBus` singleton with `emit()` and `subscribe()` methods
- [x] Write unit tests for EventBus pub/sub behaviour in `tests/test_event_bus.py`

#### 1.2 Orchestrator instrumentation

- [x] Import and call `bus.emit("session.start", {...})` on wake word detection
- [x] Emit `state.changed` when `_mode` or `_code_project` changes
- [x] Emit `turn.stt` after transcription (include `text`, `blank` flag)
- [x] Emit `turn.llm` after Ollama response (include `text`, `latency_ms`)
- [x] Emit `turn.code` after `ask_claude_code` returns (include `project`, `task`, `response`, `status`, `latency_ms`)
- [x] Emit `turn.tts` after synthesis completes (include `latency_ms`)
- [x] Emit `session.end` on goodbye or session exception (include `reason`)
- [x] Emit `error` in all existing `except` blocks (include `module`, `message`, `traceback`)
- [x] Emit `metric` at the end of each `_run_turn()` with the full latency breakdown dict

#### 1.3 API — state and projects endpoints

- [x] Create `app/api/v1/endpoints/state.py` — `GET /api/v1/state` returns current mode, project, session_id, uptime
- [x] Create `app/api/v1/endpoints/projects.py` — `GET /api/v1/projects` proxies to `_find_projects()` in the MCP server module
- [x] Register both routers in `app/api/v1/router.py`

#### 1.4 API — WebSocket event stream

- [x] Create `app/api/v1/endpoints/events.py` — `WS /api/v1/events`
- [x] On connection, subscribe to EventBus and forward each event as a JSON frame
- [x] On disconnect, unsubscribe cleanly
- [x] Register router in `app/api/v1/router.py`

#### 1.5 Process model

- [x] Run orchestrator in a background thread from `app/main.py` so both share the same EventBus instance (resolve open question §9 item 1)
- [x] Add `--no-voice` flag to FastAPI startup to skip launching the orchestrator (useful for UI development without hardware)
- [x] Update `dann api` CLI command to start the combined process

---

### Phase 2 — Core UI: scaffold + status + projects

**Goal:** Browser shows the live mode indicator and project list. This is the
visible proof that the backend pipeline works end-to-end.

#### 2.1 Project scaffold

- [x] Initialise `ui/` with Vite + React + TypeScript template
- [x] Install and configure Tailwind CSS
- [x] Install Zustand for state management
- [x] Install Recharts (ready for Phase 3, no components yet)
- [x] Add `vite.config.ts` proxy rule so `/api` forwards to `localhost:8000` (includes WS)
- [x] Add `ui` section to `.gitignore` for `node_modules` and `dist`
- [x] Add `dann ui` command to CLI helper (runs `npm run dev`)

#### 2.2 WebSocket hook and global store

- [x] Create `ui/src/hooks/useDannEvents.ts` — connects to `WS /api/v1/events`, reconnects with exponential backoff
- [x] Create `ui/src/hooks/useDannState.ts` — Zustand store; seed from `GET /api/v1/state` on mount, update from WebSocket events, refresh projects every 60 s
- [x] Create `ui/src/lib/api.ts` — typed fetch wrappers for state, projects, and open-project endpoints

#### 2.3 StatusBar component

- [x] Create `ui/src/components/StatusBar.tsx`
- [x] Show mode pill: Idle (grey) / Listening (green pulse dot) / Code: \<project\> (blue) /
- [x] Show WebSocket connection indicator
- [x] Clicking the CODE pill scrolls the active project card into view

#### 2.4 ProjectPanel component

- [x] Create `ui/src/components/ProjectPanel.tsx`
- [x] Fetch project list from `GET /api/v1/projects` on mount and every 60 s
- [x] Render one card per project: name, path, turn count, last-turn time
- [x] Highlight active project card when `state.project` matches (blue border + pulse dot)
- [x] "Open" button calls `POST /api/v1/projects/{name}/open`
- [x] Expand/collapse toggle per card (collapsed by default)

#### 2.5 TerminalOutput component

- [x] Create `ui/src/components/TerminalOutput.tsx`
- [x] Rendered inside the expanded project card
- [x] Displays `turn.code` events for that project as timestamped entries
- [x] Each entry shows: relative timestamp, task text, response, status badge, latency
- [x] Auto-scrolls to latest entry
- [x] "Copy" button per response entry (hover to reveal)

---

### Phase 3 — Metrics and logging

**Goal:** Every turn is measured and surfaced. Errors are visible without
reading raw terminal output.

#### 3.1 Metrics persistence

- [x] Create `app/services/metrics_service.py` — appends each `metric` event to `~/.dann/metrics.jsonl`
- [x] Wire MetricsService into the EventBus subscriber in `app/main.py`
- [x] Define the metric record schema (session_id, timestamp, stt_ms, llm_ms, tts_ms, mode, blank, status)

#### 3.2 Metrics API

- [x] Create `app/api/v1/endpoints/metrics.py`
- [x] `GET /api/v1/metrics?period=session|day|week|all` — returns aggregated counts and averages
- [x] Register router

#### 3.3 MetricsBar component

- [x] Create `ui/src/components/MetricsBar.tsx`
- [x] Horizontal strip below the project panel showing: wake events, sessions today, avg turn latency, code mode entries
- [x] Values update on each `metric` WebSocket event without a page reload

#### 3.4 MetricsPage

- [x] Create `ui/src/components/MetricsPage.tsx` (accessible via a "Metrics" nav link)
- [x] Sessions-per-day bar chart (Recharts `BarChart`, last 7 days)
- [x] Latency breakdown stacked bar per session (STT / LLM / TTS segments)
- [x] Mode distribution pie chart (NORMAL vs CODE turns)
- [x] Summary table: total sessions, total turns, total wake events, avg session length

#### 3.5 Log persistence

- [x] Create `app/services/log_service.py` — writes `error` and `warning` EventBus events to `~/.dann/dann.log`
- [x] Wire LogService into the EventBus subscriber

#### 3.6 Logs API

- [x] Create `app/api/v1/endpoints/logs.py`
- [x] `GET /api/v1/logs?level=INFO&module=orchestrator&search=text&limit=200&offset=0`
- [x] Register router

#### 3.7 LogPanel component

- [x] Create `ui/src/components/LogPanel.tsx`
- [x] Collapsible panel fixed to the bottom of the dashboard
- [x] Streams `error` and `warning` events in real-time via WebSocket
- [x] Filter controls: level slider, module checkboxes, free-text search input, "Errors only" toggle
- [x] Each entry: relative timestamp (absolute on hover), level badge, module tag, message, expandable stack trace
- [x] "Download log" button exports current filtered view as `.txt`
- [x] Error count badge on the panel header tab; clears when opened

---

### Phase 4 — Task tracking

**Goal:** CODE-mode responses that imply follow-up work are captured as tasks
the user can manage.

#### 4.1 Task service

- [ ] Create `app/services/task_service.py` — reads/writes `~/.dann/tasks.json`
- [ ] Implement task extraction heuristic: scan `turn.code` responses for keywords ("you should", "next step", "TODO", "implement", "fix", "consider")
- [ ] Auto-create a `Suggested` task when a keyword match is found; include source turn metadata

#### 4.2 Tasks API

- [ ] Create `app/api/v1/endpoints/tasks.py`
- [ ] `GET /api/v1/tasks` — all tasks, optionally filtered by project or status
- [ ] `POST /api/v1/tasks` — create a task manually
- [ ] `PATCH /api/v1/tasks/{id}` — update status (`suggested`, `in_progress`, `done`) or text
- [ ] `DELETE /api/v1/tasks/{id}` — remove a task
- [ ] Register router

#### 4.3 TaskBoard component

- [ ] Create `ui/src/components/TaskBoard.tsx` (accessible via a "Tasks" nav link)
- [ ] Three columns: Suggested / In Progress / Done
- [ ] Drag-and-drop between columns (use `@dnd-kit/core`)
- [ ] Each card: project name, task text, source turn timestamp and transcript snippet
- [ ] "Ask Dann" button copies a suggested follow-up prompt to the clipboard
- [ ] Manual task creation form at the top of the Suggested column
- [ ] New tasks appear in real-time via `task.created` WebSocket event (add to EventBus)

---

### Phase 5 — Polish and integration

**Goal:** Production-quality feel; single command to run everything.

#### 5.1 Layout and theming

- [ ] Implement dark theme as default (Tailwind `dark` class strategy)
- [ ] Responsive two-column layout: project panel left, terminal output right
- [ ] Navigation bar with links: Dashboard / Metrics / Tasks / Logs
- [ ] Empty states for all panels (e.g. "No projects found", "No errors today")
- [ ] Loading skeletons while API data is fetching

#### 5.2 Browser tab enhancements

- [ ] Dynamic `<title>` updates to show current mode ("Dann — Code: dev-diary")
- [ ] Error badge on favicon using Canvas API when unread errors exist

#### 5.3 Static build integration

- [ ] Add `npm run build` step output to `ui/dist/`
- [ ] Mount `ui/dist/` as a StaticFiles route in `app/main.py` at `/`
- [ ] Ensure API routes take priority over static file catchall

#### 5.4 CLI integration

- [ ] Update `dann api` to run `npm run build` if `ui/dist/` is stale before starting uvicorn
- [ ] Add `dann ui` shortcut that runs `npm run dev` (Vite HMR for development)
- [ ] Document in README: `dann` (voice only), `dann api` (voice + dashboard), `dann ui` (frontend dev mode)

---

## 9. Open Questions

1. **Process model**: Should the orchestrator and FastAPI run as one process
   (simplest — shared EventBus object) or as separate processes communicating
   over a Unix socket or Redis pub/sub? Single process is simpler but means
   the API server blocks the GIL during heavy STT/TTS work.

2. **Task extraction**: The heuristic for auto-creating tasks from Claude Code
   responses needs tuning. An alternative is to ask the user to confirm via
   the UI rather than auto-create.

3. **Authentication**: The API currently uses a static `SECRET_KEY`. For a
   local-only dashboard this is probably fine, but should be documented
   clearly so users don't accidentally expose it.

4. **Historical sessions**: Should the terminal output panel show sessions
   from previous runs (requires persisting turn events to disk), or only the
   current run?
