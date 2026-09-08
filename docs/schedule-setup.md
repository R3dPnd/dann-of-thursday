# Schedule Module Setup (Google Calendar)

The `schedule` MCP module (`src/mcp_servers/schedule_server.py`) talks to
Google Calendar. It's an **on-demand module** (`always_on: false` in
`config.yaml`) — nothing about it runs until Dann's LLM calls
`enable_module("schedule")`, and it won't work at all until you complete
this one-time setup.

## 1. Create a Google Cloud project

- Go to https://console.cloud.google.com/ and create a new project (or
  reuse an existing personal one).

## 2. Enable the Calendar API

- In the project, go to **APIs & Services → Library**, search for
  "Google Calendar API", and enable it.

## 3. Create an OAuth client ID

- Go to **APIs & Services → Credentials → Create Credentials → OAuth
  client ID**.
- If prompted, configure the OAuth consent screen first — choose
  **External**, fill in the required fields, and add yourself as a test
  user (this keeps it out of Google's review process since only you use
  it).
- Application type: **Desktop app**.
- Download the resulting JSON.

## 4. Install it

```bash
mkdir -p ~/.dann
mv ~/Downloads/client_secret_*.json ~/.dann/google_credentials.json
```

## 5. First run

The first time any `schedule` tool actually calls the Calendar API, it
opens a browser window for you to log in and grant access. After that,
the resulting token is cached at `~/.dann/google_token.json` and reused
(refreshed automatically) — you won't be prompted again unless you revoke
access or delete that file.

Neither `~/.dann/google_credentials.json` nor `~/.dann/google_token.json`
live in this repo — they're personal, per-machine secrets, kept entirely
outside git (see `src/mcp_servers/_store.py`).

## Enabling the module

Once set up, say something that needs the calendar (e.g. "what's on my
schedule today") — Dann's routing model should call `enable_module`
automatically. You can also ask "what modules are available" to confirm
`schedule` shows up.
