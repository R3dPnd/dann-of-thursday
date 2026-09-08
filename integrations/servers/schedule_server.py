#!/usr/bin/env python3
"""MCP server for Google Calendar — Dann's schedule module.

One-time setup required before this module works — see docs/schedule-setup.md:
  1. Create a Google Cloud project, enable the Calendar API.
  2. Create an OAuth client ID (type: Desktop app), download the JSON.
  3. Save it as ~/.dann/google_credentials.json.
  4. The first tool call opens a browser for one-time consent; the
     resulting token is cached at ~/.dann/google_token.json so you're not
     re-prompted after that.

Exposed tools:
  list_events   — upcoming events in a date range
  create_event  — create a calendar event
  find_event    — search upcoming events by title (to get an id for delete)
  delete_event  — delete an event by id
"""
import datetime as dt

from mcp.server.fastmcp import FastMCP

from integrations.servers._store import dann_home

mcp = FastMCP("schedule")

_CREDENTIALS_FILE = dann_home() / "google_credentials.json"
_TOKEN_FILE = dann_home() / "google_token.json"
_SCOPES = ["https://www.googleapis.com/auth/calendar"]

_service = None


def _get_service():
    """Lazily build (and cache) an authenticated Calendar API client."""
    global _service
    if _service is not None:
        return _service

    if not _CREDENTIALS_FILE.exists():
        raise RuntimeError(
            f"No Google OAuth credentials at {_CREDENTIALS_FILE}. "
            "See docs/schedule-setup.md to set up the schedule module."
        )

    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds = None
    if _TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(_TOKEN_FILE), _SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(_CREDENTIALS_FILE), _SCOPES)
            creds = flow.run_local_server(port=0)
        _TOKEN_FILE.write_text(creds.to_json())

    _service = build("calendar", "v3", credentials=creds)
    return _service


@mcp.tool()
def list_events(days_ahead: int = 7, calendar_id: str = "primary") -> str:
    """List upcoming calendar events.

    Args:
        days_ahead: How many days ahead to look.
        calendar_id: Calendar to query (default: primary).
    """
    try:
        service = _get_service()
    except RuntimeError as exc:
        return str(exc)

    now = dt.datetime.utcnow().isoformat() + "Z"
    end = (dt.datetime.utcnow() + dt.timedelta(days=days_ahead)).isoformat() + "Z"
    events_result = service.events().list(
        calendarId=calendar_id, timeMin=now, timeMax=end,
        singleEvents=True, orderBy="startTime", maxResults=25,
    ).execute()
    events = events_result.get("items", [])
    if not events:
        return f"No events in the next {days_ahead} day(s)."
    lines = []
    for e in events:
        start = e["start"].get("dateTime", e["start"].get("date"))
        lines.append(f"[{e['id']}] {start} — {e.get('summary', '(no title)')}")
    return "\n".join(lines)


@mcp.tool()
def create_event(
    summary: str, start: str, end: str, description: str = "", calendar_id: str = "primary"
) -> str:
    """Create a calendar event.

    Args:
        summary: Event title.
        start: Start time, ISO 8601 (e.g. "2026-09-01T09:00:00").
        end: End time, ISO 8601.
        description: Optional event description.
        calendar_id: Calendar to create the event on (default: primary).
    """
    try:
        service = _get_service()
    except RuntimeError as exc:
        return str(exc)

    body = {
        "summary": summary,
        "description": description,
        "start": {"dateTime": start},
        "end": {"dateTime": end},
    }
    event = service.events().insert(calendarId=calendar_id, body=body).execute()
    return f"Created event '{summary}' ({event['id']})."


@mcp.tool()
def find_event(query: str, days_ahead: int = 30, calendar_id: str = "primary") -> str:
    """Search upcoming events by title — use this to get an id for delete_event.

    Args:
        query: Text to search for in event titles.
        days_ahead: How many days ahead to search.
        calendar_id: Calendar to search (default: primary).
    """
    try:
        service = _get_service()
    except RuntimeError as exc:
        return str(exc)

    now = dt.datetime.utcnow().isoformat() + "Z"
    end = (dt.datetime.utcnow() + dt.timedelta(days=days_ahead)).isoformat() + "Z"
    events_result = service.events().list(
        calendarId=calendar_id, timeMin=now, timeMax=end,
        q=query, singleEvents=True, orderBy="startTime", maxResults=10,
    ).execute()
    events = events_result.get("items", [])
    if not events:
        return f"No events matching '{query}'."
    lines = []
    for e in events:
        start = e["start"].get("dateTime", e["start"].get("date"))
        lines.append(f"[{e['id']}] {start} — {e.get('summary', '(no title)')}")
    return "\n".join(lines)


@mcp.tool()
def delete_event(event_id: str, calendar_id: str = "primary") -> str:
    """Delete a calendar event by id (use find_event to look one up first).

    Args:
        event_id: The event id.
        calendar_id: Calendar the event is on (default: primary).
    """
    try:
        service = _get_service()
    except RuntimeError as exc:
        return str(exc)

    try:
        service.events().delete(calendarId=calendar_id, eventId=event_id).execute()
        return f"Deleted event {event_id}."
    except Exception as exc:
        return f"Couldn't delete event {event_id}: {exc}"


if __name__ == "__main__":
    mcp.run()
