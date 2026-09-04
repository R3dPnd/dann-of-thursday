#!/usr/bin/env python3
"""MCP server for tracking your own garden — plantings, harvests, and notes.

Local-only storage (~/.dann/gardening.json) — no external account needed.
This deliberately doesn't try to answer general gardening questions (frost
dates, pest identification, etc.) — the base LLM already knows that. What it
adds is the thing a stock LLM can't have: a record of what *you* planted,
where, and how it went.

Exposed tools:
  log_planting     — record something you planted
  list_plantings   — list recent plantings, optionally filtered by location
  record_harvest   — record a harvest against a plant
  list_harvests    — list recent harvests, optionally filtered by plant
  add_garden_note  — free-form observation (pests, watering, care notes)
  search_garden_log — substring search across plantings, harvests, and notes
"""
import uuid
from datetime import date, datetime

from mcp.server.fastmcp import FastMCP

from src.mcp_servers._store import load_json, save_json

mcp = FastMCP("gardening")

_FILE = "gardening.json"
_DEFAULT = {"plantings": [], "harvests": [], "notes": []}


def _load() -> dict:
    data = load_json(_FILE, _DEFAULT)
    for key in _DEFAULT:
        data.setdefault(key, [])
    return data


def _save(data: dict) -> None:
    save_json(_FILE, data)


@mcp.tool()
def log_planting(plant: str, location: str = "", planted_on: str = "") -> str:
    """Record something you planted.

    Args:
        plant: What you planted (e.g. "roma tomatoes").
        location: Optional bed/container/zone (e.g. "raised bed 2").
        planted_on: Date planted, ISO 8601 (e.g. "2026-09-04"). Defaults to today.
    """
    data = _load()
    entry = {
        "id": uuid.uuid4().hex[:8],
        "plant": plant,
        "location": location,
        "planted_on": planted_on or date.today().isoformat(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    data["plantings"].append(entry)
    _save(data)
    return f"Logged planting of {plant}" + (f" in {location}" if location else "") + "."


@mcp.tool()
def list_plantings(location: str = "", limit: int = 20) -> str:
    """List recent plantings, most recent first.

    Args:
        location: Optional bed/container/zone to filter by.
        limit: Max plantings to return.
    """
    plantings = _load()["plantings"]
    if location:
        plantings = [p for p in plantings if location.lower() in p.get("location", "").lower()]
    plantings = sorted(plantings, key=lambda p: p["planted_on"], reverse=True)[:limit]
    if not plantings:
        return "No plantings found."
    return "\n".join(
        f"[{p['id']}] {p['plant']} — {p['location'] or 'no location'} ({p['planted_on']})"
        for p in plantings
    )


@mcp.tool()
def record_harvest(plant: str, amount: str = "", notes: str = "") -> str:
    """Record a harvest.

    Args:
        plant: What you harvested (e.g. "roma tomatoes").
        amount: Optional quantity, free text (e.g. "2 lbs", "a dozen").
        notes: Optional notes (e.g. "first ripe batch, a bit early this year").
    """
    data = _load()
    entry = {
        "id": uuid.uuid4().hex[:8],
        "plant": plant,
        "amount": amount,
        "notes": notes,
        "harvested_on": date.today().isoformat(),
    }
    data["harvests"].append(entry)
    _save(data)
    return f"Logged harvest of {plant}" + (f" — {amount}" if amount else "") + "."


@mcp.tool()
def list_harvests(plant: str = "", limit: int = 20) -> str:
    """List recent harvests, most recent first.

    Args:
        plant: Optional plant name to filter by.
        limit: Max harvests to return.
    """
    harvests = _load()["harvests"]
    if plant:
        harvests = [h for h in harvests if plant.lower() in h["plant"].lower()]
    harvests = sorted(harvests, key=lambda h: h["harvested_on"], reverse=True)[:limit]
    if not harvests:
        return "No harvests found."
    return "\n".join(
        f"[{h['id']}] {h['plant']} — {h['amount'] or 'no amount noted'} ({h['harvested_on']})"
        for h in harvests
    )


@mcp.tool()
def add_garden_note(text: str, plant: str = "") -> str:
    """Save a free-form garden observation — pests, watering, weather damage, etc.

    Args:
        text: The note content.
        plant: Optional plant this note is about.
    """
    data = _load()
    entry = {
        "id": uuid.uuid4().hex[:8],
        "text": text,
        "plant": plant,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    data["notes"].append(entry)
    _save(data)
    return f"Saved garden note {entry['id']}."


@mcp.tool()
def search_garden_log(query: str) -> str:
    """Search plantings, harvests, and notes for a substring match.

    Args:
        query: Text to search for (matches plant names, locations, note text).
    """
    data = _load()
    q = query.lower()
    hits = []
    for p in data["plantings"]:
        if q in p["plant"].lower() or q in p.get("location", "").lower():
            hits.append(f"[planting {p['id']}] {p['plant']} — {p['location'] or 'no location'} ({p['planted_on']})")
    for h in data["harvests"]:
        if q in h["plant"].lower() or q in h.get("notes", "").lower():
            hits.append(f"[harvest {h['id']}] {h['plant']} — {h['amount'] or 'no amount'} ({h['harvested_on']})")
    for n in data["notes"]:
        if q in n["text"].lower() or q in n.get("plant", "").lower():
            hits.append(f"[note {n['id']}] {n['text']}")
    if not hits:
        return f"Nothing in the garden log matching '{query}'."
    return "\n".join(hits)


if __name__ == "__main__":
    mcp.run()
