#!/usr/bin/env python3
"""MCP server for saving notes to a focus area's own context directory.

See shared/focus_areas_store.py — each focus area has a directory under
~/.dann/focus_areas/<name>/ whose contents are automatically folded into
every turn's system prompt for that focus area's work streams (see
app/services/chat_service.py). Reading is already automatic; this is what
lets Dann write to it directly, without the user going through a text
editor or the terminal.

Exposed tools:
  save_focus_area_note — save a note to the current conversation's focus area
"""

from mcp.server.fastmcp import FastMCP

from shared.focus_areas_store import save_note

mcp = FastMCP("focus-areas")


@mcp.tool()
def save_focus_area_note(content: str, title: str = "", focus_area: str = "") -> str:
    """Save a note to the current conversation's focus area — durable
    context that's automatically included in every future conversation
    about this focus area. Use this when the user asks you to remember,
    note down, or save something for later, or when you learn something
    worth keeping (a decision, a preference, a fact) that isn't already
    captured in this conversation's existing notes.

    Args:
        content: What to remember.
        title: Optional short title for the note.
        focus_area: Do not set this — supplied automatically.
    """
    if not focus_area:
        return "No focus area context available for this conversation — can't save a note here."
    note = save_note(focus_area, content, title)
    return (
        f"Saved to '{focus_area}' notes ({note['title']}). "
        f"It'll be included automatically in future conversations about this focus area."
    )


if __name__ == "__main__":
    mcp.run()
