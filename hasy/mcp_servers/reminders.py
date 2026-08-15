"""Reminders as an MCP tool, backed by HASY's own memory.

Phase 4 item 5 asks for calendar and reminders via MCP rather than new tool
plumbing. Reminders are the half HASY can own outright: the `threads` table
built in Phase 3 *is* a list of open loops, so exposing it as tools means
"remind me to X" and "what's outstanding?" work against the same state that
drives proactive speech — say it out loud and HASY may raise it later by itself.

Calendar is deliberately not here: it needs a provider (Google / Outlook) and
OAuth credentials, which is a decision plus secrets, not something to guess at.
See docs/presence.md.

Run standalone (upstream's MCP client launches it this way):

    uv run python -m hasy.mcp_servers.reminders
"""

from __future__ import annotations

import os
from typing import Optional

from ..memory.store import MemoryStore
from ..memory.types import Thread, normalize

DB_PATH = os.environ.get("HASY_MEMORY_DB", "hasy_memory.db")

_store: Optional[MemoryStore] = None


def store() -> MemoryStore:
    """Opened lazily so importing this module never touches the disk."""
    global _store
    if _store is None:
        _store = MemoryStore(DB_PATH)
    return _store


# --------------------------------------------------------------- operations
# Kept as plain functions so they are testable without an MCP client.


def list_reminders(limit: int = 20) -> list[dict]:
    """Open loops, most recently touched first."""
    return [
        {
            "id": t.id,
            "topic": t.topic,
            "opened_at": t.opened_at,
            "last_touched": t.last_touched,
        }
        for t in store().open_threads(limit=limit)
    ]


def add_reminder(topic: str) -> dict:
    """Open a loop. Re-adding an existing one touches it instead of duplicating."""
    topic = (topic or "").strip()
    if not topic:
        return {"ok": False, "error": "empty topic"}

    key = normalize(topic)
    for existing in store().open_threads(limit=100):
        if normalize(existing.topic) == key:
            store().touch_thread(existing.id)
            return {"ok": True, "id": existing.id, "topic": existing.topic, "already_open": True}

    thread_id = store().open_thread(Thread(topic=topic))
    return {"ok": True, "id": thread_id, "topic": topic, "already_open": False}


def resolve_reminder(topic: str = "", reminder_id: Optional[int] = None) -> dict:
    """Close a loop by id, or by matching its topic."""
    if reminder_id is not None:
        store().resolve_thread(int(reminder_id))
        return {"ok": True, "id": int(reminder_id)}

    key = normalize(topic)
    if not key:
        return {"ok": False, "error": "give a topic or a reminder_id"}

    for existing in store().open_threads(limit=100):
        if normalize(existing.topic) == key:
            store().resolve_thread(existing.id)
            return {"ok": True, "id": existing.id, "topic": existing.topic}
    return {"ok": False, "error": f"no open reminder matching {topic!r}"}


# ------------------------------------------------------------- MCP wrapper


def build_server():
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("hasy-reminders")

    @mcp.tool()
    def list_reminders_tool(limit: int = 20) -> list[dict]:
        """List the things the human has left unresolved. Use when asked what is
        outstanding, pending, or still to do."""
        return list_reminders(limit)

    @mcp.tool()
    def add_reminder_tool(topic: str) -> dict:
        """Remember something to follow up on later. Use when the human asks to
        be reminded, or says they will deal with something later."""
        return add_reminder(topic)

    @mcp.tool()
    def resolve_reminder_tool(topic: str = "", reminder_id: int = 0) -> dict:
        """Close out a reminder once it is done. Use when the human says
        something is finished, handled, or no longer needed."""
        return resolve_reminder(topic, reminder_id or None)

    return mcp


def main() -> None:  # pragma: no cover - process entry point
    build_server().run()


if __name__ == "__main__":  # pragma: no cover
    main()
