"""Reminders MCP server — operations run against the real threads table."""

from __future__ import annotations

import pytest

from hasy.memory import MemoryStore, Thread
from hasy.mcp_servers import reminders


@pytest.fixture(autouse=True)
def isolated_store(monkeypatch):
    store = MemoryStore(":memory:")
    monkeypatch.setattr(reminders, "_store", store)
    yield store
    store.close()


def test_add_then_list(isolated_store):
    reminders.add_reminder("book the flights")
    items = reminders.list_reminders()
    assert [i["topic"] for i in items] == ["book the flights"]


def test_adding_the_same_reminder_twice_does_not_duplicate(isolated_store):
    first = reminders.add_reminder("book the flights")
    second = reminders.add_reminder("Book The Flights")  # different casing

    assert second["already_open"] is True
    assert second["id"] == first["id"]
    assert len(reminders.list_reminders()) == 1


def test_resolve_by_topic(isolated_store):
    reminders.add_reminder("book the flights")
    out = reminders.resolve_reminder("book the flights")
    assert out["ok"] is True
    assert reminders.list_reminders() == []


def test_resolve_by_id(isolated_store):
    added = reminders.add_reminder("call the dentist")
    out = reminders.resolve_reminder(reminder_id=added["id"])
    assert out["ok"] is True
    assert reminders.list_reminders() == []


def test_resolving_something_unknown_reports_rather_than_silently_passing(isolated_store):
    out = reminders.resolve_reminder("something never mentioned")
    assert out["ok"] is False
    assert "no open reminder" in out["error"]


def test_empty_topic_is_rejected(isolated_store):
    assert reminders.add_reminder("   ")["ok"] is False
    assert reminders.resolve_reminder("")["ok"] is False


def test_reminders_are_the_same_threads_that_drive_proactive_speech(isolated_store):
    """The point of backing reminders with the threads table: saying it out loud
    and asking HASY to remember it land in the same place."""
    from datetime import datetime, timedelta, timezone

    from hasy.presence.proactive import ProactiveConfig, ProactivePolicy

    reminders.add_reminder("book the flights")

    now = datetime.now(timezone.utc) + timedelta(hours=6)
    # Time-of-day is tested in test_proactive.py; neutralise it here so this
    # test is about the shared table and nothing else.
    config = ProactiveConfig(quiet_before_hour=0, quiet_after_hour=24)
    decision = ProactivePolicy(config).decide(
        now, isolated_store.open_threads(), last_interaction=now - timedelta(hours=2)
    )
    assert decision.should_speak
    assert decision.thread.topic == "book the flights"


def test_mcp_server_exposes_the_three_tools():
    server = reminders.build_server()
    assert server is not None
    assert server.name == "hasy-reminders"
