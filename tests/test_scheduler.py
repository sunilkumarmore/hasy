"""Proactive scheduler: does it actually decide, fire, and stay out of the way?"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from hasy.memory import MemoryStore, Thread
from hasy.presence.proactive import ProactiveConfig, ProactivePolicy
from hasy.presence.scheduler import ProactiveScheduler

NOON = datetime(2026, 8, 14, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def store() -> MemoryStore:
    s = MemoryStore(":memory:")
    yield s
    s.close()


def scheduler_for(store: MemoryStore, now=NOON, **cfg) -> ProactiveScheduler:
    return ProactiveScheduler(
        store,
        policy=ProactivePolicy(ProactiveConfig(**cfg)),
        tick_s=0.01,
        now_fn=lambda: now,
    )


def aged_thread(store: MemoryStore, topic: str, hours_old: float) -> None:
    ts = (NOON - timedelta(hours=hours_old)).isoformat()
    store.open_thread(Thread(topic=topic, opened_at=ts, last_touched=ts))


def test_consider_reads_open_threads_from_the_store(store: MemoryStore):
    aged_thread(store, "book the flights", 6)
    d = scheduler_for(store).consider()
    assert d.should_speak
    assert d.thread.topic == "book the flights"


def test_consider_is_quiet_with_an_empty_store(store: MemoryStore):
    assert not scheduler_for(store).consider().should_speak


def test_recent_interaction_suppresses(store: MemoryStore):
    aged_thread(store, "x", 6)
    s = scheduler_for(store)
    s.note_interaction()  # human is right here
    assert not s.consider().should_speak


async def test_loop_fires_the_trigger_and_records_the_cooldown(store: MemoryStore):
    aged_thread(store, "book the flights", 6)
    s = scheduler_for(store)

    fired = asyncio.Event()

    async def trigger():
        fired.set()

    class FakeAgent:
        _pending_proactive = None

    agent = FakeAgent()
    task = asyncio.create_task(s.run(trigger, agent=agent))
    try:
        await asyncio.wait_for(fired.wait(), timeout=2.0)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert s.last_proactive is not None, "cooldown was not recorded"
    assert agent._pending_proactive is not None, "decision never reached the agent"
    assert agent._pending_proactive.thread.topic == "book the flights"


async def test_loop_stays_silent_when_policy_says_no(store: MemoryStore):
    # No threads at all -> nothing to say.
    s = scheduler_for(store)
    calls = []

    async def trigger():
        calls.append(1)

    task = asyncio.create_task(s.run(trigger))
    await asyncio.sleep(0.08)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert calls == []


async def test_a_failing_tick_does_not_kill_the_loop(store: MemoryStore):
    aged_thread(store, "x", 6)
    s = scheduler_for(store)
    attempts = []

    async def flaky_trigger():
        attempts.append(1)
        raise RuntimeError("websocket died")

    task = asyncio.create_task(s.run(flaky_trigger))
    await asyncio.sleep(0.1)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert len(attempts) >= 2, "loop stopped after the first failure"


async def test_loop_cancels_cleanly(store: MemoryStore):
    s = scheduler_for(store)

    async def trigger():
        pass

    task = asyncio.create_task(s.run(trigger))
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
