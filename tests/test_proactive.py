"""Proactive-speech policy: silence is the default, and must be earned out of."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from hasy.memory.types import Thread
from hasy.presence.proactive import (
    ProactiveConfig,
    ProactivePolicy,
    compose_proactive_block,
)

NOON = datetime(2026, 8, 14, 12, 0, 0, tzinfo=timezone.utc)


def iso(dt: datetime) -> str:
    return dt.isoformat()


def thread(topic: str, hours_old: float, resolved: bool = False) -> Thread:
    touched = NOON - timedelta(hours=hours_old)
    return Thread(
        id=1,
        topic=topic,
        opened_at=iso(touched),
        last_touched=iso(touched),
        resolved_at=iso(NOON) if resolved else None,
    )


def policy(**overrides) -> ProactivePolicy:
    return ProactivePolicy(ProactiveConfig(**overrides))


# ------------------------------------------------------------- speaks when ok


def test_speaks_about_an_aging_open_thread():
    d = policy().decide(
        NOON,
        [thread("book the flights", hours_old=6)],
        last_interaction=NOON - timedelta(hours=2),
    )
    assert d.should_speak
    assert d.thread.topic == "book the flights"


def test_picks_the_most_neglected_thread():
    """The loop most at risk of being forgotten is the quietest one."""
    d = policy().decide(
        NOON,
        [thread("recent thing", hours_old=1), thread("forgotten thing", hours_old=48)],
        last_interaction=NOON - timedelta(hours=2),
    )
    assert d.thread.topic == "forgotten thing"


# ------------------------------------------------------------ stays quiet when


def test_silent_during_quiet_hours():
    night = NOON.replace(hour=3)
    d = policy().decide(night, [thread("x", hours_old=6)], last_interaction=night - timedelta(hours=5))
    assert not d.should_speak
    assert "quiet hours" in d.reason


def test_silent_mid_conversation():
    d = policy().decide(
        NOON,
        [thread("x", hours_old=6)],
        last_interaction=NOON - timedelta(seconds=30),
    )
    assert not d.should_speak
    assert "since last exchange" in d.reason


def test_silent_during_cooldown():
    d = policy().decide(
        NOON,
        [thread("x", hours_old=6)],
        last_interaction=NOON - timedelta(hours=2),
        last_proactive=NOON - timedelta(minutes=5),
    )
    assert not d.should_speak
    assert "cooldown" in d.reason


def test_silent_when_there_is_nothing_to_raise():
    d = policy().decide(NOON, [], last_interaction=NOON - timedelta(hours=2))
    assert not d.should_speak


def test_does_not_parrot_back_a_just_mentioned_thread():
    d = policy().decide(
        NOON,
        [thread("just said this", hours_old=0.1)],
        last_interaction=NOON - timedelta(hours=2),
    )
    assert not d.should_speak, "raised a topic from minutes ago"


def test_stops_nagging_about_stale_threads():
    d = policy().decide(
        NOON,
        [thread("ancient history", hours_old=24 * 30)],
        last_interaction=NOON - timedelta(hours=2),
    )
    assert not d.should_speak


def test_resolved_threads_are_never_raised():
    d = policy().decide(
        NOON,
        [thread("already done", hours_old=6, resolved=True)],
        last_interaction=NOON - timedelta(hours=2),
    )
    assert not d.should_speak


def test_disabled_means_never():
    d = policy(enabled=False).decide(
        NOON, [thread("x", hours_old=6)], last_interaction=NOON - timedelta(hours=9)
    )
    assert not d.should_speak


def test_first_ever_run_has_no_history_to_block_it():
    d = policy().decide(NOON, [thread("x", hours_old=6)])
    assert d.should_speak


def test_quiet_window_can_wrap_past_midnight():
    p = policy(quiet_before_hour=22, quiet_after_hour=6)  # awake at night
    assert p.decide(NOON.replace(hour=23), [thread("x", 6)], NOON - timedelta(hours=9)).should_speak
    assert not p.decide(NOON.replace(hour=12), [thread("x", 6)], NOON - timedelta(hours=9)).should_speak


def test_thread_with_unparseable_timestamps_is_skipped_not_crashed():
    bad = Thread(id=1, topic="corrupt", opened_at="not-a-date", last_touched="also-bad")
    d = policy().decide(NOON, [bad], last_interaction=NOON - timedelta(hours=2))
    assert not d.should_speak


# --------------------------------------------------------------- the steer


def test_block_names_the_topic_without_scripting_the_line():
    d = policy().decide(
        NOON, [thread("book the flights", 6)], last_interaction=NOON - timedelta(hours=2)
    )
    block = compose_proactive_block(d)
    assert "book the flights" in block
    assert "do not recite it verbatim" in block.lower()


def test_no_block_when_not_speaking():
    d = policy().decide(NOON, [], last_interaction=NOON)
    assert compose_proactive_block(d) == ""


def test_threadless_mode_is_opt_in():
    d = policy(require_thread=False).decide(NOON, [], last_interaction=NOON - timedelta(hours=2))
    assert d.should_speak
    assert compose_proactive_block(d)
