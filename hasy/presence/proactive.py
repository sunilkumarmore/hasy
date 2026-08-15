"""Proactive speaking, driven by unresolved threads and time-of-day.

Upstream already has proactive speech, but its trigger source is the *client*:
the frontend sends `ai-speak-signal` and the server loads a static
"say something" prompt. Nothing decides *when* it is worth speaking, or *what*
about.

This module supplies that decision. The policy is kept pure and separate from
the wiring so it can be tested without a server, a clock, or a websocket:

    ProactivePolicy.decide(now, threads, last_interaction, last_proactive)
        -> ProactiveDecision(should_speak, thread, reason)

Design bias: **silence is the default.** A companion that pipes up too often is
worse than one that never does, so every rule here is a reason *not* to speak,
and a thread must earn its interruption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence

from ..memory.types import Thread


def _parse(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        parsed = datetime.fromisoformat(ts)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


@dataclass
class ProactiveConfig:
    enabled: bool = True

    #: Never speak unprompted before / after these local hours.
    quiet_before_hour: int = 9
    quiet_after_hour: int = 22

    #: Don't speak while the human is mid-conversation — wait for a real lull.
    min_silence_s: float = 300.0

    #: Never proactively speak more often than this.
    cooldown_s: float = 3600.0

    #: A thread must be at least this old before it's worth raising, so we
    #: don't parrot back something said thirty seconds ago.
    min_thread_age_s: float = 1800.0

    #: A thread nobody has touched in this long is stale; stop nagging.
    max_thread_age_s: float = 14 * 24 * 3600.0

    #: Skip proactive speech entirely if there are no open threads.
    require_thread: bool = True


@dataclass
class ProactiveDecision:
    should_speak: bool
    reason: str
    thread: Optional[Thread] = None

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.should_speak


@dataclass
class ProactivePolicy:
    config: ProactiveConfig = field(default_factory=ProactiveConfig)

    def decide(
        self,
        now: datetime,
        threads: Sequence[Thread],
        last_interaction: Optional[datetime] = None,
        last_proactive: Optional[datetime] = None,
    ) -> ProactiveDecision:
        """Should HASY speak unprompted right now, and about what?

        `now` is injected rather than read from the clock so this is testable
        and so the caller owns the timezone question.
        """
        c = self.config

        if not c.enabled:
            return ProactiveDecision(False, "proactive speech disabled")

        # --- time of day -------------------------------------------------
        hour = now.hour
        if c.quiet_before_hour <= c.quiet_after_hour:
            awake = c.quiet_before_hour <= hour < c.quiet_after_hour
        else:  # window wraps past midnight
            awake = hour >= c.quiet_before_hour or hour < c.quiet_after_hour
        if not awake:
            return ProactiveDecision(
                False, f"quiet hours ({hour:02d}:00 outside {c.quiet_before_hour}-{c.quiet_after_hour})"
            )

        # --- don't interrupt an active conversation ----------------------
        if last_interaction is not None:
            silence = (now - last_interaction).total_seconds()
            if silence < c.min_silence_s:
                return ProactiveDecision(
                    False, f"only {silence:.0f}s since last exchange (need {c.min_silence_s:.0f}s)"
                )

        # --- don't nag ----------------------------------------------------
        if last_proactive is not None:
            since = (now - last_proactive).total_seconds()
            if since < c.cooldown_s:
                return ProactiveDecision(
                    False, f"cooldown, {since:.0f}s since last proactive (need {c.cooldown_s:.0f}s)"
                )

        # --- pick a thread worth raising ----------------------------------
        candidate = self._pick_thread(now, threads)
        if candidate is None:
            if c.require_thread:
                return ProactiveDecision(False, "no unresolved thread worth raising")
            return ProactiveDecision(True, "no thread, but proactive speech allowed without one")

        age = (now - (_parse(candidate.last_touched) or now)).total_seconds()
        return ProactiveDecision(
            True,
            f"thread '{candidate.topic}' untouched for {age / 3600:.1f}h",
            thread=candidate,
        )

    def _pick_thread(self, now: datetime, threads: Sequence[Thread]) -> Optional[Thread]:
        """Oldest-but-not-stale open thread.

        Oldest first because the loop most at risk of being forgotten is the
        one that has gone quietest — not the one most recently discussed.
        """
        c = self.config
        eligible: list[tuple[float, Thread]] = []
        for t in threads:
            if not t.is_open:
                continue
            touched = _parse(t.last_touched) or _parse(t.opened_at)
            if touched is None:
                continue
            age = (now - touched).total_seconds()
            if age < c.min_thread_age_s or age > c.max_thread_age_s:
                continue
            eligible.append((age, t))

        if not eligible:
            return None
        eligible.sort(key=lambda pair: pair[0], reverse=True)
        return eligible[0][1]


def compose_proactive_block(decision: ProactiveDecision) -> str:
    """The steer handed to the model on a proactive turn.

    Deliberately permissive: HASY is told it *may* raise the topic, not that it
    must recite it. An unprompted line that sounds scripted is worse than none.
    """
    if not decision.should_speak:
        return ""
    if decision.thread is None:
        return (
            "=== Speaking first ===\n"
            "You are choosing to say something unprompted. Keep it to one short, "
            "natural line. Do not announce that you are speaking proactively.\n"
            "======================"
        )
    return (
        "=== Speaking first ===\n"
        f"There is an unresolved topic from earlier: {decision.thread.topic}\n"
        "You may bring this up if it feels natural. One short line, the way a "
        "person would if it just came to mind. Do not recite it verbatim, do not "
        "list it, and do not announce that you are consulting memory.\n"
        "======================"
    )
