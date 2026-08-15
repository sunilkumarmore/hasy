"""Runs the proactive policy against a live client.

Upstream's proactive path is triggered by the frontend sending `ai-speak-signal`
and carries no payload. So this module:

    1. runs a slow background loop per connected client,
    2. asks ProactivePolicy whether to speak (and about what),
    3. stashes the decision on the agent so it can steer the turn,
    4. fires the same `ai-speak-signal` upstream already understands.

No upstream file is edited — WebSocketHandler is patched from here.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

from ..memory.store import MemoryStore
from .proactive import ProactiveConfig, ProactivePolicy

#: How often to *consider* speaking. Cheap: a store read and some arithmetic.
DEFAULT_TICK_S = 60.0


class ProactiveScheduler:
    """One loop per connected client."""

    def __init__(
        self,
        store: MemoryStore,
        policy: Optional[ProactivePolicy] = None,
        tick_s: float = DEFAULT_TICK_S,
        now_fn=lambda: datetime.now(timezone.utc),
    ):
        self.store = store
        self.policy = policy or ProactivePolicy()
        self.tick_s = tick_s
        self._now = now_fn
        self.last_interaction: Optional[datetime] = None
        self.last_proactive: Optional[datetime] = None

    def note_interaction(self) -> None:
        """Called on every real turn so the loop knows the human is present."""
        self.last_interaction = self._now()

    def consider(self):
        """One evaluation. Separated from the loop so it is testable."""
        threads = self.store.open_threads(limit=50)
        return self.policy.decide(
            now=self._now(),
            threads=threads,
            last_interaction=self.last_interaction,
            last_proactive=self.last_proactive,
        )

    async def run(self, trigger, agent=None) -> None:
        """Loop until cancelled. `trigger` is an async callable that speaks."""
        logger.info(f"HASY presence: proactive loop started (tick {self.tick_s:.0f}s)")
        try:
            while True:
                await asyncio.sleep(self.tick_s)
                try:
                    decision = self.consider()
                    if not decision.should_speak:
                        logger.trace(f"HASY presence: staying quiet — {decision.reason}")
                        continue
                    # Hand the chosen thread to the agent before triggering.
                    if agent is not None:
                        agent._pending_proactive = decision
                    logger.info(f"HASY presence: speaking first — {decision.reason}")
                    await trigger()
                    # Cooldown starts only once the utterance actually went out.
                    # A trigger that raised means the human heard nothing, and
                    # burning an hour of silence for a failed attempt is worse
                    # than retrying on the next tick.
                    self.last_proactive = self._now()
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    # A bad tick must never kill the loop or the connection.
                    logger.warning(f"HASY presence: proactive tick failed ({e})")
        except asyncio.CancelledError:
            logger.debug("HASY presence: proactive loop stopped")
            raise


def install(config: Optional[ProactiveConfig] = None, tick_s: float = DEFAULT_TICK_S) -> None:
    """Attach a proactive loop to every client connection."""
    try:
        from src.open_llm_vtuber.websocket_handler import WebSocketHandler
    except Exception as e:  # pragma: no cover
        logger.warning(f"HASY presence: cannot patch WebSocketHandler ({e})")
        return

    from ..memory.install import build_memory  # reuses the configured store

    cfg = config or ProactiveConfig()
    if not cfg.enabled:
        logger.info("HASY presence: proactive speech disabled")
        return

    original_connect = WebSocketHandler.handle_new_connection
    original_disconnect = WebSocketHandler.handle_disconnect
    tasks: dict[str, asyncio.Task] = {}
    schedulers: dict[str, ProactiveScheduler] = {}

    async def connect_wrapper(self, websocket, client_uid, *args, **kwargs):
        result = await original_connect(self, websocket, client_uid, *args, **kwargs)
        try:
            context = self.client_contexts.get(client_uid)
            agent = getattr(context, "agent_engine", None)
            store = _store_from_agent(agent)
            if store is None:
                logger.debug("HASY presence: no memory store on this agent; proactive off")
                return result

            scheduler = ProactiveScheduler(store, ProactivePolicy(cfg), tick_s=tick_s)
            schedulers[client_uid] = scheduler

            async def trigger():
                await self._handle_conversation_trigger(
                    websocket, client_uid, {"type": "ai-speak-signal"}
                )

            tasks[client_uid] = asyncio.create_task(scheduler.run(trigger, agent=agent))
        except Exception as e:
            logger.warning(f"HASY presence: could not start proactive loop ({e})")
        return result

    async def disconnect_wrapper(self, client_uid, *args, **kwargs):
        task = tasks.pop(client_uid, None)
        schedulers.pop(client_uid, None)
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        return await original_disconnect(self, client_uid, *args, **kwargs)

    WebSocketHandler.handle_new_connection = connect_wrapper
    WebSocketHandler.handle_disconnect = disconnect_wrapper

    # Every real turn resets the "is the human here" clock.
    _patch_interaction_marker(schedulers)

    logger.info("HASY presence: proactive speaking installed.")


def _store_from_agent(agent) -> Optional[MemoryStore]:
    """Pull the live store out of the agent's writer/retriever, if present."""
    for attr in ("_writer", "_retriever"):
        component = getattr(agent, attr, None)
        store = getattr(component, "store", None)
        if store is not None:
            return store
    return None


def _patch_interaction_marker(schedulers: dict[str, "ProactiveScheduler"]) -> None:
    """Mark the last human interaction so the policy can require a real lull."""
    try:
        from src.open_llm_vtuber.conversations import conversation_handler as ch
    except Exception:  # pragma: no cover
        return

    original = ch.handle_conversation_trigger

    async def wrapper(msg_type: str, *args, **kwargs):
        if msg_type != "ai-speak-signal":
            client_uid = kwargs.get("client_uid")
            if client_uid is None and len(args) >= 2:
                client_uid = args[1]
            scheduler = schedulers.get(client_uid)
            if scheduler is not None:
                scheduler.note_interaction()
        return await original(msg_type, *args, **kwargs)

    ch.handle_conversation_trigger = wrapper
    # websocket_handler imported the symbol directly — rebind it there too.
    try:
        from src.open_llm_vtuber import websocket_handler as wh

        if getattr(wh, "handle_conversation_trigger", None) is not None:
            wh.handle_conversation_trigger = wrapper
    except Exception:  # pragma: no cover
        pass
