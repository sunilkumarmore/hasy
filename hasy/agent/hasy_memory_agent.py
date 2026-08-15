"""HASY's agent: upstream's BasicMemoryAgent plus entity-resolved memory.

Deliberately a **subclass**, not a replacement. ASR, TTS, the Live2D pipeline,
interruption handling, and transport are all untouched — the only two things
added are:

    read  — retrieved context is appended to the system prompt for this turn
    write — after the response is already streaming, the turn is committed async

Both are wrapped so that a memory failure degrades HASY to plain conversation
rather than breaking it.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Optional, Union

from loguru import logger

from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
from src.open_llm_vtuber.agent.input_types import BatchInput
from src.open_llm_vtuber.agent.output_types import SentenceOutput

from ..memory.retrieval import MemoryRetriever
from ..memory.writer import MemoryWriter


class HasyMemoryAgent(BasicMemoryAgent):
    """BasicMemoryAgent with a hybrid read path and an async write path."""

    def __init__(
        self,
        *args,
        retriever: Optional[MemoryRetriever] = None,
        writer: Optional[MemoryWriter] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # BasicMemoryAgent._set_llm assigns `self.chat` as an *instance*
        # attribute, which would shadow this class's chat() and silently skip
        # the whole memory layer. Drop it so method resolution reaches us.
        self.__dict__.pop("chat", None)
        self._retriever = retriever
        self._writer = writer
        self._turn_counter = 0
        logger.info(
            f"HasyMemoryAgent ready (retrieval={'on' if retriever else 'off'}, "
            f"write={'on' if writer else 'off'})."
        )

    async def chat(
        self, input_data: BatchInput
    ) -> AsyncIterator[Union[SentenceOutput, Dict[str, Any]]]:
        user_text = ""
        try:
            user_text = self._to_text_prompt(input_data)
        except Exception as e:
            logger.debug(f"HASY memory: could not read user text ({e})")

        base_system = self._system
        memory_block = ""

        # --- proactive turn? -------------------------------------------------
        # The scheduler stashes its decision on the agent just before triggering,
        # because upstream's ai-speak-signal path carries no room for a payload.
        proactive_block = ""
        if self._is_proactive(input_data):
            decision = getattr(self, "_pending_proactive", None)
            if decision is not None:
                from ..presence.proactive import compose_proactive_block

                proactive_block = compose_proactive_block(decision)
                self._pending_proactive = None
                if decision.thread is not None:
                    logger.info(
                        f"HASY presence: speaking first about '{decision.thread.topic}' "
                        f"({decision.reason})"
                    )

        # --- read path (on the latency path — measured in MemoryRetriever) ---
        if self._retriever is not None and user_text:
            try:
                ctx = self._retriever.retrieve(user_text)
                memory_block = self._retriever.compose(ctx)
                if memory_block:
                    logger.debug(
                        f"HASY memory: recalled {len(ctx.entities)} entities, "
                        f"{len(ctx.episodes)} episodes, {len(ctx.threads)} threads "
                        f"in {ctx.elapsed_ms:.1f} ms"
                    )
            except Exception as e:
                logger.warning(f"HASY memory: read path failed ({e})")

        extras = [b for b in (memory_block, proactive_block) if b]
        if extras:
            # Appended after the persona so retrieved facts never displace it.
            self._system = base_system + "\n\n" + "\n\n".join(extras)

        self._turn_counter += 1
        turn_id = f"turn-{self._turn_counter}"

        try:
            async for output in super().chat(input_data):
                yield output
        finally:
            self._system = base_system
            # Upstream marks proactive turns skip_memory/skip_history; honour
            # that. Storing "human said: please say something" would poison the
            # transcript and teach HASY to talk to itself.
            if not self._is_proactive(input_data):
                self._schedule_write(user_text, turn_id)

    @staticmethod
    def _is_proactive(input_data: BatchInput) -> bool:
        try:
            return bool((input_data.metadata or {}).get("proactive_speak"))
        except Exception:
            return False

    def _schedule_write(self, user_text: str, turn_id: str) -> None:
        """Commit the turn after the response has already gone out.

        The assistant's text is read from the agent's own memory list rather
        than by consuming the output stream — consuming it here would starve
        the TTS pipeline downstream.
        """
        if self._writer is None or not user_text:
            return
        try:
            assistant_text = ""
            if self._memory and self._memory[-1].get("role") == "assistant":
                assistant_text = self._memory[-1].get("content", "")
            if assistant_text:
                self._writer.schedule(user_text, assistant_text, turn_id=turn_id)
        except Exception as e:
            logger.warning(f"HASY memory: could not schedule write ({e})")
