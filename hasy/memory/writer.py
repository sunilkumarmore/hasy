"""Write path — committing a turn to memory, after the response is streaming.

Order matters:
    1. extract candidates from the turn (cheap model call)
    2. **resolve** every mention against existing entities before writing
    3. commit — facts supersede rather than accumulate; threads open/close

Step 2 is the point of the whole design. Writing first and deduplicating later
is what produces three separate 'Phani's.

Everything here is async and fire-and-forget: `schedule()` returns immediately
so the conversation is never blocked, and any failure is logged rather than
raised into the response path.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from loguru import logger

from .embeddings import Embedder, HashEmbedder
from .extractor import Extractor, StubExtractor
from .resolver import EntityResolver
from .store import MemoryStore
from .types import Episode, Fact, Thread, normalize


class MemoryWriter:
    def __init__(
        self,
        store: MemoryStore,
        resolver: Optional[EntityResolver] = None,
        extractor: Optional[Extractor] = None,
        embedder: Optional[Embedder] = None,
    ):
        self.store = store
        self.resolver = resolver or EntityResolver(store)
        self.extractor = extractor or StubExtractor()
        self.embedder = embedder or HashEmbedder()
        self._tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------- scheduling

    def schedule(self, user_text: str, assistant_text: str, turn_id: Optional[str] = None) -> None:
        """Fire-and-forget. Returns immediately; the response is already streaming."""
        try:
            task = asyncio.create_task(self.write_turn(user_text, assistant_text, turn_id))
        except RuntimeError:
            # No running loop (e.g. called from sync context) — skip rather than raise.
            logger.debug("HASY memory: no event loop; skipping async write")
            return
        # Hold a reference so the task isn't garbage-collected mid-flight.
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def drain(self) -> None:
        """Await outstanding writes. For tests and clean shutdown."""
        if self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    # ------------------------------------------------------------- write path

    async def write_turn(
        self, user_text: str, assistant_text: str, turn_id: Optional[str] = None
    ) -> None:
        try:
            extraction = await self.extractor.extract(user_text, assistant_text)
        except Exception as e:
            logger.warning(f"HASY memory: extraction raised ({e})")
            return

        try:
            context = f"{user_text}\n{assistant_text}"[:500]

            # --- 1. resolve every mention BEFORE writing anything -----------
            resolved: dict[str, int] = {}
            for mention in extraction.entities:
                decision = self.resolver.resolve(
                    mention.surface_form, type=mention.type, context=context
                )
                if decision.entity_id is not None:
                    resolved[normalize(mention.surface_form)] = decision.entity_id
                    self.store.touch_entity(decision.entity_id)
                    logger.debug(
                        f"HASY memory: '{mention.surface_form}' -> "
                        f"#{decision.entity_id} via {decision.method} "
                        f"({decision.rationale})"
                    )
                else:
                    logger.info(
                        f"HASY memory: left '{mention.surface_form}' unresolved — "
                        f"{decision.rationale}"
                    )

            # --- 2. facts, superseding rather than appending ----------------
            for fact in extraction.facts:
                entity_id = resolved.get(normalize(fact.subject))
                if entity_id is None:
                    # The fact's subject was never extracted as an entity; resolve
                    # it now rather than dropping the fact on the floor.
                    decision = self.resolver.resolve(fact.subject, context=context)
                    entity_id = decision.entity_id
                    if entity_id is not None:
                        resolved[normalize(fact.subject)] = entity_id
                if entity_id is None:
                    logger.debug(f"HASY memory: dropped fact with unresolved subject {fact.subject!r}")
                    continue
                self.store.upsert_fact(
                    Fact(
                        entity_id=entity_id,
                        predicate=fact.predicate,
                        value=fact.value,
                        confidence=fact.confidence,
                        source_turn_id=turn_id,
                    )
                )

            # --- 3. episode, with its embedding for later recall ------------
            if extraction.episode_summary:
                embedding = None
                try:
                    embedding = self.embedder.embed([extraction.episode_summary])[0]
                except Exception as e:
                    logger.debug(f"HASY memory: embedding failed ({e})")
                self.store.add_episode(
                    Episode(
                        summary=extraction.episode_summary,
                        entities_mentioned=sorted(set(resolved.values())),
                        valence=extraction.valence,
                        source_turn_id=turn_id,
                    ),
                    embedding=embedding,
                )

            # --- 4. threads -------------------------------------------------
            self._apply_threads(extraction, resolved)

        except Exception as e:
            logger.warning(f"HASY memory: write failed ({e})")

    def _apply_threads(self, extraction, resolved: dict[str, int]) -> None:
        open_threads = self.store.open_threads(limit=50)
        by_topic = {normalize(t.topic): t for t in open_threads}

        for candidate in extraction.threads:
            key = normalize(candidate.topic)
            existing = by_topic.get(key)

            if candidate.status == "resolved":
                if existing and existing.id is not None:
                    self.store.resolve_thread(existing.id)
                    logger.debug(f"HASY memory: closed thread '{existing.topic}'")
                continue

            if existing and existing.id is not None:
                self.store.touch_thread(existing.id)
            else:
                self.store.open_thread(
                    Thread(
                        topic=candidate.topic,
                        entity_ids=sorted(set(resolved.values())),
                    )
                )
                logger.debug(f"HASY memory: opened thread '{candidate.topic}'")
