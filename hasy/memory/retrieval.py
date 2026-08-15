"""Read path — assembling what HASY should know before it answers.

Hybrid, not top-k cosine:
    1. vector search over episodes
    2. graph walk from entities mentioned in the current utterance
    3. recency-weighted unresolved threads

The result is composed into a system-prompt block under a fixed token budget.

This runs **on the latency path**, so it is measured. The Phase 3 spec sets a
~120 ms ceiling; `MemoryContext.elapsed_ms` reports the real number and
`retrieve()` logs a warning when it is exceeded, rather than letting the cost
hide.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from .embeddings import Embedder, HashEmbedder
from .store import MemoryStore
from .types import Entity, Episode, Fact, Thread, normalize

#: Spec ceiling for the whole read path. Exceeding it is reported, not hidden.
RETRIEVAL_BUDGET_MS = 120.0

#: Rough chars-per-token. Deliberately conservative so the block never overruns.
CHARS_PER_TOKEN = 3.6


@dataclass
class MemoryContext:
    """What retrieval found, plus what it cost."""

    entities: list[Entity] = field(default_factory=list)
    facts: dict[int, list[Fact]] = field(default_factory=dict)
    episodes: list[Episode] = field(default_factory=list)
    threads: list[Thread] = field(default_factory=list)
    elapsed_ms: float = 0.0
    truncated: bool = False

    @property
    def is_empty(self) -> bool:
        return not (self.entities or self.episodes or self.threads)

    @property
    def within_budget(self) -> bool:
        return self.elapsed_ms <= RETRIEVAL_BUDGET_MS


class MemoryRetriever:
    def __init__(
        self,
        store: MemoryStore,
        embedder: Optional[Embedder] = None,
        token_budget: int = 600,
        max_episodes: int = 4,
        max_entities: int = 6,
        max_threads: int = 3,
    ):
        self.store = store
        self.embedder = embedder or HashEmbedder()
        self.token_budget = token_budget
        self.max_episodes = max_episodes
        self.max_entities = max_entities
        self.max_threads = max_threads

    # ------------------------------------------------------------------ read

    def retrieve(self, utterance: str) -> MemoryContext:
        t0 = time.perf_counter()
        ctx = MemoryContext()

        try:
            ctx.entities = self._entities_in(utterance)
            ctx.facts = {
                e.id: self.store.current_facts(e.id) for e in ctx.entities if e.id is not None
            }
            ctx.episodes = self._recall_episodes(utterance, ctx.entities)
            ctx.threads = self.store.open_threads(limit=self.max_threads)
        except Exception as e:
            # Memory is an enhancement. If it fails, HASY still talks.
            logger.warning(f"HASY memory: retrieval failed ({e})")

        ctx.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if not ctx.within_budget:
            logger.warning(
                f"HASY memory: retrieval took {ctx.elapsed_ms:.1f} ms "
                f"(budget {RETRIEVAL_BUDGET_MS:.0f} ms) — this is on the latency path"
            )
        else:
            logger.debug(f"HASY memory: retrieval {ctx.elapsed_ms:.1f} ms")
        return ctx

    # -------------------------------------------------------------- internals

    def _entities_in(self, utterance: str) -> list[Entity]:
        """Graph-walk seed: entities whose known aliases appear in the utterance.

        Blocking-style substring match over normalized aliases. Cheap on
        purpose — this is on the latency path, so no model call and no vector
        search happens here.
        """
        key = normalize(utterance)
        if not key:
            return []
        padded = f" {key} "

        hits: list[tuple[Entity, float]] = []
        for entity in self.store.all_entities():
            if entity.id is None:
                continue
            forms = {entity.canonical_name, *entity.aliases}
            matched = any(
                (norm := normalize(form)) and f" {norm} " in padded for form in forms
            )
            if matched:
                hits.append((entity, entity.salience))

        # Most salient first: the thing referenced often is the thing that matters.
        hits.sort(key=lambda t: t[1], reverse=True)
        return [e for e, _ in hits[: self.max_entities]]

    def _recall_episodes(self, utterance: str, entities: list[Entity]) -> list[Episode]:
        """Vector recall, falling back to recency when vectors are unavailable."""
        episodes: list[Episode] = []
        try:
            vec = self.embedder.embed([utterance])[0]
            episodes = [ep for ep, _ in self.store.search_episodes(vec, limit=self.max_episodes)]
        except Exception as e:
            logger.debug(f"HASY memory: vector recall unavailable ({e})")

        if not episodes:
            episodes = self.store.recent_episodes(limit=self.max_episodes)

        # Pull in episodes that mention the entities in play, even if the
        # wording of this utterance is nothing like the wording back then.
        ids = {e.id for e in entities if e.id is not None}
        if ids:
            seen = {ep.id for ep in episodes}
            for ep in self.store.recent_episodes(limit=self.max_episodes * 3):
                if ep.id in seen:
                    continue
                if ids.intersection(ep.entities_mentioned):
                    episodes.append(ep)
                    if len(episodes) >= self.max_episodes:
                        break
        return episodes[: self.max_episodes]

    # ------------------------------------------------------------- composition

    def compose(self, ctx: MemoryContext) -> str:
        """Render context into a system-prompt block within the token budget.

        Ordered by value-per-token: who is being talked about, then open loops,
        then episodic colour. Truncation drops from the bottom.
        """
        if ctx.is_empty:
            return ""

        max_chars = int(self.token_budget * CHARS_PER_TOKEN)
        sections: list[str] = []

        if ctx.entities:
            lines = ["What you know about who's being discussed:"]
            for e in ctx.entities:
                facts = ctx.facts.get(e.id or -1, [])
                detail = "; ".join(f"{f.predicate}: {f.value}" for f in facts)
                aka = (
                    f" (also called {', '.join(a for a in e.aliases if a != e.canonical_name)})"
                    if len(e.aliases) > 1
                    else ""
                )
                lines.append(f"- {e.canonical_name}{aka}" + (f" — {detail}" if detail else ""))
            sections.append("\n".join(lines))

        if ctx.threads:
            lines = ["Unresolved topics you could follow up on:"]
            lines += [f"- {t.topic} (since {(t.opened_at or '')[:10]})" for t in ctx.threads]
            sections.append("\n".join(lines))

        if ctx.episodes:
            lines = ["Relevant moments from earlier:"]
            lines += [f"- {(ep.ts or '')[:16].replace('T', ' ')}: {ep.summary}" for ep in ctx.episodes]
            sections.append("\n".join(lines))

        block = ""
        for section in sections:
            candidate = f"{block}\n\n{section}" if block else section
            if len(candidate) > max_chars:
                ctx.truncated = True
                break
            block = candidate

        if not block:
            return ""
        return (
            "=== What you remember ===\n"
            f"{block}\n"
            "Use this naturally if it is relevant. Do not recite it, and do not "
            "mention that you are consulting memory.\n"
            "=========================="
        )
