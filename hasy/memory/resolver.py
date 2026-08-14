"""Entity resolution — deciding whether a mention is someone we already know.

This is the step the whole memory design exists for. Resolution runs
cheapest-first and stops as soon as it is confident:

    1. blocking      exact match on a normalized alias        (free)
    2. embedding     lexical/vector similarity over candidates (cheap)
    3. adjudication  an LLM call, only for genuine ambiguity   (expensive)
    4. create        a new entity, when nothing plausible fits

Every outcome is written to `resolution_log` with the method and rationale, so
duplicate entities can be audited rather than guessed at.
"""

from __future__ import annotations

from typing import Callable, Optional, Protocol, Sequence

from loguru import logger

from .embeddings import Embedder, HashEmbedder, cosine
from .store import MemoryStore
from .types import Entity, EntityType, ResolutionDecision, normalize

#: Above this similarity, accept the candidate without an LLM call.
AUTO_MATCH_THRESHOLD = 0.92
#: Below this, do not even offer the candidate to the adjudicator.
CANDIDATE_FLOOR = 0.45


class Adjudicator(Protocol):
    """Resolves genuine ambiguity. Injected so tests never need an API key."""

    def __call__(
        self,
        surface_form: str,
        candidates: Sequence[Entity],
        context: str,
    ) -> Optional[int]:
        """Return the matching entity id, or None to mean 'none of these'."""
        ...


class EntityResolver:
    def __init__(
        self,
        store: MemoryStore,
        embedder: Optional[Embedder] = None,
        adjudicator: Optional[Adjudicator] = None,
        auto_match_threshold: float = AUTO_MATCH_THRESHOLD,
        candidate_floor: float = CANDIDATE_FLOOR,
    ):
        self.store = store
        self.embedder = embedder or HashEmbedder()
        self.adjudicator = adjudicator
        self.auto_match_threshold = auto_match_threshold
        self.candidate_floor = candidate_floor

    # ------------------------------------------------------------------ api

    def resolve(
        self,
        surface_form: str,
        type: EntityType = "person",
        context: str = "",
        create_if_missing: bool = True,
        learn_alias: bool = True,
    ) -> ResolutionDecision:
        """Resolve a mention to an entity id, creating one only if warranted."""
        surface_form = (surface_form or "").strip()
        if not surface_form:
            return self._log(
                ResolutionDecision(
                    surface_form=surface_form,
                    entity_id=None,
                    method="created",
                    score=0.0,
                    rationale="empty surface form; ignored",
                )
            )

        decision = self._resolve_inner(
            surface_form, type, context, create_if_missing, learn_alias
        )
        return self._log(decision)

    def _resolve_inner(
        self,
        surface_form: str,
        type: EntityType,
        context: str,
        create_if_missing: bool,
        learn_alias: bool,
    ) -> ResolutionDecision:
        # --- 1. blocking on exact normalized alias -------------------------
        blocked = self.store.find_by_alias(surface_form, type=type)
        if len(blocked) == 1:
            e = blocked[0]
            method = (
                "exact"
                if normalize(e.canonical_name) == normalize(surface_form)
                else "alias"
            )
            return ResolutionDecision(
                surface_form=surface_form,
                entity_id=e.id,
                method=method,  # type: ignore[arg-type]
                score=1.0,
                rationale=f"exact alias match to '{e.canonical_name}' (#{e.id})",
            )

        # Same name, more than one person. Blocking cannot settle this; only
        # context can, so go straight to adjudication over the tied candidates.
        if len(blocked) > 1:
            return self._adjudicate(
                surface_form,
                blocked,
                context,
                [(e.id, 1.0) for e in blocked if e.id is not None],
                reason="multiple entities share this exact name",
                create_if_missing=create_if_missing,
                type=type,
                learn_alias=learn_alias,
            )

        # --- 2. embedding similarity over existing entities ----------------
        pool = self.store.all_entities(type=type)
        if not pool:
            return self._create(
                surface_form, type, "no existing entities of this type", create_if_missing
            )

        scored = self._score_candidates(surface_form, pool)
        top_id, top_score = scored[0]

        if top_score >= self.auto_match_threshold:
            entity = self.store.get_entity(top_id)
            if learn_alias and entity:
                # Cache the win: next time this phrasing is a free blocking hit.
                self.store.add_alias(top_id, surface_form)
            return ResolutionDecision(
                surface_form=surface_form,
                entity_id=top_id,
                method="embedding",
                score=top_score,
                rationale=(
                    f"similarity {top_score:.3f} >= {self.auto_match_threshold} "
                    f"to '{entity.canonical_name if entity else top_id}'"
                ),
                candidates=scored[:5],
            )

        plausible = [(eid, s) for eid, s in scored if s >= self.candidate_floor]
        if plausible:
            candidates = [
                e for e in (self.store.get_entity(eid) for eid, _ in plausible[:5]) if e
            ]
            return self._adjudicate(
                surface_form,
                candidates,
                context,
                plausible[:5],
                reason=f"top similarity {top_score:.3f} below auto-match",
                create_if_missing=create_if_missing,
                type=type,
                learn_alias=learn_alias,
            )

        # Nothing cleared the floor. If we can ask, ask before creating: a
        # duplicate identity is a permanent, compounding error, while a single
        # adjudication is a bounded cost paid on the async write path.
        if self.adjudicator is not None:
            candidates = [
                e for e in (self.store.get_entity(eid) for eid, _ in scored[:3]) if e
            ]
            if candidates:
                return self._adjudicate(
                    surface_form,
                    candidates,
                    context,
                    scored[:3],
                    reason=(
                        f"nothing above floor (best {top_score:.3f}); "
                        "confirming before creating"
                    ),
                    create_if_missing=create_if_missing,
                    type=type,
                    learn_alias=learn_alias,
                )

        return self._create(
            surface_form,
            type,
            f"no candidate above floor (best {top_score:.3f})",
            create_if_missing,
            candidates=scored[:5],
        )

    # -------------------------------------------------------------- helpers

    def _entity_profile(self, entity: Entity) -> str:
        """A descriptive blob for an entity: its names plus what we know about it.

        Descriptors ('the platform architect') have almost no lexical overlap
        with names ('Phani Meduri'), but they do overlap with the entity's
        *facts* (role=architect). Scoring against facts is what lets a
        description find its subject.
        """
        parts = [entity.canonical_name, *entity.aliases]
        if entity.id is not None:
            parts += [f.value for f in self.store.current_facts(entity.id)]
        return " ".join(dict.fromkeys(p for p in parts if p))

    def _score_candidates(
        self, surface_form: str, pool: Sequence[Entity]
    ) -> list[tuple[int, float]]:
        """Score every entity by its best-matching surface form or fact.

        'Phani' matches the entity canonically named 'Phani Meduri' via its
        alias list, so we take the max over every form we know rather than
        comparing to the canonical name alone.
        """
        query = self.embedder.embed([surface_form])[0]
        scored: list[tuple[int, float]] = []
        for entity in pool:
            if entity.id is None:
                continue
            forms = {entity.canonical_name, *entity.aliases, self._entity_profile(entity)}
            vecs = self.embedder.embed(sorted(f for f in forms if f))
            scored.append((entity.id, max(cosine(query, v) for v in vecs)))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored

    def _adjudicate(
        self,
        surface_form: str,
        candidates: Sequence[Entity],
        context: str,
        scored: list[tuple[int, float]],
        reason: str,
        create_if_missing: bool,
        type: EntityType,
        learn_alias: bool,
    ) -> ResolutionDecision:
        if self.adjudicator is None:
            # No adjudicator wired: refuse to guess. Creating a duplicate is the
            # failure mode this whole design exists to prevent, so when we
            # cannot tell, we say so rather than silently forking an identity.
            return ResolutionDecision(
                surface_form=surface_form,
                entity_id=None,
                method="llm",
                score=scored[0][1] if scored else 0.0,
                rationale=f"ambiguous ({reason}) and no adjudicator configured",
                candidates=scored,
            )

        chosen = self.adjudicator(surface_form, list(candidates), context)
        if chosen is not None:
            if learn_alias:
                self.store.add_alias(chosen, surface_form)
            entity = self.store.get_entity(chosen)
            return ResolutionDecision(
                surface_form=surface_form,
                entity_id=chosen,
                method="llm",
                score=dict(scored).get(chosen, 0.0),
                rationale=(
                    f"adjudicator chose '{entity.canonical_name if entity else chosen}'"
                    f" (#{chosen}); {reason}"
                ),
                candidates=scored,
            )

        return self._create(
            surface_form,
            type,
            f"adjudicator rejected all candidates; {reason}",
            create_if_missing,
            candidates=scored,
        )

    def _create(
        self,
        surface_form: str,
        type: EntityType,
        why: str,
        create_if_missing: bool,
        candidates: Optional[list[tuple[int, float]]] = None,
    ) -> ResolutionDecision:
        if not create_if_missing:
            return ResolutionDecision(
                surface_form=surface_form,
                entity_id=None,
                method="created",
                score=0.0,
                rationale=f"no match ({why}) and creation disabled",
                candidates=candidates or [],
            )
        entity_id = self.store.create_entity(
            Entity(canonical_name=surface_form, type=type)
        )
        logger.debug(f"HASY memory: created {type} #{entity_id} '{surface_form}' — {why}")
        return ResolutionDecision(
            surface_form=surface_form,
            entity_id=entity_id,
            method="created",
            score=0.0,
            rationale=f"created new entity; {why}",
            created=True,
            candidates=candidates or [],
        )

    def _log(self, decision: ResolutionDecision) -> ResolutionDecision:
        try:
            self.store.log_resolution(decision)
        except Exception as e:  # auditing must never break the write path
            logger.warning(f"HASY memory: failed to log resolution ({e})")
        return decision
