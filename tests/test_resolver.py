"""Entity resolution, with the adversarial cases from the Phase 3 spec.

No network and no API key: the adjudicator is a stub, and the embedder is the
dependency-free HashEmbedder.
"""

from __future__ import annotations

from typing import Optional, Sequence

import pytest

from hasy.memory import Entity, Fact, MemoryStore
from hasy.memory.resolver import EntityResolver


@pytest.fixture
def store() -> MemoryStore:
    s = MemoryStore(":memory:")
    yield s
    s.close()


class RecordingAdjudicator:
    """Stub LLM adjudicator. Returns a preset answer and records its calls."""

    def __init__(self, answer: Optional[int] | list[Optional[int]] = None):
        self._answers = answer if isinstance(answer, list) else [answer]
        self.calls: list[tuple[str, list[str], str]] = []

    def __call__(
        self, surface_form: str, candidates: Sequence[Entity], context: str
    ) -> Optional[int]:
        self.calls.append(
            (surface_form, [c.canonical_name for c in candidates], context)
        )
        return self._answers[min(len(self.calls) - 1, len(self._answers) - 1)]


# --------------------------------------------------------------- basic paths


def test_exact_match_costs_nothing(store: MemoryStore):
    eid = store.create_entity(Entity(canonical_name="Phani Meduri", type="person"))
    adj = RecordingAdjudicator(None)
    r = EntityResolver(store, adjudicator=adj).resolve("Phani Meduri")

    assert r.entity_id == eid
    assert r.method == "exact"
    assert adj.calls == [], "exact match should not reach the adjudicator"


def test_known_alias_resolves_without_adjudicator(store: MemoryStore):
    eid = store.create_entity(
        Entity(canonical_name="Phani Meduri", type="person", aliases=["Phani"])
    )
    adj = RecordingAdjudicator(None)
    r = EntityResolver(store, adjudicator=adj).resolve("phani")  # case-insensitive

    assert r.entity_id == eid
    assert r.method == "alias"
    assert adj.calls == []


def test_unknown_name_creates_entity(store: MemoryStore):
    r = EntityResolver(store).resolve("Ravi Kumar")
    assert r.created is True
    assert r.entity_id is not None
    assert store.get_entity(r.entity_id).canonical_name == "Ravi Kumar"


# ------------------------------------------- adversarial: one person, 3 names


def test_same_person_three_aliases_resolves_to_one_entity(store: MemoryStore):
    """'Phani', 'Phani Meduri', 'the platform architect' -> a single entity.

    The descriptor is not lexically similar to the name, so it must go through
    adjudication; afterwards it is learned as an alias and becomes free.
    """
    eid = store.create_entity(
        Entity(canonical_name="Phani Meduri", type="person", aliases=["Phani"])
    )
    adj = RecordingAdjudicator(eid)
    resolver = EntityResolver(store, adjudicator=adj)

    a = resolver.resolve("Phani Meduri")
    b = resolver.resolve("Phani")
    c = resolver.resolve("the platform architect", context="Phani leads the platform")

    assert a.entity_id == b.entity_id == c.entity_id == eid
    assert len(store.all_entities()) == 1, "aliases forked into duplicate entities"


def test_adjudicated_alias_is_learned_and_not_re_adjudicated(store: MemoryStore):
    eid = store.create_entity(Entity(canonical_name="Phani Meduri", type="person"))
    adj = RecordingAdjudicator(eid)
    resolver = EntityResolver(store, adjudicator=adj)

    first = resolver.resolve("the platform architect", context="Phani leads platform")
    assert first.entity_id == eid
    assert len(adj.calls) == 1

    second = resolver.resolve("the platform architect")
    assert second.entity_id == eid
    assert second.method == "alias", "learned alias should short-circuit to blocking"
    assert len(adj.calls) == 1, "adjudicator called again for a learned alias"


# --------------------------------------- adversarial: same name, two people


def test_same_name_different_people_are_kept_distinct(store: MemoryStore):
    """Two people genuinely named 'Ravi' must not be merged."""
    ravi_eng = store.create_entity(Entity(canonical_name="Ravi", type="person"))
    ravi_design = store.create_entity(Entity(canonical_name="Ravi", type="person"))
    store.upsert_fact(Fact(entity_id=ravi_eng, predicate="team", value="engineering"))
    store.upsert_fact(Fact(entity_id=ravi_design, predicate="team", value="design"))

    adj = RecordingAdjudicator(ravi_design)
    r = EntityResolver(store, adjudicator=adj).resolve(
        "Ravi", context="the designer who owns the mockups"
    )

    assert r.entity_id == ravi_design
    assert r.method == "llm"
    assert len(adj.calls) == 1, "an exact-name tie must go to adjudication"
    assert len(adj.calls[0][1]) == 2, "both same-named candidates should be offered"


def test_ambiguity_without_adjudicator_refuses_rather_than_guessing(
    store: MemoryStore,
):
    """Silently forking an identity is the failure this design exists to prevent."""
    store.create_entity(Entity(canonical_name="Ravi", type="person"))
    store.create_entity(Entity(canonical_name="Ravi", type="person"))

    r = EntityResolver(store, adjudicator=None).resolve("Ravi")

    assert r.entity_id is None
    assert r.created is False
    assert "no adjudicator" in r.rationale
    assert len(store.all_entities()) == 2, "ambiguity must not create a third entity"


def test_adjudicator_rejecting_all_candidates_creates_new_entity(store: MemoryStore):
    store.create_entity(Entity(canonical_name="Ravi Kumar", type="person"))
    adj = RecordingAdjudicator(None)  # "none of these"
    r = EntityResolver(store, adjudicator=adj).resolve(
        "Ravi Shankar", context="the musician"
    )

    assert r.created is True
    assert len(store.all_entities()) == 2


# ------------------------------------------------------------------ scoping


def test_types_do_not_collide(store: MemoryStore):
    """A project and a person may share a name without being confused."""
    person = store.create_entity(Entity(canonical_name="Atlas", type="person"))
    project = store.create_entity(Entity(canonical_name="Atlas", type="project"))
    resolver = EntityResolver(store, adjudicator=RecordingAdjudicator(None))

    assert resolver.resolve("Atlas", type="person").entity_id == person
    assert resolver.resolve("Atlas", type="project").entity_id == project


# -------------------------------------------------------------- audit trail


def test_every_resolution_is_logged_with_rationale(store: MemoryStore):
    eid = store.create_entity(Entity(canonical_name="Phani Meduri", type="person"))
    # Third call: adjudicator rejects, so that mention becomes a new entity.
    resolver = EntityResolver(store, adjudicator=RecordingAdjudicator([eid, None]))

    resolver.resolve("Phani Meduri")
    resolver.resolve("the platform architect", context="leads platform")
    resolver.resolve("Someone New Entirely")

    log = store.resolution_log()
    assert len(log) == 3
    assert {e["method"] for e in log} == {"exact", "llm", "created"}
    assert all(e["rationale"] for e in log), "every decision needs a rationale"
    assert any(e["created"] == 1 for e in log)


def test_empty_surface_form_is_ignored(store: MemoryStore):
    r = EntityResolver(store).resolve("   ")
    assert r.entity_id is None
    assert len(store.all_entities()) == 0
