"""Temporal fact versioning: the 'never append a contradiction' contract."""

from __future__ import annotations

import pytest

from hasy.memory import Entity, Fact, MemoryStore


@pytest.fixture
def store() -> MemoryStore:
    s = MemoryStore(":memory:")
    yield s
    s.close()


@pytest.fixture
def phani(store: MemoryStore) -> int:
    return store.create_entity(
        Entity(canonical_name="Phani Meduri", type="person", aliases=["Phani"])
    )


def test_new_fact_is_current(store: MemoryStore, phani: int):
    store.upsert_fact(Fact(entity_id=phani, predicate="role", value="architect"))
    facts = store.current_facts(phani)
    assert len(facts) == 1
    assert facts[0].value == "architect"
    assert facts[0].is_current


def test_changed_fact_supersedes_rather_than_accumulates(
    store: MemoryStore, phani: int
):
    """The core requirement: one current value, old value retained as history."""
    store.upsert_fact(Fact(entity_id=phani, predicate="role", value="architect"))
    store.upsert_fact(Fact(entity_id=phani, predicate="role", value="VP Engineering"))

    current = store.current_facts(phani)
    assert len(current) == 1, "contradiction accumulated instead of superseding"
    assert current[0].value == "VP Engineering"

    history = store.fact_history(phani, "role")
    assert len(history) == 2
    assert history[0].value == "architect"
    assert history[0].valid_to is not None, "old row was not closed"
    assert history[1].valid_to is None


def test_reasserting_same_value_is_not_a_change(store: MemoryStore, phani: int):
    store.upsert_fact(Fact(entity_id=phani, predicate="role", value="architect"))
    store.upsert_fact(Fact(entity_id=phani, predicate="role", value="architect"))
    assert len(store.fact_history(phani, "role")) == 1


def test_reasserting_raises_confidence_but_keeps_one_row(
    store: MemoryStore, phani: int
):
    store.upsert_fact(
        Fact(entity_id=phani, predicate="role", value="architect", confidence=0.5)
    )
    store.upsert_fact(
        Fact(entity_id=phani, predicate="role", value="architect", confidence=0.9)
    )
    facts = store.current_facts(phani)
    assert len(facts) == 1
    assert facts[0].confidence == pytest.approx(0.9)


def test_distinct_predicates_coexist(store: MemoryStore, phani: int):
    store.upsert_fact(Fact(entity_id=phani, predicate="role", value="architect"))
    store.upsert_fact(Fact(entity_id=phani, predicate="employer", value="ThoughtSpark"))
    assert len(store.current_facts(phani)) == 2


def test_contradictory_facts_arriving_out_of_order(store: MemoryStore, phani: int):
    """Adversarial: yesterday's truth is learned after today's.

    The store applies changes in arrival order, so the last write wins as the
    current value. What must NOT happen is two rows both claiming to be current.
    """
    store.upsert_fact(
        Fact(entity_id=phani, predicate="city", value="Bangalore"),
        as_of="2026-08-10T00:00:00+00:00",
    )
    store.upsert_fact(
        Fact(entity_id=phani, predicate="city", value="Hyderabad"),
        as_of="2026-08-01T00:00:00+00:00",  # older event, learned later
    )

    current = store.current_facts(phani)
    assert len(current) == 1, "two rows claim to be current simultaneously"
    assert current[0].value == "Hyderabad"

    history = store.fact_history(phani, "city")
    assert len(history) == 2
    assert sum(1 for f in history if f.is_current) == 1


def test_facts_are_scoped_per_entity(store: MemoryStore, phani: int):
    other = store.create_entity(Entity(canonical_name="Ravi", type="person"))
    store.upsert_fact(Fact(entity_id=phani, predicate="role", value="architect"))
    store.upsert_fact(Fact(entity_id=other, predicate="role", value="designer"))

    assert store.current_facts(phani)[0].value == "architect"
    assert store.current_facts(other)[0].value == "designer"


def test_source_turn_id_is_retained(store: MemoryStore, phani: int):
    store.upsert_fact(
        Fact(entity_id=phani, predicate="role", value="architect", source_turn_id="t-42")
    )
    assert store.current_facts(phani)[0].source_turn_id == "t-42"
