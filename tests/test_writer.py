"""Write path: resolution before writing, supersession, threads, async safety."""

from __future__ import annotations

import pytest

from hasy.memory import Entity, Fact, MemoryStore
from hasy.memory.extractor import ExtractionOut, FactOut, MentionOut, StubExtractor, ThreadOut
from hasy.memory.resolver import EntityResolver
from hasy.memory.writer import MemoryWriter


@pytest.fixture
def store() -> MemoryStore:
    s = MemoryStore(":memory:")
    yield s
    s.close()


def writer_for(store: MemoryStore, extraction: ExtractionOut, adjudicator=None) -> MemoryWriter:
    return MemoryWriter(
        store=store,
        resolver=EntityResolver(store, adjudicator=adjudicator),
        extractor=StubExtractor([extraction]),
    )


@pytest.mark.asyncio
async def test_writes_entity_fact_episode_and_thread(store: MemoryStore):
    w = writer_for(
        store,
        ExtractionOut(
            entities=[MentionOut(surface_form="Phani Meduri", type="person")],
            facts=[FactOut(subject="Phani Meduri", predicate="role", value="architect", confidence=0.9)],
            threads=[ThreadOut(topic="review the platform migration plan", status="open")],
            episode_summary="Discussed the platform migration with Phani.",
            valence=0.3,
        ),
    )
    await w.write_turn("...", "...", turn_id="t-1")

    entities = store.all_entities()
    assert len(entities) == 1
    assert store.current_facts(entities[0].id)[0].value == "architect"
    assert len(store.recent_episodes()) == 1
    assert len(store.open_threads()) == 1


@pytest.mark.asyncio
async def test_second_mention_does_not_create_a_duplicate(store: MemoryStore):
    """The whole point: resolution happens before the write."""
    extraction = ExtractionOut(
        entities=[MentionOut(surface_form="Phani", type="person")],
        episode_summary="Talked about Phani again.",
    )
    store.create_entity(
        Entity(canonical_name="Phani Meduri", type="person", aliases=["Phani"])
    )

    w = writer_for(store, extraction)
    await w.write_turn("...", "...")

    assert len(store.all_entities()) == 1, "resolution ran after the write, not before"


@pytest.mark.asyncio
async def test_changed_fact_supersedes_through_the_write_path(store: MemoryStore):
    eid = store.create_entity(Entity(canonical_name="Phani Meduri", type="person"))
    store.upsert_fact(Fact(entity_id=eid, predicate="role", value="architect"))

    w = writer_for(
        store,
        ExtractionOut(
            entities=[MentionOut(surface_form="Phani Meduri", type="person")],
            facts=[FactOut(subject="Phani Meduri", predicate="role", value="VP Engineering", confidence=1.0)],
            episode_summary="Phani got promoted.",
        ),
    )
    await w.write_turn("...", "...")

    current = store.current_facts(eid)
    assert len(current) == 1
    assert current[0].value == "VP Engineering"
    assert len(store.fact_history(eid, "role")) == 2


@pytest.mark.asyncio
async def test_resolved_thread_is_closed_not_duplicated(store: MemoryStore):
    store.open_thread(__import__("hasy.memory", fromlist=["Thread"]).Thread(topic="book the flights"))

    w = writer_for(
        store,
        ExtractionOut(
            threads=[ThreadOut(topic="book the flights", status="resolved")],
            episode_summary="Flights are booked.",
        ),
    )
    await w.write_turn("...", "...")

    assert store.open_threads() == []


@pytest.mark.asyncio
async def test_repeating_an_open_thread_touches_rather_than_duplicates(store: MemoryStore):
    extraction = ExtractionOut(
        threads=[ThreadOut(topic="book the flights", status="open")],
        episode_summary="Flights still not booked.",
    )
    w = MemoryWriter(store=store, extractor=StubExtractor([extraction, extraction]))
    await w.write_turn("...", "...")
    await w.write_turn("...", "...")

    assert len(store.open_threads()) == 1


@pytest.mark.asyncio
async def test_fact_about_an_unextracted_subject_is_still_resolved(store: MemoryStore):
    """The extractor may report a fact whose subject it forgot to list as an entity."""
    w = writer_for(
        store,
        ExtractionOut(
            entities=[],
            facts=[FactOut(subject="Ravi", predicate="team", value="design", confidence=0.8)],
            episode_summary="Mentioned Ravi.",
        ),
    )
    await w.write_turn("...", "...")

    entities = store.all_entities()
    assert len(entities) == 1
    assert store.current_facts(entities[0].id)[0].value == "design"


@pytest.mark.asyncio
async def test_extractor_failure_does_not_raise(store: MemoryStore):
    class Exploding:
        async def extract(self, user_text, assistant_text):
            raise RuntimeError("model unavailable")

    w = MemoryWriter(store=store, extractor=Exploding())
    await w.write_turn("...", "...")  # must not raise
    assert store.all_entities() == []


@pytest.mark.asyncio
async def test_schedule_does_not_block_and_completes(store: MemoryStore):
    w = writer_for(
        store,
        ExtractionOut(
            entities=[MentionOut(surface_form="Phani", type="person")],
            episode_summary="A turn happened.",
        ),
    )
    w.schedule("...", "...")          # returns immediately
    assert store.all_entities() == []  # nothing written yet
    await w.drain()
    assert len(store.all_entities()) == 1


@pytest.mark.asyncio
async def test_unresolvable_mention_is_logged_not_forced(store: MemoryStore):
    """Ambiguity with no adjudicator must not silently fork an identity."""
    store.create_entity(Entity(canonical_name="Ravi", type="person"))
    store.create_entity(Entity(canonical_name="Ravi", type="person"))

    w = writer_for(store, ExtractionOut(
        entities=[MentionOut(surface_form="Ravi", type="person")],
        episode_summary="Ravi came up.",
    ))
    await w.write_turn("...", "...")

    assert len(store.all_entities()) == 2, "ambiguity created a third entity"
