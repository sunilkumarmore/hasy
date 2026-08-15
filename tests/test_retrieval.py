"""Read path: hybrid recall, prompt composition, and the ~120 ms budget."""

from __future__ import annotations

import pytest

from hasy.memory import Entity, Episode, Fact, MemoryStore, Thread
from hasy.memory.embeddings import HashEmbedder
from hasy.memory.retrieval import RETRIEVAL_BUDGET_MS, MemoryRetriever


@pytest.fixture
def store() -> MemoryStore:
    s = MemoryStore(":memory:")
    yield s
    s.close()


def test_entity_mentioned_in_utterance_is_recalled_with_its_facts(store: MemoryStore):
    eid = store.create_entity(
        Entity(canonical_name="Phani Meduri", type="person", aliases=["Phani"])
    )
    store.upsert_fact(Fact(entity_id=eid, predicate="role", value="architect"))

    ctx = MemoryRetriever(store).retrieve("how is Phani doing")

    assert [e.id for e in ctx.entities] == [eid]
    assert ctx.facts[eid][0].value == "architect"


def test_unmentioned_entities_are_not_recalled(store: MemoryStore):
    store.create_entity(Entity(canonical_name="Phani Meduri", type="person"))
    ctx = MemoryRetriever(store).retrieve("what's for lunch")
    assert ctx.entities == []


def test_partial_word_does_not_false_match(store: MemoryStore):
    """'Ravi' must not match inside 'ravioli'."""
    store.create_entity(Entity(canonical_name="Ravi", type="person"))
    ctx = MemoryRetriever(store).retrieve("I had ravioli for dinner")
    assert ctx.entities == []


def test_more_salient_entities_come_first(store: MemoryStore):
    a = store.create_entity(Entity(canonical_name="Atlas", type="project"))
    b = store.create_entity(Entity(canonical_name="Beacon", type="project"))
    for _ in range(5):
        store.touch_entity(b)

    ctx = MemoryRetriever(store).retrieve("how are Atlas and Beacon going")
    assert ctx.entities[0].id == b


def test_open_threads_are_surfaced(store: MemoryStore):
    store.open_thread(Thread(topic="book the flights"))
    ctx = MemoryRetriever(store).retrieve("anything I should know")
    assert [t.topic for t in ctx.threads] == ["book the flights"]


def test_resolved_threads_are_not_surfaced(store: MemoryStore):
    tid = store.open_thread(Thread(topic="book the flights"))
    store.resolve_thread(tid)
    ctx = MemoryRetriever(store).retrieve("anything I should know")
    assert ctx.threads == []


def test_episodes_are_recalled_by_similarity(store: MemoryStore):
    emb = HashEmbedder()
    store.add_episode(
        Episode(summary="Discussed the platform migration timeline."),
        embedding=emb.embed(["Discussed the platform migration timeline."])[0],
    )
    store.add_episode(
        Episode(summary="Talked about lunch options."),
        embedding=emb.embed(["Talked about lunch options."])[0],
    )

    ctx = MemoryRetriever(store, embedder=emb, max_episodes=1).retrieve(
        "the platform migration timeline"
    )
    assert ctx.episodes and "migration" in ctx.episodes[0].summary


def test_compose_includes_persona_safe_block(store: MemoryStore):
    eid = store.create_entity(Entity(canonical_name="Phani Meduri", type="person", aliases=["Phani"]))
    store.upsert_fact(Fact(entity_id=eid, predicate="role", value="architect"))
    store.open_thread(Thread(topic="review the migration plan"))

    r = MemoryRetriever(store)
    block = r.compose(r.retrieve("how is Phani"))

    assert "Phani Meduri" in block
    assert "architect" in block
    assert "review the migration plan" in block
    assert "do not mention that you are consulting memory" in block.lower()


def test_compose_is_empty_when_nothing_recalled(store: MemoryStore):
    r = MemoryRetriever(store)
    assert r.compose(r.retrieve("hello there")) == ""


def test_compose_respects_the_token_budget(store: MemoryStore):
    for i in range(40):
        eid = store.create_entity(Entity(canonical_name=f"Person{i}", type="person"))
        store.upsert_fact(
            Fact(entity_id=eid, predicate="bio", value="x" * 200)
        )
    mentions = " ".join(f"Person{i}" for i in range(40))

    r = MemoryRetriever(store, token_budget=100)
    ctx = r.retrieve(mentions)
    block = r.compose(ctx)

    assert len(block) < 100 * 3.6 + 400  # budget + fixed wrapper text


def test_retrieval_reports_its_own_cost(store: MemoryStore):
    store.create_entity(Entity(canonical_name="Phani", type="person"))
    ctx = MemoryRetriever(store).retrieve("how is Phani")
    assert ctx.elapsed_ms > 0


def test_retrieval_stays_within_budget_at_realistic_scale(store: MemoryStore):
    """The spec's ceiling: retrieval must not blow the latency budget.

    100 entities / 200 facts / 200 episodes is well past a year of desk use.
    """
    emb = HashEmbedder()
    for i in range(100):
        eid = store.create_entity(
            Entity(canonical_name=f"Person{i}", type="person", aliases=[f"P{i}"])
        )
        store.upsert_fact(Fact(entity_id=eid, predicate="role", value=f"role{i}"))
        store.upsert_fact(Fact(entity_id=eid, predicate="team", value=f"team{i}"))
    for i in range(200):
        s = f"Talked about topic {i} with Person{i % 100}."
        store.add_episode(Episode(summary=s, entities_mentioned=[i % 100 + 1]),
                          embedding=emb.embed([s])[0])
    for i in range(10):
        store.open_thread(Thread(topic=f"follow up on {i}"))

    r = MemoryRetriever(store, embedder=emb)
    r.retrieve("warm up the caches")            # discard first call
    ctx = r.retrieve("how is Person42 getting on with topic 17")

    assert ctx.entities, "should have recalled Person42"
    assert ctx.elapsed_ms < RETRIEVAL_BUDGET_MS, (
        f"retrieval took {ctx.elapsed_ms:.1f} ms, over the "
        f"{RETRIEVAL_BUDGET_MS:.0f} ms budget"
    )


def test_retrieval_survives_a_broken_store(store: MemoryStore):
    store.close()  # every query will now raise
    ctx = MemoryRetriever(store).retrieve("hello")
    assert ctx.is_empty, "a broken store must degrade, not raise"
