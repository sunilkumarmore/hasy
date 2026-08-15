"""Desk-merge verification: Phase 1 latency + Phase 3 memory in one process.

Structural coexistence isn't enough — these drive a real turn through the
patched Claude LLM class with a memory-enabled agent, and assert that the
latency stage timings are still recorded.
"""

from __future__ import annotations

import pytest

from hasy.agent.hasy_memory_agent import HasyMemoryAgent
from hasy.memory import Entity, Fact, MemoryStore
from hasy.memory.retrieval import MemoryRetriever

from src.open_llm_vtuber.agent.input_types import BatchInput, TextData, TextSource
from src.open_llm_vtuber.agent.stateless_llm import claude_llm
from src.open_llm_vtuber.config_manager import TTSPreprocessorConfig
from src.open_llm_vtuber.config_manager.tts_preprocessor import TranslatorConfig


# ----------------------------------------------------- fake Anthropic client


class _Delta:
    def __init__(self, text):
        self.type = "text_delta"
        self.text = text


class _Event:
    def __init__(self, type_, delta=None, index=0):
        self.type = type_
        self.delta = delta
        self.index = index  # upstream logs event.index on content_block_delta


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def __aiter__(self):
        for c in self._chunks:
            yield _Event("content_block_delta", _Delta(c))
        yield _Event("message_stop")


class _FakeMessages:
    def __init__(self, chunks):
        self._chunks = chunks
        self.systems: list[str] = []

    def stream(self, *, messages, system, model, max_tokens, tools=None):
        self.systems.append(system or "")
        return _FakeStream(self._chunks)


class FakeAnthropic:
    def __init__(self, chunks):
        self.messages = _FakeMessages(chunks)


def tts_config() -> TTSPreprocessorConfig:
    return TTSPreprocessorConfig(
        remove_special_char=True,
        ignore_brackets=True,
        ignore_parentheses=True,
        ignore_asterisks=True,
        ignore_angle_brackets=True,
        translator_config=TranslatorConfig(translate_audio=False, translate_provider="deeplx"),
    )


class FakeLive2D:
    def extract_emotion(self, text):
        return []


@pytest.fixture
def store() -> MemoryStore:
    s = MemoryStore(":memory:")
    yield s
    s.close()


@pytest.fixture
def latency_installed():
    """Install the latency patches once for this module's tests."""
    from hasy.latency import install

    install()
    yield


def build(store: MemoryStore) -> tuple[HasyMemoryAgent, FakeAnthropic]:
    llm = claude_llm.AsyncLLM(model="claude-haiku-4-5", llm_api_key="test-key")
    fake = FakeAnthropic(["Hello ", "there. ", "All good."])
    llm.client = fake  # no network
    agent = HasyMemoryAgent(
        llm=llm,
        system="You are HASY.",
        live2d_model=FakeLive2D(),
        tts_preprocessor_config=tts_config(),
        retriever=MemoryRetriever(store),
        writer=None,
    )
    return agent, fake


def turn(text: str) -> BatchInput:
    return BatchInput(
        texts=[TextData(source=TextSource.INPUT, content=text, from_name="Human")],
        images=None,
    )


async def test_latency_records_first_token_for_a_memory_agent(
    store: MemoryStore, latency_installed
):
    from hasy.latency import TurnMetrics, _current_turn

    store.create_entity(Entity(canonical_name="Phani", type="person"))
    agent, _ = build(store)

    m = TurnMetrics(turn=1)
    _current_turn.set(m)

    outputs = [o async for o in agent.chat(turn("how is Phani"))]

    assert outputs, "no output produced"
    assert m.llm_first_token_ms is not None, (
        "latency instrumentation did not observe the LLM through the memory agent"
    )
    assert m.llm_first_token_ms >= 0


async def test_memory_context_reaches_the_real_llm_call(store: MemoryStore, latency_installed):
    eid = store.create_entity(Entity(canonical_name="Phani Meduri", type="person", aliases=["Phani"]))
    store.upsert_fact(Fact(entity_id=eid, predicate="role", value="architect"))

    agent, fake = build(store)
    [o async for o in agent.chat(turn("what is Phani up to"))]

    system = fake.messages.systems[0]
    assert "You are HASY." in system
    assert "architect" in system, "memory never reached the actual API call"


async def test_memory_injection_costs_prompt_tokens(store: MemoryStore, latency_installed):
    """Recall is not free — it enlarges the system prompt, which is on the
    time-to-first-token path. Measured here so the cost stays visible."""
    eid = store.create_entity(Entity(canonical_name="Phani", type="person"))
    for i in range(5):
        store.upsert_fact(Fact(entity_id=eid, predicate=f"attr{i}", value=f"value {i}"))

    agent_cold, fake_cold = build(store)
    [o async for o in agent_cold.chat(turn("nothing relevant here"))]
    baseline = len(fake_cold.messages.systems[0])

    agent_warm, fake_warm = build(store)
    [o async for o in agent_warm.chat(turn("tell me about Phani"))]
    with_memory = len(fake_warm.messages.systems[0])

    assert with_memory > baseline
    # Bounded by the retriever's token budget, not unbounded growth.
    assert with_memory - baseline < 600 * 3.6 + 400
