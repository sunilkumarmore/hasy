"""HasyMemoryAgent: does the memory layer actually run during a turn?

These drive the agent with a fake LLM so no API key or network is needed.
"""

from __future__ import annotations

import pytest

from hasy.agent.hasy_memory_agent import HasyMemoryAgent
from hasy.memory import Entity, Fact, MemoryStore
from hasy.memory.extractor import ExtractionOut, MentionOut, StubExtractor
from hasy.memory.retrieval import MemoryRetriever
from hasy.memory.writer import MemoryWriter

from src.open_llm_vtuber.agent.input_types import BatchInput, TextData, TextSource
from src.open_llm_vtuber.config_manager import TTSPreprocessorConfig
from src.open_llm_vtuber.config_manager.tts_preprocessor import TranslatorConfig


def tts_config() -> TTSPreprocessorConfig:
    """Upstream's tts_filter builds a default config if given None, and that
    model has required fields — so supply a real one."""
    return TTSPreprocessorConfig(
        remove_special_char=True,
        ignore_brackets=True,
        ignore_parentheses=True,
        ignore_asterisks=True,
        ignore_angle_brackets=True,
        translator_config=TranslatorConfig(translate_audio=False, translate_provider="deeplx"),
    )


class FakeLLM:
    """Stands in for the Claude backend; records the system prompt it was given."""

    def __init__(self, reply: str = "Sure thing."):
        self.reply = reply
        self.systems: list[str] = []

    async def chat_completion(self, messages, system=None, tools=None):
        self.systems.append(system or "")
        for chunk in self.reply.split(" "):
            yield chunk + " "


class FakeLive2D:
    def extract_emotion(self, text):
        return []

    def to_dict(self):
        return {}


def build_agent(store: MemoryStore, llm: FakeLLM, extraction: ExtractionOut | None = None):
    retriever = MemoryRetriever(store)
    writer = MemoryWriter(store=store, extractor=StubExtractor([extraction or ExtractionOut()]))
    agent = HasyMemoryAgent(
        llm=llm,
        system="You are HASY.",
        live2d_model=FakeLive2D(),
        tts_preprocessor_config=tts_config(),
        retriever=retriever,
        writer=writer,
    )
    return agent, writer


def user_turn(text: str) -> BatchInput:
    return BatchInput(
        texts=[TextData(source=TextSource.INPUT, content=text, from_name="Human")],
        images=None,
    )


@pytest.fixture
def store() -> MemoryStore:
    s = MemoryStore(":memory:")
    yield s
    s.close()


async def drain(agent, batch) -> list:
    return [out async for out in agent.chat(batch)]


async def test_chat_override_actually_runs(store: MemoryStore):
    """Guards the instance-attribute shadowing bug.

    BasicMemoryAgent.__init__ sets self.chat as an instance attribute; if that
    isn't cleared, HasyMemoryAgent.chat never executes and memory silently
    does nothing while everything still appears to work.
    """
    llm = FakeLLM()
    agent, writer = build_agent(
        store,
        llm,
        ExtractionOut(
            entities=[MentionOut(surface_form="Phani", type="person")],
            episode_summary="Talked to Phani.",
        ),
    )
    assert "chat" not in agent.__dict__, "instance attribute would shadow the override"

    await drain(agent, user_turn("hello"))
    await writer.drain()

    assert store.all_entities(), "write path never ran — chat() was shadowed"


async def test_known_entity_is_injected_into_the_system_prompt(store: MemoryStore):
    eid = store.create_entity(
        Entity(canonical_name="Phani Meduri", type="person", aliases=["Phani"])
    )
    store.upsert_fact(Fact(entity_id=eid, predicate="role", value="architect"))

    llm = FakeLLM()
    agent, writer = build_agent(store, llm)
    await drain(agent, user_turn("what is Phani working on"))
    await writer.drain()

    system = llm.systems[0]
    assert "You are HASY." in system, "persona was displaced"
    assert "Phani Meduri" in system
    assert "architect" in system


async def test_unknown_topic_injects_nothing(store: MemoryStore):
    llm = FakeLLM()
    agent, writer = build_agent(store, llm)
    await drain(agent, user_turn("what's the weather"))
    await writer.drain()

    assert llm.systems[0] == "You are HASY.\n\nIf you received `[interrupted by user]` signal, you were interrupted."


async def test_system_prompt_is_restored_between_turns(store: MemoryStore):
    """Injected memory must not accumulate turn over turn."""
    eid = store.create_entity(Entity(canonical_name="Phani", type="person"))
    store.upsert_fact(Fact(entity_id=eid, predicate="role", value="architect"))

    llm = FakeLLM()
    agent, writer = build_agent(store, llm)
    await drain(agent, user_turn("tell me about Phani"))
    await drain(agent, user_turn("tell me about Phani"))
    await writer.drain()

    assert llm.systems[0] == llm.systems[1], "memory block accumulated across turns"
    assert llm.systems[0].count("What you remember") == 1


async def test_memory_failure_does_not_break_the_conversation(store: MemoryStore):
    class ExplodingRetriever:
        def retrieve(self, utterance):
            raise RuntimeError("store is on fire")

        def compose(self, ctx):
            raise RuntimeError("store is still on fire")

    llm = FakeLLM(reply="Still talking.")
    agent = HasyMemoryAgent(
        llm=llm,
        system="You are HASY.",
        live2d_model=FakeLive2D(),
        tts_preprocessor_config=tts_config(),
        retriever=ExplodingRetriever(),
        writer=None,
    )
    outputs = await drain(agent, user_turn("hello"))
    assert outputs, "a memory failure silenced the agent"


async def test_write_is_scheduled_not_awaited(store: MemoryStore):
    """The response must not wait on the write path."""
    llm = FakeLLM()
    agent, writer = build_agent(
        store,
        llm,
        ExtractionOut(
            entities=[MentionOut(surface_form="Ravi", type="person")],
            episode_summary="Met Ravi.",
        ),
    )
    await drain(agent, user_turn("hello"))
    # Streaming finished; the write may still be in flight.
    await writer.drain()
    assert len(store.all_entities()) == 1
