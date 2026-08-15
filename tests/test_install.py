"""The wiring seam: does install() actually upgrade the agent?

Registration happens by monkeypatching AgentFactory rather than editing
upstream's if/elif chain, so this is exactly the kind of seam that can silently
not fire. Tested directly.
"""

from __future__ import annotations

import pytest

from hasy.agent.hasy_memory_agent import HasyMemoryAgent
from hasy.memory import MemoryStore
from hasy.memory.retrieval import MemoryRetriever
from hasy.memory.writer import MemoryWriter

from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
from src.open_llm_vtuber.config_manager import TTSPreprocessorConfig
from src.open_llm_vtuber.config_manager.tts_preprocessor import TranslatorConfig


class FakeLLM:
    async def chat_completion(self, messages, system=None, tools=None):
        yield "ok"


class FakeLive2D:
    def extract_emotion(self, text):
        return []


def tts_config() -> TTSPreprocessorConfig:
    return TTSPreprocessorConfig(
        remove_special_char=True,
        ignore_brackets=True,
        ignore_parentheses=True,
        ignore_asterisks=True,
        ignore_angle_brackets=True,
        translator_config=TranslatorConfig(translate_audio=False, translate_provider="deeplx"),
    )


@pytest.fixture
def store() -> MemoryStore:
    s = MemoryStore(":memory:")
    yield s
    s.close()


def plain_agent() -> BasicMemoryAgent:
    return BasicMemoryAgent(
        llm=FakeLLM(),
        system="You are HASY.",
        live2d_model=FakeLive2D(),
        tts_preprocessor_config=tts_config(),
    )


def test_upstream_agent_really_does_shadow_chat():
    """Documents the upstream behaviour the upgrade path has to work around."""
    agent = plain_agent()
    assert "chat" in agent.__dict__, (
        "upstream no longer sets chat as an instance attribute — "
        "the pop() in HasyMemoryAgent/install may now be unnecessary"
    )


def test_upgrade_path_produces_a_working_memory_agent(store: MemoryStore):
    """Mirrors what install()'s wrapper does to a freshly built agent."""
    agent = plain_agent()

    agent.__class__ = HasyMemoryAgent
    agent.__dict__.pop("chat", None)
    agent._retriever = MemoryRetriever(store)
    agent._writer = MemoryWriter(store=store)
    agent._turn_counter = 0

    assert isinstance(agent, HasyMemoryAgent)
    assert "chat" not in agent.__dict__
    # The bound method must be HasyMemoryAgent's, not the shadowing closure.
    assert agent.chat.__func__ is HasyMemoryAgent.chat


def test_install_patches_the_agent_factory(monkeypatch):
    import hasy.memory.install as install_mod
    from src.open_llm_vtuber.agent.agent_factory import AgentFactory

    original = AgentFactory.create_agent
    monkeypatch.setattr(install_mod, "_installed", False)
    # Keep the test off disk and off the network.
    monkeypatch.setattr(
        install_mod, "_load_settings", lambda: {**install_mod.DEFAULTS, "extraction": "off"}
    )
    try:
        install_mod.install()
        assert AgentFactory.create_agent is not original, "factory was not patched"
    finally:
        AgentFactory.create_agent = original
        install_mod._installed = False


def test_install_respects_disabled_setting(monkeypatch):
    import hasy.memory.install as install_mod
    from src.open_llm_vtuber.agent.agent_factory import AgentFactory

    original = AgentFactory.create_agent
    monkeypatch.setattr(install_mod, "_installed", False)
    monkeypatch.setattr(
        install_mod, "_load_settings", lambda: {**install_mod.DEFAULTS, "enabled": False}
    )
    try:
        install_mod.install()
        assert AgentFactory.create_agent is original, "patched despite being disabled"
    finally:
        AgentFactory.create_agent = original
        install_mod._installed = False


def test_non_basic_agent_types_are_left_alone(monkeypatch, store: MemoryStore):
    import hasy.memory.install as install_mod
    from src.open_llm_vtuber.agent.agent_factory import AgentFactory

    original = AgentFactory.create_agent
    sentinel = object()
    monkeypatch.setattr(install_mod, "_installed", False)
    monkeypatch.setattr(
        install_mod, "_load_settings", lambda: {**install_mod.DEFAULTS, "extraction": "off"}
    )
    monkeypatch.setattr(
        AgentFactory, "create_agent", staticmethod(lambda choice, *a, **k: sentinel)
    )
    try:
        install_mod.install()
        assert AgentFactory.create_agent("letta_agent") is sentinel
    finally:
        AgentFactory.create_agent = original
        install_mod._installed = False
