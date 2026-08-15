"""Wire HASY's memory agent into upstream without editing upstream.

Upstream registers agent types in an if/elif chain in `agent_factory.py`. That
is the one place the Phase 3 spec's "subclass, don't modify" rule genuinely
collides with upstream's shape — so instead of editing that file, we wrap
`AgentFactory.create_agent` from here, the same monkeypatch approach used by
`hasy/latency.py`.

Config lives in an optional `hasy.yaml` at the repo root (git-ignored), so no
upstream config schema is touched:

    memory:
      enabled: true
      db_path: hasy_memory.db
      token_budget: 600
      extraction: claude          # 'claude' | 'off'
      extraction_model: claude-haiku-4-5
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from .embeddings import HashEmbedder
from .resolver import EntityResolver
from .retrieval import MemoryRetriever
from .store import MemoryStore
from .writer import MemoryWriter

_installed = False

DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "db_path": "hasy_memory.db",
    "token_budget": 600,
    "extraction": "claude",
    "extraction_model": "claude-haiku-4-5",
}


def _load_settings() -> dict[str, Any]:
    settings = dict(DEFAULTS)
    path = Path("hasy.yaml")
    if path.exists():
        try:
            import yaml

            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            settings.update((data.get("memory") or {}))
        except Exception as e:
            logger.warning(f"HASY memory: could not read hasy.yaml ({e}); using defaults")
    return settings


def _claude_credentials() -> tuple[Optional[str], Optional[str]]:
    """Reuse the Claude key already configured for the main LLM.

    Read straight from conf.yaml rather than adding a second place to put a
    secret. Falls back to ANTHROPIC_API_KEY.
    """
    try:
        import yaml

        conf = yaml.safe_load(Path("conf.yaml").read_text(encoding="utf-8")) or {}
        claude = (
            conf.get("character_config", {})
            .get("agent_config", {})
            .get("llm_configs", {})
            .get("claude_llm", {})
        )
        key = claude.get("llm_api_key")
        base_url = claude.get("base_url")
        if key and not key.startswith("PASTE_"):
            return key, base_url
    except Exception as e:
        logger.debug(f"HASY memory: could not read conf.yaml for credentials ({e})")
    return os.environ.get("ANTHROPIC_API_KEY"), None


def build_memory(settings: Optional[dict[str, Any]] = None):
    """Construct the store/retriever/writer trio. Returns (retriever, writer)."""
    settings = settings or _load_settings()
    store = MemoryStore(settings["db_path"])
    embedder = HashEmbedder()

    extractor = None
    adjudicator = None
    if settings.get("extraction") == "claude":
        key, base_url = _claude_credentials()
        if key:
            from .extractor import ClaudeExtractor

            extractor = ClaudeExtractor(
                api_key=key, model=settings["extraction_model"], base_url=base_url
            )
            logger.info(
                f"HASY memory: extraction via {settings['extraction_model']} (async write path)"
            )
        else:
            logger.warning(
                "HASY memory: no Claude API key found; extraction disabled. "
                "Memory will retrieve but never learn."
            )

    retriever = MemoryRetriever(
        store, embedder=embedder, token_budget=settings["token_budget"]
    )
    writer = MemoryWriter(
        store=store,
        resolver=EntityResolver(store, embedder=embedder, adjudicator=adjudicator),
        extractor=extractor,
        embedder=embedder,
    ) if extractor else None

    return retriever, writer


def install() -> None:
    """Patch AgentFactory so `basic_memory_agent` builds HasyMemoryAgent instead.

    NOTE (upstream boundary): this replaces the agent that upstream's config
    name selects, rather than registering a new name, precisely so that no
    upstream file needs editing. Setting `extraction: off` in hasy.yaml, or
    `memory.enabled: false`, restores plain upstream behaviour.
    """
    global _installed
    if _installed:
        return

    settings = _load_settings()
    if not settings.get("enabled", True):
        logger.info("HASY memory: disabled in hasy.yaml; using upstream agent.")
        return

    try:
        from src.open_llm_vtuber.agent.agent_factory import AgentFactory
    except Exception as e:
        logger.warning(f"HASY memory: cannot patch AgentFactory ({e})")
        return

    from ..agent.hasy_memory_agent import HasyMemoryAgent

    original = AgentFactory.create_agent

    def wrapper(conversation_agent_choice: str, *args, **kwargs):
        agent = original(conversation_agent_choice, *args, **kwargs)
        if conversation_agent_choice != "basic_memory_agent":
            return agent
        try:
            retriever, writer = build_memory(settings)
            agent.__class__ = HasyMemoryAgent
            # BasicMemoryAgent.__init__ set `chat` as an instance attribute;
            # it would shadow HasyMemoryAgent.chat and silently disable memory.
            agent.__dict__.pop("chat", None)
            agent._retriever = retriever
            agent._writer = writer
            agent._turn_counter = 0
            logger.info("HASY memory: agent upgraded to HasyMemoryAgent.")
        except Exception as e:
            logger.error(f"HASY memory: failed to attach memory ({e}); using plain agent")
        return agent

    AgentFactory.create_agent = staticmethod(wrapper)
    _installed = True
    logger.info("HASY memory installed.")
