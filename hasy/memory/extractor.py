"""Turn → memory candidates.

The write path's first step: a cheap model call that reads one conversation turn
and proposes entities, facts, thread updates, and an episode summary. Nothing
here writes to the store — resolution happens afterwards, in `resolver.py`, so
that identity decisions are made against what we already know.

The `Extractor` protocol keeps the model call injectable: tests use
`StubExtractor` and never need an API key or a network.
"""

from __future__ import annotations

import json
from typing import Literal, Optional, Protocol, Sequence

from loguru import logger
from pydantic import BaseModel, Field

from .types import EntityType

#: Cheap by design. This runs once per turn on the async write path, never on
#: the latency path, but an always-on companion makes a lot of turns.
DEFAULT_EXTRACTION_MODEL = "claude-haiku-4-5"


class MentionOut(BaseModel):
    surface_form: str = Field(description="The exact phrase used, e.g. 'Phani' or 'the architect'")
    type: EntityType = Field(description="person, project, place, or thing")


class FactOut(BaseModel):
    subject: str = Field(description="Surface form of the entity this fact is about")
    predicate: str = Field(description="Short snake_case attribute name, e.g. 'role' or 'employer'")
    value: str = Field(description="The value of the attribute")
    confidence: float = Field(description="0.0-1.0 confidence this fact was actually stated")


class ThreadOut(BaseModel):
    topic: str = Field(description="Short description of the open loop or unresolved topic")
    status: Literal["open", "resolved"] = Field(description="Whether this topic is still open")


class ExtractionOut(BaseModel):
    """What one turn contributed to memory."""

    entities: list[MentionOut] = Field(default_factory=list)
    facts: list[FactOut] = Field(default_factory=list)
    threads: list[ThreadOut] = Field(default_factory=list)
    episode_summary: str = Field(default="", description="One sentence summarising this exchange")
    valence: float = Field(default=0.0, description="-1.0 negative to 1.0 positive emotional tone")


EXTRACTION_SYSTEM = """\
You extract durable memory from a single turn of spoken conversation between a \
human and their desk companion.

Extract only what was actually said. Do not infer, embellish, or invent detail.
If the turn is small talk with nothing worth remembering, return empty lists and \
a brief summary.

- entities: people, projects, places, or things referred to. Use the exact \
surface form spoken, including descriptions like "the platform architect".
- facts: durable attributes only. "Phani is the architect" is a fact; "Phani is \
in a meeting right now" is not. Use short snake_case predicates.
- threads: unresolved topics or open loops worth following up on later. Mark \
status "resolved" only if the turn closes a loop.
- episode_summary: one sentence, past tense, describing what was discussed.
- valence: emotional tone of the exchange, -1.0 to 1.0.
"""


class Extractor(Protocol):
    async def extract(self, user_text: str, assistant_text: str) -> ExtractionOut:
        """Propose memory candidates from one turn."""
        ...


class StubExtractor:
    """Returns preset results. For tests and for running with extraction off."""

    def __init__(self, results: Optional[Sequence[ExtractionOut]] = None):
        self._results = list(results or [])
        self.calls: list[tuple[str, str]] = []

    async def extract(self, user_text: str, assistant_text: str) -> ExtractionOut:
        self.calls.append((user_text, assistant_text))
        if not self._results:
            return ExtractionOut()
        return self._results[min(len(self.calls) - 1, len(self._results) - 1)]


class ClaudeExtractor:
    """Structured extraction via the Anthropic API.

    Uses `messages.parse()` so the response is schema-validated by the SDK
    rather than hand-parsed out of free text.
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_EXTRACTION_MODEL,
        base_url: Optional[str] = None,
        max_tokens: int = 4096,
    ):
        from anthropic import AsyncAnthropic

        self.client = AsyncAnthropic(api_key=api_key, base_url=base_url or None)
        self.model = model
        self.max_tokens = max_tokens

    async def extract(self, user_text: str, assistant_text: str) -> ExtractionOut:
        turn = (
            f"Human said:\n{user_text.strip()}\n\n"
            f"Companion replied:\n{assistant_text.strip()}"
        )
        try:
            response = await self.client.messages.parse(
                model=self.model,
                max_tokens=self.max_tokens,
                system=EXTRACTION_SYSTEM,
                messages=[{"role": "user", "content": turn}],
                output_format=ExtractionOut,
            )
            result = response.parsed_output
            if result is None:
                logger.warning("HASY memory: extractor returned no parsed output")
                return ExtractionOut()
            return result
        except Exception as e:
            # The write path must never take down a conversation.
            logger.warning(f"HASY memory: extraction failed ({e})")
            return ExtractionOut()


ADJUDICATION_SYSTEM = """\
You decide whether a mention refers to someone already known.

You are given a surface form, the conversational context it appeared in, and a \
list of candidate entities with what is known about each.

Reply with the id of the matching candidate, or null if the mention refers to \
someone or something not in the list. Prefer null over a wrong match: creating \
a duplicate is recoverable, merging two different people is not.
"""


class AdjudicationOut(BaseModel):
    entity_id: Optional[int] = Field(description="Matching candidate id, or null for none of them")
    reason: str = Field(description="One short sentence explaining the decision")


class ClaudeAdjudicator:
    """Resolves ambiguous mentions with a model call.

    Called only when blocking and embedding similarity are both inconclusive,
    and always on the async write path — never during a response.
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_EXTRACTION_MODEL,
        base_url: Optional[str] = None,
    ):
        from anthropic import AsyncAnthropic

        self.client = AsyncAnthropic(api_key=api_key, base_url=base_url or None)
        self.model = model

    async def adjudicate(
        self, surface_form: str, candidates: Sequence[dict], context: str
    ) -> AdjudicationOut:
        prompt = (
            f"Mention: {surface_form!r}\n"
            f"Context: {context or '(none)'}\n\n"
            f"Candidates:\n{json.dumps(list(candidates), indent=2)}"
        )
        try:
            response = await self.client.messages.parse(
                model=self.model,
                max_tokens=1024,
                system=ADJUDICATION_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                output_format=AdjudicationOut,
            )
            return response.parsed_output or AdjudicationOut(
                entity_id=None, reason="no parsed output"
            )
        except Exception as e:
            logger.warning(f"HASY memory: adjudication failed ({e})")
            return AdjudicationOut(entity_id=None, reason=f"adjudication error: {e}")
