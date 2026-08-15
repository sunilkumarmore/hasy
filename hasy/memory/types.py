"""Core value types for HASY memory.

Kept deliberately dumb: no persistence logic, no LLM calls. The store owns SQL,
the resolver owns identity decisions, these just carry data between them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Optional

EntityType = Literal["person", "project", "place", "thing"]

#: How an entity mention was resolved to an entity. Ordered cheapest-first;
#: the resolver stops at the first confident answer.
ResolutionMethod = Literal["exact", "alias", "embedding", "llm", "created"]


def utcnow() -> str:
    """ISO-8601 UTC timestamp. Stored as TEXT so SQLite sorts it lexically."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")


def normalize(name: str) -> str:
    """Fold a surface form to a blocking key.

    'Phani Meduri' / 'phani  meduri!' / 'PHANI MEDURI' all collapse to the same
    key, so exact-match blocking catches trivial variation before we spend an
    embedding or an LLM call on it.
    """
    if not name:
        return ""
    return _WS.sub(" ", _PUNCT.sub(" ", name.lower())).strip()


@dataclass
class Entity:
    canonical_name: str
    type: EntityType
    id: Optional[int] = None
    aliases: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    salience: float = 0.0
    last_referenced: Optional[str] = None
    created_at: Optional[str] = None


@dataclass
class Fact:
    """A single (entity, predicate) → value assertion, valid over a time range.

    `valid_to is None` means "currently true". Superseding a fact closes the old
    row rather than deleting it, so history stays queryable.
    """

    entity_id: int
    predicate: str
    value: str
    confidence: float = 1.0
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    source_turn_id: Optional[str] = None
    id: Optional[int] = None

    @property
    def is_current(self) -> bool:
        return self.valid_to is None


@dataclass
class Episode:
    """A summarised moment of conversation, the unit of vector recall."""

    summary: str
    ts: Optional[str] = None
    participants: list[str] = field(default_factory=list)
    entities_mentioned: list[int] = field(default_factory=list)
    valence: float = 0.0
    source_turn_id: Optional[str] = None
    id: Optional[int] = None


@dataclass
class Thread:
    """An open loop — the thing that makes proactive follow-up possible."""

    topic: str
    entity_ids: list[int] = field(default_factory=list)
    opened_at: Optional[str] = None
    last_touched: Optional[str] = None
    resolved_at: Optional[str] = None
    id: Optional[int] = None

    @property
    def is_open(self) -> bool:
        return self.resolved_at is None


@dataclass
class ResolutionDecision:
    """Why the resolver believed a mention was (or wasn't) a known entity.

    Persisted for every write so duplicate-entity bugs can be audited after the
    fact instead of guessed at.
    """

    surface_form: str
    entity_id: Optional[int]
    method: ResolutionMethod
    score: float
    rationale: str
    created: bool = False
    candidates: list[tuple[int, float]] = field(default_factory=list)
