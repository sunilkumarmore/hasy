"""SQLite-backed memory store for HASY.

Single file, sqlite-vec for vector recall. Owns all SQL; callers work in the
dataclasses from `types.py`.

Design commitments (from the Phase 3 spec):
- Facts are **temporally versioned**. Changing a fact closes the old row's
  `valid_to` and inserts a new one. Contradictions are never appended.
- Every identity decision is written to `resolution_log`, so duplicate entities
  can be audited rather than guessed at.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from loguru import logger

from .types import (
    Entity,
    Episode,
    Fact,
    ResolutionDecision,
    Thread,
    normalize,
    utcnow,
)

SCHEMA_VERSION = 1

_DDL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entities (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    type            TEXT NOT NULL CHECK (type IN ('person','project','place','thing')),
    canonical_name  TEXT NOT NULL,
    attributes      TEXT NOT NULL DEFAULT '{}',
    salience        REAL NOT NULL DEFAULT 0.0,
    last_referenced TEXT,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);

-- Aliases are a table, not a JSON array, so blocking is an index lookup.
CREATE TABLE IF NOT EXISTS aliases (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id  INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    alias      TEXT NOT NULL,
    normalized TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (entity_id, normalized)
);
CREATE INDEX IF NOT EXISTS idx_aliases_normalized ON aliases(normalized);

CREATE TABLE IF NOT EXISTS facts (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id      INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    predicate      TEXT NOT NULL,
    value          TEXT NOT NULL,
    confidence     REAL NOT NULL DEFAULT 1.0,
    valid_from     TEXT NOT NULL,
    valid_to       TEXT,              -- NULL == currently true
    source_turn_id TEXT,
    created_at     TEXT NOT NULL
);
-- Partial index: the hot query is "current value of this predicate".
CREATE INDEX IF NOT EXISTS idx_facts_current
    ON facts(entity_id, predicate) WHERE valid_to IS NULL;
CREATE INDEX IF NOT EXISTS idx_facts_entity ON facts(entity_id);

CREATE TABLE IF NOT EXISTS episodes (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                 TEXT NOT NULL,
    summary            TEXT NOT NULL,
    participants       TEXT NOT NULL DEFAULT '[]',
    entities_mentioned TEXT NOT NULL DEFAULT '[]',
    valence            REAL NOT NULL DEFAULT 0.0,
    source_turn_id     TEXT
);
CREATE INDEX IF NOT EXISTS idx_episodes_ts ON episodes(ts DESC);

CREATE TABLE IF NOT EXISTS threads (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    topic        TEXT NOT NULL,
    entity_ids   TEXT NOT NULL DEFAULT '[]',
    opened_at    TEXT NOT NULL,
    last_touched TEXT NOT NULL,
    resolved_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_threads_open
    ON threads(last_touched DESC) WHERE resolved_at IS NULL;

CREATE TABLE IF NOT EXISTS resolution_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           TEXT NOT NULL,
    surface_form TEXT NOT NULL,
    entity_id    INTEGER,
    method       TEXT NOT NULL,
    score        REAL NOT NULL,
    created      INTEGER NOT NULL DEFAULT 0,
    rationale    TEXT NOT NULL,
    candidates   TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_resolution_ts ON resolution_log(ts DESC);
"""


class MemoryStore:
    """Owns the SQLite file. Not thread-safe; use one per event loop."""

    def __init__(self, path: str | Path = "hasy_memory.db", embedding_dim: int = 384):
        self.path = str(path)
        self.embedding_dim = embedding_dim
        self._vec_enabled = False
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        # WAL lets the async write path commit without blocking reads.
        if self.path != ":memory:":
            self.conn.execute("PRAGMA journal_mode = WAL")
        self._load_vec()
        self._migrate()

    # ---------------------------------------------------------------- setup

    def _load_vec(self) -> None:
        """Load sqlite-vec. Degrade to non-vector operation if unavailable."""
        try:
            import sqlite_vec

            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
            self.conn.enable_load_extension(False)
            self._vec_enabled = True
        except Exception as e:
            logger.warning(
                f"HASY memory: sqlite-vec unavailable ({e}). "
                "Vector recall disabled; entity/fact/thread memory still works."
            )

    def _migrate(self) -> None:
        self.conn.executescript(_DDL)
        if self._vec_enabled:
            self.conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS episode_vectors "
                f"USING vec0(episode_id INTEGER PRIMARY KEY, "
                f"embedding float[{self.embedding_dim}])"
            )
        self.conn.execute(
            "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # -------------------------------------------------------------- entities

    def create_entity(self, entity: Entity) -> int:
        now = utcnow()
        cur = self.conn.execute(
            "INSERT INTO entities(type, canonical_name, attributes, salience,"
            " last_referenced, created_at) VALUES (?,?,?,?,?,?)",
            (
                entity.type,
                entity.canonical_name,
                json.dumps(entity.attributes),
                entity.salience,
                entity.last_referenced or now,
                now,
            ),
        )
        entity_id = int(cur.lastrowid)
        # The canonical name is always an alias of itself, so blocking finds it.
        for alias in {entity.canonical_name, *entity.aliases}:
            self.add_alias(entity_id, alias, commit=False)
        self.conn.commit()
        return entity_id

    def add_alias(self, entity_id: int, alias: str, commit: bool = True) -> None:
        if not alias or not alias.strip():
            return
        self.conn.execute(
            "INSERT OR IGNORE INTO aliases(entity_id, alias, normalized, created_at)"
            " VALUES (?,?,?,?)",
            (entity_id, alias, normalize(alias), utcnow()),
        )
        if commit:
            self.conn.commit()

    def get_entity(self, entity_id: int) -> Optional[Entity]:
        row = self.conn.execute(
            "SELECT * FROM entities WHERE id = ?", (entity_id,)
        ).fetchone()
        return self._row_to_entity(row) if row else None

    def find_by_alias(
        self, surface_form: str, type: Optional[str] = None
    ) -> list[Entity]:
        """Exact blocking lookup on the normalized alias. The cheap first pass."""
        key = normalize(surface_form)
        if not key:
            return []
        sql = (
            "SELECT DISTINCT e.* FROM entities e"
            " JOIN aliases a ON a.entity_id = e.id"
            " WHERE a.normalized = ?"
        )
        params: list[Any] = [key]
        if type:
            sql += " AND e.type = ?"
            params.append(type)
        return [self._row_to_entity(r) for r in self.conn.execute(sql, params)]

    def all_entities(self, type: Optional[str] = None) -> list[Entity]:
        sql = "SELECT * FROM entities"
        params: list[Any] = []
        if type:
            sql += " WHERE type = ?"
            params.append(type)
        return [self._row_to_entity(r) for r in self.conn.execute(sql, params)]

    def touch_entity(self, entity_id: int, salience_delta: float = 1.0) -> None:
        """Mark an entity as referenced now, bumping its salience."""
        self.conn.execute(
            "UPDATE entities SET last_referenced = ?, salience = salience + ?"
            " WHERE id = ?",
            (utcnow(), salience_delta, entity_id),
        )
        self.conn.commit()

    def get_aliases(self, entity_id: int) -> list[str]:
        return [
            r["alias"]
            for r in self.conn.execute(
                "SELECT alias FROM aliases WHERE entity_id = ? ORDER BY id",
                (entity_id,),
            )
        ]

    def _row_to_entity(self, row: sqlite3.Row) -> Entity:
        return Entity(
            id=row["id"],
            type=row["type"],
            canonical_name=row["canonical_name"],
            attributes=json.loads(row["attributes"]),
            salience=row["salience"],
            last_referenced=row["last_referenced"],
            created_at=row["created_at"],
            aliases=self.get_aliases(row["id"]),
        )

    # ----------------------------------------------------------------- facts

    def upsert_fact(self, fact: Fact, as_of: Optional[str] = None) -> int:
        """Assert a fact, superseding any conflicting current value.

        This is the temporal-versioning contract:
        - same predicate, same value  → no-op (returns the existing row id)
        - same predicate, new value   → close the old row, insert the new one
        - no current value            → plain insert

        `as_of` lets callers replay history out of order without lying about
        when the change happened.
        """
        now = as_of or utcnow()
        existing = self.conn.execute(
            "SELECT * FROM facts WHERE entity_id = ? AND predicate = ?"
            " AND valid_to IS NULL",
            (fact.entity_id, fact.predicate),
        ).fetchone()

        if existing is not None:
            if existing["value"] == fact.value:
                # Same claim again: keep the earlier valid_from, take the higher
                # confidence. Re-asserting is evidence, not a change.
                if fact.confidence > existing["confidence"]:
                    self.conn.execute(
                        "UPDATE facts SET confidence = ? WHERE id = ?",
                        (fact.confidence, existing["id"]),
                    )
                    self.conn.commit()
                return int(existing["id"])

            # A genuine change: close the old row rather than contradict it.
            self.conn.execute(
                "UPDATE facts SET valid_to = ? WHERE id = ?", (now, existing["id"])
            )
            logger.debug(
                f"HASY memory: superseded entity={fact.entity_id} "
                f"{fact.predicate}: '{existing['value']}' -> '{fact.value}'"
            )

        cur = self.conn.execute(
            "INSERT INTO facts(entity_id, predicate, value, confidence, valid_from,"
            " valid_to, source_turn_id, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (
                fact.entity_id,
                fact.predicate,
                fact.value,
                fact.confidence,
                fact.valid_from or now,
                None,
                fact.source_turn_id,
                utcnow(),
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def current_facts(self, entity_id: int) -> list[Fact]:
        return [
            self._row_to_fact(r)
            for r in self.conn.execute(
                "SELECT * FROM facts WHERE entity_id = ? AND valid_to IS NULL"
                " ORDER BY predicate",
                (entity_id,),
            )
        ]

    def fact_history(self, entity_id: int, predicate: str) -> list[Fact]:
        """Every value this predicate has held, oldest first."""
        return [
            self._row_to_fact(r)
            for r in self.conn.execute(
                "SELECT * FROM facts WHERE entity_id = ? AND predicate = ?"
                " ORDER BY valid_from, id",
                (entity_id, predicate),
            )
        ]

    def _row_to_fact(self, row: sqlite3.Row) -> Fact:
        return Fact(
            id=row["id"],
            entity_id=row["entity_id"],
            predicate=row["predicate"],
            value=row["value"],
            confidence=row["confidence"],
            valid_from=row["valid_from"],
            valid_to=row["valid_to"],
            source_turn_id=row["source_turn_id"],
        )

    # -------------------------------------------------------------- episodes

    def add_episode(
        self, episode: Episode, embedding: Optional[Sequence[float]] = None
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO episodes(ts, summary, participants, entities_mentioned,"
            " valence, source_turn_id) VALUES (?,?,?,?,?,?)",
            (
                episode.ts or utcnow(),
                episode.summary,
                json.dumps(episode.participants),
                json.dumps(episode.entities_mentioned),
                episode.valence,
                episode.source_turn_id,
            ),
        )
        episode_id = int(cur.lastrowid)
        if embedding is not None and self._vec_enabled:
            import sqlite_vec

            self.conn.execute(
                "INSERT INTO episode_vectors(episode_id, embedding) VALUES (?, ?)",
                (episode_id, sqlite_vec.serialize_float32(list(embedding))),
            )
        self.conn.commit()
        return episode_id

    def search_episodes(
        self, embedding: Sequence[float], limit: int = 5
    ) -> list[tuple[Episode, float]]:
        """Vector recall over episodes. Returns (episode, distance), nearest first."""
        if not self._vec_enabled:
            return []
        import sqlite_vec

        rows = self.conn.execute(
            "SELECT episode_id, distance FROM episode_vectors"
            " WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (sqlite_vec.serialize_float32(list(embedding)), limit),
        ).fetchall()
        out: list[tuple[Episode, float]] = []
        for r in rows:
            ep = self.get_episode(r["episode_id"])
            if ep:
                out.append((ep, float(r["distance"])))
        return out

    def get_episode(self, episode_id: int) -> Optional[Episode]:
        row = self.conn.execute(
            "SELECT * FROM episodes WHERE id = ?", (episode_id,)
        ).fetchone()
        if not row:
            return None
        return Episode(
            id=row["id"],
            ts=row["ts"],
            summary=row["summary"],
            participants=json.loads(row["participants"]),
            entities_mentioned=json.loads(row["entities_mentioned"]),
            valence=row["valence"],
            source_turn_id=row["source_turn_id"],
        )

    def recent_episodes(self, limit: int = 5) -> list[Episode]:
        return [
            self.get_episode(r["id"])  # type: ignore[misc]
            for r in self.conn.execute(
                "SELECT id FROM episodes ORDER BY ts DESC LIMIT ?", (limit,)
            )
        ]

    # --------------------------------------------------------------- threads

    def open_thread(self, thread: Thread) -> int:
        now = utcnow()
        cur = self.conn.execute(
            "INSERT INTO threads(topic, entity_ids, opened_at, last_touched,"
            " resolved_at) VALUES (?,?,?,?,NULL)",
            (
                thread.topic,
                json.dumps(thread.entity_ids),
                thread.opened_at or now,
                thread.last_touched or now,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def touch_thread(self, thread_id: int) -> None:
        self.conn.execute(
            "UPDATE threads SET last_touched = ? WHERE id = ?", (utcnow(), thread_id)
        )
        self.conn.commit()

    def resolve_thread(self, thread_id: int) -> None:
        self.conn.execute(
            "UPDATE threads SET resolved_at = ? WHERE id = ? AND resolved_at IS NULL",
            (utcnow(), thread_id),
        )
        self.conn.commit()

    def open_threads(self, limit: int = 10) -> list[Thread]:
        """Unresolved loops, most recently touched first — the proactive-speak feed."""
        return [
            Thread(
                id=r["id"],
                topic=r["topic"],
                entity_ids=json.loads(r["entity_ids"]),
                opened_at=r["opened_at"],
                last_touched=r["last_touched"],
                resolved_at=r["resolved_at"],
            )
            for r in self.conn.execute(
                "SELECT * FROM threads WHERE resolved_at IS NULL"
                " ORDER BY last_touched DESC LIMIT ?",
                (limit,),
            )
        ]

    # ----------------------------------------------------------------- audit

    def log_resolution(self, decision: ResolutionDecision) -> None:
        self.conn.execute(
            "INSERT INTO resolution_log(ts, surface_form, entity_id, method, score,"
            " created, rationale, candidates) VALUES (?,?,?,?,?,?,?,?)",
            (
                utcnow(),
                decision.surface_form,
                decision.entity_id,
                decision.method,
                decision.score,
                int(decision.created),
                decision.rationale,
                json.dumps(decision.candidates),
            ),
        )
        self.conn.commit()

    def resolution_log(self, limit: int = 50) -> list[dict[str, Any]]:
        return [
            dict(r)
            for r in self.conn.execute(
                "SELECT * FROM resolution_log ORDER BY ts DESC, id DESC LIMIT ?",
                (limit,),
            )
        ]
