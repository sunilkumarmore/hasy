# Phase 3 — HASY memory layer

Built after the Phase 2 verdict of **insufficient** recorded in
[`memory-evaluation.md`](memory-evaluation.md). Lives entirely in HASY-owned
code under `hasy/memory/` and `hasy/agent/`; upstream's ASR, TTS, Live2D,
interruption, and transport code is untouched.

## Shape

```
turn ─┬─ READ  (on the latency path, ~2.4 ms)
      │    alias blocking → current facts → vector recall → open threads
      │    → composed into the system prompt under a token budget
      │
      └─ WRITE (async, after the response is already streaming)
           extract → RESOLVE → commit
```

**Resolution happens before the write.** That ordering is the entire design:
writing first and deduplicating later is what produces three separate "Phani"s.

| Module | Role |
|---|---|
| `types.py` | Entity / Fact / Episode / Thread / ResolutionDecision; surface-form normalization |
| `store.py` | SQLite + sqlite-vec. All SQL lives here |
| `embeddings.py` | `Embedder` protocol + dependency-free `HashEmbedder` |
| `resolver.py` | Identity decisions, cheapest-first |
| `extractor.py` | Turn → candidates, via a cheap model call |
| `retrieval.py` | The read path |
| `writer.py` | The async write path |
| `install.py` | Wiring, by monkeypatch |
| `../agent/hasy_memory_agent.py` | `BasicMemoryAgent` subclass |

## The four evaluation questions, now answered

These are the questions Phase 2 found upstream could not answer at all.

| Question | How this design answers it |
|---|---|
| Duplicate records for the same entity? | Resolution runs *before* every write: exact alias blocking → embedding similarity → LLM adjudication → create. Adjudicated matches are **learned as aliases**, so an expensive decision is paid once. |
| Changed fact — supersede or accumulate? | `upsert_fact` closes the old row's `valid_to` and inserts a new one. There is never more than one current value for an (entity, predicate). History stays queryable. |
| Right context, or just similar text? | Hybrid: entities mentioned in the utterance drive a fact lookup; vector search covers episodic recall; open threads are surfaced by recency. Not top-k cosine. |
| Unresolved topics across sessions? | The `threads` table — opened, touched, and resolved by the write path. This is what makes proactive follow-up possible in Phase 4. |

## Measured retrieval cost

Spec ceiling: **~120 ms**. Measured on 100 entities / 200 facts / 200 episodes /
10 open threads — well past a year of desk use:

```
retrieval mean=2.4 ms   p50=2.3 ms   max=3.2 ms
```

**~37× headroom.** Retrieval is not a latency concern at this scale, and
`MemoryContext.elapsed_ms` reports the real number every turn, with a warning
logged if the budget is ever exceeded. It is measured, not assumed.

Two reasons it is this cheap, both deliberate:
- The read path makes **no model call** — extraction and adjudication are on the
  async write path only.
- Entity lookup is indexed alias blocking, not a vector scan.

## Auditability

Every identity decision is written to `resolution_log` with the method
(`exact` / `alias` / `embedding` / `llm` / `created`), a similarity score, the
candidates considered, and a plain-English rationale:

```sql
SELECT ts, surface_form, entity_id, method, created, rationale
FROM resolution_log ORDER BY ts DESC LIMIT 20;
```

If duplicates ever appear, this says exactly why — no guessing.

## Failure behaviour

Memory is an enhancement, never a dependency:
- Retrieval failure → logged, HASY talks without context.
- Extraction failure → logged, the turn is simply not learned.
- Ambiguity with no adjudicator → **refuses to guess** rather than forking an
  identity. An unresolved mention is recoverable; a merged identity is not.
- Write failures never propagate into the response path.

## Upstream boundary — one flagged deviation

Upstream registers agent types in an `if/elif` chain in `agent_factory.py`.
Registering a new name requires editing that file, which the boundary rule in
[`../CLAUDE.md`](../CLAUDE.md) exists to prevent.

**Resolution:** `install.py` wraps `AgentFactory.create_agent` by monkeypatch —
the same technique as `hasy/latency.py` — and upgrades the agent that
`basic_memory_agent` builds. **No upstream file is modified.** The cost is that
HASY replaces the agent upstream's config name selects rather than adding a new
name; `memory.enabled: false` in `hasy.yaml` restores upstream behaviour exactly.

## Configuration

See [`../hasy.example.yaml`](../hasy.example.yaml). Copy it to `hasy.yaml`
(git-ignored) to override. An absent `hasy.yaml` is a working default.

The Claude API key is **read from `conf.yaml`** so there is only ever one place
a secret lives; `ANTHROPIC_API_KEY` is the fallback.

## Tests

52 tests, no API key and no network required — the extractor and adjudicator are
injected protocols, and the embedder is dependency-free.

Adversarial cases from the spec: one person via three aliases (including a
description that shares no letters with the name), two different people sharing
a name, contradictory facts arriving out of order, ambiguity with no
adjudicator, and a broken store.

Two bugs the tests caught before they shipped, both worth remembering:

1. **`"the platform architect"` scored 0.066 against `"Phani Meduri"`** — below
   the candidate floor, so it silently created a duplicate. Entities are now
   scored against their **facts**, not just their names, and when nothing
   clears the floor the adjudicator is asked before creating.
2. **`BasicMemoryAgent.__init__` assigns `self.chat` as an instance attribute**
   (via `_set_llm`), which shadows a subclass's `chat()`. The entire memory
   layer would have silently never run while everything appeared to work.
   `test_chat_override_actually_runs` fails if this regresses, and
   `test_upstream_agent_really_does_shadow_chat` fails if upstream ever stops
   doing it — so the workaround gets removed rather than lingering forever.
