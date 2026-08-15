# Phase 2 — Evaluation of the existing memory layer

## VERDICT: INSUFFICIENT — Phase 3 authorised (2026-08-14)

The human reviewed the findings below, including the assessment of Letta as the
"don't reinvent it" alternative, and chose to build the custom memory layer.
**The Phase 2 → 3 gate is satisfied.** Phase 3 work proceeds on the
`hasy-memory` branch.

Reasoning, in short:
- `basic_memory_agent` is transcript replay, not memory — it cannot answer any
  of the four evaluation questions because it has no entity, fact, or thread
  model at all.
- Letta is a capable memory system, but *upstream's adapter to it* is thin: a
  no-op `handle_interrupt`, silently dropped images, and no persona plumbing.
  Adopting it would mean repairing that adapter — i.e. upstream edits, which the
  boundary rule in `CLAUDE.md` exists to avoid — and it regresses Phase 4 work
  (barge-in, face tracking) before that work has started.
- A custom layer built as an **Agent subclass** leaves the ASR/TTS/interruption/
  vision/transport pipeline untouched, which is the cheaper architectural bet.

Findings below were written from reading the vendored snapshot; no code was
written during Phase 2.

---

## What actually ships upstream

Three agent types are registered in `agent/agent_factory.py`. Only one is both
real and self-contained.

### 1. `basic_memory_agent` — the default, and what HASY runs today

| Aspect | Reality |
|---|---|
| Storage (working) | A plain Python list in RAM: `self._memory`, `{role, content}` dicts (`basic_memory_agent.py:55`) |
| Storage (persistent) | JSON files at `chat_history/<conf_uid>/<history_uid>.json`, written by `chat_history_manager.py` |
| Retrieval | **None.** The full message list is replayed into the prompt every turn |
| Entity model | **None** |
| Embeddings / vector search | **None.** No sqlite, no embedding model, no vector index anywhere in `src/` |
| Fact versioning | **None** |
| Cross-session recall | Only if the client explicitly requests a `history_uid`. `ServiceContext.history_uid` defaults to `""` (`service_context.py:70`), so a fresh session starts blank |

It is **conversation-transcript replay**, not a memory system. That is not a
criticism of the code — it does what it says — but it is not what "long-term
memory" implies.

### 2. `letta_agent` — a client, not an implementation

`agents/letta_agent.py` is ~130 lines of HTTP client against a **Letta server the
user must run separately** (default `localhost:8283`). All memory behaviour lives
in Letta. `set_memory_from_history()` is a deliberate no-op, commented:
*"The Letta Server automatically stores historical messages, so this part is not
needed."*

This is the credible "don't reinvent it" option — Letta is a mature memory system
with entity and fact handling already built. The cost is an external service
dependency in the critical path of a desk appliance.

### 3. `mem0_agent` — broken

`agents/mem0_llm.py` is **0 bytes**, but `agent_factory.py:89` does
`from .agents.mem0_llm import LLM as Mem0LLM`. Selecting `mem0_agent` raises
`ImportError`. Treat this agent type as non-existent.

---

## The four evaluation questions

These were intended to be answered after a week of real use. Three are settled by
reading the code; a week of logging would add nothing.

| Question | Answer |
|---|---|
| Duplicate records for the same entity ("Phani", "Phani Meduri", "the platform architect")? | **Moot** — no entity records exist to duplicate. Raw transcript only. |
| Does a changed fact supersede, or accumulate a contradiction? | **Accumulates.** No fact model; both statements remain in the transcript and the LLM sees both. |
| Does retrieval surface the right context, or just similar text? | **Neither** — there is no retrieval. Full replay. Perfect recall *within* a session, zero *across* sessions. |
| Anything tracking unresolved topics across sessions? | **No.** Nothing resembling threads/open loops. |

---

## Finding not on the original list: replay degrades latency over time

Because every turn re-sends the entire conversation, prompt size grows linearly
with session length. That directly inflates time-to-first-token — the stage
measured at ~719 ms mean in the Phase 1 baseline
([`latency-backlog.md`](latency-backlog.md)) — and will eventually hit the model's
context limit.

This is a present-tense problem, not a future one, and it is independent of
whichever memory direction is chosen. Any solution that replaces full replay with
bounded retrieval also *improves* latency.

---

## Recommendation

Reading the code has already answered the questions the week-long harness was
meant to answer, so **waiting a week does not improve this decision**. The real
fork is:

1. **Evaluate Letta** — honours "don't reinvent it". If it resolves entities and
   supersedes facts well, Phase 3 is skipped entirely. Cost: an external server
   the desk appliance depends on.
2. **Declare "insufficient" and build Phase 3** — the SQLite + `sqlite-vec`
   entity-resolved design. Full control, no external service, more code to own.
3. **Defer** — accept transcript replay, move to Phase 4 (presence), revisit
   after living with HASY daily.

**Pending the human's decision. Do not start Phase 3 on the strength of this
document alone — the gate requires an explicit verdict.**
