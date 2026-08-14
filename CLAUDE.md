# HASY — Desk Hologram AI Companion

HASY is a desk-resident AI companion that renders into a Pepper's Ghost holographic box.
It is built **on top of a vendored snapshot of [Open-LLM-VTuber](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber)** (MIT).

- **Runtime:** Open-LLM-VTuber (this repo is a point-in-time copy; we do **not** pull upstream updates).
- **Brain:** Claude API (Anthropic), streaming. LLM provider `claude_llm` is already supported upstream.
- **Display (v1):** a tablet lying **screen-down** on a plywood Pepper's Ghost box. Render-only surface — its mic/speaker/touch are unusable.
- **Brain host:** the laptop. Mic/speaker on the laptop (ReSpeaker Mic Array v2.0, USB, beamforming + AEC).
- **Render target:** pure `#000000` fullscreen, no UI chrome, no cursor.

## Hard constraints (do not regress)

- **Latency budget:** wake-word → first spoken syllable **< 900 ms**. Instrumented from Phase 1.
- Everything the avatar renders sits on `#000000` — no taskbar, no cursor, no browser chrome.
- Prefer **config changes and subclasses over new code**. Write as little novel code as possible.

## THE UPSTREAM BOUNDARY (read before editing anything)

Treat everything that shipped with the vendored snapshot as **read-mostly upstream code**:
`src/open_llm_vtuber/**`, `run_server.py`, `upgrade*.py`, `config_templates/**`, `prompts/**`, etc.

**Do not modify upstream files if a config file, a subclass, or a new module will do.**
- Configuration → user-owned `conf.yaml` (copied from `config_templates/conf.default.yaml`, git-ignored).
- New behavior → HASY-owned code under **`hasy/`** (top-level package, importable because the server runs from repo root).
- If a change *genuinely* cannot be done without touching an upstream file (e.g. registering a brand-new
  agent type in `agent/agent_factory.py`), **stop and flag it** in the PR/commit and keep the edit as small
  and localized as possible. This is the "abstraction leaked" signal called out in the project plan.

Upstream's own architecture notes are preserved at [`docs/CLAUDE.upstream.md`](docs/CLAUDE.upstream.md).

## Where things live

- `src/open_llm_vtuber/` — upstream runtime (read-mostly).
- `hasy/` — all HASY custom code (latency instrumentation, custom agent/memory when Phase 3 lands, presence hooks). *Created as phases need it.*
- `conf.yaml` — user config with secrets. **Git-ignored. Never commit.**
- `.env` — secrets (API keys). **Git-ignored. Never commit.**
- `docs/` — HASY docs, including the preserved upstream CLAUDE.md.

## Secrets

- API keys live only in `conf.yaml` or `.env` on the local machine.
- **Never** put keys or secrets in cloud-session environment variables — they are visible to anyone in that environment.

## Git workflow

- **Never commit or push to `main`.** Work on:
  - `hasy-integration` — **local** sessions, hardware-touching phases (1, 4·webcam, 5, 6).
  - `hasy-memory` — **cloud** sessions, the memory layer (2, 3). Reviewed as PRs.
- Commit after each **working increment**, not at the end of a phase.
- Conventional commits: `feat:`, `fix:`, `refactor:`, `chore:`, `test:`.
- Push after each commit so diffs are reviewable from a phone.
- **Never force-push. Never rewrite history.** Never hard-reset shared branches. Never delete branches without explicit ask.
- If a change belongs on `main`, open a **PR** — do not merge it yourself.
- Merge the two branches "at the desk" (locally), deliberately.

`.claude/settings.json` encodes allow rules for read/commit/push-to-working-branches and deny rules for
main-push / force-push / hard-reset / branch-deletion. These are belt-and-suspenders; the real guardrail is
GitHub branch protection on `main` (enable it on the remote).

## Phase gates (from the project plan — enforce these)

- **Phase 1 → 2:** a voice conversation must work end-to-end and the human must have seen the latency numbers.
- **Phase 2 → 3:** build a custom memory layer **only** if the Phase 2 written verdict is "insufficient." If asked
  to build it before that verdict exists, **refuse and cite this gate.**
- **Latency first:** if latency is failing budget, fix latency before adding features.
- **Phase 6 (Pi / HDMI cabinet):** deferred. Do not start until the human explicitly says so *and* has lived with
  the tablet build for ≥2 weeks. If raised early, refuse and point back here. If Phase 6 ever requires changes
  above the display layer, stop — the abstraction leaked.

## Running (upstream)

- Install deps: `uv sync`
- Run server: `uv run run_server.py` (add `--verbose` for debug logs)
- Config: `conf.yaml` (copy from `config_templates/conf.default.yaml`)
- The web frontend is a git submodule (`frontend/`) — not yet populated in this snapshot; needed to render/interact.
