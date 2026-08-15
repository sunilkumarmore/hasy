# Phase 4 — Presence

The things that make HASY feel like something in the room rather than a chatbot
with a face. Ordered as the project plan orders them: value per line of code.

Config: [`../hasy.example.yaml`](../hasy.example.yaml) → copy to `hasy.yaml`.

---

## 1. Face tracking — turn toward the human

**Upstream had nothing.** No vision module, no camera capture, no face position.

**What HASY adds:** laptop webcam → OpenCV Haar cascade → largest face →
normalized gaze in `[-1, 1]` → the browser → `setDragging(x, y)` on the Live2D
model. That is the same call the vendored bundle already makes on `pointermove`,
so the renderer needs no changes at all.

The transport seam was the interesting part. The frontend is a minified bundle
with no "look here" message type — but it exposes `window.getLive2DManager()`
and `window.getLAppAdapter()`, which is enough. So HASY adds its own routes to
the same FastAPI app and ships a companion script:

| Route | Purpose |
|---|---|
| `GET /hasy/health` | Status JSON — checkable from a phone. Also Phase 5. |
| `GET /hasy/presence.js` | The companion script |
| `WS /hasy/presence-ws` | Gaze targets, server → browser |

### Calibration — you will have to do this physically

The webcam and the hologram box are in **different places**, so where the camera
sees you is not where the avatar should look. That is what `offset_*` and
`gain_*` are for, and why they are config rather than code:

```bash
uv run python scripts/check_face_tracking.py --preview
```

Sit where you actually sit. Aim for gaze ≈ `(0.0, 0.0)` looking straight at the
box. Then:

| Symptom | Knob |
|---|---|
| Consistently looks left/right of you | `offset_x` |
| Consistently looks above/below you | `offset_y` |
| Barely moves | raise `gain_x` / `gain_y` |
| Moves the wrong way | `invert_x` / `invert_y` |
| Twitchy | raise `smoothing` |
| Laggy | lower `smoothing` |

Put the values you settle on into `hasy.yaml`, then set `enabled: true`.

**Not verified on hardware.** The geometry is unit-tested; the camera path has
never run here. Expect to spend a few minutes with the calibration script.

---

## 2. Proactive speaking — driven by unresolved threads

**Upstream has proactive speech, but its trigger source is the client.** The
frontend sends `ai-speak-signal` and the server loads a static "say something"
prompt. Nothing decided *when* it was worth speaking, or *what about*.

**What HASY adds** is that decision, wired to the `threads` table from Phase 3:

```
every 60s → policy.decide(now, open_threads, last_interaction, last_proactive)
          → pick the most neglected thread
          → stash it on the agent
          → fire the same ai-speak-signal upstream already understands
```

**Silence is the default.** Every rule is a reason *not* to speak:

| Rule | Why |
|---|---|
| Quiet hours (default 09:00–22:00) | Don't talk to an empty room at 3am |
| ≥5 min since the last exchange | Don't interrupt an actual conversation |
| ≥1 h since the last proactive line | Don't nag |
| Thread ≥30 min old | Don't parrot back something said minutes ago |
| Thread ≤14 days old | Stop raising things that have gone cold |
| A thread must exist | Nothing to say → say nothing |

It picks the **most neglected** open thread, not the most recent — the loop at
risk of being forgotten is the one that has gone quietest.

The chosen thread is injected as a permissive steer ("you *may* bring this up"),
not a script, because an unprompted line that sounds recited is worse than none.
Proactive turns are **not** written to memory — upstream marks them
`skip_memory`, and storing "human said: please say something" would teach HASY
to talk to itself.

---

## 3. Idle behaviour — already upstream

**Verified rather than rebuilt.** The vendored frontend already handles Live2D
motion groups and calls `startRandomMotion`, and the shipped models include idle
motions. The avatar breathes and shifts between turns without HASY doing
anything.

Face tracking composes with this rather than fighting it: `setDragging` drives
the head/eye parameters while idle motions drive the body, so both run at once.

**Nothing was written for this item.** If the avatar ever does look frozen, the
cause is the renderer, not HASY.

---

## 4. Barge-in — a hardware test, not code

Upstream's interruption path exists and is two-sided:

- the frontend can send `interrupt-signal`
- the server's VAD emits `<|PAUSE|>` mid-playback and pushes
  `{"type": "control", "text": "interrupt"}` to the client

The claim in upstream's docs is that this works **without headphones** — i.e.
the AI doesn't interrupt itself on its own voice. Whether that survives your
ReSpeaker's AEC is an empirical question about a specific microphone in a
specific room, and it cannot be answered from code.

### Adversarial test procedure

Run the server, then work down this list. Anything that fails is worth knowing
*before* the box is built.

| # | Test | Pass looks like |
|---|---|---|
| 1 | Ask a question with a long answer; stay silent | HASY finishes uninterrupted. **Failing here means it is hearing itself** — the AEC is not holding. |
| 2 | Interrupt mid-sentence at normal volume | Stops within ~1 s; your words are transcribed |
| 3 | Interrupt at low volume / from across the room | Does it still hear you, or only shouting? |
| 4 | Interrupt with a single word ("stop") | Short utterances are the hard case for VAD |
| 5 | Play music/TV nearby while it speaks | Does background audio trigger a false interrupt? |
| 6 | Interrupt, then say nothing | Does it recover, or hang waiting? |
| 7 | Interrupt twice in quick succession | Does state get stuck? |
| 8 | Raise speaker volume to near-max, repeat #1 | AEC usually fails first at high output volume |

Note what happens in each case. If #1 or #8 fails, the fix is likely the
ReSpeaker's own AEC configuration or speaker placement, not this codebase.

---

## 5. MCP tools — reminders now, calendar deferred

Upstream already has the MCP plumbing (`mcp_servers.json` + `use_mcpp`), so
nothing new was built for tool calling itself.

**Reminders are implemented**, backed by the same `threads` table that drives
proactive speech ([`../hasy/mcp_servers/reminders.py`](../hasy/mcp_servers/reminders.py)).
That means "remind me to book the flights" and HASY raising it unprompted two
days later are *the same state*, not two parallel systems. Three tools:
`list_reminders`, `add_reminder`, `resolve_reminder`.

**Calendar is deliberately not implemented.** It needs a provider (Google /
Outlook) and OAuth credentials — a decision plus secrets, not something to
guess at. When you pick one, add it to `mcp_servers.json` alongside the others;
no code change is required.

### ⚠️ MCP costs latency

`use_mcpp: True` sends tool definitions on **every** request, which lands
directly on time-to-first-token — the stage measured at ~719 ms in the Phase 1
baseline. Set `use_mcpp: False` in `conf.yaml` to restore the baseline for
comparison. This is a real trade, and it is being made deliberately rather than
by accident.

---

## Upstream boundary

Everything above is patched in from `hasy/presence/`, with **one exception**:

**`frontend/index.html` gains a single `<script>` tag.** The frontend is
vendored, and there is no way to load a companion script into a minified bundle
without it. It is one line, injected idempotently on boot, and the script itself
is served from `/hasy/presence.js` so no file is written into the vendored
directory.

No file under `src/open_llm_vtuber/` is modified.
