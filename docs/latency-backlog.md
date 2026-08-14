# Latency backlog

Deferred optimizations from the Phase 1 baseline. **Status: consciously accepted
as-is.** The human saw these numbers on 2026-08-14, judged the current latency
acceptable for now, and chose to move to Phase 2. Nothing here is a blocker —
it is the ordered list to work through when latency becomes the priority again.

The Phase 1 → 2 gate was satisfied: a voice conversation worked end-to-end and
the numbers below were reviewed.

---

## Measured baseline (2026-08-14, 8 real spoken turns)

Windows 11 ARM64 laptop, x64 CPython 3.10 under emulation. Claude Haiku 4.5 +
Groq `whisper-large-v3-turbo` + `edge_tts`. MCP disabled. Budget: **900 ms**.

| turn | asr | llm_first_token | tts_first_chunk\* | **total_to_first_audio** |
|---:|---:|---:|---:|---:|
| 1 | 652 | 1060 | 605 | **3041** |
| 2 | 429 | 671 | 1038 | **2721** |
| 3 | 421 | 704 | 1044 | **2745** |
| 4 | 620 | 606 | 1658 | **3022** |
| 5 | 400 | 812 | 2879 | **4636** |
| 6 | 396 | 589 | 894 | **2331** |
| 7 | 525 | 600 | 3715 | **5204** |
| 8 | 430 | 710 | 535 | **1805** |
| **mean** | **484** | **719** | ~1300 | **3188** |

`n=8, mean=3188 ms, p50=3022 ms, max=5204 ms. Within budget: 0/8.`

\* `tts_first_chunk` in this run was affected by a since-fixed race (parallel TTS
tasks overwrote each other). Treat as approximate. `asr`, `llm_first_token` and
the totals are sound. Raw data: `latency_logs/latency_20260814_103800.csv`.

### Two reasons the real number is worse than the table

1. **`vad_endpoint_ms` is unmeasured (`-` on every turn).** The web frontend runs
   VAD **client-side** and sends a complete utterance, so upstream's server-side
   Silero VAD never executes and there is nothing to hook. The clock therefore
   starts *after* the browser decided speech ended. Real perceived latency
   includes that trailing-silence detection.
2. **Measurement ends when audio leaves the server** — the LAN hop to the tablet
   and browser decode/playback are on top. Phase 5 measures the hop separately.

---

## Backlog, highest value first

### 1. ~~Shorten the first sentence~~ — DONE 2026-08-14
The largest single cost was ~1400 ms of `sentence_wait_ms`: TTS cannot begin
until the LLM completes its first *sentence*, not its first token. Addressed via
a prompt constraint — see [`hasy/prompts/speaking_style.md`](../hasy/prompts/speaking_style.md).
**Not yet re-measured.** Re-run the instrumented server to quantify the gain.

### 2. Replace `edge_tts` — biggest remaining win
Free but slow and wildly inconsistent: 535 ms best, **3715 ms worst**. That
variance is fatal for presence — the avatar feels broken, not just slow.
- **ElevenLabs Flash v2.5** (`eleven_flash_v2_5`) targets ~75–150 ms and is
  already implemented upstream (`tts/elevenlabs_tts.py`) — a config-only swap.
  Costs credits; an always-on companion will burn them steadily.
- Do **not** use the config default `eleven_multilingual_v2` — it is the
  high-quality/slow model and lands directly on the critical path.
- Cartesia is also implemented upstream and is another low-latency option.

### 3. Verify `faster_first_response` actually works
It is `True` and is documented to flush at the first comma to cut latency, yet
we still measured ~1400 ms of sentence wait. Either it is not behaving as
advertised with `segment_method: pysbd`, or the persona's long opening sentences
defeated it. Read `utils/sentence_divider.py` and confirm empirically before
optimizing around it. Possible free win.

### 4. Streaming TTS instead of file-per-sentence
Upstream synthesizes each sentence to a **complete file** (`generate_audio`
returns a path) before sending anything. Even a streaming-capable vendor gains
nothing through this interface. Changing it means touching the TTS interface and
`tts_manager` — a real upstream edit, so weigh it against the boundary rule in
CLAUDE.md.

### 5. Accept or renegotiate the cloud floor
`asr` (484 ms) + `llm_first_token` (719 ms) ≈ **1200 ms of unavoidable cloud
round-trips**, already over the 900 ms budget before TTS. With this architecture
900 ms is not reachable. Options: local ASR (slower here — emulated x64 CPU, no
CUDA), a lower-latency LLM, prompt caching to cut time-to-first-token, or
accepting a higher target. **This is the honest structural limit — the budget
needs revisiting, not just tuning.**

### 6. Prompt caching for the system prompt
The persona + expression prompts are re-sent every turn. Anthropic prompt caching
could reduce time-to-first-token. Requires a small change to how `claude_llm.py`
builds requests.

### 7. Instrument the missing stages
- **Client-side VAD endpoint** — needs a frontend-side timestamp shipped to the
  server; currently invisible and it is real user-perceived latency.
- **LAN hop to the tablet** (Phase 5).
- **Browser decode/playback** to first audible sample.

### 8. Re-check ARM64 emulation overhead
Everything runs as emulated x64. If server-side VAD is ever re-enabled (torch,
CPU) this will matter. Not currently on the critical path since VAD runs in the
browser.

---

## Instrumentation notes

`hasy/latency.py` patches upstream by monkeypatch only — no upstream file is
modified. Two defects found and fixed during Phase 1, worth remembering:

- **Patching a function on its defining module is not enough.** Modules that did
  `from x import f` hold their own reference; `conversation_handler.py` was
  calling the original and *all* measurement silently returned nothing. Use the
  `_rebind()` helper across every holder module.
- **Claim "first" slots synchronously.** TTS tasks run concurrently; checking
  `is None` after an `await` lets every parallel task write, and the slowest
  wins.
