# HASY speaking-style prompt block

`conf.yaml` is git-ignored (it holds API keys), so this file is the versioned
source of truth for the latency-critical prompt text. If you rebuild `conf.yaml`
from `config_templates/conf.default.yaml`, re-append this block to
`character_config.persona_prompt`.

**Why it exists:** the Phase 1 baseline measured ~1400 ms per turn of
`sentence_wait_ms` — time spent waiting for the LLM to finish its *first
sentence*, because TTS cannot start until a sentence boundary is reached. It was
the single largest cost in the turn. Constraining the opening sentence attacks
it directly, with no infrastructure change.

See [`docs/latency-backlog.md`](../../docs/latency-backlog.md) for the full
measured baseline and the remaining optimizations.

```text
SPEAKING STYLE (latency-critical, do not ignore):
- Your FIRST sentence must be short: 10 words or fewer. Nothing you say is
  spoken aloud until that first sentence is complete, so a long opening
  sentence makes you audibly slow to respond.
- Keep the whole reply to 1-3 short sentences unless explicitly asked for
  more detail.
- Never open with a preamble, a restatement of the question, or a list.
  Answer immediately, then elaborate only if needed.
- You are spoken aloud, not read. No markdown, no bullet points, no emoji,
  no code blocks.
```

## Not yet done

The character is still upstream's demo persona (`Mao` / "sarcastic AI VTuber
Mili") from `conf.default.yaml`. Only the speaking-style block above is HASY's.
Replacing the persona with an actual HASY character is a deliberate, separate
decision — it changes the product, not the latency.
