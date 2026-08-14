"""Per-turn latency instrumentation for HASY.

Measures the path from VAD speech-endpoint to first audio chunk leaving the
server, broken down by stage. Installed by monkeypatching upstream at a few
well-defined boundaries -- no upstream file is modified.

Stages emitted per turn:
    vad_endpoint_ms        VAD endpoint -> conversation turn starts
    asr_ms                 speech-to-text
    llm_first_token_ms     LLM request -> first text token
    tts_first_chunk_ms     first sentence handed to TTS -> audio file ready
    total_to_first_audio_ms  VAD endpoint (or turn start) -> first audio payload

Budget: total_to_first_audio_ms < 900 ms.

NOTE on what "first audio" means: this is measured server-side, at the moment
the first audio payload is queued for the websocket. It excludes network
transport to the client and client-side decode/playback. Phase 5 measures the
LAN hop separately.

Usage:
    from hasy.latency import install
    install()          # before the server starts
"""

from __future__ import annotations

import atexit
import contextvars
import csv
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

BUDGET_MS = 900.0

# Directory for the per-run CSV (git-ignored).
LOG_DIR = Path("latency_logs")

# Per-turn metrics, carried through the async call chain. Each conversation turn
# runs as its own asyncio task, and contextvars are copied into tasks spawned
# from it -- so TTS tasks created inside a turn see the right object.
_current_turn: contextvars.ContextVar[Optional["TurnMetrics"]] = contextvars.ContextVar(
    "hasy_current_turn", default=None
)

# VAD endpoints happen on a different websocket message than the turn itself,
# so they are stashed per client_uid and claimed when the turn starts.
_vad_endpoint_at: dict[str, float] = {}

_turns: list["TurnMetrics"] = []
_installed = False


def _ms(start: float, end: float) -> float:
    return (end - start) * 1000.0


@dataclass
class TurnMetrics:
    """Timings for a single conversation turn. All values in milliseconds."""

    turn: int
    started_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    vad_endpoint_ms: Optional[float] = None
    asr_ms: Optional[float] = None
    llm_first_token_ms: Optional[float] = None
    tts_first_chunk_ms: Optional[float] = None
    total_to_first_audio_ms: Optional[float] = None

    transcript: str = ""

    # Internal reference points (perf_counter), not exported.
    _t_vad: Optional[float] = field(default=None, repr=False)
    _t_start: float = field(default_factory=time.perf_counter, repr=False)
    _first_audio_recorded: bool = field(default=False, repr=False)

    @property
    def within_budget(self) -> Optional[bool]:
        if self.total_to_first_audio_ms is None:
            return None
        return self.total_to_first_audio_ms < BUDGET_MS

    def export(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if not k.startswith("_")}
        d["within_budget"] = self.within_budget
        return d

    def log(self) -> None:
        def fmt(v: Optional[float]) -> str:
            return f"{v:7.1f}" if v is not None else "      -"

        verdict = (
            "OK " if self.within_budget else "OVER"
        ) if self.within_budget is not None else "??? "

        # ASCII only: this runs mid-turn, and a UnicodeEncodeError on a cp1252
        # Windows console would surface inside the TTS task.
        try:
            logger.bind(hasy_latency=True).info(
                f"[HASY] turn {self.turn:>2} [{verdict}] "
                f"vad={fmt(self.vad_endpoint_ms)} "
                f"asr={fmt(self.asr_ms)} "
                f"llm_ft={fmt(self.llm_first_token_ms)} "
                f"tts_fc={fmt(self.tts_first_chunk_ms)} "
                f"-> first_audio={fmt(self.total_to_first_audio_ms)} ms "
                f"(budget {BUDGET_MS:.0f})"
            )
        except Exception:  # never let instrumentation break a conversation
            pass


def mark_vad_endpoint(client_uid: str) -> None:
    """Record that VAD just closed an utterance for this client."""
    _vad_endpoint_at[client_uid] = time.perf_counter()


def record_first_audio() -> None:
    """Called when the first audio payload of a turn is ready to send."""
    m = _current_turn.get()
    if m is None or m._first_audio_recorded:
        return
    m._first_audio_recorded = True
    origin = m._t_vad if m._t_vad is not None else m._t_start
    m.total_to_first_audio_ms = _ms(origin, time.perf_counter())
    m.log()


def summary_table() -> str:
    """Render the collected turns as a text table."""
    if not _turns:
        return "No turns recorded."

    cols = [
        ("turn", "turn", 4),
        ("vad_endpoint_ms", "vad", 8),
        ("asr_ms", "asr", 8),
        ("llm_first_token_ms", "llm_ft", 8),
        ("tts_first_chunk_ms", "tts_fc", 8),
        ("total_to_first_audio_ms", "TOTAL", 9),
    ]
    head = " ".join(f"{label:>{w}}" for _, label, w in cols) + "   verdict"
    lines = [head, "-" * len(head)]

    for m in _turns:
        row = []
        for attr, _, w in cols:
            v = getattr(m, attr)
            if isinstance(v, bool) or v is None:
                row.append(f"{'-':>{w}}")
            elif isinstance(v, int):
                row.append(f"{v:>{w}d}")
            elif isinstance(v, float):
                row.append(f"{v:>{w}.1f}")
            else:
                row.append(f"{'-':>{w}}")
        verdict = "-"
        if m.within_budget is not None:
            verdict = "OK" if m.within_budget else "OVER BUDGET"
        lines.append(" ".join(row) + f"   {verdict}")

    totals = [m.total_to_first_audio_ms for m in _turns if m.total_to_first_audio_ms is not None]
    if totals:
        ordered = sorted(totals)
        mean = sum(totals) / len(totals)
        p50 = ordered[len(ordered) // 2]
        worst = ordered[-1]
        n_ok = sum(1 for t in totals if t < BUDGET_MS)
        lines += [
            "-" * len(head),
            f"n={len(totals)}  mean={mean:.1f} ms  p50={p50:.1f} ms  max={worst:.1f} ms",
            f"within {BUDGET_MS:.0f} ms budget: {n_ok}/{len(totals)}",
        ]

        # Which stage dominates, on average?
        stages = {
            "vad_endpoint": "vad_endpoint_ms",
            "asr": "asr_ms",
            "llm_first_token": "llm_first_token_ms",
            "tts_first_chunk": "tts_first_chunk_ms",
        }
        means = {}
        for name, attr in stages.items():
            vals = [getattr(m, attr) for m in _turns if getattr(m, attr) is not None]
            if vals:
                means[name] = sum(vals) / len(vals)
        if means:
            slowest = max(means, key=means.get)
            detail = "  ".join(f"{k}={v:.1f}" for k, v in means.items())
            lines += [f"stage means (ms): {detail}", f"bottleneck: {slowest}"]

    return "\n".join(lines)


def write_csv(path: Optional[Path] = None) -> Optional[Path]:
    """Write collected turns to CSV. Returns the path, or None if nothing to write."""
    if not _turns:
        return None
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = path or LOG_DIR / f"latency_{datetime.now():%Y%m%d_%H%M%S}.csv"
    rows = [m.export() for m in _turns]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _report_at_exit() -> None:
    if not _turns:
        return
    logger.info("\n=== HASY latency summary ===\n" + summary_table())
    p = write_csv()
    if p:
        logger.info(f"Latency CSV written to {p}")


def install() -> None:
    """Monkeypatch upstream boundaries to collect per-turn latency.

    Safe to call once, before the server starts. Every patch is defensive: if
    upstream's shape has drifted, instrumentation degrades rather than breaking
    the conversation.
    """
    global _installed
    if _installed:
        return
    _installed = True

    _patch_vad_endpoint()
    _patch_turn_and_asr()
    _patch_llm_first_token()
    _patch_tts_first_chunk()

    atexit.register(_report_at_exit)
    logger.info(f"HASY latency instrumentation installed (budget {BUDGET_MS:.0f} ms).")


def _patch_vad_endpoint() -> None:
    """Stamp the moment VAD closes an utterance.

    Upstream appends to received_data_buffers[client_uid] and sends
    'mic-audio-end' exactly when an utterance closes, so buffer growth across
    the call is a faithful proxy for the endpoint.
    """
    try:
        from src.open_llm_vtuber.websocket_handler import WebSocketHandler
    except Exception as e:  # pragma: no cover
        logger.warning(f"HASY latency: cannot patch VAD endpoint ({e})")
        return

    original = WebSocketHandler._handle_raw_audio_data

    async def wrapper(self, websocket, client_uid, data):
        try:
            before = len(self.received_data_buffers.get(client_uid, ()))
        except Exception:
            before = None
        result = await original(self, websocket, client_uid, data)
        if before is not None:
            try:
                if len(self.received_data_buffers.get(client_uid, ())) > before:
                    mark_vad_endpoint(client_uid)
            except Exception:
                pass
        return result

    WebSocketHandler._handle_raw_audio_data = wrapper


def _rebind(name: str, new_obj, modules) -> int:
    """Rebind a module-level function everywhere it was imported.

    `from x import f` copies the reference into the importing module's
    namespace, so patching only the defining module leaves real call sites
    still pointing at the original. Rebind every holder.
    """
    count = 0
    for mod in modules:
        if getattr(mod, name, None) is not None:
            setattr(mod, name, new_obj)
            count += 1
    return count


def _patch_turn_and_asr() -> None:
    """Open a metrics object per turn, and time the ASR call inside it."""
    try:
        from src.open_llm_vtuber.conversations import single_conversation as sc
        from src.open_llm_vtuber.conversations import conversation_utils as cu
        from src.open_llm_vtuber.conversations import conversation_handler as ch
        from src.open_llm_vtuber.conversations import group_conversation as gc
    except Exception as e:  # pragma: no cover
        logger.warning(f"HASY latency: cannot patch conversation ({e})")
        return

    original_turn = sc.process_single_conversation

    async def turn_wrapper(context, websocket_send, client_uid, user_input, *args, **kwargs):
        m = TurnMetrics(turn=len(_turns) + 1)
        t_vad = _vad_endpoint_at.pop(client_uid, None)
        if t_vad is not None:
            m._t_vad = t_vad
            m.vad_endpoint_ms = _ms(t_vad, m._t_start)
        _turns.append(m)
        _current_turn.set(m)
        try:
            return await original_turn(
                context, websocket_send, client_uid, user_input, *args, **kwargs
            )
        finally:
            if not m._first_audio_recorded:
                # Turn produced no audio (error, or text-only). Still report it.
                m.log()

    # conversation_handler holds its own reference and is the real call site.
    n = _rebind("process_single_conversation", turn_wrapper, (sc, ch))
    logger.debug(f"HASY latency: turn wrapper bound in {n} module(s)")

    # ASR: time it, and capture the transcript for the report.
    original_asr = cu.process_user_input

    async def asr_wrapper(user_input, asr_engine, websocket_send):
        m = _current_turn.get()
        # Text input skips ASR entirely; only time the audio path.
        import numpy as np

        is_audio = isinstance(user_input, np.ndarray)
        t0 = time.perf_counter()
        text = await original_asr(user_input, asr_engine, websocket_send)
        if m is not None:
            if is_audio:
                m.asr_ms = _ms(t0, time.perf_counter())
            m.transcript = (text or "")[:120]
        return text

    n = _rebind("process_user_input", asr_wrapper, (cu, sc, gc))
    logger.debug(f"HASY latency: asr wrapper bound in {n} module(s)")


def _patch_llm_first_token() -> None:
    """Time from LLM request to first streamed text token (Claude + OpenAI-compatible)."""
    targets = [
        ("src.open_llm_vtuber.agent.stateless_llm.claude_llm", "claude"),
        ("src.open_llm_vtuber.agent.stateless_llm.openai_compatible_llm", "openai"),
    ]
    import importlib

    for module_path, kind in targets:
        try:
            mod = importlib.import_module(module_path)
            cls = mod.AsyncLLM
        except Exception as e:  # pragma: no cover
            logger.warning(f"HASY latency: cannot patch {kind} LLM ({e})")
            continue

        original = cls.chat_completion

        def make_wrapper(original):
            async def wrapper(self, *args, **kwargs):
                m = _current_turn.get()
                t0 = time.perf_counter()
                seen = False
                async for event in original(self, *args, **kwargs):
                    if not seen:
                        # Claude yields dict events; OpenAI-compatible yields str.
                        is_text = (
                            isinstance(event, str)
                            and event != "__API_NOT_SUPPORT_TOOLS__"
                        ) or (
                            isinstance(event, dict)
                            and event.get("type") == "text_delta"
                        )
                        if is_text:
                            seen = True
                            if m is not None and m.llm_first_token_ms is None:
                                m.llm_first_token_ms = _ms(t0, time.perf_counter())
                    yield event

            return wrapper

        cls.chat_completion = make_wrapper(original)


def _patch_tts_first_chunk() -> None:
    """Time the first TTS synthesis of a turn, and stamp first audio out."""
    try:
        from src.open_llm_vtuber.conversations.tts_manager import TTSTaskManager
    except Exception as e:  # pragma: no cover
        logger.warning(f"HASY latency: cannot patch TTS ({e})")
        return

    original_generate = TTSTaskManager._generate_audio

    async def generate_wrapper(self, tts_engine, text):
        m = _current_turn.get()
        first = m is not None and m.tts_first_chunk_ms is None
        t0 = time.perf_counter()
        result = await original_generate(self, tts_engine, text)
        if first:
            m.tts_first_chunk_ms = _ms(t0, time.perf_counter())
        return result

    TTSTaskManager._generate_audio = generate_wrapper

    # First payload queued == first audio ready to leave the server.
    original_put = TTSTaskManager._process_tts

    async def process_wrapper(self, *args, **kwargs):
        result = await original_put(self, *args, **kwargs)
        try:
            record_first_audio()
        except Exception as e:  # instrumentation must never break audio
            logger.debug(f"HASY latency: record_first_audio failed ({e})")
        return result

    TTSTaskManager._process_tts = process_wrapper
