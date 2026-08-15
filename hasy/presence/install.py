"""One entry point for all of Phase 4.

Reads `hasy.yaml` (git-ignored, optional) and installs whichever presence
features are enabled. Everything is opt-in-shaped: an absent config gives you
proactive speech on and face tracking off, because face tracking opens a camera
and that should never happen by surprise.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from .face_tracking import FaceTracker, TrackerConfig
from .proactive import ProactiveConfig

_installed = False
_tracker: Optional[FaceTracker] = None

DEFAULTS: dict[str, Any] = {
    "proactive": {
        "enabled": True,
        "tick_s": 60.0,
        "quiet_before_hour": 9,
        "quiet_after_hour": 22,
        "min_silence_s": 300.0,
        "cooldown_s": 3600.0,
        "min_thread_age_s": 1800.0,
        "max_thread_age_s": 14 * 24 * 3600.0,
        "require_thread": True,
    },
    "face_tracking": {
        "enabled": False,  # opens a camera — opt in deliberately
        "camera_index": 0,
        "fps": 10.0,
        "offset_x": 0.0,
        "offset_y": 0.0,
        "gain_x": 1.0,
        "gain_y": 1.0,
        "invert_x": False,
        "invert_y": False,
        "smoothing": 0.7,
        "lost_face_timeout_s": 1.5,
        "min_face_fraction": 0.08,
    },
    "transport": {"enabled": True},
}


def _load_settings() -> dict[str, Any]:
    settings = {k: dict(v) for k, v in DEFAULTS.items()}
    path = Path("hasy.yaml")
    if path.exists():
        try:
            import yaml

            data = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("presence") or {}
            for section, values in data.items():
                if section in settings and isinstance(values, dict):
                    settings[section].update(values)
        except Exception as e:
            logger.warning(f"HASY presence: could not read hasy.yaml ({e}); using defaults")
    return settings


def install(frontend_dir: Path = Path("frontend")) -> None:
    global _installed, _tracker
    if _installed:
        return
    _installed = True

    settings = _load_settings()

    # --- transport: routes + companion script -----------------------------
    if settings["transport"].get("enabled", True):
        try:
            from . import transport

            transport.install(frontend_dir)
        except Exception as e:
            logger.warning(f"HASY presence: transport install failed ({e})")

    # --- proactive speaking ------------------------------------------------
    proactive = settings["proactive"]
    if proactive.get("enabled", True):
        try:
            from .scheduler import install as install_scheduler

            tick = float(proactive.pop("tick_s", 60.0))
            install_scheduler(ProactiveConfig(**proactive), tick_s=tick)
        except Exception as e:
            logger.warning(f"HASY presence: proactive install failed ({e})")

    # --- face tracking -----------------------------------------------------
    face = settings["face_tracking"]
    if face.get("enabled", False):
        try:
            from . import transport

            config = TrackerConfig(**face)
            _tracker = FaceTracker(config, on_gaze=transport.hub.send_gaze_threadsafe)
            _bind_hub_loop_on_startup()
            if _tracker.start():
                logger.info(
                    "HASY presence: face tracking on "
                    f"(offset {config.offset_x:+.2f},{config.offset_y:+.2f} "
                    f"gain {config.gain_x:.2f},{config.gain_y:.2f})"
                )
        except Exception as e:
            logger.warning(f"HASY presence: face tracking install failed ({e})")
    else:
        logger.debug("HASY presence: face tracking off (enable in hasy.yaml)")


def _bind_hub_loop_on_startup() -> None:
    """The tracker runs on a thread; it needs the server's loop to send.

    Bound when the first renderer connects, but also bound here on server
    startup so gaze can be dispatched even before that.
    """
    try:
        from src.open_llm_vtuber.server import WebSocketServer

        from . import transport

        original_init = WebSocketServer.__init__

        def wrapper(self, *args, **kwargs):
            original_init(self, *args, **kwargs)

            @self.app.on_event("startup")
            async def _bind():  # pragma: no cover - server lifecycle
                transport.hub.bind_loop(asyncio.get_running_loop())

        WebSocketServer.__init__ = wrapper
    except Exception as e:  # pragma: no cover
        logger.debug(f"HASY presence: could not bind hub loop early ({e})")


def shutdown() -> None:
    """Release the camera. Called on exit."""
    global _tracker
    if _tracker is not None:
        _tracker.stop()
        _tracker = None
