"""Getting gaze targets to the avatar.

Upstream's `/client-ws` protocol has no message type for "look here", and the
frontend is a minified bundle. Rather than patch either, HASY adds its own
endpoints to the same FastAPI app and ships a tiny companion script that the
page loads alongside the bundle:

    GET  /hasy/health        — status JSON, for checking from a phone (Phase 5)
    GET  /hasy/presence.js   — the companion script
    WS   /hasy/presence-ws   — gaze targets, server -> browser

The script drives `window.getLive2DManager().getModel(0).setDragging(x, y)` —
the same call the bundle already makes on `pointermove`, so nothing new has to
be understood by the renderer.

The one intrusion is a `<script>` tag appended to `frontend/index.html`. That
file is vendored, so it is edited rather than patched — kept to a single tag,
and documented in docs/presence.md.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

from loguru import logger

PRESENCE_JS = """\
/* HASY presence companion. Loaded alongside the vendored frontend bundle.
   Drives the Live2D model's head via the same setDragging() call the bundle
   uses for mouse-follow, so the renderer needs no changes. */
(function () {
  var RECONNECT_MS = 2000;
  var model = null;

  function findModel() {
    try {
      if (window.getLive2DManager) {
        var mgr = window.getLive2DManager();
        if (mgr && mgr.getModel) return mgr.getModel(0);
      }
      if (window.getLAppAdapter) {
        var ad = window.getLAppAdapter();
        if (ad && ad.getModel) return ad.getModel();
      }
    } catch (e) { /* renderer not ready yet */ }
    return null;
  }

  function apply(x, y) {
    if (!model) model = findModel();
    if (!model) return;
    try {
      if (typeof model.setDragging === 'function') model.setDragging(x, y);
    } catch (e) { model = null; }
  }

  function connect() {
    var proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    var ws = new WebSocket(proto + '//' + location.host + '/hasy/presence-ws');

    ws.onmessage = function (ev) {
      var msg;
      try { msg = JSON.parse(ev.data); } catch (e) { return; }
      if (msg.type === 'gaze') apply(msg.x, msg.y);
    };
    // Resilience: the tablet's wifi will drop (Phase 5). Just keep retrying.
    ws.onclose = function () { setTimeout(connect, RECONNECT_MS); };
    ws.onerror = function () { try { ws.close(); } catch (e) {} };
  }

  if (document.readyState === 'complete') connect();
  else window.addEventListener('load', connect);
})();
"""

SCRIPT_TAG = '<script src="/hasy/presence.js"></script>'


class PresenceHub:
    """Fans gaze targets out to every connected renderer."""

    def __init__(self) -> None:
        self._clients: set = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.last_gaze: tuple[float, float] = (0.0, 0.0)

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def register(self, websocket) -> None:
        self._clients.add(websocket)
        logger.debug(f"HASY presence: renderer connected ({len(self._clients)} total)")

    async def unregister(self, websocket) -> None:
        self._clients.discard(websocket)

    @property
    def client_count(self) -> int:
        return len(self._clients)

    def send_gaze_threadsafe(self, x: float, y: float) -> None:
        """Called from the face-tracker thread; hops to the event loop."""
        self.last_gaze = (x, y)
        loop = self._loop
        if loop is None or not self._clients:
            return
        try:
            asyncio.run_coroutine_threadsafe(self.broadcast_gaze(x, y), loop)
        except Exception as e:
            logger.debug(f"HASY presence: could not dispatch gaze ({e})")

    async def broadcast_gaze(self, x: float, y: float) -> None:
        if not self._clients:
            return
        payload = json.dumps({"type": "gaze", "x": round(x, 4), "y": round(y, 4)})
        dead = []
        for ws in list(self._clients):
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)


hub = PresenceHub()


def ensure_script_tag(index_html: Path) -> bool:
    """Add the companion <script> to the vendored index.html, once.

    Returns True if the file was modified. Idempotent — safe to call on boot.
    """
    try:
        if not index_html.exists():
            return False
        html = index_html.read_text(encoding="utf-8")
        if "/hasy/presence.js" in html:
            return False
        if "</body>" not in html:
            logger.warning("HASY presence: index.html has no </body>; script not injected")
            return False
        index_html.write_text(
            html.replace("</body>", f"  {SCRIPT_TAG}\n</body>"), encoding="utf-8"
        )
        logger.info("HASY presence: companion script tag added to frontend/index.html")
        return True
    except Exception as e:
        logger.warning(f"HASY presence: could not inject script tag ({e})")
        return False


def write_presence_js(frontend_dir: Path) -> bool:
    """Write the companion script next to the bundle."""
    try:
        target = frontend_dir / "hasy-presence.js"
        if target.exists() and target.read_text(encoding="utf-8") == PRESENCE_JS:
            return False
        target.write_text(PRESENCE_JS, encoding="utf-8")
        return True
    except Exception as e:
        logger.warning(f"HASY presence: could not write presence.js ({e})")
        return False


def install(frontend_dir: Path = Path("frontend")) -> None:
    """Add HASY's routes to upstream's FastAPI app, without editing it."""
    try:
        from src.open_llm_vtuber.server import WebSocketServer
    except Exception as e:  # pragma: no cover
        logger.warning(f"HASY presence: cannot patch server ({e})")
        return

    # Served from the /hasy/presence.js route, so the only change to the
    # vendored frontend is a single <script> tag.
    ensure_script_tag(frontend_dir / "index.html")

    original_init = WebSocketServer.__init__

    def init_wrapper(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        try:
            _add_routes(self.app)
        except Exception as e:
            logger.warning(f"HASY presence: could not add routes ({e})")

    WebSocketServer.__init__ = init_wrapper
    logger.info("HASY presence: transport installed (/hasy/health, /hasy/presence-ws)")


def _add_routes(app) -> None:
    from fastapi import WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse, PlainTextResponse

    @app.get("/hasy/health")
    async def hasy_health():  # noqa: D401 - route
        """Status, checkable from a phone. Also used in Phase 5."""
        from ..latency import _turns

        recent = [t.export() for t in _turns[-5:]]
        return JSONResponse(
            {
                "status": "ok",
                "renderers_connected": hub.client_count,
                "last_gaze": {"x": hub.last_gaze[0], "y": hub.last_gaze[1]},
                "turns_recorded": len(_turns),
                "recent_turns": recent,
            }
        )

    @app.get("/hasy/presence.js")
    async def hasy_presence_js():  # noqa: D401 - route
        return PlainTextResponse(PRESENCE_JS, media_type="application/javascript")

    @app.websocket("/hasy/presence-ws")
    async def hasy_presence_ws(websocket: WebSocket):  # noqa: D401 - route
        await websocket.accept()
        hub.bind_loop(asyncio.get_running_loop())
        await hub.register(websocket)
        try:
            while True:
                # The renderer never sends; this just keeps the socket open.
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            await hub.unregister(websocket)
