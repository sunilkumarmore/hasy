"""Face tracking: the geometry, which is the part that must be right.

Capture needs a camera, so the hardware path is exercised separately by
scripts/check_face_tracking.py. Everything here runs with no webcam.
"""

from __future__ import annotations

import math

import pytest

from hasy.presence.face_tracking import TrackerConfig, map_to_gaze, smooth
from hasy.presence.transport import PRESENCE_JS, SCRIPT_TAG, ensure_script_tag, write_presence_js

W, H = 640, 480


def cfg(**kw) -> TrackerConfig:
    return TrackerConfig(**kw)


# ----------------------------------------------------------------- geometry


def test_centre_face_looks_straight_ahead():
    x, y = map_to_gaze(W / 2, H / 2, W, H, cfg())
    assert x == pytest.approx(0.0, abs=1e-6)
    assert y == pytest.approx(0.0, abs=1e-6)


def test_face_on_the_right_of_frame_gives_positive_x():
    x, _ = map_to_gaze(W * 0.9, H / 2, W, H, cfg())
    assert x > 0


def test_image_y_down_is_flipped_to_live2d_y_up():
    """A face high in frame (small pixel y) must look UP, not down."""
    _, y_high = map_to_gaze(W / 2, H * 0.1, W, H, cfg())
    _, y_low = map_to_gaze(W / 2, H * 0.9, W, H, cfg())
    assert y_high > 0, "face near the top of frame should look up"
    assert y_low < 0


def test_output_is_clamped_to_the_live2d_range():
    for gain in (1.0, 5.0, 20.0):
        x, y = map_to_gaze(0, 0, W, H, cfg(gain_x=gain, gain_y=gain))
        assert -1.0 <= x <= 1.0
        assert -1.0 <= y <= 1.0


def test_offset_moves_the_neutral_point():
    """The webcam and the box are in different places — this is that knob."""
    x, _ = map_to_gaze(W / 2, H / 2, W, H, cfg(offset_x=0.3))
    assert x == pytest.approx(0.3)


def test_gain_scales_the_swing():
    x_small, _ = map_to_gaze(W * 0.75, H / 2, W, H, cfg(gain_x=0.5))
    x_large, _ = map_to_gaze(W * 0.75, H / 2, W, H, cfg(gain_x=1.0))
    assert abs(x_large) > abs(x_small)


def test_inversion_flips_each_axis():
    base = map_to_gaze(W * 0.75, H * 0.25, W, H, cfg())
    flipped_x = map_to_gaze(W * 0.75, H * 0.25, W, H, cfg(invert_x=True))
    flipped_y = map_to_gaze(W * 0.75, H * 0.25, W, H, cfg(invert_y=True))
    assert flipped_x[0] == pytest.approx(-base[0])
    assert flipped_y[1] == pytest.approx(-base[1])


def test_degenerate_frame_size_is_safe():
    assert map_to_gaze(10, 10, 0, 0, cfg()) == (0.0, 0.0)


# ---------------------------------------------------------------- smoothing


def test_first_sample_is_taken_as_is():
    assert smooth(None, (0.5, -0.5), 0.7) == (0.5, -0.5)


def test_smoothing_moves_toward_the_target_without_snapping():
    out = smooth((0.0, 0.0), (1.0, 1.0), 0.7)
    assert 0 < out[0] < 1.0
    assert out[0] == pytest.approx(0.3)


def test_repeated_smoothing_converges():
    p = (0.0, 0.0)
    for _ in range(60):
        p = smooth(p, (1.0, -1.0), 0.7)
    assert p[0] == pytest.approx(1.0, abs=1e-3)
    assert p[1] == pytest.approx(-1.0, abs=1e-3)


def test_smoothing_alpha_is_bounded_so_it_always_converges():
    """alpha=1.0 would freeze the head forever; it must be clamped."""
    p = (0.0, 0.0)
    for _ in range(200):
        p = smooth(p, (1.0, 1.0), 1.0)
    assert p[0] > 0.5, "head never moved — smoothing was not clamped below 1.0"


# ---------------------------------------------------------------- transport


def test_companion_script_targets_the_real_renderer_api():
    assert "getLive2DManager" in PRESENCE_JS
    assert "setDragging" in PRESENCE_JS
    assert "presence-ws" in PRESENCE_JS


def test_script_tag_injection_is_idempotent(tmp_path):
    index = tmp_path / "index.html"
    index.write_text("<html><body><div id='live2d'></div></body></html>", encoding="utf-8")

    assert ensure_script_tag(index) is True
    once = index.read_text(encoding="utf-8")
    assert SCRIPT_TAG in once

    assert ensure_script_tag(index) is False, "injected twice"
    assert index.read_text(encoding="utf-8").count("/hasy/presence.js") == 1


def test_script_tag_injection_handles_missing_file(tmp_path):
    assert ensure_script_tag(tmp_path / "nope.html") is False


def test_script_tag_injection_handles_missing_body(tmp_path):
    index = tmp_path / "index.html"
    index.write_text("<html><div>no body tag</div></html>", encoding="utf-8")
    assert ensure_script_tag(index) is False


def test_presence_js_is_written_once(tmp_path):
    assert write_presence_js(tmp_path) is True
    assert (tmp_path / "hasy-presence.js").read_text(encoding="utf-8") == PRESENCE_JS
    assert write_presence_js(tmp_path) is False


async def test_hub_broadcast_survives_a_dead_renderer():
    from hasy.presence.transport import PresenceHub

    class DeadWS:
        async def send_text(self, _):
            raise RuntimeError("socket closed")

    class LiveWS:
        def __init__(self):
            self.sent = []

        async def send_text(self, payload):
            self.sent.append(payload)

    hub = PresenceHub()
    live = LiveWS()
    await hub.register(DeadWS())
    await hub.register(live)

    await hub.broadcast_gaze(0.5, -0.25)

    assert live.sent, "live renderer did not receive the gaze"
    assert hub.client_count == 1, "dead renderer was not reaped"


# ------------------------------------------------------------------- routes


def _app():
    from fastapi import FastAPI

    from hasy.presence.transport import _add_routes

    app = FastAPI()
    _add_routes(app)
    return app


def test_health_endpoint_reports_status():
    from fastapi.testclient import TestClient

    r = TestClient(_app()).get("/hasy/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "renderers_connected" in body
    assert "last_gaze" in body


def test_presence_js_is_served_so_no_file_lands_in_the_vendored_frontend():
    from fastapi.testclient import TestClient

    r = TestClient(_app()).get("/hasy/presence.js")
    assert r.status_code == 200
    assert "setDragging" in r.text


def test_renderer_can_connect_and_is_tracked():
    """Starlette's TestClient re-raises the close on context exit, so the
    disconnect is expected — what matters is that the renderer was registered
    while connected, and reaped afterwards."""
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    from hasy.presence.transport import hub

    before = hub.client_count
    try:
        with TestClient(_app()).websocket_connect("/hasy/presence-ws"):
            assert hub.client_count == before + 1
    except WebSocketDisconnect:
        pass
    assert hub.client_count == before, "renderer was not unregistered on disconnect"
