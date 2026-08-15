"""Face tracking — make HASY turn toward whoever is in the room.

Reportedly the single most convincing presence cue, and cheap: a webcam frame,
a face box, two numbers.

Geometry note that matters for HASY specifically: the **laptop webcam and the
hologram box sit in different places**, so where the camera sees a face is not
where the avatar should look. `TrackerConfig` carries a calibration offset and
gain you tune in config, never in code — because you will tune it physically,
by sitting in the chair and nudging numbers until it feels right.

Split deliberately:
    FaceTracker   — capture + detect + smooth (needs a camera)
    map_to_gaze() — pure geometry (testable with no hardware)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from loguru import logger


@dataclass
class TrackerConfig:
    enabled: bool = False  # opt-in: it opens a camera
    camera_index: int = 0
    fps: float = 10.0  # plenty for head movement; keeps CPU low

    #: Calibration. Offset shifts the neutral point, gain scales the swing.
    #: Tune these by sitting where you actually sit.
    offset_x: float = 0.0
    offset_y: float = 0.0
    gain_x: float = 1.0
    gain_y: float = 1.0
    invert_x: bool = False  # camera mirroring vs. the reflected hologram
    invert_y: bool = False

    #: 0 = no smoothing (twitchy), 0.9 = heavy (laggy). Head motion is slow.
    smoothing: float = 0.7

    #: Give up on a face after this long and drift back to neutral, so HASY
    #: doesn't stare at where someone used to be.
    lost_face_timeout_s: float = 1.5

    #: Detection tuning (OpenCV Haar cascade).
    min_face_fraction: float = 0.08  # ignore faces smaller than this of frame width


def map_to_gaze(
    face_cx: float,
    face_cy: float,
    frame_w: int,
    frame_h: int,
    config: TrackerConfig,
) -> tuple[float, float]:
    """Map a face centre in pixels to Live2D drag coordinates in [-1, 1].

    Live2D's `setDragging(x, y)` takes -1..1 with +x right and +y **up**, while
    image coordinates run +y **down** — so the vertical axis is flipped here
    rather than in config, where it would be a confusing thing to have to know.
    """
    if frame_w <= 0 or frame_h <= 0:
        return 0.0, 0.0

    nx = (face_cx / frame_w) * 2.0 - 1.0
    ny = -((face_cy / frame_h) * 2.0 - 1.0)  # image y-down -> Live2D y-up

    if config.invert_x:
        nx = -nx
    if config.invert_y:
        ny = -ny

    x = nx * config.gain_x + config.offset_x
    y = ny * config.gain_y + config.offset_y

    return _clamp(x), _clamp(y)


def _clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def smooth(previous: Optional[tuple[float, float]], target: tuple[float, float], alpha: float):
    """Exponential smoothing so the head glides instead of snapping."""
    if previous is None:
        return target
    a = _clamp(alpha, 0.0, 0.99)
    return (
        previous[0] * a + target[0] * (1 - a),
        previous[1] * a + target[1] * (1 - a),
    )


class FaceTracker:
    """Captures from a webcam in a background thread and emits gaze targets.

    A thread rather than asyncio because OpenCV's capture call blocks; the
    callback is handed back to the event loop by the caller.
    """

    def __init__(
        self,
        config: Optional[TrackerConfig] = None,
        on_gaze: Optional[Callable[[float, float], None]] = None,
    ):
        self.config = config or TrackerConfig()
        self.on_gaze = on_gaze
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._last: Optional[tuple[float, float]] = None
        self._last_seen: float = 0.0
        self.frames = 0
        self.detections = 0

    # ------------------------------------------------------------- lifecycle

    def start(self) -> bool:
        if not self.config.enabled:
            logger.info("HASY presence: face tracking disabled")
            return False
        if self._thread is not None:
            return True
        try:
            import cv2  # noqa: F401
        except ImportError:
            logger.warning(
                "HASY presence: opencv-python not installed; face tracking off"
            )
            return False

        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="hasy-face", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # ----------------------------------------------------------------- loop

    def _run(self) -> None:
        import cv2

        cap = cv2.VideoCapture(self.config.camera_index)
        if not cap.isOpened():
            logger.warning(
                f"HASY presence: could not open camera {self.config.camera_index}; "
                "face tracking off"
            )
            return

        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if cascade.empty():
            logger.warning("HASY presence: face cascade failed to load; tracking off")
            cap.release()
            return

        interval = 1.0 / max(self.config.fps, 1.0)
        logger.info(
            f"HASY presence: face tracking started (camera {self.config.camera_index}, "
            f"{self.config.fps:.0f} fps)"
        )

        try:
            while not self._stop.is_set():
                started = time.perf_counter()
                ok, frame = cap.read()
                if not ok:
                    time.sleep(interval)
                    continue
                self.frames += 1
                self._process(frame, cascade, cv2)
                elapsed = time.perf_counter() - started
                if (remaining := interval - elapsed) > 0:
                    self._stop.wait(remaining)
        except Exception as e:  # pragma: no cover - hardware path
            logger.warning(f"HASY presence: face tracking stopped ({e})")
        finally:
            cap.release()

    def _process(self, frame, cascade, cv2) -> None:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(w * self.config.min_face_fraction), int(w * self.config.min_face_fraction)),
        )

        if len(faces) == 0:
            self._maybe_recentre()
            return

        # Largest face wins — the nearest person is the one being talked to.
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        self.detections += 1
        self._last_seen = time.time()

        target = map_to_gaze(fx + fw / 2.0, fy + fh / 2.0, w, h, self.config)
        self._emit(smooth(self._last, target, self.config.smoothing))

    def _maybe_recentre(self) -> None:
        """Drift back to neutral once the face has been gone a moment."""
        if self._last is None:
            return
        if time.time() - self._last_seen < self.config.lost_face_timeout_s:
            return
        neutral = smooth(self._last, (0.0, 0.0), self.config.smoothing)
        self._emit(neutral)
        if abs(neutral[0]) < 0.01 and abs(neutral[1]) < 0.01:
            self._last = None

    def _emit(self, gaze: tuple[float, float]) -> None:
        self._last = gaze
        if self.on_gaze is not None:
            try:
                self.on_gaze(gaze[0], gaze[1])
            except Exception as e:
                logger.debug(f"HASY presence: gaze callback failed ({e})")
