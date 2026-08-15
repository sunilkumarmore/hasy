"""Webcam check and calibration aid for HASY face tracking.

Run this before trusting face tracking in the real thing. It opens the camera,
detects faces, and prints the gaze values HASY would send — so you can sit where
you actually sit and tune `offset_x/offset_y/gain_x/gain_y` until the avatar
looks at you rather than past you.

    uv run python scripts/check_face_tracking.py
    uv run python scripts/check_face_tracking.py --preview        # show video
    uv run python scripts/check_face_tracking.py --camera 1
    uv run python scripts/check_face_tracking.py --offset-x 0.2 --gain-x 1.4

Ctrl+C to stop. Nothing here touches the server.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hasy.presence.face_tracking import TrackerConfig, map_to_gaze, smooth  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="HASY face-tracking check / calibration")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--preview", action="store_true", help="show the video window")
    p.add_argument("--offset-x", type=float, default=0.0)
    p.add_argument("--offset-y", type=float, default=0.0)
    p.add_argument("--gain-x", type=float, default=1.0)
    p.add_argument("--gain-y", type=float, default=1.0)
    p.add_argument("--invert-x", action="store_true")
    p.add_argument("--invert-y", action="store_true")
    p.add_argument("--smoothing", type=float, default=0.7)
    args = p.parse_args()

    try:
        import cv2
    except ImportError:
        print("opencv-python is not installed.  uv pip install 'opencv-python<5'")
        return 1

    config = TrackerConfig(
        enabled=True,
        camera_index=args.camera,
        offset_x=args.offset_x,
        offset_y=args.offset_y,
        gain_x=args.gain_x,
        gain_y=args.gain_y,
        invert_x=args.invert_x,
        invert_y=args.invert_y,
        smoothing=args.smoothing,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}.")
        print("Try a different --camera index, and check no other app is using it.")
        return 1

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if cascade.empty():
        print("Face cascade failed to load — the opencv install looks broken.")
        cap.release()
        return 1

    print("Camera open. Sit where you normally sit.")
    print("Aim for gaze ~ (0.0, 0.0) when you are looking straight at the box.")
    print("  too far left/right  -> adjust --offset-x")
    print("  moves too little    -> raise --gain-x / --gain-y")
    print("  moves the wrong way -> add --invert-x or --invert-y")
    print("Ctrl+C to stop.\n")

    smoothed = None
    frames = detections = 0
    started = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            frames += 1
            h, w = frame.shape[:2]
            gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            faces = cascade.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5,
                minSize=(int(w * 0.08), int(w * 0.08)),
            )

            if len(faces):
                detections += 1
                fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                target = map_to_gaze(fx + fw / 2, fy + fh / 2, w, h, config)
                smoothed = smooth(smoothed, target, config.smoothing)
                sys.stdout.write(
                    f"\rface at ({fx + fw // 2:4d},{fy + fh // 2:4d})px  "
                    f"raw=({target[0]:+.2f},{target[1]:+.2f})  "
                    f"smoothed=({smoothed[0]:+.2f},{smoothed[1]:+.2f})  "
                    f"detect-rate={detections / max(frames, 1):.0%}   "
                )
                if args.preview:
                    cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
            else:
                sys.stdout.write(
                    f"\rno face   frames={frames}  "
                    f"detect-rate={detections / max(frames, 1):.0%}            "
                )
            sys.stdout.flush()

            if args.preview:
                cv2.imshow("HASY face tracking (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if args.preview:
            cv2.destroyAllWindows()

    elapsed = time.time() - started
    print(f"\n\n{frames} frames in {elapsed:.1f}s ({frames / max(elapsed, 1e-6):.1f} fps)")
    print(f"face detected in {detections / max(frames, 1):.0%} of frames")
    if detections == 0:
        print("\nNo faces detected at all. Check lighting, and that you are facing")
        print("the camera — the Haar cascade only finds front-facing faces.")
    else:
        print("\nPut the values you settled on into hasy.yaml under presence.face_tracking.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
