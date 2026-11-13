#!/usr/bin/env python3
"""Grab stills from the Pi camera for offline piece-detect tuning.

Usage on the Pi:
  ./scripts/run-pi.sh python -m gamebot.capture_frames
  ./scripts/run-pi.sh python -m gamebot.capture_frames --n 5

Copies cal board ROI into a sidecar JSON next to each JPG so laptop tuning
matches the live ROI. Then scp the folder to your laptop, e.g.:

  scp -r max@gamebot:~/src/gamebot/captures ./captures
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2

from gamebot.adapters import DEFAULT_CAMERA_ZOOM, create_camera
from gamebot.calibration import CalibrationData


def _board_roi() -> list[list[float]] | None:
    path = Path.home() / ".gamebot" / "calibration.json"
    if not path.is_file():
        return None
    try:
        corners = CalibrationData.load(path).board_corners_camera
        return corners.astype(float).tolist()
    except Exception:
        return None


def main() -> None:
    p = argparse.ArgumentParser(description="Capture camera stills for piece-detect tuning")
    p.add_argument("--camera", choices=["picam", "mock"], default="picam")
    p.add_argument("--n", type=int, default=3, help="Number of frames to save")
    p.add_argument("--warmup", type=int, default=8, help="Discard this many frames first")
    p.add_argument("--delay", type=float, default=0.35, help="Seconds between saved frames")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("captures"),
        help="Output directory (default: ./captures)",
    )
    p.add_argument("--size", default="1280x720", help="WxH")
    p.add_argument(
        "--zoom",
        type=float,
        default=DEFAULT_CAMERA_ZOOM,
        help=f"Digital center zoom (default {DEFAULT_CAMERA_ZOOM})",
    )
    args = p.parse_args()

    w, h = (int(x) for x in args.size.lower().split("x"))
    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    roi = _board_roi()
    zoom = float(args.zoom)

    cam = create_camera(args.camera, (w, h), zoom=zoom)
    cam.open()
    try:
        for _ in range(max(0, args.warmup)):
            cam.read()
            time.sleep(0.05)

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        saved: list[Path] = []
        for i in range(max(1, args.n)):
            frame = cam.read()
            name = f"{stamp}_{i + 1:02d}.jpg"
            path = out_dir / name
            ok = cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if not ok:
                raise RuntimeError(f"failed to write {path}")
            meta = {
                "image": name,
                "size": [int(frame.shape[1]), int(frame.shape[0])],
                "camera_zoom": zoom,
                "board_corners_camera": roi,
                "captured_at": datetime.now(timezone.utc).isoformat(),
            }
            path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
            saved.append(path)
            print(f"saved {path}")
            if i + 1 < args.n:
                time.sleep(args.delay)
    finally:
        cam.close()

    print()
    print(f"{len(saved)} frame(s) in {out_dir}")
    print("copy to laptop, then tune:")
    print(f"  scp -r '{out_dir}' ./captures")
    print("  .venv/bin/python -m gamebot.tune_piece_detect captures/")


if __name__ == "__main__":
    main()
