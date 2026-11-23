#!/usr/bin/env python3
"""Grab stills from the Pi camera for offline piece-detect tuning.

Usage on the Pi:
  ./scripts/run-pi.sh python -m gamebot.capture_frames
  ./scripts/run-pi.sh python -m gamebot.capture_frames --n 5

Copies cal board ROI into a sidecar JSON next to each JPG so laptop tuning
matches the live ROI. Default out is ~/.gamebot/captures (outside the git
tree). Pull into gamebot-vision with ./scripts/pull-captures.sh.
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
from gamebot.paths import gamebot_home


def save_move_still(
    frame_bgr: np.ndarray,
    out_dir: Path,
    *,
    uci: str,
    who: str,
    fen: str,
    camera_zoom: float,
    board_corners_camera: list | None,
    set_name: str = "04_play",
) -> Path:
    """Write one play still plus a sidecar the training pull already picks up.

    Lands in ``captures/sets/04_play`` so ``pull-captures.sh`` copies it with
    the guided sets. The board area is unlit during play; a move arrow can
    still be in the frame.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    safe_uci = "".join(ch for ch in uci if ch.isalnum()) or "move"
    stem = f"{stamp}_{safe_uci}_{who}"
    path = out_dir / f"{stem}.jpg"
    n = 2
    while path.exists():
        path = out_dir / f"{stem}_{n}.jpg"
        n += 1
    if not cv2.imwrite(str(path), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
        raise RuntimeError(f"failed to write {path}")
    meta = {
        "image": path.name,
        "size": [int(frame_bgr.shape[1]), int(frame_bgr.shape[0])],
        "camera_zoom": camera_zoom,
        "board_corners_camera": board_corners_camera,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "uci": uci,
        "who": who,
        "fen": fen,
        "set": set_name,
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    return path


def _board_roi() -> list[list[float]] | None:
    path = gamebot_home() / "calibration.json"
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
        default=None,
        help="Output directory (default: <user>/.gamebot/captures — SUDO_USER under run-pi.sh)",
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
    out_dir = (args.out or (gamebot_home() / "captures")).expanduser().resolve()
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
    print("  cd ../gamebot-vision && ./scripts/pull-captures.sh max@gamebot")


if __name__ == "__main__":
    main()
