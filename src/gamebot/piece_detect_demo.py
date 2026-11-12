#!/usr/bin/env python3
"""Visual test: project boxes on the best white + black piece (strict + smoothed).

Usage:
  export DISPLAY=:0
  python -m gamebot.piece_detect_demo --camera picam --projector hdmi
  python -m gamebot.piece_detect_demo --camera picam --projector hdmi --full-frame

Press q / Esc to quit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from gamebot.adapters import create_camera, create_projector
from gamebot.calibration import CalibrationData


def _try_load_board_roi() -> np.ndarray | None:
    path = Path.home() / ".gamebot" / "calibration.json"
    if not path.is_file():
        return None
    try:
        return CalibrationData.load(path).board_corners_camera
    except Exception:
        return None


class _SmoothTrack:
    """Hold last lock; only jump if a new blob is clearly better / far away for a bit."""

    def __init__(self, alpha: float = 0.35) -> None:
        self.alpha = alpha
        self.xy: tuple[float, float] | None = None
        self._miss = 0
        self._cand: tuple[float, float] | None = None
        self._cand_n = 0

    def update(self, xy: tuple[float, float] | None) -> tuple[float, float] | None:
        if xy is None:
            self._miss += 1
            if self._miss > 15:
                self.xy = None
                self._cand = None
                self._cand_n = 0
            return self.xy
        self._miss = 0
        if self.xy is None:
            self.xy = xy
            return self.xy
        dist = float(np.linalg.norm(np.array(xy) - np.array(self.xy)))
        if dist < 35:
            a = self.alpha
            self.xy = (self.xy[0] * (1 - a) + xy[0] * a, self.xy[1] * (1 - a) + xy[1] * a)
            self._cand = None
            self._cand_n = 0
            return self.xy
        # Big jump — require 5 frames agreeing before teleport
        if self._cand is not None and float(np.linalg.norm(np.array(xy) - np.array(self._cand))) < 40:
            self._cand_n += 1
            self._cand = (
                self._cand[0] * 0.5 + xy[0] * 0.5,
                self._cand[1] * 0.5 + xy[1] * 0.5,
            )
        else:
            self._cand = xy
            self._cand_n = 1
        if self._cand_n >= 5:
            self.xy = self._cand
            self._cand = None
            self._cand_n = 0
        return self.xy


def _overlay(
    size: tuple[int, int],
    white: tuple[float, float] | None,
    black: tuple[float, float] | None,
    roi: np.ndarray | None,
    status: str,
    raw_w: int,
    raw_b: int,
) -> np.ndarray:
    w, h = size
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    if roi is not None:
        pts = roi.astype(np.int32)
        cv2.polylines(frame, [pts], True, (0, 200, 140), 3)

    if white is not None:
        cx, cy = int(white[0]), int(white[1])
        cv2.rectangle(frame, (cx - 28, cy - 28), (cx + 28, cy + 28), (255, 255, 255), 4)
        cv2.putText(frame, "W", (cx - 12, cy - 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    if black is not None:
        cx, cy = int(black[0]), int(black[1])
        cv2.rectangle(frame, (cx - 28, cy - 28), (cx + 28, cy + 28), (40, 40, 255), 4)
        cv2.putText(frame, "B", (cx - 12, cy - 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 255), 2)

    cv2.putText(frame, status, (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(
        frame,
        f"raw candidates W={raw_w} B={raw_b} (showing locked 1+1)",
        (40, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (180, 180, 180),
        2,
    )
    cv2.putText(frame, "piece detect — q quit", (40, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    return frame


def main() -> None:
    p = argparse.ArgumentParser(description="Project boxes on best white/black pieces")
    p.add_argument("--camera", choices=["mock", "picam", "file"], default="picam")
    p.add_argument("--projector", choices=["mock", "hdmi"], default="hdmi")
    p.add_argument("--full-frame", action="store_true", help="Ignore cal board ROI")
    args = p.parse_args()

    from chess_plugin.vision import find_piece_blobs

    size = (1280, 720)
    cam = create_camera(args.camera, size)
    proj = create_projector(args.projector, size)
    cam.open()
    proj.open()
    roi = None if args.full_frame else _try_load_board_roi()
    tw, tb = _SmoothTrack(), _SmoothTrack()
    print(f"piece detect demo — ROI={'cal board' if roi is not None else 'full frame'} (strict+smooth)")

    try:
        while True:
            frame = cam.read()
            # Peek at how many pass filters before cutting to 1
            raw_w, raw_b = find_piece_blobs(frame, roi, max_each=8)
            whites, blacks = find_piece_blobs(frame, roi, max_each=1)
            w_xy = tw.update(whites[0].xy if whites else None)
            b_xy = tb.update(blacks[0].xy if blacks else None)
            status = "locked"
            if w_xy:
                status += f"  W=({w_xy[0]:.0f},{w_xy[1]:.0f})"
            else:
                status += "  W=—"
            if b_xy:
                status += f"  B=({b_xy[0]:.0f},{b_xy[1]:.0f})"
            else:
                status += "  B=—"
            print(status, end="\r", flush=True)
            proj.show(_overlay(size, w_xy, b_xy, roi, status, len(raw_w), len(raw_b)))
            if proj.poll_key(30) in (ord("q"), 27):
                break
    finally:
        print()
        proj.close()
        cam.close()


if __name__ == "__main__":
    main()
