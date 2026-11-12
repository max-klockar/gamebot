#!/usr/bin/env python3
"""Visual test: project boxes on every detected white/black piece (strict + tracked).

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


class _MultiTrack:
    """Greedy nearest-neighbour tracks — one per piece, not capped to 1+1."""

    def __init__(self, alpha: float = 0.35, match_dist: float = 55.0, max_miss: int = 12) -> None:
        self.alpha = alpha
        self.match_dist = match_dist
        self.max_miss = max_miss
        self._xy: list[tuple[float, float]] = []
        self._miss: list[int] = []

    def update(self, detections: list[tuple[float, float]]) -> list[tuple[float, float]]:
        unused = list(range(len(detections)))
        new_xy: list[tuple[float, float] | None] = [None] * len(self._xy)
        new_miss = list(self._miss)

        for ti, txy in enumerate(self._xy):
            best_j, best_d = -1, self.match_dist
            for j in unused:
                d = float(np.linalg.norm(np.array(detections[j]) - np.array(txy)))
                if d < best_d:
                    best_d, best_j = d, j
            if best_j >= 0:
                dx, dy = detections[best_j]
                a = self.alpha
                new_xy[ti] = (txy[0] * (1 - a) + dx * a, txy[1] * (1 - a) + dy * a)
                new_miss[ti] = 0
                unused.remove(best_j)
            else:
                new_miss[ti] = new_miss[ti] + 1
                new_xy[ti] = txy if new_miss[ti] <= self.max_miss else None

        kept_xy = [xy for xy, m in zip(new_xy, new_miss) if xy is not None and m <= self.max_miss]
        kept_miss = [m for xy, m in zip(new_xy, new_miss) if xy is not None and m <= self.max_miss]
        for j in unused:
            kept_xy.append(detections[j])
            kept_miss.append(0)
        self._xy = kept_xy
        self._miss = kept_miss
        return list(self._xy)


def _overlay(
    size: tuple[int, int],
    whites: list[tuple[float, float]],
    blacks: list[tuple[float, float]],
    roi: np.ndarray | None,
    status: str,
) -> np.ndarray:
    w, h = size
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    if roi is not None:
        pts = roi.astype(np.int32)
        cv2.polylines(frame, [pts], True, (0, 200, 140), 3)

    for i, (cx, cy) in enumerate(whites, start=1):
        x, y = int(cx), int(cy)
        cv2.rectangle(frame, (x - 24, y - 24), (x + 24, y + 24), (255, 255, 255), 3)
        cv2.putText(frame, f"W{i}", (x - 16, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    for i, (cx, cy) in enumerate(blacks, start=1):
        x, y = int(cx), int(cy)
        cv2.rectangle(frame, (x - 24, y - 24), (x + 24, y + 24), (40, 40, 255), 3)
        cv2.putText(frame, f"B{i}", (x - 16, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 255), 2)

    cv2.putText(frame, status, (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(frame, "piece detect — q quit", (40, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    return frame


def main() -> None:
    p = argparse.ArgumentParser(description="Project boxes on detected white/black pieces")
    p.add_argument("--camera", choices=["mock", "picam", "file"], default="picam")
    p.add_argument("--camera-file", type=Path, default=None, help="Still/video when --camera file")
    p.add_argument("--projector", choices=["mock", "hdmi"], default="hdmi")
    p.add_argument("--full-frame", action="store_true", help="Ignore cal board ROI")
    args = p.parse_args()

    from chess_plugin.vision import find_piece_blobs

    size = (1280, 720)
    cam = create_camera(args.camera, size, args.camera_file)
    proj = create_projector(args.projector, size)
    cam.open()
    proj.open()
    roi = None if args.full_frame else _try_load_board_roi()
    tw, tb = _MultiTrack(), _MultiTrack()
    print(f"piece detect demo — ROI={'cal board' if roi is not None else 'full frame'}")

    try:
        while True:
            frame = cam.read()
            whites, blacks = find_piece_blobs(frame, roi)  # up to 16 each
            w_xy = tw.update([b.xy for b in whites])
            b_xy = tb.update([b.xy for b in blacks])
            status = f"W={len(w_xy)} B={len(b_xy)}"
            print(status, end="\r", flush=True)
            proj.show(_overlay(size, w_xy, b_xy, roi, status))
            if proj.poll_key(30) in (ord("q"), 27):
                break
    finally:
        print()
        proj.close()
        cam.close()


if __name__ == "__main__":
    main()
