#!/usr/bin/env python3
"""Generate synthetic fixture images for vision tests."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "fixtures"


def board_frame(occupancy: np.ndarray | None = None, token_xy: tuple[int, int] | None = None) -> np.ndarray:
    w, h = 1280, 720
    frame = np.full((h, w, 3), 40, dtype=np.uint8)
    side = min(w, h) * 0.55
    x0 = int((w - side) / 2)
    y0 = int((h - side) / 2)
    sq = side / 8
    if occupancy is None:
        occupancy = np.zeros((8, 8), dtype=bool)
        occupancy[0:2, :] = True
        occupancy[6:8, :] = True
    for r in range(8):
        for c in range(8):
            color = (210, 210, 210) if (r + c) % 2 == 0 else (90, 120, 70)
            x1, y1 = int(x0 + c * sq), int(y0 + r * sq)
            x2, y2 = int(x0 + (c + 1) * sq), int(y0 + (r + 1) * sq)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
            if occupancy[r, c]:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                piece = (30, 30, 30) if r < 4 else (240, 240, 240)
                cv2.circle(frame, (cx, cy), int(sq * 0.3), piece, -1)
    cv2.rectangle(frame, (x0, y0), (int(x0 + side), int(y0 + side)), (255, 255, 255), 2)
    if token_xy:
        cv2.circle(frame, token_xy, 18, (0, 0, 255), -1)
    return frame


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    start = board_frame()
    cv2.imwrite(str(ROOT / "board_start.png"), start)

    occ = np.zeros((8, 8), dtype=bool)
    occ[0:2, :] = True
    occ[6:8, :] = True
    # e2e4: vacate rank2 file e (row6,col4), fill rank4 file e (row4,col4)
    occ[6, 4] = False
    occ[4, 4] = True
    cv2.imwrite(str(ROOT / "board_e2e4.png"), board_frame(occ))

    # Token over right-side menu area
    cv2.imwrite(str(ROOT / "token_menu.png"), board_frame(token_xy=(1050, 200)))

    # Calibration-like white dots
    cal = np.zeros((720, 1280, 3), dtype=np.uint8)
    for y in np.linspace(80, 640, 4):
        for x in np.linspace(80, 1200, 6):
            cv2.circle(cal, (int(x), int(y)), 12, (255, 255, 255), -1)
    cv2.imwrite(str(ROOT / "calibration_dots.png"), cal)
    print(f"Wrote fixtures to {ROOT}")


if __name__ == "__main__":
    main()
