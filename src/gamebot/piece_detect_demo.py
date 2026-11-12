#!/usr/bin/env python3
"""Temporary visual test: find white/black pieces and project boxes around them.

Usage on Pi:
  export DISPLAY=:0
  python -m gamebot.piece_detect_demo --camera picam --projector hdmi
  # or via sudo wrapper if needed for display:
  # sudo -E .venv/bin/python -m gamebot.piece_detect_demo --camera picam --projector hdmi

Press q / Esc to quit.
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from gamebot.adapters import create_camera, create_projector
from gamebot.calibration import CalibrationData
from pathlib import Path


def _try_load_board_roi() -> np.ndarray | None:
    path = Path.home() / ".gamebot" / "calibration.json"
    if not path.is_file():
        return None
    try:
        cal = CalibrationData.load(path)
        return cal.board_corners_camera
    except Exception:
        return None


def _detect_blobs(frame_bgr: np.ndarray, roi: np.ndarray | None) -> tuple[list[tuple], list[tuple]]:
    """Return (white_blobs, black_blobs) as (cx, cy, area) in full-frame coords."""
    work = frame_bgr
    offset = np.array([0.0, 0.0])
    if roi is not None and len(roi) >= 4:
        xs, ys = roi[:, 0], roi[:, 1]
        x0 = max(0, int(xs.min()) - 30)
        y0 = max(0, int(ys.min()) - 30)
        x1 = min(frame_bgr.shape[1], int(xs.max()) + 30)
        y1 = min(frame_bgr.shape[0], int(ys.max()) + 30)
        if x1 - x0 >= 40 and y1 - y0 >= 40:
            work = frame_bgr[y0:y1, x0:x1]
            offset = np.array([float(x0), float(y0)])

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    whites: list[tuple] = []
    blacks: list[tuple] = []

    # Multi-blob: find several bright / dark regions
    w_thr = max(165.0, float(np.percentile(blur, 90)))
    b_thr = min(85.0, float(np.percentile(blur, 15)))
    w_mask = cv2.threshold(blur, w_thr, 255, cv2.THRESH_BINARY)[1]
    b_mask = cv2.threshold(blur, b_thr, 255, cv2.THRESH_BINARY_INV)[1]
    w_mask = cv2.morphologyEx(w_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    b_mask = cv2.morphologyEx(b_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    for mask, out in ((w_mask, whites), (b_mask, blacks)):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < 80 or area > 30000:
                continue
            m = cv2.moments(cnt)
            if m["m00"] == 0:
                continue
            cx = m["m10"] / m["m00"] + offset[0]
            cy = m["m01"] / m["m00"] + offset[1]
            out.append((cx, cy, area))
        out.sort(key=lambda t: t[2], reverse=True)

    return whites[:6], blacks[:6]


def _overlay(
    size: tuple[int, int],
    whites: list[tuple],
    blacks: list[tuple],
    roi: np.ndarray | None,
    status: str,
) -> np.ndarray:
    w, h = size
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    if roi is not None:
        pts = roi.astype(np.int32)
        cv2.polylines(frame, [pts], True, (0, 200, 140), 3)
        cv2.putText(frame, "board ROI", tuple(pts[0] + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 140), 2)

    for cx, cy, area in whites:
        side = max(40, int(np.sqrt(area) * 1.4))
        x0, y0 = int(cx - side / 2), int(cy - side / 2)
        cv2.rectangle(frame, (x0, y0), (x0 + side, y0 + side), (255, 255, 255), 4)
        cv2.putText(frame, "W", (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    for cx, cy, area in blacks:
        side = max(40, int(np.sqrt(area) * 1.4))
        x0, y0 = int(cx - side / 2), int(cy - side / 2)
        cv2.rectangle(frame, (x0, y0), (x0 + side, y0 + side), (40, 40, 255), 4)
        cv2.putText(frame, "B", (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 255), 2)

    cv2.putText(frame, status, (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(
        frame,
        "piece detect test — q to quit",
        (40, h - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (200, 200, 200),
        2,
    )
    return frame


def main() -> None:
    p = argparse.ArgumentParser(description="Project boxes around detected chess pieces")
    p.add_argument("--camera", choices=["mock", "picam", "file"], default="picam")
    p.add_argument("--projector", choices=["mock", "hdmi"], default="hdmi")
    p.add_argument("--full-frame", action="store_true", help="Ignore cal board ROI; search whole image")
    args = p.parse_args()

    size = (1280, 720)
    cam = create_camera(args.camera, size)
    proj = create_projector(args.projector, size)
    cam.open()
    proj.open()
    roi = None if args.full_frame else _try_load_board_roi()
    print(f"piece detect demo — ROI={'cal board' if roi is not None else 'full frame'}")

    try:
        while True:
            frame = cam.read()
            whites, blacks = _detect_blobs(frame, roi)
            status = f"white={len(whites)}  black={len(blacks)}"
            if whites:
                status += f"  W1=({whites[0][0]:.0f},{whites[0][1]:.0f})"
            if blacks:
                status += f"  B1=({blacks[0][0]:.0f},{blacks[0][1]:.0f})"
            print(status, end="\r", flush=True)
            proj.show(_overlay(size, whites, blacks, roi, status))
            key = proj.poll_key(30)
            if key in (ord("q"), 27):
                break
    finally:
        print()
        proj.close()
        cam.close()


if __name__ == "__main__":
    main()
