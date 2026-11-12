#!/usr/bin/env python3
"""Guided capture sets for piece-detect training (SSH-friendly).

Three sets × N stills. Projector shows black or a chessboard; you rearrange
pieces, then press Enter in the SSH terminal to shoot.

  ./scripts/run-pi.sh python -m gamebot.capture_sets

Controls (stdin):
  Enter  — capture next still
  s      — skip rest of this set
  q      — quit
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from gamebot.adapters import create_camera, create_projector
from gamebot.calibration import CalibrationData
from gamebot.geometry import TableLayout
from gamebot.table_surface import draw_projected_chessboard


@dataclass(frozen=True)
class CaptureSet:
    folder: str
    project: str  # "black" | "chess"
    physical_board: bool
    title: str
    prep: str


SETS: tuple[CaptureSet, ...] = (
    CaptureSet(
        folder="01_noboard_blackproj",
        project="black",
        physical_board=False,
        title="Set 1/3 — no physical board, project BLACK",
        prep="Clear the table (no chessboard). Place 2–3 pieces on the green table.",
    ),
    CaptureSet(
        folder="02_board_blackproj",
        project="black",
        physical_board=True,
        title="Set 2/3 — physical board, project BLACK",
        prep="Put the physical chessboard on the table. Place pieces on various squares.",
    ),
    CaptureSet(
        folder="03_noboard_chessproj",
        project="chess",
        physical_board=False,
        title="Set 3/3 — no physical board, project CHESSBOARD",
        prep="Remove the physical board. Place pieces where the projected squares are.",
    ),
)


def _load_cal() -> CalibrationData | None:
    path = Path.home() / ".gamebot" / "calibration.json"
    if not path.is_file():
        return None
    try:
        return CalibrationData.load(path)
    except Exception:
        return None


def _proj_frame(size: tuple[int, int], kind: str, cal: CalibrationData | None) -> np.ndarray:
    w, h = size
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    if kind == "black":
        return frame
    layout = TableLayout()
    if cal is not None:
        draw_projected_chessboard(frame, layout, cal.H_table_to_projector)
    else:
        # No cal: centred board in projector pixels (rough, still useful).
        side = int(min(w, h) * 0.62)
        x0 = (w - side) // 2
        y0 = (h - side) // 2
        sq = side / 8
        for r in range(8):
            for c in range(8):
                color = (245, 245, 245) if (r + c) % 2 == 0 else (45, 110, 45)
                x1, y1 = int(x0 + c * sq), int(y0 + r * sq)
                x2, y2 = int(x0 + (c + 1) * sq), int(y0 + (r + 1) * sq)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        cv2.polylines(
            frame,
            [np.array([[x0, y0], [x0 + side, y0], [x0 + side, y0 + side], [x0, y0 + side]], dtype=np.int32)],
            True,
            (0, 220, 140),
            4,
        )
        cv2.putText(frame, "NO CAL — centred board", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return frame


def _prompt(msg: str) -> str:
    try:
        return input(msg).strip().lower()
    except EOFError:
        return "q"


def main() -> None:
    p = argparse.ArgumentParser(description="Guided piece-capture sets (Enter to shoot)")
    p.add_argument("--n", type=int, default=5, help="Stills per set")
    p.add_argument("--out", type=Path, default=Path("captures/sets"))
    p.add_argument("--camera", choices=["picam", "mock"], default="picam")
    p.add_argument("--projector", choices=["hdmi", "mock"], default="hdmi")
    p.add_argument("--size", default="1280x720")
    p.add_argument("--warmup", type=int, default=8)
    args = p.parse_args()

    size = tuple(int(x) for x in args.size.lower().split("x"))
    assert len(size) == 2
    out_root = args.out.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    cal = _load_cal()
    session = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    cam = create_camera(args.camera, size)
    proj = create_projector(args.projector, size)
    cam.open()
    proj.open()

    print()
    print("Guided capture — rearrange pieces, then press Enter in THIS SSH session.")
    print("Commands: Enter=shoot  s=skip set  q=quit")
    print(f"Output: {out_root}")
    print(f"Calibration: {'yes' if cal else 'NO (chess proj will be centred guess)'}")
    print()

    try:
        for _ in range(max(0, args.warmup)):
            cam.read()
            time.sleep(0.04)

        for aset in SETS:
            set_dir = out_root / aset.folder
            set_dir.mkdir(parents=True, exist_ok=True)
            frame = _proj_frame(size, aset.project, cal)
            proj.show(frame)

            print("=" * 60)
            print(aset.title)
            print(aset.prep)
            print(f"Will take {args.n} shots. Move pieces between shots.")
            print("=" * 60)
            ready = _prompt("Ready for this set? [Enter=go, q=quit] ")
            if ready == "q":
                break
            proj.show(frame)

            for i in range(1, args.n + 1):
                cmd = _prompt(f"  [{aset.folder}] shot {i}/{args.n} — Enter=capture, s=skip set, q=quit: ")
                if cmd == "q":
                    print("quit")
                    return
                if cmd == "s":
                    print("  skip set")
                    break
                # Refresh projection in case the display blanked
                proj.show(frame)
                time.sleep(0.15)
                img = cam.read()
                name = f"{session}_{i:02d}.jpg"
                path = set_dir / name
                if not cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
                    raise RuntimeError(f"failed to write {path}")
                meta = {
                    "image": name,
                    "set": aset.folder,
                    "project": aset.project,
                    "physical_board": aset.physical_board,
                    "size": [int(img.shape[1]), int(img.shape[0])],
                    "board_corners_camera": None
                    if cal is None
                    else cal.board_corners_camera.astype(float).tolist(),
                    "captured_at": datetime.now(timezone.utc).isoformat(),
                }
                path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
                print(f"  saved {path}")
                if i < args.n:
                    print("  → move pieces to new spots, then continue")

        print()
        print("done.")
        print(f"scp -r max@gamebot:{out_root} ./captures/sets")
    finally:
        proj.close()
        cam.close()


if __name__ == "__main__":
    main()
