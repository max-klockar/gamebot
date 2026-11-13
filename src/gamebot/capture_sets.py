#!/usr/bin/env python3
"""Guided capture sets for piece-detect training (SSH-friendly).

Three phases × N stills (N can differ per phase). Projector shows black or a
chessboard; you rearrange pieces, then press Enter in the SSH terminal to shoot.

  ./scripts/run-pi.sh python -m gamebot.capture_sets
  ./scripts/run-pi.sh python -m gamebot.capture_sets --n 10
  ./scripts/run-pi.sh python -m gamebot.capture_sets --n 0,0,10   # phase 3 only

Controls (stdin):
  Enter     — capture next still (or start phase)
  <number>  — at phase start: override shot count for this phase
  s         — skip rest of this phase / skip phase
  q         — quit

Training projection ignores calibration pose/size (cal boards often sit
off-centre and clip). Chess is a centred square in projector pixels with mild
size jitter. Phase-1 guide border is inset ~10% so projector overscan still
shows it; it goes black for each shot.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from gamebot.adapters import DEFAULT_CAMERA_ZOOM, create_camera, create_projector
from gamebot.calibration import CalibrationData


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
        title="Phase 1/3 — no physical board, project BLACK",
        prep="Clear the table (no chessboard). Place pieces on the green table.",
    ),
    CaptureSet(
        folder="02_board_blackproj",
        project="black",
        physical_board=True,
        title="Phase 2/3 — physical board, project BLACK",
        prep="Put the physical chessboard on the table. Place pieces on various squares.",
    ),
    CaptureSet(
        folder="03_noboard_chessproj",
        project="chess",
        physical_board=False,
        title="Phase 3/3 — no physical board, project CHESSBOARD",
        prep="Remove the physical board. Place pieces where the projected squares are.",
    ),
)


def _parse_counts(raw: str, n_sets: int) -> list[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise SystemExit("--n needs at least one count")
    try:
        nums = [int(p) for p in parts]
    except ValueError as e:
        raise SystemExit(f"bad --n {raw!r}: need integers") from e
    if any(n < 0 for n in nums):
        raise SystemExit("--n counts must be >= 0")
    if len(nums) == 1:
        return nums * n_sets
    if len(nums) != n_sets:
        raise SystemExit(f"--n needs 1 value or {n_sets} comma-separated counts, got {len(nums)}")
    return nums


def _load_cal() -> CalibrationData | None:
    path = Path.home() / ".gamebot" / "calibration.json"
    if not path.is_file():
        return None
    try:
        return CalibrationData.load(path)
    except Exception:
        return None


def _guide_border_frame(size: tuple[int, int]) -> np.ndarray:
    """FOV outline inset enough to survive typical projector overscan.

    Only shown while arranging — cleared to black before the camera fires.
    """
    w, h = size
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # ~10% inset: a 8px rim is invisible on many projectors
    inset = max(48, int(round(min(w, h) * 0.10)))
    thickness = max(10, inset // 6)
    color = (0, 255, 180)
    pts = np.array(
        [
            [inset, inset],
            [w - 1 - inset, inset],
            [w - 1 - inset, h - 1 - inset],
            [inset, h - 1 - inset],
        ],
        dtype=np.int32,
    )
    cv2.polylines(frame, [pts], True, color, thickness)
    tick = max(40, inset)
    for x, y, dx, dy in (
        (inset, inset, 1, 1),
        (w - 1 - inset, inset, -1, 1),
        (w - 1 - inset, h - 1 - inset, -1, -1),
        (inset, h - 1 - inset, 1, -1),
    ):
        cv2.line(frame, (x, y), (x + dx * tick, y), color, thickness)
        cv2.line(frame, (x, y), (x, y + dy * tick), color, thickness)
    cv2.putText(
        frame,
        "GUIDE (overscan-safe) — goes BLACK for each shot",
        (inset + 12, inset + 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (240, 240, 240),
        2,
    )
    return frame


def _training_chessboard(
    size: tuple[int, int],
    rng: random.Random,
    *,
    side_frac: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Centred square 8×8 in projector pixels — ignores calibration pose/size.

    Mild size (+ tiny centre) jitter for variety; always fully inside the frame
    with overscan margin so the board never clips off the projector.
    """
    w, h = size
    short = min(w, h)
    margin = max(40, int(round(short * 0.08)))
    max_side = short - 2 * margin
    frac = float(side_frac) if side_frac is not None else rng.uniform(0.50, 0.68)
    side = int(round(short * frac))
    side = max(160, min(side, max_side))

    x0 = (w - side) // 2 + rng.randint(-margin // 3, margin // 3)
    y0 = (h - side) // 2 + rng.randint(-margin // 3, margin // 3)
    x0 = int(np.clip(x0, margin, w - side - margin))
    y0 = int(np.clip(y0, margin, h - side - margin))

    frame = np.zeros((h, w, 3), dtype=np.uint8)
    sq = side / 8.0
    light = (245, 245, 245)
    dark = (45, 110, 45)
    for r in range(8):
        for c in range(8):
            color = light if (r + c) % 2 == 0 else dark
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
    meta = {
        "board_px": [x0, y0, side],
        "side_frac": round(side / short, 4),
        "source": "training_centred",
    }
    return frame, meta


def _black_frame(size: tuple[int, int]) -> np.ndarray:
    w, h = size
    return np.zeros((h, w, 3), dtype=np.uint8)


def _prompt(msg: str) -> str:
    try:
        return input(msg).strip().lower()
    except EOFError:
        return "q"


def _phase_plan(default_n: int) -> int | None:
    """Return shot count for this phase, or None to skip, or raise SystemExit on quit."""
    while True:
        cmd = _prompt(
            f"Ready? [Enter={default_n} shots, number=override, s=skip phase, q=quit] "
        )
        if cmd in ("", "y", "yes", "go"):
            return default_n
        if cmd == "q":
            raise SystemExit(0)
        if cmd in ("s", "skip"):
            return None
        if cmd.isdigit():
            return int(cmd)
        print("  type Enter, a number, s, or q")


def main() -> None:
    p = argparse.ArgumentParser(description="Guided piece-capture sets (Enter to shoot)")
    p.add_argument(
        "--n",
        type=str,
        default="5",
        help="Stills per phase: one number for all, or comma list matching phases "
        f"(default 5). Example: --n 0,0,10 for phase 3 only",
    )
    p.add_argument("--out", type=Path, default=Path("captures/sets"))
    p.add_argument("--camera", choices=["picam", "mock"], default="picam")
    p.add_argument("--projector", choices=["hdmi", "mock"], default="hdmi")
    p.add_argument("--size", default="1280x720")
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument(
        "--zoom",
        type=float,
        default=DEFAULT_CAMERA_ZOOM,
        help=f"Digital center zoom (default {DEFAULT_CAMERA_ZOOM}; match gamebot --zoom)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for chessboard size jitter (default: random)",
    )
    args = p.parse_args()

    counts = _parse_counts(args.n, len(SETS))
    size = tuple(int(x) for x in args.size.lower().split("x"))
    assert len(size) == 2
    out_root = args.out.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    cal = _load_cal()  # optional meta only — not used for projection pose
    session = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    zoom = float(args.zoom)
    rng = random.Random(args.seed)

    cam = create_camera(args.camera, size, zoom=zoom)
    proj = create_projector(args.projector, size)
    cam.open()
    proj.open()

    print()
    print("Guided capture — rearrange pieces, then press Enter in THIS SSH session.")
    print("Commands: Enter=shoot  s=next phase  q=quit")
    print(f"Plan: " + " | ".join(f"{s.folder}×{n}" for s, n in zip(SETS, counts)))
    print(f"Output: {out_root}")
    print(f"Zoom: {zoom}× (must match play / cal)")
    print("Chess proj: centred training square (ignores cal pose — avoids off-screen boards)")
    print(f"Cal on disk: {'yes (meta only)' if cal else 'no'}")
    print()

    try:
        for _ in range(max(0, args.warmup)):
            cam.read()
            time.sleep(0.04)

        for aset, default_n in zip(SETS, counts):
            set_dir = out_root / aset.folder
            set_dir.mkdir(parents=True, exist_ok=True)
            use_guide = not aset.physical_board and aset.project == "black"
            chess_meta: dict | None = None

            if aset.project == "chess":
                arrange, chess_meta = _training_chessboard(size, rng)
                shoot = arrange
            elif use_guide:
                arrange = _guide_border_frame(size)
                shoot = _black_frame(size)
            else:
                arrange = _black_frame(size)
                shoot = arrange

            proj.show(arrange)

            print("=" * 60)
            print(aset.title)
            print(aset.prep)
            if use_guide:
                print("Guide border ON (inset ~10%) while you arrange — BLACK for each shot.")
            if aset.project == "chess" and chess_meta is not None:
                print(
                    f"Training board side≈{chess_meta['side_frac']:.0%} of short edge "
                    f"(re-rolls slightly each shot)."
                )
            print(f"Default for this phase: {default_n} shots (vary piece counts/angles).")
            print("=" * 60)
            try:
                n = _phase_plan(default_n)
            except SystemExit:
                print("quit")
                break
            if n is None or n == 0:
                print("  skip phase")
                continue
            proj.show(arrange)

            for i in range(1, n + 1):
                cmd = _prompt(
                    f"  [{aset.folder}] shot {i}/{n} — Enter=capture, s=next phase, q=quit: "
                )
                if cmd == "q":
                    print("quit")
                    return
                if cmd in ("s", "skip"):
                    print("  next phase")
                    break

                proj.show(shoot)
                time.sleep(0.20)
                img = cam.read()
                name = f"{session}_{i:03d}.jpg"
                path = set_dir / name
                if not cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
                    raise RuntimeError(f"failed to write {path}")
                meta = {
                    "image": name,
                    "set": aset.folder,
                    "project": aset.project,
                    "physical_board": aset.physical_board,
                    "size": [int(img.shape[1]), int(img.shape[0])],
                    "camera_zoom": zoom,
                    "guide_cleared_for_shot": use_guide,
                    "projected_board": chess_meta,
                    "captured_at": datetime.now(timezone.utc).isoformat(),
                }
                path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
                print(f"  saved {path}")
                if i < n:
                    if aset.project == "chess":
                        arrange, chess_meta = _training_chessboard(size, rng)
                        shoot = arrange
                    proj.show(arrange)
                    print("  → move pieces to new spots, then continue")

        print()
        print("done.")
        print(f"scp -r max@gamebot:{out_root} ./captures/sets")
    finally:
        proj.close()
        cam.close()


if __name__ == "__main__":
    main()
