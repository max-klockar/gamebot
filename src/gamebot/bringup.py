#!/usr/bin/env python3
"""Smoke-test hardware adapters without starting a full game.

Usage on Pi:
  python -m gamebot.bringup --camera picam --projector hdmi --led
  python -m gamebot.bringup --led-demo              # headless: cycle blue / purple / red
  python -m gamebot.bringup --calibrate             # auto colour-corner cal + save verify jpg
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from gamebot.adapters import create_camera, create_projector
from gamebot.adapters.led import (
    LedMode,
    LedRing,
    Ws2812Backend,
    create_led_backend,
)
from gamebot.calibration import Calibrator, make_grid_pattern
from gamebot.geometry import TableLayout


def _draw_led_swatch(frame: np.ndarray, led: LedRing, label: str) -> None:
    r, g, b = led.current_rgb
    cv2.rectangle(frame, (frame.shape[1] - 90, 20), (frame.shape[1] - 20, 90), (b, g, r), -1)
    cv2.putText(frame, label, (frame.shape[1] - 160, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1)


def _make_led_ring(kind: str, led_count: int) -> LedRing:
    if kind == "ws2812" or (kind == "auto" and _ws2812_available()):
        backend = Ws2812Backend(count=led_count)
    else:
        backend = create_led_backend(kind)
    return LedRing(backend=backend, enabled=True)


def _ws2812_available() -> bool:
    try:
        import RPi.GPIO  # noqa: F401
        import rpi_ws281x  # noqa: F401

        return True
    except ImportError:
        return False


def run_led_demo(seconds: float, led_count: int, loop: bool) -> None:
    """Pulse blue → purple → red on the strip. No projector/camera needed."""
    led = _make_led_ring("auto", led_count)
    led.open()
    modes = [
        (LedMode.IDLE, "idle blue", None),
        (LedMode.THINKING, "thinking purple", None),
        (LedMode.OVERHEAT, "overheat red", 90.0),
    ]
    print(f"LED demo on {led_count} LEDs — Ctrl+C to stop")
    try:
        while True:
            per = max(1.0, seconds / len(modes))
            for mode, label, mock_temp in modes:
                led.thermal.set_mock(mock_temp)
                led.set_mode(mode)
                led.tick_thermal()
                print(f"  → {label}")
                t_end = time.time() + per
                while time.time() < t_end:
                    time.sleep(0.05)
            if not loop:
                break
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        led.close()


def run_auto_calibrate(camera: str, projector: str, out_jpg: Path, settle_s: float = 1.0) -> int:
    """Search coloured corners onto the table, solve, verify grid, write debug JPG."""
    from gamebot.calibration import render_color_corner_pattern

    size = (1280, 720)
    layout = TableLayout()
    calibrator = Calibrator(layout, size, size)
    cam = create_camera(camera, size)
    proj = create_projector(projector, size)
    cam.open()
    proj.open()

    def show_and_grab(points: np.ndarray, status: str) -> np.ndarray:
        pattern = render_color_corner_pattern(size, points, status=status)
        for _ in range(2):
            proj.show(pattern)
            proj.poll_key(20)
            cam.read()
        proj.show(pattern)
        proj.poll_key(20)
        return cam.read()

    def show_frame(image: np.ndarray) -> np.ndarray:
        for _ in range(2):
            proj.show(image)
            proj.poll_key(25)
            cam.read()
        proj.show(image)
        proj.poll_key(25)
        return cam.read()

    print("searching for colour corners (start tight/center, then steer)…")
    result = calibrator.auto_calibrate_with_search(show_and_grab, show_frame)
    if result is None:
        # Last frame dump
        frame = cam.read()
        print("FAILED: could not get all 4 coloured dots onto the visible table")
        cv2.imwrite(str(out_jpg), frame)
        print(f"wrote raw camera frame → {out_jpg}")
        proj.close()
        cam.close()
        return 1
    out_jpg = Path(out_jpg)
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_jpg), result.camera_debug_bgr)
    cal_path = out_jpg.with_suffix(".json")
    result.calibration.save(cal_path)
    print(result.message)
    print(f"wrote {out_jpg}")
    print(f"wrote {cal_path}")
    print("showing green verify grid on projector for 8s — check table alignment")
    t_end = time.time() + 8.0
    while time.time() < t_end:
        proj.show(result.verify_projector_bgr)
        if proj.poll_key(30) in (ord("q"), 27):
            break
    proj.close()
    cam.close()
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Schack hardware bring-up smoke test")
    p.add_argument("--camera", choices=["mock", "picam", "file"], default="mock")
    p.add_argument("--projector", choices=["mock", "hdmi"], default="mock")
    p.add_argument("--led", action="store_true", help="Enable LED ring hardware backend")
    p.add_argument("--led-demo", action="store_true", help="Cycle blue / purple / red LED modes (headless OK)")
    p.add_argument("--calibrate", action="store_true", help="Auto colour-corner calibration + verify image")
    p.add_argument("--out", type=Path, default=Path.home() / "cal_verify.jpg", help="Debug JPG path")
    p.add_argument("--led-count", type=int, default=90, help="WS2812 LED count (default 90)")
    p.add_argument("--loop", action="store_true", help="With --led-demo, repeat until Ctrl+C")
    p.add_argument("--seconds", type=float, default=9.0, help="Total demo length (split across 3 colors)")
    args = p.parse_args()

    if args.led_demo:
        run_led_demo(args.seconds, args.led_count, args.loop)
        return

    if args.calibrate:
        raise SystemExit(run_auto_calibrate(args.camera, args.projector, args.out))

    cam = create_camera(args.camera, (1280, 720))
    proj = create_projector(args.projector, (1280, 720))
    led = _make_led_ring("auto" if args.led else "mock", args.led_count)
    cam.open()
    proj.open()
    led.open()
    led.set_mode(LedMode.IDLE)

    pattern, _ = make_grid_pattern((1280, 720))
    t_end = time.time() + args.seconds
    print(f"Showing calibration grid for {args.seconds}s; press q to quit early")
    while time.time() < t_end:
        frame = cam.read()
        out = pattern.copy()
        thumb = cv2.resize(frame, (320, 180))
        out[20:200, 20:340] = thumb
        cv2.putText(out, "bring-up", (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        _draw_led_swatch(out, led, led._effective_mode().value)
        proj.show(out)
        if proj.poll_key(30) in (ord("q"), 27):
            break

    led.close()
    proj.close()
    cam.close()
    print("bring-up done")


if __name__ == "__main__":
    main()
