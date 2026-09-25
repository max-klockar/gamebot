#!/usr/bin/env python3
"""Smoke-test hardware adapters without starting a full game.

Usage on Pi:
  python -m gamebot.bringup --camera picam --projector hdmi --led
  python -m gamebot.bringup --led-demo   # cycle idle / thinking / overheat
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from gamebot.adapters import create_camera, create_projector
from gamebot.adapters.led import LedMode, LedRing, create_led_backend
from gamebot.calibration import make_grid_pattern


def _draw_led_swatch(frame: np.ndarray, led: LedRing, label: str) -> None:
    r, g, b = led.current_rgb
    cv2.rectangle(frame, (frame.shape[1] - 90, 20), (frame.shape[1] - 20, 90), (b, g, r), -1)
    cv2.putText(frame, label, (frame.shape[1] - 160, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1)


def run_led_demo(seconds: float) -> None:
    led = LedRing(backend=create_led_backend("auto"), enabled=True)
    led.open()
    proj = create_projector("mock", (640, 360))
    proj.open()
    modes = [
        (LedMode.IDLE, "idle blue", None),
        (LedMode.THINKING, "thinking purple", None),
        (LedMode.OVERHEAT, "overheat red", 90.0),
    ]
    try:
        per = max(1.0, seconds / len(modes))
        for mode, label, mock_temp in modes:
            led.thermal.set_mock(mock_temp)
            led.set_mode(mode)
            led.tick_thermal()
            t_end = time.time() + per
            while time.time() < t_end:
                frame = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(
                    frame,
                    f"LED demo: {label}",
                    (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                )
                _draw_led_swatch(frame, led, led._effective_mode().value)
                proj.show(frame)
                if proj.poll_key(30) in (ord("q"), 27):
                    return
    finally:
        proj.close()
        led.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Schack hardware bring-up smoke test")
    p.add_argument("--camera", choices=["mock", "picam", "file"], default="mock")
    p.add_argument("--projector", choices=["mock", "hdmi"], default="mock")
    p.add_argument("--led", action="store_true", help="Enable LED ring hardware backend")
    p.add_argument("--led-demo", action="store_true", help="Cycle blue / purple / red LED modes")
    p.add_argument("--seconds", type=float, default=5.0)
    args = p.parse_args()

    if args.led_demo:
        run_led_demo(args.seconds)
        return

    cam = create_camera(args.camera, (1280, 720))
    proj = create_projector(args.projector, (1280, 720))
    led = LedRing(backend=create_led_backend("auto" if args.led else "mock"), enabled=True)
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
