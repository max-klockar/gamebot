#!/usr/bin/env python3
"""Smoke-test hardware adapters without starting a full game.

Usage on Pi:
  python -m gamebot.bringup --camera picam --projector hdmi --led
  python -m gamebot.bringup --led-demo              # headless: cycle blue / purple / red
  python -m gamebot.bringup --led-demo --seconds 30
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from gamebot.adapters import create_camera, create_projector
from gamebot.adapters.led import (
    LedMode,
    LedRing,
    Ws2812Backend,
    create_led_backend,
)
from gamebot.calibration import make_grid_pattern


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


def main() -> None:
    p = argparse.ArgumentParser(description="Schack hardware bring-up smoke test")
    p.add_argument("--camera", choices=["mock", "picam", "file"], default="mock")
    p.add_argument("--projector", choices=["mock", "hdmi"], default="mock")
    p.add_argument("--led", action="store_true", help="Enable LED ring hardware backend")
    p.add_argument("--led-demo", action="store_true", help="Cycle blue / purple / red LED modes (headless OK)")
    p.add_argument("--led-count", type=int, default=16, help="WS2812 LED count (default 16)")
    p.add_argument("--loop", action="store_true", help="With --led-demo, repeat until Ctrl+C")
    p.add_argument("--seconds", type=float, default=9.0, help="Total demo length (split across 3 colors)")
    args = p.parse_args()

    if args.led_demo:
        run_led_demo(args.seconds, args.led_count, args.loop)
        return

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
