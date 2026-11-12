#!/usr/bin/env python3
"""Capture a still via picamera2 and dump colour metadata (AWB gains, etc.).

Usage on the Pi:
  ./scripts/run-pi.sh python -m gamebot.camera_color_check
  ./scripts/run-pi.sh python -m gamebot.camera_color_check --awb daylight
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description="Picamera2 colour diagnostic still")
    p.add_argument("--out", type=Path, default=Path("captures/color_check"))
    p.add_argument("--size", default="1280x720")
    p.add_argument(
        "--awb",
        choices=["auto", "daylight", "cloudy", "tungsten", "fluorescent", "indoor", "custom"],
        default="auto",
    )
    p.add_argument("--warmup", type=float, default=1.5, help="Seconds to let AWB settle")
    args = p.parse_args()

    from picamera2 import Picamera2
    from libcamera import controls

    w, h = (int(x) for x in args.size.lower().split("x"))
    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cam = Picamera2()
    config = cam.create_still_configuration(main={"size": (w, h), "format": "RGB888"})
    cam.configure(config)

    awb_map = {
        "auto": controls.AwbModeEnum.Auto,
        "daylight": controls.AwbModeEnum.Daylight,
        "cloudy": controls.AwbModeEnum.Cloudy,
        "tungsten": controls.AwbModeEnum.Tungsten,
        "fluorescent": controls.AwbModeEnum.Fluorescent,
        "indoor": controls.AwbModeEnum.Indoor,
        "custom": controls.AwbModeEnum.Custom,
    }
    cam.set_controls({"AwbEnable": True, "AwbMode": awb_map[args.awb]})
    cam.start()
    try:
        time.sleep(max(0.2, args.warmup))
        # Drain a few frames so AWB can converge
        for _ in range(8):
            cam.capture_array()
        req = cam.capture_request()
        try:
            # Same as PicameraAdapter: do not RGB2BGR — buffer is already OpenCV order.
            bgr = req.make_array("main")
            md = req.get_metadata()
        finally:
            req.release()
    finally:
        cam.stop()
        cam.close()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    jpg = out_dir / f"{stamp}_{args.awb}.jpg"
    cv2.imwrite(str(jpg), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    # Swapped copy for regression: should look *wrong* (blue pliers → orange).
    swapped = bgr[:, :, ::-1]
    swap_path = out_dir / f"{stamp}_{args.awb}_swapped_wrong.jpg"
    cv2.imwrite(str(swap_path), swapped, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    b, g, r = cv2.split(bgr)
    summary = {
        "image": str(jpg),
        "awb_mode": args.awb,
        "size": [w, h],
        "mean_bgr": [float(b.mean()), float(g.mean()), float(r.mean())],
        "ColourGains": md.get("ColourGains"),
        "ColourTemperature": md.get("ColourTemperature"),
        "AeLocked": md.get("AeLocked"),
        "AwbLocked": md.get("AwbLocked"),
        "ExposureTime": md.get("ExposureTime"),
        "AnalogueGain": md.get("AnalogueGain"),
        "note": (
            f"Main {jpg.name} should match reality (blue pliers). "
            f"{swap_path.name} is intentional R/B swap and should look wrong."
        ),
    }
    meta_path = jpg.with_suffix(".json")
    meta_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nsaved {jpg}")
    print(f"saved {swap_path}")
    print(f"saved {meta_path}")


if __name__ == "__main__":
    main()
