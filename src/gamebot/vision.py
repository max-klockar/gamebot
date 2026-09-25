from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class TokenTapDetector:
    """Find a distinct colored token (default red) for projected button presses."""

    hsv_low: tuple[int, int, int] = (0, 120, 80)
    hsv_high: tuple[int, int, int] = (10, 255, 255)
    hsv_low2: tuple[int, int, int] = (170, 120, 80)
    hsv_high2: tuple[int, int, int] = (180, 255, 255)
    min_area: int = 80

    def detect(self, frame_bgr: np.ndarray) -> tuple[int, int] | None:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, np.array(self.hsv_low), np.array(self.hsv_high)),
            cv2.inRange(hsv, np.array(self.hsv_low2), np.array(self.hsv_high2)),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_area = 0.0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area <= best_area:
                continue
            (x, y), _r = cv2.minEnclosingCircle(cnt)
            best = (int(x), int(y))
            best_area = area
        return best
