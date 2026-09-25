from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np


class CameraAdapter(ABC):
    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def read(self) -> np.ndarray:
        """Return BGR frame."""

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self) -> CameraAdapter:
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class MockCamera(CameraAdapter):
    """Synthetic top-down table view for laptop development."""

    def __init__(self, size: tuple[int, int] = (1280, 720), board_size_mm: float = 400.0) -> None:
        self.size = size
        self.board_size_mm = board_size_mm
        self._frame: np.ndarray | None = None
        self._token_xy: tuple[int, int] | None = None
        self._occupancy: np.ndarray | None = None

    def open(self) -> None:
        if self._occupancy is None:
            self._occupancy = self._starting_occupancy()
        self._frame = self._render()

    def set_token(self, xy: tuple[int, int] | None) -> None:
        self._token_xy = xy
        self._frame = self._render()

    def set_occupancy(self, grid: np.ndarray) -> None:
        self._occupancy = grid.astype(bool)
        self._frame = self._render()

    def read(self) -> np.ndarray:
        if self._frame is None:
            self.open()
        assert self._frame is not None
        return self._frame.copy()

    def close(self) -> None:
        self._frame = None

    def _starting_occupancy(self) -> np.ndarray:
        g = np.zeros((8, 8), dtype=bool)
        g[0:2, :] = True
        g[6:8, :] = True
        return g

    def _render(self) -> np.ndarray:
        w, h = self.size
        frame = np.full((h, w, 3), 40, dtype=np.uint8)
        side = min(w, h) * 0.55
        x0 = int((w - side) / 2)
        y0 = int((h - side) / 2)
        sq = side / 8
        for r in range(8):
            for c in range(8):
                color = (210, 210, 210) if (r + c) % 2 == 0 else (90, 120, 70)
                x1, y1 = int(x0 + c * sq), int(y0 + r * sq)
                x2, y2 = int(x0 + (c + 1) * sq), int(y0 + (r + 1) * sq)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
                if self._occupancy is not None and self._occupancy[r, c]:
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    piece_color = (30, 30, 30) if r < 4 else (240, 240, 240)
                    cv2.circle(frame, (cx, cy), int(sq * 0.3), piece_color, -1)
        cv2.rectangle(frame, (x0, y0), (int(x0 + side), int(y0 + side)), (255, 255, 255), 2)
        if self._token_xy is not None:
            cv2.circle(frame, self._token_xy, 18, (0, 0, 255), -1)
            cv2.circle(frame, self._token_xy, 18, (255, 255, 255), 2)
        return frame


class FileCamera(CameraAdapter):
    """Replay a still image or video file as the camera."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._cap: cv2.VideoCapture | None = None
        self._still: np.ndarray | None = None

    def open(self) -> None:
        suffix = self.path.suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            img = cv2.imread(str(self.path))
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {self.path}")
            self._still = img
        else:
            self._cap = cv2.VideoCapture(str(self.path))
            if not self._cap.isOpened():
                raise FileNotFoundError(f"Cannot open video: {self.path}")

    def read(self) -> np.ndarray:
        if self._still is not None:
            return self._still.copy()
        assert self._cap is not None
        ok, frame = self._cap.read()
        if not ok:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError("Failed to read from video")
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._still = None


class PicameraAdapter(CameraAdapter):
    """Live Raspberry Pi Camera Module 3 (picamera2)."""

    def __init__(self, size: tuple[int, int] = (1280, 720)) -> None:
        self.size = size
        self._cam = None

    def open(self) -> None:
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise ImportError(
                "picamera2 is required for PicameraAdapter. Install with: pip install -e '.[pi]'"
            ) from exc
        cam = Picamera2()
        config = cam.create_preview_configuration(
            main={"size": self.size, "format": "RGB888"}
        )
        cam.configure(config)
        cam.start()
        self._cam = cam

    def read(self) -> np.ndarray:
        if self._cam is None:
            raise RuntimeError("Camera not open")
        rgb = self._cam.capture_array()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def close(self) -> None:
        if self._cam is not None:
            self._cam.stop()
            self._cam.close()
            self._cam = None


def create_camera(kind: str, size: tuple[int, int], path: Path | None = None) -> CameraAdapter:
    kind = kind.lower()
    if kind == "mock":
        return MockCamera(size=size)
    if kind == "picam":
        return PicameraAdapter(size=size)
    if kind == "file":
        if path is None:
            raise ValueError("camera=file requires a path")
        return FileCamera(path)
    raise ValueError(f"Unknown camera adapter: {kind}")
