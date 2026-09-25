from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np


class ProjectorAdapter(ABC):
    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def show(self, frame_bgr: np.ndarray) -> None:
        """Display a full projector frame (BGR)."""

    @abstractmethod
    def poll_key(self, delay_ms: int = 1) -> int:
        """Return OpenCV waitKey code, or -1."""

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self) -> ProjectorAdapter:
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class MockProjector(ProjectorAdapter):
    """Desktop window standing in for the HDMI projector."""

    def __init__(self, size: tuple[int, int] = (1280, 720), window_name: str = "schack-projector") -> None:
        self.size = size
        self.window_name = window_name
        self._opened = False

    def open(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, *self.size)
        self._opened = True

    def show(self, frame_bgr: np.ndarray) -> None:
        if not self._opened:
            self.open()
        if frame_bgr.shape[1] != self.size[0] or frame_bgr.shape[0] != self.size[1]:
            frame_bgr = cv2.resize(frame_bgr, self.size)
        cv2.imshow(self.window_name, frame_bgr)

    def poll_key(self, delay_ms: int = 1) -> int:
        return cv2.waitKey(delay_ms) & 0xFF if delay_ms >= 0 else -1

    def close(self) -> None:
        if self._opened:
            cv2.destroyWindow(self.window_name)
            self._opened = False


class HdmiProjector(ProjectorAdapter):
    """Fullscreen HDMI output intended for the HY300-clone projector."""

    def __init__(
        self,
        size: tuple[int, int] = (1280, 720),
        window_name: str = "schack-hdmi",
        display_index: int = 0,
    ) -> None:
        self.size = size
        self.window_name = window_name
        self.display_index = display_index
        self._opened = False

    def open(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self._opened = True

    def show(self, frame_bgr: np.ndarray) -> None:
        if not self._opened:
            self.open()
        if frame_bgr.shape[1] != self.size[0] or frame_bgr.shape[0] != self.size[1]:
            frame_bgr = cv2.resize(frame_bgr, self.size)
        cv2.imshow(self.window_name, frame_bgr)

    def poll_key(self, delay_ms: int = 1) -> int:
        return cv2.waitKey(delay_ms) & 0xFF if delay_ms >= 0 else -1

    def close(self) -> None:
        if self._opened:
            cv2.destroyWindow(self.window_name)
            self._opened = False


def create_projector(kind: str, size: tuple[int, int]) -> ProjectorAdapter:
    kind = kind.lower()
    if kind == "mock":
        return MockProjector(size=size)
    if kind == "hdmi":
        return HdmiProjector(size=size)
    raise ValueError(f"Unknown projector adapter: {kind}")
