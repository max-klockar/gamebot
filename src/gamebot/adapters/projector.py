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
        """Return OpenCV waitKey-style code, or -1."""

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self) -> ProjectorAdapter:
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class MockProjector(ProjectorAdapter):
    """Desktop window standing in for the HDMI projector (needs OpenCV GUI build)."""

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
    """Fullscreen HDMI via pygame (works with opencv-python-headless on the Pi)."""

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
        self._screen = None
        self._pygame = None

    def open(self) -> None:
        try:
            import pygame
        except ImportError as exc:
            raise ImportError(
                "pygame is required for HDMI projection. On the Pi: pip install pygame"
            ) from exc

        self._pygame = pygame
        pygame.display.init()
        pygame.font.init()
        pygame.mouse.set_visible(False)
        self._screen = pygame.display.set_mode(self.size, pygame.FULLSCREEN)
        pygame.display.set_caption(self.window_name)
        self._opened = True

    def show(self, frame_bgr: np.ndarray) -> None:
        if not self._opened:
            self.open()
        assert self._pygame is not None and self._screen is not None
        if frame_bgr.shape[1] != self.size[0] or frame_bgr.shape[0] != self.size[1]:
            frame_bgr = cv2.resize(frame_bgr, self.size)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # pygame wants (w, h, 3); OpenCV is (h, w, 3)
        surface = self._pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        self._screen.blit(surface, (0, 0))
        self._pygame.display.flip()

    def poll_key(self, delay_ms: int = 1) -> int:
        if self._pygame is None:
            return -1
        pg = self._pygame
        for event in pg.event.get():
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    return 27
                if event.key == pg.K_q:
                    return ord("q")
                if event.unicode and len(event.unicode) == 1:
                    return ord(event.unicode)
            if event.type == pg.QUIT:
                return 27
        pg.time.wait(max(0, int(delay_ms)))
        return -1

    def close(self) -> None:
        if self._pygame is not None:
            self._pygame.display.quit()
            self._pygame = None
        self._screen = None
        self._opened = False


def create_projector(kind: str, size: tuple[int, int]) -> ProjectorAdapter:
    kind = kind.lower()
    if kind == "mock":
        return MockProjector(size=size)
    if kind == "hdmi":
        return HdmiProjector(size=size)
    raise ValueError(f"Unknown projector adapter: {kind}")
