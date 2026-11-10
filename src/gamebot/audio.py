from __future__ import annotations

from importlib.resources import files
from pathlib import Path


def default_ambient_path() -> Path:
    return Path(str(files("gamebot").joinpath("assets/ambient.mp3")))


class AmbientPlayer:
    """Optional looping music. Disabled by default — Pi+sudo has no usable HDMI audio."""

    def __init__(
        self,
        path: Path | None = None,
        volume: float = 0.4,
        *,
        enabled: bool = False,
    ) -> None:
        self.path = path if path is not None else default_ambient_path()
        self.volume = float(volume)
        self.enabled = enabled
        self._playing = False

    def start(self) -> None:
        if not self.enabled or self._playing:
            return
        try:
            import pygame
        except ImportError:
            print("ambient: pygame not installed — skipping")
            return
        if not self.path.is_file():
            print(f"ambient: file missing ({self.path})")
            return
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.music.load(str(self.path))
            pygame.mixer.music.set_volume(self.volume)
            pygame.mixer.music.play(loops=-1)
            self._playing = True
            print(f"ambient: looping {self.path.name} @ {self.volume:.0%}")
        except Exception as exc:  # noqa: BLE001 — audio must never kill the host
            print(f"ambient: failed to start ({exc})")
            self._playing = False

    def stop(self) -> None:
        if not self._playing:
            return
        try:
            import pygame

            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
        except Exception:  # noqa: BLE001
            pass
        self._playing = False
