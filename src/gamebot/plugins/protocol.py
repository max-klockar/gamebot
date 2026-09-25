from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from gamebot.geometry import TableLayout
from gamebot.ui import Scene


@dataclass
class IdentifyContext:
    """Camera frame plus any calibrated geometry the plugin may use."""

    layout: TableLayout
    board_corners_camera: np.ndarray | None = None


@dataclass
class RuntimeHandles:
    """Shared host services passed into a plugin session (kept intentionally thin)."""

    data_dir: Path
    layout: TableLayout
    thinking: object  # callable(bool) — LED thinking pulse
    config: object


@runtime_checkable
class GameSession(Protocol):
    def on_enter(self) -> None: ...
    def build_scene(self) -> Scene: ...
    def handle_zone(self, zone_id: str) -> None: ...
    def handle_key(self, key: int) -> None: ...
    def tick(self, frame_bgr: np.ndarray) -> None: ...
    def wants_exit(self) -> bool: ...


@runtime_checkable
class GamePlugin(Protocol):
    id: str
    name: str

    def icon(self) -> np.ndarray:
        """BGR icon image for the chooser (e.g. 128x128)."""

    def identify(self, frame_bgr: np.ndarray, ctx: IdentifyContext) -> bool:
        """Return True if this plugin believes the table shows its game."""

    def create_session(self, runtime: RuntimeHandles) -> GameSession: ...
