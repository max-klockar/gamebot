from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class AppConfig:
    """Runtime configuration for laptop mock or Pi bring-up."""

    data_dir: Path = field(default_factory=lambda: Path.home() / ".gamebot")
    db_path: Path | None = None
    calibration_path: Path | None = None
    camera: str = "mock"  # mock | picam | file
    projector: str = "mock"  # mock | hdmi
    camera_file: Path | None = None
    projector_size: tuple[int, int] = (1280, 720)
    camera_size: tuple[int, int] = (1280, 720)
    board_size_mm: float = 400.0
    table_margin_mm: float = 120.0
    dwell_ms: int = 600
    dwell_motion_px: float = 8.0
    engine_path: str | None = None
    engine_threads: int = 2
    engine_hash_mb: int = 64
    debug_overlay: bool = True
    identity_homography: bool = True
    led_backend: str = "auto"  # auto | mock | rgb_pwm | ws2812
    hot_celsius: float = 75.0

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if self.db_path is None:
            self.db_path = self.data_dir / "gamebot.db"
        if self.calibration_path is None:
            self.calibration_path = self.data_dir / "calibration.json"

    @property
    def table_size_mm(self) -> tuple[float, float]:
        side = self.board_size_mm + 2 * self.table_margin_mm
        return (side + self.table_margin_mm, side)

    def identity_H(self) -> np.ndarray:
        """Map table mm coords roughly into projector pixels (mock default)."""
        tw, th = self.table_size_mm
        pw, ph = self.projector_size
        sx, sy = pw / tw, ph / th
        return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)
