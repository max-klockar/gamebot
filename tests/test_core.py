from __future__ import annotations

from pathlib import Path

import chess
import cv2
import numpy as np
import pytest

from chess_plugin.db import Database
from chess_plugin.game import ChessSession, Phase, SkillBand, skill_to_stockfish
from chess_plugin.plugin import ChessPlugin, looks_like_chessboard
from chess_plugin.ui import load_icon
from chess_plugin.vision import hypothesize_moves, occupancy_from_board
from gamebot.adapters.camera import MockCamera
from gamebot.adapters.led import LedMode, LedRing, MockLedBackend, ThermalSensor, pulse_brightness
from gamebot.calibration import Calibrator, CalibrationData, make_grid_pattern
from gamebot.config import AppConfig
from gamebot.geometry import TableLayout
from gamebot.plugins import IdentifyContext, discover_plugins, identify_matches
from gamebot.ui import DwellTapTracker, HitZone, build_game_chooser_scene
from gamebot.vision import TokenTapDetector


@pytest.fixture
def db(tmp_path: Path) -> Database:
    return Database(tmp_path / "test.db")


def test_discover_chess_plugin() -> None:
    plugins = discover_plugins()
    assert any(p.id == "chess" for p in plugins)


def test_chess_icon() -> None:
    icon = load_icon()
    assert icon.shape[0] >= 32 and icon.shape[2] == 3


def test_identify_mock_board() -> None:
    cam = MockCamera()
    cam.open()
    frame = cam.read()
    plugin = ChessPlugin()
    ctx = IdentifyContext(layout=TableLayout())
    assert plugin.identify(frame, ctx) is True


def test_identify_blank_is_false() -> None:
    blank = np.full((720, 1280, 3), 40, dtype=np.uint8)
    assert looks_like_chessboard(blank) is False


def test_chooser_when_zero_or_many() -> None:
    plugins = discover_plugins()
    blank = np.full((720, 1280, 3), 40, dtype=np.uint8)
    ctx = IdentifyContext(layout=TableLayout())
    matches = identify_matches(plugins, blank, ctx)
    assert matches == []
    scene = build_game_chooser_scene(
        TableLayout(), [(p.id, p.name, p.icon()) for p in plugins]
    )
    assert any(z.id.startswith("game:") for z in scene.zones)


def test_skill_mapping() -> None:
    assert skill_to_stockfish(1)[0] == 0
    assert skill_to_stockfish(10)[0] == 20


def test_profiles_and_game(db: Database) -> None:
    p = db.create_profile("Alice", "beginner", 2)
    assert any(x.name == "Guest" for x in db.list_profiles())
    g = db.start_game(p.id, 2)
    db.add_move(g.id, 1, "e2e4", "e4", False)
    moves = db.moves_for_game(g.id)
    assert moves[0]["uci"] == "e2e4"


def test_illegal_and_promotion(db: Database) -> None:
    s = ChessSession(db)
    guest = next(p for p in db.list_profiles() if p.name == "Guest")
    s.select_profile(guest)
    s.select_skill_band(SkillBand.BEGINNER)
    s.select_skill_level(1)
    assert not s.try_human_move(chess.Move.from_uci("e2e5"))
    assert s.revert_hint is not None

    s.board = chess.Board("8/P7/8/8/8/8/8/4K2k w - - 0 1")
    s.phase = Phase.PLAY
    s.game = db.start_game(guest.id, 1)
    assert s.try_human_move(chess.Move.from_uci("a7a8"))
    assert s.phase == Phase.PROMOTE
    assert s.confirm_promotion(chess.QUEEN)


def test_dwell_and_token() -> None:
    poly = np.array([[0, 0], [50, 0], [50, 50], [0, 50]], dtype=np.float64)
    tracker = DwellTapTracker(dwell_ms=100, motion_mm=5)
    t0 = 1000.0
    assert tracker.update([HitZone("a", "A", poly)], (25, 25), now=t0) is None
    assert tracker.update([HitZone("a", "A", poly)], (25, 25), now=t0 + 0.12) == "a"
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(frame, (100, 100), 15, (0, 0, 255), -1)
    assert TokenTapDetector().detect(frame) is not None


def test_hypothesize_and_calibration(tmp_path: Path) -> None:
    board = chess.Board()
    before = occupancy_from_board(board)
    board.push_uci("e2e4")
    after = occupancy_from_board(board)
    board.pop()
    hyps = hypothesize_moves(before, after, board)
    assert hyps[0].move.uci() == "e2e4"
    cal = Calibrator(TableLayout(), (1280, 720), (1280, 720)).mock_calibrate()
    cal.save(tmp_path / "cal.json")
    assert CalibrationData.load(tmp_path / "cal.json").projector_size == (1280, 720)
    img, pts = make_grid_pattern((640, 480), cols=4, rows=3)
    assert len(pts) == 12


def test_led_modes() -> None:
    led = LedRing(backend=MockLedBackend(), enabled=True)
    led.set_mode(LedMode.IDLE)
    r, g, b = led.sample(0.0)
    assert b > r
    led.set_mode(LedMode.THINKING)
    r, g, b = led.sample(0.0)
    assert r > 50 and b > 50
    led.thermal = ThermalSensor(hot_celsius=75)
    led.thermal.set_mock(90)
    led.tick_thermal()
    assert led._effective_mode() == LedMode.OVERHEAT
    assert pulse_brightness(0.0, 0.12, 1.0) <= 1.0


def test_config(tmp_path: Path) -> None:
    cfg = AppConfig(data_dir=tmp_path / "data")
    assert cfg.data_dir.exists()
