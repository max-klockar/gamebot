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
from gamebot.calibration import (
    Calibrator,
    CalibrationData,
    detect_chessboard_outer_corners,
    detect_color_corners,
    detect_dot_by_difference,
    make_grid_pattern,
    order_quad_tl_tr_br_bl,
    quad_looks_sane,
    render_color_corner_pattern,
)
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
    from chess_plugin.game import elo_to_skill, SkillBand

    band, level = elo_to_skill(500)
    assert band == SkillBand.BEGINNER and level == 1


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
    messy = np.array([[200.0, 10.0], [10.0, 10.0], [10.0, 200.0], [200.0, 200.0]])
    ordered = order_quad_tl_tr_br_bl(messy)
    assert ordered[0].tolist() == [10.0, 10.0]
    assert ordered[2].tolist() == [200.0, 200.0]
    # Synthetic camera view of the colour pattern should detect 4 blobs.
    pattern = render_color_corner_pattern((640, 360))
    found = detect_color_corners(pattern)
    assert found is not None and len(found) == 4
    cal = Calibrator(TableLayout(), (640, 360), (640, 360)).auto_calibrate(pattern)
    assert cal is not None


def test_difference_dot_ignores_static_lamp() -> None:
    dark = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.circle(dark, (40, 40), 12, (255, 255, 255), -1)  # static lamp
    lit = dark.copy()
    cv2.circle(lit, (200, 150), 18, (0, 0, 255), -1)  # projected dot
    pt = detect_dot_by_difference(dark, lit)
    assert pt is not None
    assert abs(pt[0] - 200) < 8 and abs(pt[1] - 150) < 8
    assert not quad_looks_sane(np.array([[10, 10], [12, 10], [11, 11], [10, 40]], dtype=np.float64))


def test_chessboard_center_with_start_pieces() -> None:
    """Empty middle ranks (start position) should still yield an outer board quad."""
    size = 400
    sq = size // 8
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for r in range(8):
        for c in range(8):
            if (r + c) % 2 == 0:
                img[r * sq : (r + 1) * sq, c * sq : (c + 1) * sq] = 220
            else:
                img[r * sq : (r + 1) * sq, c * sq : (c + 1) * sq] = 40
    # Cover ranks 1–2 and 7–8 (top/bottom two rows) like starting pieces.
    img[0 : 2 * sq, :] = 30
    img[6 * sq : 8 * sq, :] = 200
    outer = detect_chessboard_outer_corners(img)
    assert outer is not None
    ordered = order_quad_tl_tr_br_bl(outer)
    assert ordered[0, 0] < 30 and ordered[0, 1] < 30
    assert ordered[2, 0] > size - 30 and ordered[2, 1] > size - 30


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
