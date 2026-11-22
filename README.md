# Gamebot

Generic projected tabletop game machine. Games live in separate plugin packages.

```bash
cd gamebot
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" -e "../chess-plugin[dev]"
gamebot --mock
```

After calibrate, plugins run `identify()`. One match starts that game; zero or many matches shows an icon chooser on the table.

## Chess vision (pieces + board)

**Runtime board lock:** YOLO AABB (`board.onnx` via onnxruntime) + classical
corner refine (chessboard corners → wood contour → Hough). **Do not**
`pip install ultralytics` on the Pi — it pulls Torch/CUDA and fills the card.

```bash
pip install -e ".[vision]"   # onnxruntime only (~tens of MB)
python -m gamebot.board_detect_demo path/to/still.jpg
# on Pi: ./scripts/run-pi.sh python -m gamebot.board_detect_demo --camera
```

**Training** (LabelMe, YOLO, datasets) lives in laptop repo `gamebot-vision`.
Stills are **not** kept in this git tree:

- Pi (temp): `~/.gamebot/captures/sets/`
- Laptop: `gamebot-vision/captures/sets/` (gitignored)

```bash
# on laptop
cd ../gamebot-vision
./scripts/pull-captures.sh max@gamebot --purge
./scripts/label.sh captures/sets/02_board_blackproj
./scripts/train-board.sh                     # AABB finder
./scripts/train-board-pose.sh                # optional; AABB fallback only
./scripts/train-pieces.sh                    # skips chessproj
./scripts/export-weights.sh ../gamebot
```

On the Pi (same `--zoom` as play; default **1.2×** center crop).
`run-pi.sh` is root for LEDs but captures land in `/home/<user>/.gamebot/` via `SUDO_USER`:

```bash
./scripts/run-pi.sh python -m gamebot.capture_sets --n 0,20,0   # phase 2 only
./scripts/run-pi.sh python -m gamebot.capture_sets --n 10
# Enter=shoot, number=override, s=next phase, q=quit
```

Weights in `models/vision/`: `board.onnx` (Pi AABB), `board.pt` / `pieces.pt` (laptop train).

## License

[PolyForm Noncommercial 1.0.0](LICENSE.md) — personal / hobby / education use OK; **selling or other commercial use is not** allowed without a separate license from the copyright holder. See [NOTICE](NOTICE).
