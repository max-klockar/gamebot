# Gamebot

Generic projected tabletop game machine. Games live in separate plugin packages.

```bash
cd gamebot
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" -e "../chess-plugin[dev]"
gamebot --mock
```

After calibrate, plugins run `identify()`. One match starts that game; zero or many matches shows an icon chooser on the table.

## Operator page

The camera can only be open in the game process, so the page is served from there.

```bash
./scripts/run-pi.sh --portal --camera picam --projector hdmi
```

Open `http://<pi>:8080`. From the phone: live camera, start chess at an Elo (optional skip-cal and a still after each move), capture a training frame, or shut the Pi down. During a game the camera image shows each detection and the standing point, and the board on the page is the game's pieces (rank 8 at the top). **Save these boxes** writes those rectangles, plus the locked board outline, to `captures/sets/06_accepted` as training labels. Press it when the boxes sit on the pieces and the arrow is off. Logs are the same lines as the terminal.

`--elo` still starts immediately. Add `--portal` beside it if you also want the page. Booting straight into the page is `gamebot.service` in the hardware repo.

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
