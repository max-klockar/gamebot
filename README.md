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

**Training** (LabelMe, YOLO, datasets) lives in a separate laptop repo:
`gamebot-vision` (sibling checkout). The Pi only needs capture + exported
weights under `models/vision/`.

```bash
# on laptop
cd ../gamebot-vision
./scripts/pull-captures.sh max@gamebot
./scripts/label.sh
./scripts/train-board.sh && ./scripts/train-pieces.sh
./scripts/export-weights.sh ../gamebot
```

On the Pi, stills (same `--zoom` as play; default **1.2×** center crop):

```bash
./scripts/run-pi.sh python -m gamebot.capture_sets           # 5 per phase
./scripts/run-pi.sh python -m gamebot.capture_sets --n 10    # 10 each
./scripts/run-pi.sh python -m gamebot.capture_sets --n 3,12,8  # per phase
# Enter=shoot, number at phase start=override count, s=next phase, q=quit
# Changing zoom ⇒ re-calibrate; keep capture + gamebot --zoom identical
```

## License

[PolyForm Noncommercial 1.0.0](LICENSE.md) — personal / hobby / education use OK; **selling or other commercial use is not** allowed without a separate license from the copyright holder. See [NOTICE](NOTICE).
