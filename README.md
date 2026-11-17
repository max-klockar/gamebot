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

**Training** (LabelMe, YOLO, datasets) lives in laptop repo `gamebot-vision`.
Stills are **not** kept in this git tree:

- Pi (temp): `~/.gamebot/captures/sets/`
- Laptop: `gamebot-vision/captures/sets/` (gitignored)

```bash
# on laptop
cd ../gamebot-vision
./scripts/pull-captures.sh max@gamebot          # then optional --purge
./scripts/label.sh --new-only
./scripts/train-board.sh && ./scripts/train-pieces.sh
./scripts/export-weights.sh ../gamebot
```

On the Pi (same `--zoom` as play; default **1.2×** center crop):

```bash
./scripts/run-pi.sh python -m gamebot.capture_sets           # → /home/<user>/.gamebot/captures/sets (even under sudo)
./scripts/run-pi.sh python -m gamebot.capture_sets --n 10
./scripts/run-pi.sh python -m gamebot.capture_sets --n 0,0,10
# Enter=shoot, number=override, s=next phase, q=quit
```

## License

[PolyForm Noncommercial 1.0.0](LICENSE.md) — personal / hobby / education use OK; **selling or other commercial use is not** allowed without a separate license from the copyright holder. See [NOTICE](NOTICE).
