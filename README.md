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

Strategy, labeling, perspective/8×8 math, and train commands:
[docs/vision-strategy.md](docs/vision-strategy.md).

```bash
./scripts/label-pieces.sh --missing-board   # add board field to existing labels
./scripts/train-board.sh
./scripts/train-pieces.sh
```

## License

[PolyForm Noncommercial 1.0.0](LICENSE.md) — personal / hobby / education use OK; **selling or other commercial use is not** allowed without a separate license from the copyright holder. See [NOTICE](NOTICE).
