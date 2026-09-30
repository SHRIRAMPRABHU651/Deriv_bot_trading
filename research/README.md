# Research commands

All commands run from the repository root inside the virtualenv. Nothing here places trades.

```bash
# 1. Historical ticks (public call, no token). Adjust --ws-url if Deriv changes the public endpoint.
python -m research.download_ticks --symbol R_100 --count 200000 --out data/R_100.csv

# Trade type: add --product multiplier|accumulator|turbo|vanilla|rise_fall and its terms, e.g.
#   --product turbo --horizon 10 --barrier-offset 0.5 --take-profit 0.5
#   --product multiplier --horizon 20 --multiplier 100 --take-profit 0.1 --stop-loss 0.1
#   --product accumulator --horizon 10 --growth-rate 0.01 --barrier-pct 0.0006
#   --product vanilla --horizon 10 --needed-move 1.0 --target-pct 0.5
# (defaults come from config.yaml trading.product)

# 2. (optional) inspect the supervised dataset
python -m research.build_dataset --ticks data/R_100.csv --horizon 10 --out data/R_100_h10.npz

# 3. Walk-forward report with shuffled-label control (no artifact written)
python -m research.walk_forward --ticks data/R_100.csv --horizon 10

# 4. Train + validate + write model/model.joblib and model/metadata.json
python -m research.train_model --symbol R_100 --ticks data/R_100.csv --horizon 10

# 5. Out-of-sample check on NEWER ticks
python -m research.evaluate --model-dir model --ticks data/R_100_new.csv

# 6. Manual lifecycle (see docs/ML.md)
python -m research.evaluate --model-dir model --start-demo-validation
python -m research.evaluate --model-dir model --demo-db data/derivbot.db            # dry run
python -m research.evaluate --model-dir model --demo-db data/derivbot.db --promote
```

A model whose verdict is **NO EVIDENCE OF EDGE** is stored with status `REJECTED` and the bot
refuses to trade it. That is the expected outcome for most price series (synthetic indices are
designed to be random walks). Do not lower thresholds to change that.

## Evidence report (start here)
```bash
python -m research.analyze --ticks data/R_100.csv
```
Randomness tests + fee-adjusted break-even + walk-forward test per trade setting, corrected for the number of settings tried.
The bot should only be allowed to trade a setting the report marks `CANDIDATE`, and even then only after >= 1000 demo trades.

## Using other datasets (Hugging Face, Kaggle, broker exports)
```bash
python -m research.convert_csv --in eurusd_1m.csv --time-col timestamp --price-col close --out data/eurusd.csv
python -m research.analyze --ticks data/eurusd.csv --horizon 5
```
Only data of the instrument you will actually trade is relevant; the tick spacing must match the trade horizon.
