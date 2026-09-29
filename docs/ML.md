# Machine learning

> **A model that does not demonstrate statistically credible out-of-sample edge must not trade.**
> The bot enforces this: models without a `PROMOTABLE`/`DEMO_VALIDATING` status are refused by the
> RiskManager, and a missing/corrupt/incompatible artifact means **no model → no trading** (ticks are still
> received). Only conventional supervised ML is used (scikit-learn `LogisticRegression`,
> `HistGradientBoostingClassifier`). There is **no** LLM/AI-service dependency anywhere.

## Dataset
Ticks (`epoch, quote`) from `research.download_ticks` (public `ticks_history`) or from the live feed. Row *i* of the
dataset corresponds to tick index `i + 63` (a 64-tick window). Memory at runtime is bounded: one 64-price deque per
symbol.

## Trade types
The pipeline is product-aware (`app/products.py`). A model is trained for **one `ProductSpec`** (product, horizon and
label-defining terms) which is stored in the artifact; the bot refuses a model whose terms differ from `config.yaml`.
`up`/`down` in the dataset mean *"the bullish/bearish trade reaches its profit target"* and are simulated on the tick
path with first-passage rules (ambiguity always resolved **against** the trade; a position that neither hits its
target nor its stop within the horizon counts as a **loss**, a conservative bound):

| Product | win event (per direction) | W / L used for break-even `L/(W+L)` |
|---|---|---|
| Multiplier | take-profit before stop-loss within N ticks (`ret = multiplier × ΔS/S`) | `W = TP − fee`, `L = SL + fee` |
| Accumulator | every tick within `barrier_pct` for N ticks (non-directional) | `W = (1+g)^N − 1`, `L = 1` |
| Turbo | +`tp × offset` before −`offset` within N ticks | `W = tp`, `L = 1` |
| Vanilla | terminal move ≥ `needed_move` at expiry N | `W = target`, `L = 1` |
| Rise/Fall | strictly beyond the entry tick at expiry | `W = R − 1`, `L = 1` |

Because these events are not complementary (timeouts), multipliers/turbos/vanillas train **separate bullish and bearish
heads**; accumulators use one survival model; Rise/Fall keeps `P(PUT) = 1 − P(up) − P(tie)`. The research payoff is an
assumption; at trade time W and L come from the actual proposal and the live terms must still match the trained ones.
Vectorised labels are unit-tested against a brute-force simulator for every product.

## Label and expiry mapping (Rise/Fall)
For an entry at tick `t` and horizon `N` (= contract duration in ticks, `trading.duration_ticks`):

| outcome | condition | winner |
|---|---|---|
| up | `price[t+N] > price[t]` | CALL |
| down | `price[t+N] < price[t]` | PUT |
| tie | equal | nobody (both lose) |

The model is trained on `up`. `P(PUT wins) = 1 − P(up) − tie_rate` (the empirical tie rate is stored in the artifact),
so ties are never counted in the bot's favour. The main evaluation set is **non-overlapping** (`stride = N`), so no two
evaluated samples share a future tick.

## Features (`feature_version = v1`, `app/ml/features.py`)
`return_1/3/5/10`, rolling volatility (20 steps), EMA(5) and EMA(20) relative to price, EMA gap, 20-tick trend slope,
RSI(14), signed streak length, and up-move frequency (20 steps). Every feature at time *t* is a pure function of the last
64 prices **ending at t**. Batch (training) and streaming (runtime) share the same function, and tests prove that
(a) batch = streaming, and (b) changing future prices never changes past features. The runtime rejects artifacts
whose `feature_version` differs.

## Walk-forward validation (no random splits)
```
TRAIN [0, a)  |  GAP  |  TEST            then roll forward:
                                          TRAIN [0, a+step) | GAP | TEST ...
```
Example (in days): train days 1–20, gap day 21, test days 22–25; then train 1–25, gap 26, test 27–30. In this project
the unit is ticks/samples. The gap must be **≥ the label horizon** (`walk_forward_splits` raises otherwise), because
the last training labels look `N` ticks ahead and would otherwise overlap the first test features.

## Calibration (time-aware)
Inside each training window the samples are split in time order:
`[ model fit | gap (= horizon) | calibration slice ]`. A Platt-scaling calibrator is fitted **only** on the later
calibration slice, never on test data. Reported per fold and pooled: accuracy, precision, recall, ROC-AUC, Brier score,
the calibration curve (predicted vs actual per bin) and sample sizes. A model is only `CALIBRATED` if every bin's
|predicted − actual| ≤ 0.05.

## Shuffled-label control (mandatory)
The identical pipeline is re-run `n_shuffles` times with **shuffled training labels** and evaluated on the true test
labels. The verdict is **`NO EVIDENCE OF EDGE`** — and the model is `REJECTED` — unless **all** hold:
1. at least `min_validation_trades` out-of-sample trades,
2. the win rate is significantly above break-even (test below), and
3. the real ROC-AUC exceeds the 95th percentile of the shuffled-label AUCs.

## Break-even, expected value and the edge gate
With payout ratio `R = payout / stake` (payout includes the stake):
```
break_even_probability = 1 / R          EV per unit stake = p·R − 1
```
At run time `R` comes from the **actual proposal**, never a constant. A trade is allowed only if
`calibrated_probability ≥ 1/R + EDGE_MARGIN` (default 0.03) **and** the model status allows it. `accuracy > 50 %` is
never used as a substitute. (`research.*` needs an *assumed* `R`, default 1.95, only to define the out-of-sample
trade selection; this assumption is stored in the artifact metadata.)

## Statistical test
`H0: p ≤ break_even` vs `H1: p > break_even`, exact one-sided **binomial test**, significance `alpha = 0.05`
(`--comparisons k` Bonferroni-adjusts alpha when you tried *k* variants). Reported: sample size, win rate,
Wilson 95 % confidence interval, p-value and Cohen's *h* effect size. Assumption: trades are independent (guaranteed
by the non-overlapping stride). **Small edges need thousands of trades**: detecting a true win rate of 54.3 % against a 51.3 % break-even
(one-sided, alpha 0.05, 80 % power) needs about 1,700 independent trades (`required_sample_size`); a 1.5-point edge
needs several times more.
Selecting a model/horizon/threshold after looking at test results inflates false positives — count your comparisons.

## Model status lifecycle
`UNTRAINED → BACKTESTED → WALK_FORWARD_VALID → CALIBRATED → DEMO_VALIDATING → PROMOTABLE`, or `REJECTED`.

| status | meaning | may trade? |
|---|---|---|
| `REJECTED` | no evidence of edge | never |
| `BACKTESTED` / `WALK_FORWARD_VALID` | trained / edge evidence but calibration too loose | never |
| `CALIBRATED` | passed walk-forward, shuffled control and calibration | never |
| `DEMO_VALIDATING` | operator ran `--start-demo-validation` | **DEMO only** |
| `PROMOTABLE` | ≥ `min_demo_trades` real demo trades of this exact model version, win rate significantly above the break-even actually paid | DEMO and (if separately authorised) LIVE |

Promotion beyond `CALIBRATED` is **manual** (`python -m research.evaluate …`, see `research/README.md`); nothing
promotes itself. Default with no artifact: **NO MODEL / NO TRADING**.

## Artifact (`model/model.joblib` + `model/metadata.json`)
Metadata records: model version, kind, status, feature version, label horizon, symbols, training dates, training /
validation(calibration) / test sample counts, payout assumption, break-even, edge margin, observed win rate,
confidence interval, p-value, calibration metrics, tie rate, verdict + reasons, demo statistics, git commit and the
SHA-256 of `model.joblib`. The loader refuses a file whose hash differs (pickle files are only loaded after the hash
check) or whose feature version is incompatible. `model/*.joblib` is git-ignored.

## Live monitoring and edge decay
After every settlement the rolling 100-trade window is tested; a win rate significantly **below** break-even halts the
model version (`model_halt`, CRITICAL). The threshold is never lowered automatically; loading a different model version
after manual review lifts the halt. Edges in financial series decay — re-run the walk-forward on fresh data regularly.

## Limitations
- Deriv's synthetic indices (`R_*`) are engineered to be random walks; expect `NO EVIDENCE OF EDGE`.
- Only 12 hand-built features; no order-book/latency information.
- Fixed payout assumption in research; real payouts vary by symbol/duration/time.
- A statistically significant historical edge can vanish (regime change, multiple-testing, payout changes).
- The unit tests use synthetic series with a planted signal to prove the *pipeline*; they say nothing about real markets.
