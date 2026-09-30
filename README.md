# DERIVBOT

A local, **demo-first**, 24/7 automated trading bot for the [Deriv](https://developers.deriv.com/docs/) API with
strict, persistent risk management, a conventional-ML prediction gate (no AI/LLM services), a secured local
dashboard and a mocked-broker test suite.

> ### Read this before anything else
> - **Profitability is NOT guaranteed.** Every Deriv product is priced with a built-in margin, so without a real,
>   statistically validated edge the expected value is negative. Deriv's synthetic indices are designed to be
>   random walks. Expect the model pipeline to say **`NO EVIDENCE OF EDGE`** — in that case the bot **refuses to
>   trade by design**. Risk controls bound losses; they do not create profit.
> - The bot **always starts in DEMO** and **never enables LIVE by itself**. No real-money order is ever placed during
>   development or testing.
> - **Verification status:** the full pipeline is verified against a deterministic **mock** Deriv server (228 tests).
>   It has **not yet been run against the real Deriv service** (the build environment had no access to it). See
>   [`docs/API_NOTES.md`](docs/API_NOTES.md) for what is confirmed vs. unconfirmed, and run the real-DEMO checklist in
>   [`docs/OPERATIONS.md`](docs/OPERATIONS.md) before trusting anything.

## Architecture

```
 Deriv WS ──► receiver ──► bounded tick queue ──► tick processor ──► Strategy.on_tick(symbol, tick)
 (ticks)      (O(1), never blocks)                 per-symbol state      (features → ML → calibrated p)
                                                                              │ Signal
                                                                   bounded signal queue (latest wins)
                                                                              ▼
              ┌──────────────────────── RiskManager.evaluate ────────────────────────┐
              │ 20+ rules (kill switch, halts, limits, feed, latency, model, edge…)   │
              └───────────────┬──────────────────────────────────────────────────────┘
                              ▼ approved (exposure reserved)
                   Executor: proposal ─► RiskManager.authorize_buy (edge gate with the REAL payout,
                              stale-quote check) ─► BuyPermit ─► buy (never auto-retried) ─► confirm
                              ▼
                   SettlementTracker (watch → settle | timeout → RECONCILIATION_PENDING → resolve)
                              ▼
                        SQLite (mode-tagged rows: signals, orders, trades, risk_events, latency, …)
```

One asyncio process: FastAPI/uvicorn + the trading tasks share a single event loop and a single lifecycle.
`DerivClient.buy()` requires a `BuyPermit` that only `RiskManager` can mint — there is no code path to a buy that
bypasses the risk manager (a test scans the source to enforce this).

## Trade types (the Options products in the Deriv app)
The bot trades the four products shown under **Options** in the Deriv app, plus classic Rise/Fall. Pick one with
`trading.product.product` in `config.yaml` (default **`multiplier`**); the terms below are per-product settings.

| Product | Contract | How the bot trades it | Max loss | Exit |
|---|---|---|---|---|
| **Multipliers** | `MULTUP` / `MULTDOWN` | server-side **take-profit + stop-loss** (`limit_order`) as % of stake | stop-loss (≤ stake) | TP/SL by the broker; hold cap → sell at market |
| **Accumulators** | `ACCU` | stake grows `growth_rate`/tick while the spot stays in the barrier; native take-profit = growth after N ticks | stake (knock-out) | TP by the broker; hold cap → sell at market |
| **Turbos** | `TURBOSLONG` / `TURBOSSHORT` | knock-out barrier `barrier_offset` away; **bot-managed take-profit** (sells at market at +X % of cost) | stake (knock-out) | bot TP or expiry |
| **Vanillas** | `VANILLALONGCALL` / `VANILLALONGPUT` | held to expiry; the model predicts a terminal move that pays the target | premium (stake) | expiry |
| Rise/Fall | `CALL` / `PUT` | binary payout | stake | expiry |

Every product is reduced to a **bounded two-outcome bet** so one edge gate covers them all: the trade either reaches
its profit target (net win **W**) or not (net loss **L**, valued conservatively). The gate is
`p ≥ L / (W + L) + margin`, with **W and L taken from the actual proposal at trade time** (ask price, fees, limit
orders, contracts) — for Rise/Fall this is exactly `1/R`. The live proposal must also still match the terms the model
was trained on (barrier distance, needed move, fees, …), otherwise the trade is refused (`spec_mismatch`).
Train per product: `python -m research.train_model --product turbo --barrier-offset 0.5 --take-profit 0.5 …`
(see [`docs/ML.md`](docs/ML.md)). Which products/symbols/durations your account offers must be confirmed on DEMO
([`docs/API_NOTES.md`](docs/API_NOTES.md)).

## Requirements
Python **3.11+** on Windows, macOS or Linux. Docker is optional.

## Installation
```bash
git clone https://github.com/shriramprabhu651/deriv_bot_trading.git && cd deriv_bot_trading
make install                 # creates .venv and installs requirements-dev.txt
# no make (Windows): python -m venv .venv && .venv\Scripts\pip install -r requirements-dev.txt
```

## Configuration
| File | Purpose |
|---|---|
| `.env` (from `.env.example`) | **secrets only** — never committed |
| `config.yaml` (from `config.example.yaml`) | non-secret settings (risk limits, symbols, timeouts, retention) |

```dotenv
DERIV_APP_ID=...            # required by current PAT authentication
DERIV_DEMO_TOKEN=...        # Personal Access Token created while the DEMO account is selected
DERIV_DEMO_ACCOUNT_ID=...   # your demo Options account id
ALLOW_LIVE=false            # LIVE is impossible unless this is true AND live credentials exist
DERIV_LIVE_TOKEN=           # leave empty
DERIV_LIVE_ACCOUNT_ID=      # leave empty
DASHBOARD_TOKEN=            # optional; random token printed at start if empty
TELEGRAM_BOT_TOKEN= / TELEGRAM_CHAT_ID=   # optional alerts
```

### API authentication (current Deriv API)
1. Register an app on the Deriv developer portal to get an **App ID**.
2. In your Deriv account create a **Personal Access Token** (API token) on the **demo** account.
3. The bot calls `GET /trading/v1/options/accounts` (Bearer PAT + `Deriv-App-ID`) to verify that your configured
   account exists and is of type `demo`, then `POST …/accounts/{id}/otp` for a single-use, 120-second WebSocket URL
   (a **fresh OTP for every reconnect**). Details and open questions: [`docs/API_NOTES.md`](docs/API_NOTES.md).

If credentials are missing the bot stops with: *"Put your DEMO credentials in .env using DERIV_APP_ID,
DERIV_DEMO_TOKEN and DERIV_DEMO_ACCOUNT_ID. Do not send the token in chat."*

## Running (DEMO)
```bash
make run     # dashboard on http://127.0.0.1:8000, bot stopped until you press Start
make demo    # same, but autostarts; refuses to run without DEMO credentials
```
Windows/macOS/Linux service setup, sleep prevention and logs: [`deploy/`](deploy) and
[`docs/OPERATIONS.md`](docs/OPERATIONS.md).

### Docker (optional)
```bash
cp .env.example .env         # fill in DEMO values
docker compose up -d --build # container listens on 0.0.0.0:8000; the HOST port is bound to 127.0.0.1 only
```
Volumes persist `data/` (SQLite), `logs/` and `model/`. The dashboard is never published publicly by default.

## Dashboard
Plain HTML/JS at `http://127.0.0.1:8000`: **MODE** banner (green DEMO / red LIVE), balance + timestamp, open trades,
daily/weekly P&L, drawdown, high-water mark, consecutive losses, win rate vs break-even, ML probabilities, edge margin,
model status, risk/halt status, feed health, latency (median/p95/max), recent signals/trades/risk events and a live tick
chart. Actions (`/start /stop /kill /kill/clear /mode /model/reload`) require **all** of: allowed `Host`, allowed
`Origin`, bearer dashboard token, CSRF token and a JSON body with an explicit confirmation. `GET` status is readable
locally only (Host allowlist). The server binds to `127.0.0.1`.

## Machine learning workflow
```bash
python -m research.download_ticks --symbol R_100 --count 200000 --out data/R_100.csv
python -m research.walk_forward   --ticks data/R_100.csv --product multiplier --horizon 20   # report only
python -m research.train_model    --symbol R_100 --ticks data/R_100.csv --product multiplier \
    --horizon 20 --multiplier 100 --take-profit 0.1 --stop-loss 0.1
```
Walk-forward validation (train | gap | test, rolling), time-aware calibration, a shuffled-label control, exact
binomial test against the **dynamically computed** break-even `1/R`, Wilson confidence intervals and Cohen's *h*.
Result → `REJECTED` (**NO EVIDENCE OF EDGE**) or `CALIBRATED`. Details: [`docs/ML.md`](docs/ML.md).

### Model promotion (always manual)
`CALIBRATED → DEMO_VALIDATING` (`--start-demo-validation`, trades on DEMO only) `→ PROMOTABLE` (only with ≥ 1000 real
demo trades of that exact model whose win rate is significantly above the break-even actually paid). With no model the
bot receives ticks but **does not trade**.

## Risk management (defaults; all configurable)
| | DEMO | LIVE |
|---|---|---|
| stake (of *current* balance, floored, never rounded up, never increased after a loss) | 1 % | 0.5 % |
| daily loss (realized **+ open exposure**) | 3 % | 1 % |
| weekly loss | 6 % | 3 % |
| drawdown from high-water mark (persistent halt, manual reset) | 10 % | 10 % |
| consecutive losses | 3 | 3 |

Also: max trades/day, max open trades (total/per symbol), total and correlated-symbol exposure, cooldown, stale feed,
latency, duplicate-signal protection (persisted UNIQUE id), kill switch (persistent), account-type verification and the
ML edge gate `p ≥ 1/R + 0.03`. **No martingale.** Every rejection is stored in `risk_events`. Full table:
[`docs/RISK.md`](docs/RISK.md).

**Kill switch:** stops new trades immediately, cancels queued work, persists across restarts and mode switches; clearing
needs the bot stopped plus the typed phrase `CLEAR KILL`.

## LIVE authorisation procedure
LIVE is unavailable unless **all** are true: `ALLOW_LIVE=true`, live token and account id present, bot stopped, no open
trades, no kill switch, the exact text `LIVE` typed, an explicit second confirmation on a screen that shows account,
type, balance, stake, all limits, drawdown, P&L, exposure, model and edge-gate status, **and** the API confirms the
account is really `real`. Any failure keeps DEMO and logs a CRITICAL event. LIVE is never persisted across restarts.
Nothing in this repository enables it for you.

## Development
```bash
make check      # ruff + mypy (strict) + secret/AI-dependency scan + pytest  → must exit 0
make test | lint | typecheck | security | clean
```
Test suite: unit (risk boundaries, features/leakage, walk-forward, calibration, modes, dashboard security) and
integration against a deterministic mock Deriv WebSocket/REST server (end-to-end tick→trade→settlement, ambiguous buys,
reconnects, reconciliation, rate limits, shutdown with real SIGINT/SIGTERM).

```
app/        controller, deriv/ (auth, websocket, client), strategy/, risk/, execution/, ml/, storage/, api/, dashboard/
research/   download_ticks, build_dataset, train_model, walk_forward, evaluate
tests/      unit/, integration/, mocks/ (mock Deriv server)
docs/       API_NOTES, RISK, ML, OPERATIONS         deploy/  systemd, launchd, Windows (NSSM / Task Scheduler)
```

## Troubleshooting
See the table in [`docs/OPERATIONS.md`](docs/OPERATIONS.md#troubleshooting). Quick checks: `make check`, the dashboard
*risk events* panel, and `logs/derivbot.log`.

## Disclaimer
Educational software provided as is, without warranty. Trading binary/rise-fall options can lose all your stake. Nothing
here is financial advice and no profit is promised.


## DEMO probe (pipeline test, no model)
Without a validated model the bot correctly does not trade. To see real buy → settle → P&L on the **demo** account
anyway, set `DEMO_PROBE=true` in `.env` (or `probe.enabled: true` in `config.yaml`) and restart. The dashboard then shows
**DEMO PROBE – PIPELINE TEST, NOT A STRATEGY**. It emits one alternating up/down signal every `probe.interval_ticks` ticks (default 10, about 20 s on R_100; the dashboard shows a countdown);
all risk rules still apply (stake %, loss limits, exposure, cooldown, kill switch, halts). Only the model/edge gates are
skipped, only in DEMO, and it refuses to start or switch to LIVE. It has no edge: expect a slow loss to fees. Multipliers
use native take-profit/stop-loss; other products use their normal exits. If it stops after a losing streak that is the
consecutive-loss halt (`risk.demo.max_consecutive_losses`, default 3) doing its job.

**Choosing the trade type in the dashboard.** With the bot stopped, pick *Multipliers / Accumulators / Turbos / Vanillas /
Rise-Fall* in the **Trade type** box and press **Apply**. It lasts until restart; put `trading.product.product` in
`config.yaml` to make it permanent. Before trading a new type run `python -m scripts.check_proposal R_100 accumulator`
(requests prices only, never buys) to confirm Deriv accepts the request fields.
