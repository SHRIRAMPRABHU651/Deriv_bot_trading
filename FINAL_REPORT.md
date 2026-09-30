# FINAL REPORT — DERIVBOT

_Everything below reports what was actually executed. Where something could not be done it says so._

## Update — trade types of the Deriv app (Multipliers, Accumulators, Turbos, Vanillas)
The bot originally modelled only Rise/Fall. It now supports the four **Options** products shown in the Deriv app
(plus Rise/Fall), selected by `trading.product.product` (default `multiplier`):
- `app/products.py`: per-product proposal requests (`MULTUP/MULTDOWN` with native `limit_order`, `ACCU` with growth rate and
  native take-profit, `TURBOSLONG/SHORT` with a knock-out barrier, `VANILLALONGCALL/PUT`), **payoff terms from the actual
  proposal** (fees, limit orders, contracts, barrier), a **fail-closed check that the live contract still matches the
  terms the model was trained on**, and vectorised first-passage **labels** for research.
- One generalised edge gate: `p ≥ L/(W+L) + margin` (Rise/Fall = `1/R`); every product's maximum loss is bounded by its stake.
- Exits: native TP/SL for multipliers/accumulators; **bot-managed take-profit for turbos**; hold-cap **sell at market** for
  products that never expire; vanillas held to expiry. Restarts resume the exit plan from the database.
- ML: per-product datasets/labels, separate bullish/bearish heads where wins are not complementary, product stored in the
  artifact; a model for other terms is refused. Promotion/monitoring count *target hits*, not merely profit > 0.
- Tested against the mock broker for all four products (requests, buy, TP/SL/knock-out, hold cap, turbo exit, vanilla expiry,
  spec mismatch, restart) and by unit tests (vectorised labels == brute-force simulation for every product).
- **Not confirmed against the real API** (see `docs/API_NOTES.md` §"Trade types"): new-API field names for these products,
  the units of `commission` and `tick_size_barrier_percentage`, per-symbol availability and duration limits.
  Synthetic pipeline results per product: momentum series → `EDGE EVIDENCE`; random walk / constant volatility →
  `NO EVIDENCE OF EDGE` (multiplier, turbo, vanilla, accumulator).

## Summary
| | |
|---|---|
| Local quality gate | `make check` → **exit 0** (ruff 0 errors · mypy strict 0 errors · security scan clean · **228 tests pass**) |
| Real Deriv DEMO validation | **NOT performed** (no DEMO credentials in the build environment; Deriv hosts were blocked by the egress proxy) |
| Real-money (LIVE) activity | **None.** LIVE was never enabled, no live token exists, no live order was sent |
| Profitability | **Not claimed and not guaranteed** |

## 1. Architecture
Single asyncio process (FastAPI + uvicorn lifespan). `Deriv WS → bounded tick queue → per-symbol Strategy.on_tick →
bounded signal queue → RiskManager.evaluate → Executor (proposal → RiskManager.authorize_buy → BuyPermit → buy) →
SettlementTracker → SQLite`. Tick reception never waits for proposal/buy. `DerivClient.buy()` requires a permit that
only `RiskManager` can mint; a test asserts `.buy(` is called only from `app/execution/executor.py`. See README.

## 2. Files created
`app/` (config, controller, clock, logging_setup, main; `deriv/{auth,client,protocol,rate_limiter,reconnect,websocket}`;
`strategy/{base,ema,ml_strategy}`; `risk/{limits,manager,permit,state}`; `execution/{executor,idempotency,reconciliation,
settlement}`; `ml/{artifacts,calibrate,dataset,features,labels,pipeline,predict,statistics,train,validate}`;
`storage/{database,migrations,repositories}`; `metrics/latency`; `alerts/telegram`; `api/{routes,security}`;
`dashboard/static/{index.html,app.js,styles.css}`), `research/` (5 CLIs + README), `tests/` (unit, integration, mock
Deriv server, fixtures), `docs/` (API_NOTES, RISK, ML, OPERATIONS), `deploy/` (systemd, launchd, Windows NSSM + Task
Scheduler), `scripts/` (check, run_demo, db_reset), `Dockerfile`, `docker-compose.yml`, `Makefile`, `pyproject.toml`,
`requirements*.txt`, `config.example.yaml`, `config.yaml.example`, `.env.example`, `.gitignore`, `README.md`.

## 3. Official API behaviour confirmed — and how
The documentation hosts (`developers.deriv.com`, `legacy-docs.deriv.com`, `api.deriv.com`, `ws.derivws.com`) returned
**HTTP 403 from the sandbox egress proxy**, so pages could not be fetched directly. Confirmed instead from
(a) search excerpts of the current docs and (b) the public `deriv-api-docs` JSON schemas (fetched from GitHub raw):
PAT + `Deriv-App-ID` bearer auth; `GET /trading/v1/options/accounts`; `POST …/accounts/{id}/otp` → `data.url`
(single use, 120 s); demo/real/public WebSocket paths; `proposal → buy → proposal_open_contract`; message field
names/types (e.g. `amount` and `price` are numbers, `end` is a string) ; no published rate limits; 2-minute idle
timeout. Full list, and everything **not** confirmed, in `docs/API_NOTES.md`.
One real bug found by reading the schemas: `amount` was sent as a string and `ticks_history.end` as an integer; both were
corrected to the documented types.

## 4–8. Tests executed and results
| Check | Result |
|---|---|
| `python -m compileall app tests research scripts` | OK |
| `ruff check .` | **All checks passed** (rules E,F,W,I,B,UP,SIM,C4,ASYNC; none disabled) |
| `mypy app tests research scripts` | **Success: no issues found in 89 source files** (`strict = true`; **no `type: ignore`**; only sklearn/joblib/scipy have `ignore_missing_imports` because they ship no complete typing) |
| `python -m scripts.check` | security check passed (no secrets, no live credentials, no AI/LLM imports) |
| `pytest -q` | **228 passed** (integration suite re-run 3× consecutively, stable) |
| `make check` | **exit 0** |

Test areas: strategy/EMA/features/leakage; ML (labels, walk-forward gap, calibration, shuffled control, artifacts);
risk (every rule with boundary−ε / boundary / boundary+ε, daily & weekly rollover, timezone, persistence across
restart, kill switch, drawdown/HWM, streak, edge-gate with dynamic payout, stale quote, permit forgery); LIVE gating (8
independent denial reasons + success path being memory-only); dashboard (Host/Origin/token/CSRF/JSON, action cycle);
executor + reconciliation against the mock (errors, timeouts, ambiguous buys by timeout / socket drop / never-executed,
settlement timeout, disconnect + reconnect + resubscribe, rate limits, stale feed, restart, DB-open/API-closed,
unknown broker position, failed reconciliation); shutdown (in-process ordering + **real SIGINT and SIGTERM** against a
running server); repo security (no AI libs, no hard-coded credentials, single buy path, no martingale).

## 9. Mandatory integration test
`tests/integration/test_controller_e2e.py`: a real `Tick` → `Controller.process_tick` → 64-tick feature window → model
prediction → `Signal` → risk → proposal → buy → settlement → `WON` trade with profit 9.50 and correct streak/P&L — plus
the same over the real WebSocket receiver/queue path with a slow (300 ms) proposal proving the tick loop does not block.
It exercises the real `Strategy.on_tick` interface, so the original `AttributeError: strategy.update` class of bug
cannot recur silently.

## 10. Docker
`docker compose config` → valid (host port bound to `127.0.0.1:8000`, container `0.0.0.0:8000`, `restart: always`,
volumes for data/logs/model). **`docker build` was NOT run**: the sandbox has the Docker CLI but no daemon
(`/var/run/docker.sock` missing).

## 11–20. Demo run
**Not performed.** No `DERIV_DEMO_TOKEN`/`DERIV_APP_ID`/`DERIV_DEMO_ACCOUNT_ID` were available and Deriv endpoints are
unreachable from the build environment. Therefore ticks, signals, trades, settlements, win rate, P&L, real-network
latency and risk violations against Deriv are **all unmeasured** — no numbers are fabricated here.
What *was* measured (mock broker, local, excludes network round-trips): the internal tick → signal → order → confirmation
path stays well under 1 s (asserted `< 1000 ms` in `test_latency_budget.py`).

Operator instruction (as required): *"Put your DEMO credentials in .env using DERIV_APP_ID, DERIV_DEMO_TOKEN and
DERIV_DEMO_ACCOUNT_ID. Do not send the token in chat."* Then `make demo` and follow `docs/OPERATIONS.md`.

## 21. ML validation results (synthetic data — proves the pipeline, says nothing about real markets)
Walk-forward (train | gap | test), payout assumption 1.95 (break-even 0.5128), edge margin 0.03, 10 shuffled controls:

| series | verdict | status | OOS trades | win rate | 95 % CI | p-value | AUC (real) | AUC (shuffled p95) |
|---|---|---|---|---|---|---|---|---|
| random walk (seed 21) | **NO EVIDENCE OF EDGE** | REJECTED | 106 | 0.4245 | [0.335, 0.520] | 0.972 | 0.507 | 0.519 |
| momentum series (planted signal) | EDGE EVIDENCE | WALK_FORWARD_VALID | 1318 | 0.7140 | [0.689, 0.738] | 3.8e-50 | 0.748 | 0.537 |

The planted-signal model was **not** `CALIBRATED` because its worst calibration bin deviated 0.089 (> 0.05 limit) — the
strict gate working as designed. No real-market model has been trained: no market data could be downloaded.

## 22. Known limitations
- Never run against the real Deriv service (see §3, §11-20). The first DEMO run is the real verification.
- ML: the shipped bot has **no validated model**; without one it receives ticks and does not trade. Synthetic indices
  are designed to be random, so a genuine edge is unlikely; detecting a 3-point edge needs ~1,700 independent trades.
- A `buy` with no documented idempotency key can only be reconciled, not made idempotent; a late-executing buy is
  discovered at the next reconnect/startup reconciliation (documented in OPERATIONS.md).
- Single process / SQLite by design (local use); dashboard has no multi-user accounts.
- Docker image not built here; systemd/launchd/Windows service files were not executed.
- Rate limits are a conservative bot policy (5 req/s, burst 10), not a published Deriv number.

## 23. API assumptions (all traceable in `docs/API_NOTES.md` §4)
REST base URL `https://api.derivws.com`; accounts-list JSON shape; OTP response beyond `data.url`; `symbol` field name in
`proposal`; `ticks_history` backwards paging via `end`; `status` vocabulary of `proposal_open_contract`; `loginid` in
`balance`. Each is isolated in one function (`auth.py` / `protocol.py`) so a correction is a one-line change.

## 24. What could not be tested
Real Deriv authentication and trading; `docker build`; service managers (systemd/launchd/NSSM/Task Scheduler); Telegram
delivery (unit-tested with a fake HTTP client); long-duration (24/7) soak; real-network latency.

## 25. Statement
**Profitability is NOT guaranteed.** The bot is engineered to *limit* losses and to *refuse to trade without validated,
statistically credible out-of-sample evidence*; it cannot create an edge. Trading can lose your entire stake.

## Bugs found and fixed during verification (regression tests added)
1. Settlement timer was not re-armed after a "still open" reconciliation → a lost settlement update could never be
   recovered (`test_settlement_timeout_moves_to_reconciliation_and_resolves`).
2. Walk-forward silently swallowed an unknown model kind (`ValueError`) → now validated up front.
3. Stale contract subscriptions were re-subscribed on every reconnect → now discarded when watchers are marked pending.
4. Start-up crashed on network/HTTP errors instead of reporting a clear `StartError`; account is now verified over REST
   before opening the WebSocket (`test_startup_errors.py`).
5. EMA strategy ignored a cross out of an exactly-flat state.
6. Dashboard showed *STALE FEED* while the bot was stopped.
7. `amount` / `ticks_history.end` sent with non-schema types (see §3).
