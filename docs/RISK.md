# Risk management

`RiskManager` (`app/risk/manager.py`) is the **only** gateway between a signal and a Deriv `buy`.
Structurally: `DerivClient.buy()` requires a `BuyPermit`, and a permit can only be minted by
`RiskManager.authorize_buy()` (constructing one directly raises `PermitError`). A repository test asserts that
`.buy(` is called from `app/execution/executor.py` only.

Two gates:

1. `evaluate(signal)` — every rule below except the payout-dependent edge gate. On approval it **reserves
   exposure** by inserting an `APPROVED` order (so concurrent workers cannot over-commit).
2. `authorize_buy(order, signal, proposal)` — stale-quote protection, broker min/max stake, and the ML **edge gate**
   using the *actual* proposal payout. Mints the permit.

Every rejection is written to `risk_events` (rule, message, signal, details). All state lives in SQLite; the
in-memory copy is never authoritative. Amounts use `Decimal`.

Money conventions: percentages are fractions (`0.01` = 1 %). Reference balance for loss limits is the **verified
balance at the start of the day/week**; exposure/stake limits use the **current** balance.

## Defaults (DEMO / LIVE) — configurable, never changed silently

| Setting | DEMO | LIVE |
|---|---|---|
| `stake_percent` / `stake_cap_percent` | 1 % | 0.5 % |
| `daily_loss_percent` | 3 % | 1 % |
| `weekly_loss_percent` | 6 % | 3 % |
| `max_drawdown_percent` | 10 % | 10 % |
| `max_consecutive_losses` | 3 | 3 |
| `max_trades_per_day` | 200 | 50 |
| `max_open_trades` / per symbol | 2 / 1 | 1 / 1 |
| `max_total_exposure_percent` | 3 % | 1 % |
| `max_group_exposure_percent` | 2 % | 1 % |
| `cooldown_seconds` | 5 | 10 |

## Rules

| # | Rule id | Purpose | Formula / behaviour | Persistence & reset | Tests |
|---|---|---|---|---|---|
| 1 | `stake` | Size from **current** balance | `stake = floor₂(min(bal·stake_%, bal·cap_%))`; **never rounded up** | recomputed each signal | `test_stake_*` |
| 2 | `min_stake` | Broker minimum | reject if `stake < min_stake` (never auto-raised) | – | `test_minimum_stake_boundary…` |
| 3 | `balance_stale/unknown` | Fresh balance | reject if balance missing or older than `balance_max_age_s`; refreshed at start, every 30 s, after each settlement and before executing when older than half the max age | `account_snapshots` | `test_stale_balance…` |
| 4 | `max_open_trades` | Concurrency | active orders (`APPROVED…RECONCILIATION_PENDING`) ≥ limit → reject | derived from DB | `test_max_open_trades…` |
| 5 | `max_open_per_symbol` | One contract per symbol | same, per symbol | derived from DB | same |
| 6 | `max_trades_per_day` | Overtrading | counter incremented on approval, **refunded** if the proposal gate rejects | daily reset | `test_max_trades_per_day…` |
| 7 | `daily_loss_limit` / `daily_loss_exposure` | Daily loss incl. open risk | halt when realized loss ≥ `day_start_bal·daily_%`; reject when `realized + open worst-case exposure + new stake > limit` (equality allowed) | `daily_halt` persisted; cleared at the daily rollover | `test_daily_loss_*` |
| 8 | `weekly_loss_limit` / `weekly_loss_exposure` | Weekly loss | same formula with weekly numbers | week key persisted; reset Monday 00:00 (configured tz) | `test_weekly_*` |
| 9 | `drawdown_halt` | Drawdown from high-water mark | `dd = (HWM − bal)/HWM ≥ max_dd` → **persistent halt**; HWM only rises from verified balances | never auto-reset; manual `/risk/drawdown/reset` (bot stopped) re-bases HWM | `test_drawdown_*`, `test_high_water_mark…` |
| 10 | `consecutive_losses` | Loss streak | ≥ N consecutive settled losses → reject; a win resets | persisted; also reset at daily rollover | `test_consecutive_losses…` |
| 11 | `max_exposure` | Total open exposure | `open exposure + stake ≤ bal·max_total_%` | derived | `test_total_exposure_boundary` |
| 12 | `group_exposure` | Correlated symbols | same, summed over `correlation_groups` | derived | `test_correlated_symbol_group_exposure` |
| 13 | `cooldown` | Spacing | `now − last_purchase < cooldown` → reject | `last_trade_ts` persisted | `test_cooldown_boundary` |
| 14 | `edge_gate` | Only trade a validated edge | `p ≥ 1/R + edge_margin` with **R from the actual proposal** | – | `test_edge_gate_*` |
| 15 | `model_unavailable / model_status / model_version / no_probability / model_halt` | Only validated models trade | DEMO: `DEMO_VALIDATING` or `PROMOTABLE`; LIVE: `PROMOTABLE` only | model metadata; rolling-monitor halt persisted per model version | `test_model_*`, `test_live_requires_promotable…`, `test_rolling_monitor_*` |
| 16 | `stale_feed` | No trading on stale data | any watched symbol silent > `feed_stale_after_s` → reject + CRITICAL event + optional Telegram | recovers automatically when ticks resume | `test_stale_feed…` |
| 17 | `high_latency` | Slow execution | p95 of the last 50 order-confirmation latencies > `max_latency_ms` (optional, `latency_halt_enabled`) | in-memory window, samples in DB | `test_high_latency…` |
| 18 | `kill_switch` | Emergency stop | blocks everything; cancels queued signals/tasks; CRITICAL event | **persisted**; cleared only via `/kill/clear` with the phrase `CLEAR KILL`, bot stopped; mode switch never clears it | `test_kill_switch…` |
| 19 | `bot_stopped` | Not running | reject if the controller is not running | – | `test_bot_stopped…` |
| 20 | `duplicate_signal` | Idempotency | `signal_id = sha256(symbol, tick epoch+id, strategy version, direction, model version)`; **UNIQUE** in `signals` | persisted (survives restart) | `test_duplicate_signal…` |
| 21 | `account_unverified` | Mode/account integrity | API-verified account type must match the mode (`demo`/`real`) | set at start | `test_bot_stopped_and_account_unverified` |
| 22 | `stale_quote` | Never chase a quote | buy `price` = proposal ask (× `1+max_price_slippage_percent`, default 0); the broker rejects if the price moved | – | `test_stale_quote_protection` |

## Model monitoring
After every settlement the last `monitor_window` (100) trades are tested: if the win rate is **significantly below**
the average break-even (one-sided exact binomial test, `monitor_alpha` 5 %) the current model version is halted
(`model_halt`, CRITICAL). The threshold is **never lowered automatically**; loading a *different* model version
(manual review/promotion) lifts the halt.

## What the risk manager cannot do
It bounds the *size* of losses; it cannot create an edge. Rise/Fall pays less than 2× stake, so without a real
statistical edge the expected value is negative and the bot loses money slowly, however good the risk controls.
