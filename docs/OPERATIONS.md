# Operations

## Daily routine (DEMO)
1. `make demo` (or the service/Docker) → the bot starts **stopped** in **DEMO** unless you use `make demo` (autostart).
2. Open http://127.0.0.1:8000, paste the dashboard token, check: MODE green/DEMO, account type (API-verified) = `demo`,
   balance timestamp fresh, model status, feed healthy.
3. **Start** → the bot connects, verifies the account type, refreshes the balance, **reconciles open positions**, then
   enables trading.
4. Watch *risk events*. Stop from the dashboard before host maintenance.

## Startup sequence (`Controller.start`)
kill switch check → REST account verification (type must match mode) → WebSocket (fresh OTP) → balance → build
tracker/executor → **startup reconciliation** (broker `portfolio` vs SQLite) → tick subscriptions → workers.
If reconciliation fails, trading is **not** enabled.

## Shutdown sequence (SIGINT/SIGTERM or dashboard Stop)
disable new trading → stop tick/signal processing → stop executor → reconcile unsettled positions → persist final
balance → close WebSocket → close SQLite → exit. The whole thing runs inside uvicorn's own event loop (FastAPI lifespan).
A test starts the real process, sends SIGINT/SIGTERM and checks a clean exit and a checkpointed database.

## Reconciliation semantics
| situation | action |
|---|---|
| DB open, broker open | resume watching |
| DB open, broker closed | query final state, settle (`db_open_api_closed`) |
| broker open, unknown to DB | adopt (`external-<id>`), WARNING risk event, counts as exposure |
| buy sent, no answer (timeout/disconnect) | **never retried**; look for a matching broker contract; adopt or fail |
| order approved but never sent | FAILED |
| watcher lost / settlement timeout | `RECONCILIATION_PENDING` (still counts as exposure) → resolved on reconnect / timeout |

**Residual risk:** with no documented buy idempotency key, a buy that the broker executes *after* the reconciliation
window (≈ 1.5 s + 3 s + 4.5 s) would be discovered at the next reconnect/startup reconciliation (adopted as an unknown
position), not silently doubled — but for a short time exposure could be higher than accounted for.

## Kill switch
Dashboard **KILL SWITCH**: stops new trades immediately, drops queued signals, cancels pending execution tasks,
persists, writes a CRITICAL event (+ Telegram). Clear: stop the bot, **Clear kill…**, type `CLEAR KILL`. It survives
restarts and mode switches.

## Halts and how they end
| halt | ends |
|---|---|
| daily loss | next calendar day (configured timezone) |
| weekly loss | Monday 00:00 |
| consecutive losses | a win settles, or next day |
| **drawdown** | **manual** `/risk/drawdown/reset` with the bot stopped (re-bases the high-water mark) |
| stale feed | automatically when ticks resume |
| model halt | load a different reviewed model version |
| kill switch | manual only |

## Authorising LIVE (deliberately hard)
The bot never enables LIVE by itself, and LIVE is **never persisted** — every restart is DEMO.
1. Only after a model reached `PROMOTABLE` from real demo evidence (docs/ML.md) and you accept the risk of losing money.
2. Put `ALLOW_LIVE=true`, `DERIV_LIVE_TOKEN`, `DERIV_LIVE_ACCOUNT_ID` in `.env`; restart.
3. Stop the bot, ensure no open trades and no kill switch.
4. Dashboard → **Switch to LIVE…** → review account, type, balance, stake, limits, drawdown, P&L, exposure, model and
   edge-gate status → type `LIVE` → tick the second confirmation → **Confirm LIVE**. The server re-verifies that the
   account is **real** via the API; any failure leaves DEMO and writes a CRITICAL `live_switch_denied` event.
5. **Start** (LIVE requires a `PROMOTABLE` model and uses the stricter LIVE risk profile).

## Real DEMO validation checklist (not yet performed in the build environment)
1. `.env`: `DERIV_APP_ID`, `DERIV_DEMO_TOKEN`, `DERIV_DEMO_ACCOUNT_ID` (never paste the token in chat).
2. `make demo`, run ≥ 30 minutes (preferably days). Record: ticks, signals, risk rejections, proposals, buys,
   settlements, wins/losses, P&L, balance change, latency median/p95/max (dashboard *Feed / Latency*), disconnects,
   reconnects, risk violations (should be 0). Do **not** claim profitability from a short run.
3. Confirm the items marked *Not confirmed* in `API_NOTES.md` and correct `protocol.py` if the API differs.

## Sleep prevention
- **Linux:** `sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target`
  (or `systemd-inhibit --what=sleep:idle --why=derivbot …`); laptops: `HandleLidSwitch=ignore` in `logind.conf`.
  ```bash
  sudo cp deploy/systemd/derivbot.service /etc/systemd/system/   # edit User/paths first
  sudo systemctl daemon-reload && sudo systemctl enable --now derivbot
  systemctl status derivbot ; journalctl -u derivbot -f
  ```
  `Restart=always`, `RestartSec=5`.
- **macOS:** `caffeinate -s` or *System Settings → Battery/Energy → Prevent automatic sleeping*; install the agent:
  ```bash
  cp deploy/launchd/com.derivbot.plist ~/Library/LaunchAgents/     # edit the /Users/YOU paths first
  launchctl load ~/Library/LaunchAgents/com.derivbot.plist
  launchctl list | grep derivbot ; tail -f ~/derivbot/logs/derivbot.log
  ```
- **Windows:** see `deploy/windows/NSSM.md` and `TASK_SCHEDULER.md` (`powercfg` timeouts).

## Logs
`logs/derivbot.log`, rotating (5 MB × 5) JSON lines with `timestamp, level, component, event, symbol, mode, signal_id,
order_id, contract_id, latency_ms`. Tokens, OTPs and bearer headers are redacted.

## Troubleshooting
| symptom | cause / fix |
|---|---|
| `cannot start: Missing credentials` | fill `.env` (see above) |
| `HTTP 401` / "Invalid or expired token" | run `python -m scripts.check_auth` (read-only, prints no secrets): it flags stray quotes/spaces, a token not starting with `pat_`, and shows Deriv's own message. Causes: expired/revoked/truncated token, or the wrong `DERIV_APP_ID` |
| `account … is 'real', expected 'demo'` | the demo account id points at a real account |
| `startup reconciliation failed` | broker unreachable; fix connectivity, press Start again |
| `NO MODEL / NO TRADING` | train a model or wait — this is the safe default |
| `STALE FEED` | ticks stopped; check network; trading resumes automatically |
| dashboard 400 *host not allowed* | add the host to `app.allowed_hosts` (only if you understand the DNS-rebinding risk) |
| 403 on actions | CSRF/Origin: use the dashboard page itself, or `curl` with `Authorization`, `X-CSRF-Token` (from `/api/csrf`) and JSON body |
