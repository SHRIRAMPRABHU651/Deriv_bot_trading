# Deriv API notes (provenance of every assumption)

> **Read this first.** The build environment's network proxy **blocked** `developers.deriv.com`,
> `legacy-docs.deriv.com`, `api.deriv.com` and `ws.derivws.com` (HTTP 403 from the egress proxy), so the
> documentation pages could **not be fetched directly**. What is recorded here comes from:
>
> 1. **Search-engine excerpts of the current docs** (developers.deriv.com pages: *Authentication*,
>    *API Overview*, *WebSockets — Options API*, *Options REST API*, *Trading Operations*, *Price Proposal*,
>    *Buy Contract*, *Open Contract Status*) — marked **[current-docs excerpt]**.
> 2. **The public JSON schemas** in `deriv-com/deriv-api-docs` (`config/v3/{proposal,buy,
>    proposal_open_contract,ticks,ticks_history,balance,portfolio,ping,sell}/{send,receive}.json`), fetched
>    from raw.githubusercontent.com — marked **[schema]**. These describe the *legacy* WebSocket message
>    format that the current Options WebSocket reuses per the excerpts above.
> 3. Nothing else. Anything not in (1) or (2) is listed under **Not confirmed**.
>
> **No live/demo call to Deriv has been made from the build environment.** Demo authentication is therefore
> *implemented against the documented flow and verified against a mock*, **not** verified against the real
> service. The first real DEMO run (see `OPERATIONS.md`) is the actual verification.

## 1. Confirmed current behaviour

### Authentication (PAT + App ID) **[current-docs excerpt]**
- Deriv supports OAuth 2.0 apps and **Personal Access Token (PAT)** apps. With a PAT app the user generates a
  token in Deriv and gives it to the application, which sends it as a **Bearer token**.
- For PAT authentication the **`Deriv-App-ID` header is required** on REST requests; omitting it returns
  `401 "Deriv-App-ID header is required for PAT tokens"`.
- Headers used by this project on every REST call:
  `Authorization: Bearer <PAT>` and `Deriv-App-ID: <app id>`.

### REST endpoints **[current-docs excerpt]**
| Purpose | Call |
|---|---|
| List Options accounts | `GET https://api.derivws.com/trading/v1/options/accounts` |
| Create account | `POST /trading/v1/options/accounts` (response `data`: `account_id`, `balance`, `currency`, `group`, `status`, `account_type` = `demo`\|`real`) — *not used by the bot* |
| Issue WebSocket OTP | `POST /trading/v1/options/accounts/{accountId}/otp` |

- The OTP endpoint returns `data.url`: a **ready-to-use WebSocket URL with the one-time password attached**.
- The OTP is **valid for 120 seconds and single-use** → the bot requests a **fresh OTP for every (re)connect**
  (`DerivAuth.get_ws_url`, called by the WebSocket supervisor before each connection attempt).

### WebSocket endpoints **[current-docs excerpt]**
- Public (no auth, market data): `wss://api.derivws.com/trading/v1/options/ws/public`
- Demo (OTP): `wss://api.derivws.com/trading/v1/options/ws/demo?otp=...`
- Real (OTP): `wss://api.derivws.com/trading/v1/options/ws/real?otp=...`
- The bot never builds these URLs itself; it uses the `data.url` returned by the OTP call, so the demo/real
  path always matches the account the OTP was issued for. Only `research.download_ticks` uses the public URL
  (configurable with `--ws-url`).

### Trading flow **[current-docs excerpt]**
`proposal` → `buy` (with the proposal id) → `proposal_open_contract` (with the `contract_id`), all on an
authenticated connection. `ping` is `{"ping": 1}`.

### Message formats **[schema]** (legacy schema, same messages reused by the Options WebSocket)
- **ping**: `{"ping":1}` → `{"ping":"pong"}`.
- **balance**: `{"balance":1}` → `balance.{balance, currency, loginid}`.
- **ticks** (subscribe): `{"ticks":"R_100","subscribe":1}` → `tick.{symbol, epoch, quote, id, ...}`
  and `subscription.id`.
- **ticks_history**: `{"ticks_history":"R_100","end":"latest","count":N,"style":"ticks"}` →
  `history.{times[], prices[]}`.
- **proposal**: `{"proposal":1,"amount":<stake>,"basis":"stake","contract_type":"CALL"|"PUT",
  "currency":"USD","duration":N,"duration_unit":"t","symbol":"R_100"}` →
  `proposal.{id, ask_price, payout, spot, longcode, ...}`. Fields `min_stake`/`max_stake` and
  `validation_params.stake` exist in the schema and are used if present.
- **buy**: `{"buy":"<proposal id>","price":<max price>}` → `buy.{contract_id, buy_price, payout,
  balance_after, purchase_time, start_time, shortcode, transaction_id}`. **`price` is the maximum the client
  is willing to pay**; the server rejects the buy if the price moved above it (stale-quote protection).
- **proposal_open_contract**: `{"proposal_open_contract":1,"contract_id":ID,"subscribe":1}` →
  `proposal_open_contract.{contract_id, is_sold, is_expired, status, profit, buy_price, sell_price, payout,
  entry_spot|entry_tick, exit_tick, underlying, contract_type, purchase_time, date_expiry}`; an unknown
  contract yields an empty object.
- **portfolio**: `{"portfolio":1}` → `portfolio.contracts[]` (`contract_id, symbol, contract_type,
  buy_price, payout, purchase_time, ...`) — used for reconciliation.
- **forget**: `{"forget":"<subscription id>"}`.
- **Request correlation**: every request carries an integer `req_id`; responses (and every message of a
  subscription) echo it. The client keeps a `req_id → Future` map and a `req_id → subscription` map.
- **Errors**: `{"error":{"code":..., "message":...}, "req_id":...}`. Rate-limit responses use the code
  `RateLimit` (mapped to `RateLimitError`).

### Rate limits / heartbeat **[current-docs excerpt]**
- Deriv **does not publish fixed request limits**; they vary by endpoint and load. The `api_call_limits` field
  of a server-status call reports the current limits. Exceeding them returns an error; there is no automatic
  block, but persistent abuse can lead to suspension.
- The WebSocket session **times out after 2 minutes of inactivity**, so the bot sends an application-level
  `ping` every `ping_interval_s` (default 15 s) and reconnects if the pong does not arrive in
  `ping_timeout_s`.
- **Bot policy (conservative default, not a Deriv number):** token bucket of 5 requests/s (burst 10), `RateLimit`
  errors retried with exponential backoff + jitter **only for idempotent requests**; **`buy` is never retried**.

### Contract semantics used
- Product: Rise/Fall on ticks — `CALL` wins if the exit tick (N ticks after entry) is **strictly higher** than the
  entry tick; `PUT` if strictly lower; **equal ⇒ both lose**. Basis `stake`; the stake is the price paid.
- Payout ratio `R = payout / ask_price` (payout includes the returned stake). Break-even probability is
  `1 / R`; expected value per unit stake is `p·R − 1`. Both are computed from the **actual proposal** every time.
- Stake precision: 2 decimals for USD (configurable `stake_precision`); stakes are **floored**, never rounded up.
  Minimum stake default `0.35` (configurable; the proposal's `min_stake`/`max_stake` are additionally enforced when
  present).

## 2. Endpoints/messages used by the code
`app/deriv/auth.py` (REST) · `app/deriv/protocol.py` (builders/parsers) · `app/deriv/websocket.py` (transport) ·
`app/deriv/client.py` (high-level). Messages sent: `ping`, `balance`, `ticks`, `ticks_history`, `proposal`, `buy`,
`proposal_open_contract`, `portfolio`, `forget`.

## 3. Account verification behaviour
- Before trading, `verify_account()` calls the accounts list and requires the configured account id to exist **and**
  its `account_type` to be `demo` (DEMO mode) or `real` (LIVE mode). A mismatch aborts start-up.
- LIVE switching re-verifies `account_type == "real"` server-side (`Controller.switch_mode`) even after the
  confirmation UI, and a `live_switch_denied` CRITICAL risk event is written for every refusal.

## 4. Not confirmed (could not be verified from the build environment)
1. **REST base URL** `https://api.derivws.com` — taken from an excerpt; configurable (`deriv.rest_base_url`).
2. **Exact JSON shape of the accounts-list response** (`data` as list vs `{accounts: [...]}`) — the parser accepts
   both; `account_type` values assumed `demo`/`real` (from the create-account excerpt).
3. **Shape of the OTP response beyond `data.url`.**
4. **Whether the new Options WebSocket keeps `symbol` (legacy) or renames it to `underlying_symbol` in `proposal`.**
   The bot sends `symbol`. If Deriv rejects it, change one line in `protocol.proposal()`.
5. **`ticks_history` pagination semantics** (`end` = epoch of the oldest tick minus 1, walking backwards) — assumed
   from the legacy schema; `research.download_ticks` stops after 3 chunks without progress.
6. **Minimum stake and duration limits per symbol** — configurable; the proposal's own validation is authoritative.
7. **Whether `balance` responses on the new socket contain `loginid`** — only used informationally.
8. **`proposal_open_contract` `status` vocabulary on the new API** — the bot treats `is_sold == 1` or
   `status ∈ {won, lost, sold, cancelled}` as final and otherwise keeps watching/reconciling.
9. **Rate limits** — not published; see policy above.
10. **Buy idempotency** — no idempotency key is documented. The bot therefore never retries a buy and instead
    reconciles against `portfolio` (matching symbol, contract type, buy price and purchase time); a residual race
    (the broker executes the buy *after* the reconciliation window) is documented in `OPERATIONS.md`.

## 5. Legacy SDK compatibility concerns
- `python-deriv-api` (deriv-com) targets the **legacy** `authorize`-token flow (`wss://ws.derivws.com/websockets/v3?app_id=…`).
  The current flow is **PAT → REST OTP → account-specific WebSocket** (no `authorize` message). The bot therefore
  does **not** depend on the SDK; it reuses only the *message schemas*, which are shared.
- The legacy `authorize` call, `app_id` query parameter and `loginid` switching are not used.
- The SDK's `api.subscribe()` (RxPY observables) is replaced by a small `req_id`-routed subscription registry.
