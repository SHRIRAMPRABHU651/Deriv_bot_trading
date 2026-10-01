"use strict";
// Plain JS dashboard. All dynamic text is set with textContent (never innerHTML).
const $ = (id) => document.getElementById(id);
let token = sessionStorage.getItem("derivbot_token") || "";
let csrf = "";
let pendingNonce = "";

$("token").value = token;
$("save-token").onclick = () => { token = $("token").value; sessionStorage.setItem("derivbot_token", token); say("token set"); };

function say(t) { $("msg").textContent = t; }

async function getJSON(url) {
  const r = await fetch(url, { headers: { Accept: "application/json" } });
  if (!r.ok) throw new Error(url + " -> " + r.status);
  return r.json();
}

async function post(url, body) {
  if (!csrf) csrf = (await getJSON("/api/csrf")).csrf;
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: "Bearer " + token, "X-CSRF-Token": csrf },
    body: JSON.stringify(body),
  });
  const data = await r.json().catch(() => ({}));
  if (r.status === 403 || r.status === 401) csrf = ""; // refresh CSRF token next time
  if (!r.ok) throw new Error(data.detail || ("HTTP " + r.status));
  return data;
}

function dl(el, rows) {
  el.replaceChildren();
  for (const [k, v] of rows) {
    const dt = document.createElement("dt"); dt.textContent = k;
    const dd = document.createElement("dd"); dd.textContent = v === null || v === undefined ? "—" : String(v);
    el.append(dt, dd);
  }
}

function table(el, cols, rows) {
  el.replaceChildren();
  const head = document.createElement("tr");
  for (const c of cols) { const th = document.createElement("th"); th.textContent = c; head.append(th); }
  el.append(head);
  for (const r of rows) {
    const tr = document.createElement("tr");
    for (const c of cols) { const td = document.createElement("td"); td.textContent = r[c] ?? ""; tr.append(td); }
    el.append(tr);
  }
}

const pct = (x) => (x === null || x === undefined ? "—" : (100 * x).toFixed(2) + "%");
const fmtTs = (t) => (t ? new Date(t * 1000).toISOString().slice(11, 19) + "Z" : "—");

let lastStatus = null;
function render(s) {
  lastStatus = s;
  const bar = $("modebar");
  bar.className = s.mode === "live" ? "live" : "demo";
  $("mode").textContent = "MODE: " + s.mode.toUpperCase();
  $("runstate").textContent = (s.running ? "RUNNING" : "STOPPED") + (s.trading_enabled ? "" : " · NO TRADING");

  const sel = $("product");
  if (sel && document.activeElement !== sel && s.product) sel.value = s.product.product;
  const alerts = $("alerts"); alerts.replaceChildren();
  const add = (t, cls) => { const b = document.createElement("span"); b.className = "badge " + cls; b.textContent = t; alerts.append(b, " "); };
  add("RISK: " + s.risk_status, s.risk_status === "OK" ? "ok" : "bad");
  for (const h of s.halts) add(h, "bad");
  if (s.feed.stale) add("STALE FEED", "bad");
  if (s.running && s.connection && !s.connection.connected) {
    add("DISCONNECTED" + (s.connection.last_error ? ": " + s.connection.last_error : ""), "bad");
  }
  if (s.probe) {
    add("DEMO PROBE - PIPELINE TEST, NOT A STRATEGY", "warnb");
    const nxt = Object.entries(s.probe_next_in || {}).map(([k, v]) => k + ": next signal in " + v + " ticks").join(", ");
    if (nxt) add(nxt, "ok");
  }
  if (!s.trading_enabled) add(s.model.status === "NO_MODEL" ? "NO MODEL / NO TRADING" : "MODEL " + s.model.status, "warnb");

  dl($("account"), [
    ["Account", s.account.id], ["Type (API-verified)", s.account.type],
    ["Balance", s.account.balance ? s.account.balance + " " + s.account.currency : null],
    ["Balance updated", fmtTs(s.account.balance_ts)], ["Open trades", s.open_trades],
    ["Open exposure", s.open_exposure],
  ]);
  dl($("pnl"), [
    ["Daily P&L", s.daily_pnl], ["Weekly P&L", s.weekly_pnl], ["Drawdown", pct(Number(s.drawdown))],
    ["High-water mark", s.hwm], ["Consecutive losses", s.consecutive_losses],
    ["Win rate (settled)", pct(s.win_rate) + " (" + s.settled_trades + ")"],
    ["Break-even win rate", pct(s.break_even_win_rate)],
  ]);
  const probs = Object.entries(s.probabilities).map(([k, v]) => k + " ↑" + pct(v.CALL) + " ↓" + pct(v.PUT)).join(" | ");
  dl($("model"), [
    ["Trade type", s.product.product + " · N=" + s.product.horizon_ticks + " ticks"],
    ["Model status", s.model.status], ["Model version", s.model.version], ["Model error", s.model.error],
    ["Latest probability", probs || null], ["Edge margin", s.edge_margin],
  ]);
  const lat = Object.entries(s.latency).map(([k, v]) => k + ": med " + v.median.toFixed(0) + " / p95 " + v.p95.toFixed(0) + " / max " + v.max.toFixed(0) + " ms");
  dl($("feed"), [
    ["Feed", s.feed.stale ? "STALE" : "healthy"],
    ["Reconnects / disconnects", s.feed.reconnects + " / " + s.feed.disconnects],
    ["Ticks / dropped", s.counters.ticks + " / " + s.counters.ticks_dropped],
    ["Signals / dropped", s.counters.signals_generated + " / " + s.counters.signals_dropped],
    ...lat.map((l) => ["Latency", l]),
  ]);
}

function drawChart(ticks) {
  const sel = $("symbol");
  const symbols = Object.keys(ticks);
  if (sel.options.length !== symbols.length) {
    sel.replaceChildren(...symbols.map((s) => new Option(s, s)));
  }
  const data = (ticks[sel.value || symbols[0]] || []).map((t) => t[1]);
  const c = $("chart"), g = c.getContext("2d");
  g.clearRect(0, 0, c.width, c.height);
  if (data.length < 2) return;
  const lo = Math.min(...data), hi = Math.max(...data), span = hi - lo || 1;
  g.strokeStyle = "#2f81f7"; g.beginPath();
  data.forEach((v, i) => {
    const x = (i / (data.length - 1)) * c.width, y = c.height - ((v - lo) / span) * (c.height - 10) - 5;
    i ? g.lineTo(x, y) : g.moveTo(x, y);
  });
  g.stroke();
}

async function refresh() {
  try {
    const [s, l] = await Promise.all([getJSON("/api/status"), getJSON("/api/lists")]);
    render(s);
    drawChart(l.ticks);
    table($("trades"), ["opened_at", "symbol", "direction", "stake", "state", "profit", "probability", "break_even"], l.trades);
    table($("signals"), ["created_at", "symbol", "direction", "probability", "status", "reject_reason"], l.signals);
    table($("events"), ["ts", "severity", "rule", "message"], l.risk_events);
  } catch (e) { say(String(e)); }
}

async function act(fn) { try { await fn(); say("ok"); } catch (e) { say(String(e.message || e)); } refresh(); }

$("btn-start").onclick = () => act(() => post("/start", { confirm: true }));
$("btn-stop").onclick = () => act(() => post("/stop", { confirm: true }));
$("btn-kill").onclick = () => { if (confirm("Activate the kill switch?")) act(() => post("/kill", { confirm: true, reason: "dashboard" })); };
$("btn-clear-kill").onclick = () => {
  const phrase = prompt("Bot must be stopped. Type CLEAR KILL to clear the kill switch:");
  if (phrase === "CLEAR KILL") act(() => post("/kill/clear", { confirm: true, phrase }));
};
$("btn-reload").onclick = () => act(() => post("/model/reload", { confirm: true }));
$("btn-product").onclick = () => act(() => post("/product", { confirm: true, product: $("product").value, horizon_ticks: Number($("hold").value) || null }));
$("btn-release").onclick = () => {
  const n = lastStatus ? lastStatus.open_trades : "?";
  const phrase = prompt("Bot must be stopped. The bot counts " + n + " open trade(s). If Deriv shows nothing open, release them (their result is NOT recorded).\nType RELEASE to continue:");
  if (phrase === "RELEASE") act(() => post("/orders/release", { confirm: true, phrase }));
};
$("btn-drawdown").onclick = () => {
  const dd = lastStatus ? lastStatus.drawdown : "?";
  const phrase = prompt("Bot must be stopped. Drawdown now " + dd + ". Resetting re-bases the high-water mark to the CURRENT balance.\nType RESET DRAWDOWN to continue:");
  if (phrase === "RESET DRAWDOWN") act(() => post("/risk/drawdown/reset", { confirm: true, phrase }));
};
$("btn-review").onclick = () => {
  const lines = (lastStatus && lastStatus.review ? lastStatus.review.last_trades : [])
    .map((t) => t.symbol + " " + t.direction + " " + t.profit).join("\n");
  const note = prompt("Review today's losing trades first:\n" + lines + "\n\nWhat did you review / change? (required)");
  if (!note) return;
  const phrase = prompt("Type RESUME to re-open trading for the rest of today:");
  if (phrase === "RESUME") act(() => post("/risk/review/resume", { confirm: true, phrase, note }));
};
$("btn-demo").onclick = () => act(() => post("/mode", { target: "demo" }));

$("btn-live").onclick = async () => {
  $("live-msg").textContent = "";
  try {
    const p = await post("/mode/prepare", { confirm: true });
    pendingNonce = p.nonce;
    dl($("live-summary"), [
      ["Account", p.account_id], ["Account type", p.account_type], ["Balance", p.balance],
      ["Proposed stake", p.proposed_stake], ["Daily loss limit", p.risk_limits.daily_loss_percent],
      ["Weekly loss limit", p.risk_limits.weekly_loss_percent], ["Max drawdown", p.risk_limits.max_drawdown_percent],
      ["Max exposure", p.risk_limits.max_total_exposure_percent], ["Max trades/day", p.risk_limits.max_trades_per_day],
      ["Max open trades", p.risk_limits.max_open_trades], ["Current drawdown", p.drawdown],
      ["Daily P&L", p.daily_pnl], ["Weekly P&L", p.weekly_pnl], ["Open exposure", p.open_exposure],
      ["Model status", p.model_status], ["Edge gate", (p.edge_gate.passes ? "passes" : "FAILS") + " (margin " + p.edge_gate.edge_margin + ")"],
    ]);
    $("live-typed").value = ""; $("live-second").checked = false;
    $("modal").classList.remove("hidden");
  } catch (e) { say(String(e.message || e)); }
};
$("live-cancel").onclick = () => { pendingNonce = ""; $("modal").classList.add("hidden"); };
$("live-confirm").onclick = async () => {
  try {
    await post("/mode", { target: "live", typed: $("live-typed").value, second_confirm: $("live-second").checked, nonce: pendingNonce });
    $("modal").classList.add("hidden"); say("LIVE mode set (memory only)");
  } catch (e) { $("live-msg").textContent = String(e.message || e); }
  refresh();
};

refresh();
setInterval(refresh, 1500);
