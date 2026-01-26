const COLORS = {
  ok: "#3ddc97",
  canceled: "#ffd166",
  error: "#ff5c7a",
  total: "#7aa2ff",
  grid: "rgba(255,255,255,.10)",
  text: "rgba(255,255,255,.90)",
  muted: "rgba(255,255,255,.65)",
  bg: "rgba(0,0,0,.20)",
};

function fmtInt(n) {
  return new Intl.NumberFormat("ru-RU").format(n || 0);
}

function fmtSec(s) {
  const v = Number(s || 0);
  if (!isFinite(v) || v <= 0) return "0s";
  if (v < 60) return `${v.toFixed(1)}s`;
  const m = Math.floor(v / 60);
  const r = v - m * 60;
  return `${m}m ${r.toFixed(0)}s`;
}

function fmtIso(iso) {
  if (!iso) return "—";
  const dt = new Date(iso);
  if (Number.isNaN(dt.getTime())) return iso;
  return dt.toLocaleString("ru-RU");
}

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v);
  }
  for (const ch of children) node.appendChild(ch);
  return node;
}

function renderCards(data) {
  const s = data.summary || {};
  const cards = document.getElementById("cards");
  cards.innerHTML = "";

  const items = [
    { k: "Сессий (всего)", v: fmtInt(s.sessions_total), s: `OK: ${fmtInt(s.sessions_ok)} · Cancel: ${fmtInt(s.sessions_canceled)} · Err: ${fmtInt(s.sessions_error)}` },
    {
      k: "Пользователей",
      v: fmtInt(s.unique_users),
      sNode: el("div", {}, [
        el("span", { class: "badge internal", text: `BB ${fmtInt(s.internal_users)}` }),
        el("span", { class: "badge external", text: `EXT ${fmtInt(s.external_users)}` }),
        el("span", { class: "badge unknown", text: `? ${fmtInt(s.unknown_users)}` }),
      ]),
    },
    { k: "Чатов", v: fmtInt(s.unique_chats), s: "private + group/supergroup" },
    { k: "Пик (в час)", v: fmtInt(s.peak_sessions_per_hour), s: fmtIso(s.peak_sessions_per_hour_at) },
    { k: "Пик (конкурентно)", v: fmtInt(s.peak_concurrency), s: fmtIso(s.peak_concurrency_at) },
  ];

  for (const it of items) {
    const sEl = el("div", { class: "s" });
    if (it.sNode) sEl.appendChild(it.sNode);
    else sEl.textContent = it.s || "";
    const card = el("div", { class: "card wide" }, [
      el("div", { class: "k", text: it.k }),
      el("div", { class: "v", text: it.v }),
      sEl,
    ]);
    cards.appendChild(card);
  }
}

function drawStackedBars(canvas, labels, series) {
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(1, Math.round(rect.width));
  const h = Math.max(1, Math.round(rect.height));
  canvas.width = Math.floor(w * dpr);
  canvas.height = Math.floor(h * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = COLORS.bg;
  ctx.fillRect(0, 0, w, h);

  const pad = 14;
  const innerW = w - pad * 2;
  const innerH = h - pad * 2;
  const n = labels.length;
  if (!n) return;

  let maxTotal = 1;
  const totals = labels.map((_, i) => {
    const t = series.reduce((acc, s) => acc + (s.values[i] || 0), 0);
    maxTotal = Math.max(maxTotal, t);
    return t;
  });

  ctx.strokeStyle = COLORS.grid;
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad + innerH * (i / 4);
    ctx.beginPath();
    ctx.moveTo(pad, y);
    ctx.lineTo(pad + innerW, y);
    ctx.stroke();
  }

  const gap = 6;
  const barW = Math.max(6, innerW / n - gap);
  for (let i = 0; i < n; i++) {
    let y = pad + innerH;
    for (const s of series) {
      const v = s.values[i] || 0;
      if (!v) continue;
      const bh = (v / maxTotal) * innerH;
      y -= bh;
      ctx.fillStyle = s.color;
      ctx.fillRect(pad + i * (barW + gap), y, barW, bh);
    }
  }
}

function drawLine(canvas, labels, values, color) {
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(1, Math.round(rect.width));
  const h = Math.max(1, Math.round(rect.height));
  canvas.width = Math.floor(w * dpr);
  canvas.height = Math.floor(h * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = COLORS.bg;
  ctx.fillRect(0, 0, w, h);

  const pad = 14;
  const innerW = w - pad * 2;
  const innerH = h - pad * 2;
  const n = labels.length;
  if (!n) return;

  let maxV = 1;
  for (const v of values) maxV = Math.max(maxV, v || 0);

  ctx.strokeStyle = COLORS.grid;
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad + innerH * (i / 4);
    ctx.beginPath();
    ctx.moveTo(pad, y);
    ctx.lineTo(pad + innerW, y);
    ctx.stroke();
  }

  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = pad + (innerW * i) / Math.max(1, n - 1);
    const y = pad + innerH - ((values[i] || 0) / maxV) * innerH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function renderLegend(items) {
  const node = document.getElementById("legendDaily");
  node.innerHTML = "";
  for (const it of items) {
    const pill = el("div", { class: "pill" }, [
      el("span", { class: `dot ${it.dot}` }),
      el("span", { text: it.label }),
    ]);
    node.appendChild(pill);
  }
}

function renderTimings(data) {
  const t = data.timings_ok || {};
  const stages = (t.stages || {});
  const container = document.getElementById("timings");
  container.innerHTML = "";

  const order = [
    ["download_sec", "download"],
    ["extract_wav_sec", "extract_wav"],
    ["whisper_sec", "whisper"],
    ["gigaam_sec", "gigaam"],
    ["gemini_sec", "gemini"],
    ["total_sec", "total"],
  ];

  const max = Math.max(...order.map(([k]) => (stages[k] || {}).p95 || 0), 1);
  for (const [key, label] of order) {
    const st = stages[key] || {};
    const avg = st.avg || 0;
    const p50 = st.p50 || 0;
    const p95 = st.p95 || 0;

    const row = el("div", { class: "timing" }, [
      el("div", { class: "row" }, [
        el("div", { class: "name", text: label }),
        el("div", { class: "vals", text: `${fmtSec(avg)} / ${fmtSec(p50)} / ${fmtSec(p95)}` }),
      ]),
      el("div", { class: "bar" }, [el("div")]),
    ]);
    const bar = row.querySelector(".bar > div");
    const w = Math.max(0, Math.min(100, (p95 / max) * 100));
    bar.style.width = `${w.toFixed(1)}%`;
    container.appendChild(row);
  }
}

function renderModels(data) {
  const models = (data.breakdowns || {}).models || {};
  const container = document.getElementById("models");
  container.innerHTML = "";

  const keys = [
    ["whisper", "Whisper"],
    ["gigaam", "GigaAM"],
    ["gemini", "Gemini/LLM"],
    ["device", "Device"],
    ["language", "Language"],
  ];

  for (const [k, title] of keys) {
    const map = models[k] || {};
    const items = Object.entries(map).slice(0, 6);
    const block = el("div", { class: "model" }, [
      el("div", { class: "h", text: title }),
      el("div", { class: "items" }, items.map(([kk, vv]) => el("div", { class: "kv" }, [
        el("div", { class: "key", text: kk }),
        el("div", { class: "val", text: fmtInt(vv) }),
      ]))),
    ]);
    container.appendChild(block);
  }
}

function renderTable(nodeId, rows, kind) {
  const root = document.getElementById(nodeId);
  root.innerHTML = "";
  const table = el("table");
  const thead = el("thead");
  const tbody = el("tbody");

  const headers = kind === "users"
    ? ["Пользователь", "Сессии", "OK", "Cancel", "Err", "Последний запуск"]
    : ["Чат", "Сессии", "Тип", "Последний запуск"];

  thead.appendChild(el("tr", {}, headers.map((h, i) => el("th", { class: i === 0 ? "" : "right", text: h }))));

  if (kind === "users") {
    for (const r of rows) {
      const k = (r.kind || "unknown").toLowerCase();
      const badgeLabel = k === "internal" ? "BB" : (k === "external" ? "EXT" : "?");
      const badge = el("span", { class: `badge ${k}`, text: badgeLabel });
      const nameCell = el("td", {});
      nameCell.appendChild(badge);
      nameCell.appendChild(document.createTextNode(String(r.label || r.user_id)));

      const tr = el("tr", { class: k }, [
        nameCell,
        el("td", { class: "right", text: fmtInt(r.sessions_total) }),
        el("td", { class: "right", text: fmtInt(r.sessions_ok) }),
        el("td", { class: "right", text: fmtInt(r.sessions_canceled) }),
        el("td", { class: "right", text: fmtInt(r.sessions_error) }),
        el("td", { class: "right muted", text: fmtIso(r.last_seen_at) }),
      ]);
      tbody.appendChild(tr);
    }
  } else {
    for (const r of rows) {
      tbody.appendChild(el("tr", {}, [
        el("td", { text: r.label || r.chat_id }),
        el("td", { class: "right", text: fmtInt(r.sessions_total) }),
        el("td", { class: "right", text: String(r.type || "—") }),
        el("td", { class: "right muted", text: fmtIso(r.last_seen_at) }),
      ]));
    }
  }

  table.appendChild(thead);
  table.appendChild(tbody);
  root.appendChild(table);
}

async function loadLive() {
  const res = await fetch("/api/live", { cache: "no-store" });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const live = await res.json();

  const count = Number(live.count || 0);
  const countEl = document.getElementById("liveCount");
  const metaEl = document.getElementById("liveMeta");
  const dotEl = document.getElementById("liveDot");

  countEl.textContent = fmtInt(count);
  metaEl.textContent = `upd: ${fmtIso(live.updated_at)}`;
  if (count > 0) dotEl.classList.add("on");
  else dotEl.classList.remove("on");
}

async function load() {
  const daysEl = document.getElementById("days");
  const days = daysEl.value;
  const url = days ? `/api/analytics?days=${encodeURIComponent(days)}` : "/api/analytics";

  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();

  renderCards(data);

  document.getElementById("metaGenerated").textContent = `generated: ${fmtIso(data.generated_at)}`;

  const daily = (data.timeseries || {}).daily || [];
  const labels = daily.map((d) => d.date);
  const ok = daily.map((d) => d.ok || 0);
  const canceled = daily.map((d) => d.canceled || 0);
  const error = daily.map((d) => d.error || 0);

  drawStackedBars(document.getElementById("chartDaily"), labels, [
    { key: "error", color: COLORS.error, values: error },
    { key: "canceled", color: COLORS.canceled, values: canceled },
    { key: "ok", color: COLORS.ok, values: ok },
  ]);

  renderLegend([
    { dot: "good", label: "ok" },
    { dot: "warn", label: "canceled" },
    { dot: "bad", label: "error" },
  ]);

  const hourly = (data.timeseries || {}).hourly || [];
  const hl = hourly.map((h) => h.hour);
  const hv = hourly.map((h) => h.count || 0);
  drawLine(document.getElementById("chartHourly"), hl, hv, COLORS.total);

  const s = data.summary || {};
  document.getElementById("metaLoad").textContent = `peak/h: ${fmtInt(s.peak_sessions_per_hour)} · peak concurrent: ${fmtInt(s.peak_concurrency)}`;

  renderTimings(data);
  renderModels(data);
  renderTable("topUsers", ((data.top || {}).users || []), "users");
  renderTable("topChats", ((data.top || {}).chats || []), "chats");
}

function wire() {
  document.getElementById("refresh").addEventListener("click", () => load().catch(showError));
  document.getElementById("days").addEventListener("change", () => load().catch(showError));
  window.addEventListener("resize", () => load().catch(() => {}));
}

function showError(err) {
  console.error(err);
  alert(`Не удалось загрузить аналитику: ${String(err && err.message ? err.message : err)}`);
}

wire();
load().catch(showError);
loadLive().catch(() => {});
setInterval(() => loadLive().catch(() => {}), 2000);
