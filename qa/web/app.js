/* QA Studio front end: plain JS, no build step. Talks to the FastAPI backend under /api. */
"use strict";

const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (n, d = 1) => (n == null || isNaN(n) ? "–" : Number(n).toFixed(d));
const PALETTE = ["#6366f1", "#14b8a6", "#f59e0b", "#ec4899", "#0ea5e9"];

const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode / quota */ } },
};

const READER_HINTS = {
  extractive: "DistilBERT, fine-tuned by us, extracts the answer span from the passages.",
  generative: "Flan-T5 writes the answer from the passages; confidence is the sequence probability.",
  compare: "Both readers answer from the same evidence. Agreement is a strong confidence signal.",
};

const settings = Object.assign(
  { corpus: "squad", reader: "compare", scoring: "calibrated", retrieval: "hybrid", rerank: true, top_k: 3 },
  store.get("qa-settings", {})
);
let history = store.get("qa-history", []);
let busy = false;

/* ---------- API ---------- */
async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  let data = null;
  try { data = await res.json(); } catch { /* non-JSON */ }
  if (!res.ok) {
    const detail = data?.detail;
    const msg = Array.isArray(detail) ? detail.map((d) => d.msg).join("; ") : detail || `${res.status} ${res.statusText}`;
    throw new Error(msg);
  }
  return data;
}

function toast(msg) {
  const t = document.createElement("div");
  t.className = "toast";
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 2600);
}

/* ---------- Settings UI ---------- */
function syncSettingsUI() {
  $$(".segmented").forEach((seg) => {
    const val = settings[seg.dataset.name];
    $$("button", seg).forEach((b) => b.setAttribute("aria-checked", String(b.dataset.value === val)));
  });
  $(`input[name=corpus][value=${settings.corpus}]`).checked = true;
  $("#rerank").checked = settings.rerank;
  $("#topk").value = settings.top_k;
  $("#topkOut").textContent = settings.top_k;
  $("#readerHint").textContent = READER_HINTS[settings.reader];
  $("#scoringField").hidden = settings.reader === "generative";
  const parts = [
    settings.corpus === "squad" ? "SQuAD" : "My documents",
    { hybrid: "Hybrid search", dense: "Dense search", bm25: "BM25" }[settings.retrieval] + (settings.rerank ? " + re-rank" : ""),
    `top-${settings.top_k}`,
    { extractive: "DistilBERT", generative: "Flan-T5", compare: "DistilBERT vs Flan-T5" }[settings.reader],
  ];
  $("#composerMeta").textContent = parts.join(" · ");
  store.set("qa-settings", settings);
}

function initSettings() {
  $$(".segmented").forEach((seg) => {
    seg.addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      settings[seg.dataset.name] = b.dataset.value;
      syncSettingsUI();
    });
    seg.addEventListener("keydown", (e) => {
      if (!["ArrowLeft", "ArrowRight"].includes(e.key)) return;
      const btns = $$("button", seg);
      const i = btns.findIndex((b) => b.dataset.value === settings[seg.dataset.name]);
      const next = btns[(i + (e.key === "ArrowRight" ? 1 : btns.length - 1)) % btns.length];
      settings[seg.dataset.name] = next.dataset.value;
      syncSettingsUI();
      next.focus();
    });
  });
  $$("input[name=corpus]").forEach((r) => r.addEventListener("change", () => {
    if (r.value === "docs" && !docs.length) toast("Upload a document first, then ask about it");
    settings.corpus = r.value;
    syncSettingsUI();
  }));
  $("#rerank").addEventListener("change", (e) => { settings.rerank = e.target.checked; syncSettingsUI(); });
  $("#topk").addEventListener("input", (e) => { settings.top_k = +e.target.value; syncSettingsUI(); });
  syncSettingsUI();
}

/* ---------- Health ---------- */
async function pollHealth() {
  try {
    const h = await api("/api/health");
    const ok = h.status === "ok";
    $("#statusDot").className = "status-dot " + (ok ? "ok" : "");
    $("#statusText").textContent = ok ? `Ready · ${h.extractive_model}` : "Loading models…";
    if (h.passages) $("#squadMeta").textContent = `${h.passages.toLocaleString()} passages`;
    if (!ok) setTimeout(pollHealth, 2000);
  } catch {
    $("#statusDot").className = "status-dot err";
    $("#statusText").textContent = "Server unreachable";
    setTimeout(pollHealth, 4000);
  }
}

/* ---------- Documents ---------- */
let docs = [];
function renderDocs() {
  const ul = $("#docList");
  ul.innerHTML = docs.map((d) => `<li><span aria-hidden="true">📄</span><span class="doc-name" title="${esc(d.name)}">${esc(d.name)}</span>
    <span class="doc-meta">${d.chunks} parts</span><button data-id="${d.doc_id}" aria-label="Remove ${esc(d.name)}">✕</button></li>`).join("");
  const parts = docs.reduce((a, d) => a + d.chunks, 0);
  $("#docsMeta").textContent = docs.length ? `${docs.length} file${docs.length > 1 ? "s" : ""} · ${parts} passages` : "No files yet";
}
async function loadDocs() {
  try { docs = (await api("/api/documents")).documents; renderDocs(); } catch { /* server still starting */ }
}
async function uploadFiles(files) {
  const dz = $("#dropzone");
  for (const f of files) {
    dz.classList.add("busy");
    dz.querySelector("strong").textContent = `Indexing ${f.name}…`;
    try {
      const fd = new FormData();
      fd.append("file", f);
      const res = await api("/api/documents", { method: "POST", body: fd });
      docs = res.documents;
      renderDocs();
      settings.corpus = "docs";
      syncSettingsUI();
      toast(`${f.name}: ${res.document.chunks} passages indexed`);
    } catch (err) {
      toast(`${f.name}: ${err.message}`);
    }
  }
  dz.classList.remove("busy");
  dz.querySelector("strong").textContent = "Drop PDF / TXT / MD";
}
function initDocs() {
  const dz = $("#dropzone"), input = $("#fileInput");
  dz.addEventListener("click", () => input.click());
  dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  input.addEventListener("change", () => { uploadFiles([...input.files]); input.value = ""; });
  ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => uploadFiles([...e.dataTransfer.files]));
  $("#docList").addEventListener("click", async (e) => {
    const b = e.target.closest("button[data-id]");
    if (!b) return;
    try {
      docs = (await api(`/api/documents/${b.dataset.id}`, { method: "DELETE" })).documents;
      renderDocs();
      if (!docs.length && settings.corpus === "docs") { settings.corpus = "squad"; syncSettingsUI(); }
    } catch (err) { toast(err.message); }
  });
  loadDocs();
}

/* ---------- Rendering answers ---------- */
function confClass(c) { return c == null ? "" : c >= 0.7 ? "good" : c >= 0.4 ? "warn" : "bad"; }
function confColor(c) { return { good: "var(--good)", warn: "var(--warn)", bad: "var(--bad)" }[confClass(c)] || "var(--muted)"; }
function confWord(c) { return { good: "High", warn: "Medium", bad: "Low" }[confClass(c)] || "–"; }

function ring(conf) {
  if (conf == null) return "";
  const pct = Math.round(conf * 100);
  return `<div class="ring" style="--p:${pct};--c:${confColor(conf)}" role="img" aria-label="Confidence ${pct} percent">
    <div><span>${pct}%</span><small>${confWord(conf)}</small></div></div>`;
}

function highlight(text, answer, start) {
  if (!answer) return esc(text);
  let i = start != null && text.substr(start, answer.length) === answer ? start : text.toLowerCase().indexOf(answer.toLowerCase());
  if (i < 0) return esc(text);
  return esc(text.slice(0, i)) + "<mark>" + esc(text.slice(i, i + answer.length)) + "</mark>" + esc(text.slice(i + answer.length));
}

function scoreBars(h, maxBm25) {
  const items = [];
  items.push(["P(rel)", h.prior, `${Math.round(h.prior * 100)}%`]);
  if (h.dense != null) items.push(["cos", Math.max(0, h.dense), fmt(h.dense, 2)]);
  if (h.bm25) items.push(["BM25", maxBm25 ? h.bm25 / maxBm25 : 0, fmt(h.bm25, 1)]);
  return items.map(([k, v, label]) => `<span class="score" title="${k}">${k}<span class="bar"><i style="width:${Math.min(100, v * 100)}%"></i></span>${label}</span>`).join("");
}

function evidenceHTML(r) {
  const hits = r.retrieved || [];
  const maxBm25 = Math.max(0, ...hits.map((h) => h.bm25 || 0));
  const order = hits.map((h, i) => i);
  // the passage the answer came from first, then the rest by rank
  if (r.passage_index != null) { order.splice(order.indexOf(r.passage_index), 1); order.unshift(r.passage_index); }
  const cards = order.map((i) => {
    const h = hits[i];
    const isSrc = i === r.passage_index;
    return `<div class="psg ${isSrc ? "source-psg" : ""}">
      <div class="psg-head"><span class="rank">${i + 1}</span><span class="psg-title">${esc(h.title)}</span>
        ${isSrc ? '<span class="badge accent">answer source</span>' : ""}<span class="scores">${scoreBars(h, maxBm25)}</span></div>
      <div class="psg-text ${isSrc ? "" : "clamp"}">${isSrc ? highlight(h.text, r.answer, r.char_start) : highlight(h.text, r.answer, null)}</div>
    </div>`;
  }).join("");
  const more = hits.length > 1 ? `<button class="link-btn toggle-ev">Show all ${hits.length} passages</button>` : "";
  return { cards, more };
}

function timingsHTML(t) {
  const parts = [["retrieve_ms", "Retrieve", PALETTE[0]], ["rerank_ms", "Re-rank", PALETTE[1]], ["read_ms", "Read", PALETTE[2]]];
  return parts.filter(([k]) => t?.[k] != null).map(([k, n, c]) => `<span class="timing"><i style="background:${c}"></i>${n} ${Math.round(t[k])} ms</span>`).join("");
}

function goldHTML(score, gold) {
  if (!score || !gold) return "";
  const ok = score.em === 1;
  return `<div class="gold">${ok ? "✅" : score.f1 > 0 ? "🟡" : "❌"} Gold: <b>${gold.map(esc).join(" / ")}</b>
    · EM <b>${score.em}</b> · F1 <b>${fmt(score.f1, 2)}</b></div>`;
}

function readerLabel(r) {
  return r.reader === "extractive" ? `DistilBERT · ${r.settings?.scoring === "raw" ? "raw" : "calibrated"}` : "Flan-T5 · generative";
}

function answerBlock(r, compact = false) {
  const has = r.answer && r.answer.trim();
  const alts = (r.alternatives || []).slice(1, 4).filter((a) => a.confidence >= 0.02);
  return `<div class="answer-head">
      <div class="answer-main">
        <div class="badge-row"><span class="badge accent">${readerLabel(r)}</span>
          ${r.confidence != null ? `<span class="badge ${confClass(r.confidence)}">${confWord(r.confidence)} confidence</span>` : ""}</div>
        <div class="answer-text ${has ? "" : "none"}">${has ? esc(r.answer) : "No answer found in the passages"}</div>
        ${r.source_title ? `<div class="answer-source">from <b>${esc(r.source_title)}</b></div>` : ""}
      </div>
      ${ring(r.confidence)}
    </div>
    ${!compact && alts.length ? `<div class="alts">Also considered: ${alts.map((a) => `<span class="alt">${esc(a.text)}<b>${Math.round(a.confidence * 100)}%</b></span>`).join("")}</div>` : ""}
    ${goldHTML(r.gold_score, r.gold)}`;
}

function singleCard(r) {
  const ev = evidenceHTML(r);
  return `<div class="card">
    ${answerBlock(r)}
    <div class="meta-row">${timingsHTML(r.timings)}<span class="spacer"></span>
      <button class="link-btn copy" data-text="${esc(r.answer)}">Copy</button>${ev.more}</div>
    <div class="evidence collapsed">${ev.cards}</div>
  </div>`;
}

function compareCard(res) {
  const a = res.agreement, e = res.extractive, g = res.generative;
  e.gold = g.gold = res.gold;
  const banner = a.agree
    ? `<div class="agree-banner good">✔ Both readers agree <small>· answer overlap F1 ${fmt(a.f1, 2)} · agreement is much more often correct (see Results)</small></div>`
    : `<div class="agree-banner warn">⚠ The readers disagree <small>· overlap F1 ${fmt(a.f1, 2)} · check the evidence before trusting either answer</small></div>`;
  const ev = evidenceHTML(e);
  return `<div class="card">
    ${banner}
    <div class="compare-grid"><div class="card">${answerBlock(e, true)}</div><div class="card">${answerBlock(g, true)}</div></div>
    <div class="meta-row">${timingsHTML({ ...e.timings, read_ms: (e.timings.read_ms || 0) + (g.timings.read_ms || 0) })}<span class="spacer"></span>
      <span>total ${Math.round(res.latency_ms)} ms</span>${ev.more}</div>
    <div class="evidence collapsed">${ev.cards}</div>
  </div>`;
}

function renderTurn(t) {
  const wrap = document.createElement("div");
  wrap.className = "turn-group";
  const s = t.settings;
  wrap.innerHTML = `<div class="turn user"><div class="bubble">${esc(t.question)}</div>
      <div class="bubble-meta">${s.corpus === "docs" ? "My documents" : "SQuAD"} · ${s.retrieval}${s.rerank ? " + re-rank" : ""} · top-${s.top_k}</div></div>
    <div class="turn assistant">${t.error ? `<div class="card error-card">⚠ ${esc(t.error)}</div>` : t.result.mode === "compare" ? compareCard(t.result) : singleCard(t.result)}</div>`;
  return wrap;
}

function bindTurnEvents(el) {
  el.addEventListener("click", (e) => {
    const tog = e.target.closest(".toggle-ev");
    if (tog) {
      const ev = tog.closest(".card").querySelector(".evidence");
      const collapsed = ev.classList.toggle("collapsed");
      tog.textContent = collapsed ? `Show all ${ev.children.length} passages` : "Show fewer";
      $$(".psg-text", ev).forEach((p, i) => i && p.classList.toggle("clamp", collapsed));
    }
    const cp = e.target.closest(".copy");
    if (cp) navigator.clipboard?.writeText(cp.dataset.text).then(() => toast("Answer copied"));
  });
}

function scrollToBottom() { const t = $("#thread"); t.scrollTop = t.scrollHeight; }

function renderHistory() {
  const thread = $("#thread");
  $$(".turn-group", thread).forEach((n) => n.remove());
  $("#empty").hidden = history.length > 0;
  history.forEach((t) => thread.appendChild(renderTurn(t)));
  scrollToBottom();
}

/* ---------- Asking ---------- */
async function ask(question, gold = null) {
  question = question.trim();
  if (!question || busy) return;
  if (question.length < 3) return toast("Question is too short");
  busy = true;
  $("#send").disabled = true;
  $("#empty").hidden = true;
  const turnSettings = { ...settings };
  const thread = $("#thread");

  const user = document.createElement("div");
  user.className = "turn-group";
  user.innerHTML = `<div class="turn user"><div class="bubble">${esc(question)}</div></div>`;
  thread.appendChild(user);
  const loading = $("#tpl-loading").content.firstElementChild.cloneNode(true);
  if (!turnSettings.rerank) loading.querySelector('[data-step="1"]').remove();
  user.appendChild(loading);
  scrollToBottom();
  const steps = $$(".step", loading);
  let si = 0;
  steps[0].classList.add("active");
  const timer = setInterval(() => {
    if (si < steps.length - 1) { steps[si].classList.replace("active", "done"); steps[++si].classList.add("active"); }
  }, turnSettings.rerank ? 650 : 400);

  const turn = { question, settings: turnSettings, at: Date.now() };
  try {
    turn.result = await api("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, reader: turnSettings.reader, top_k: turnSettings.top_k, retrieval: turnSettings.retrieval,
        rerank: turnSettings.rerank, scoring: turnSettings.scoring, corpus: turnSettings.corpus, gold }),
    });
  } catch (err) {
    turn.error = err.message;
  }
  clearInterval(timer);
  user.replaceWith(renderTurn(turn));
  history.push(turn);
  history = history.slice(-25);
  store.set("qa-history", history);
  scrollToBottom();
  busy = false;
  $("#send").disabled = !$("#question").value.trim();
}

async function askRandom() {
  try {
    const r = await api("/api/random");
    settings.corpus = "squad";
    syncSettingsUI();
    ask(r.question, r.gold);
  } catch (err) { toast(err.message); }
}

function initComposer() {
  const ta = $("#question"), form = $("#composer");
  const resize = () => { ta.style.height = "auto"; ta.style.height = Math.min(ta.scrollHeight, 160) + "px"; $("#send").disabled = busy || !ta.value.trim(); };
  ta.addEventListener("input", resize);
  ta.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); } });
  form.addEventListener("submit", (e) => { e.preventDefault(); const q = ta.value; ta.value = ""; resize(); ask(q); });
  $("#suggestions").addEventListener("click", (e) => { const c = e.target.closest(".chip"); if (c) ask(c.textContent); });
  $("#randomBtn").addEventListener("click", askRandom);
  $("#randomInline").addEventListener("click", askRandom);
  $("#clearChat").addEventListener("click", () => { history = []; store.set("qa-history", history); renderHistory(); });
  document.addEventListener("keydown", (e) => {
    if (e.key === "/" && document.activeElement !== ta && !/INPUT|TEXTAREA/.test(document.activeElement.tagName)) { e.preventDefault(); showView("ask"); ta.focus(); }
  });
  bindTurnEvents($("#thread"));
  resize();
}

/* ---------- Results page (pure SVG charts) ---------- */
function groupedBars({ groups, series, max = 100, height = 230, unit = "" }) {
  const W = 640, H = height, L = 36, B = 28, T = 14;
  const gw = (W - L - 8) / groups.length, bw = Math.min(26, (gw - 14) / series.length);
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img">`;
  for (let v = 0; v <= max; v += max / 4) {
    const y = T + (H - T - B) * (1 - v / max);
    s += `<line class="grid" x1="${L}" x2="${W - 4}" y1="${y}" y2="${y}"/><text x="${L - 6}" y="${y + 4}" text-anchor="end">${v}${unit}</text>`;
  }
  groups.forEach((g, gi) => {
    const x0 = L + gi * gw + (gw - bw * series.length - 2 * (series.length - 1)) / 2;
    series.forEach((se, si) => {
      const v = se.values[gi]; if (v == null) return;
      const h = (H - T - B) * (v / max), x = x0 + si * (bw + 2), y = H - B - h;
      s += `<rect x="${x}" y="${y}" width="${bw}" height="${h}" rx="3" fill="${se.color}"><title>${esc(se.name)} · ${esc(g)}: ${fmt(v)}</title></rect>`;
      if (series.length <= 3) s += `<text class="val" x="${x + bw / 2}" y="${y - 4}" text-anchor="middle">${fmt(v, 0)}</text>`;
    });
    s += `<text class="lbl" x="${L + gi * gw + gw / 2}" y="${H - 8}" text-anchor="middle">${esc(g)}</text>`;
  });
  return s + "</svg>";
}

function hBars(rows, { max = 100 } = {}) {
  const W = 640, rowH = 34, L = 250, H = rows.length * rowH + 10;
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img">`;
  rows.forEach((r, i) => {
    const y = 6 + i * rowH;
    s += `<text class="lbl" x="${L - 10}" y="${y + 13}" text-anchor="end">${esc(r.label)}</text>`;
    s += `<text x="${L - 10}" y="${y + 26}" text-anchor="end">${esc(r.sub || "")}</text>`;
    const w1 = (W - L - 60) * (r.em / max), w2 = (W - L - 60) * (r.f1 / max);
    s += `<rect x="${L}" y="${y + 2}" width="${w2}" height="12" rx="3" fill="${PALETTE[1]}" opacity=".85"><title>F1 ${fmt(r.f1)}</title></rect>`;
    s += `<rect x="${L}" y="${y + 16}" width="${w1}" height="12" rx="3" fill="${PALETTE[0]}"><title>EM ${fmt(r.em)}</title></rect>`;
    s += `<text class="val" x="${L + w2 + 6}" y="${y + 12}">${fmt(r.f1)}</text><text class="val" x="${L + w1 + 6}" y="${y + 26}">${fmt(r.em)}</text>`;
  });
  return s + "</svg>";
}

function lineChart(series, { xs, minY = 50, maxY = 100 }) {
  const W = 640, H = 230, L = 36, B = 28, T = 12, R = 16;
  const X = (x) => L + (W - L - R) * ((xs[0] - x) / (xs[0] - xs[xs.length - 1]));
  const Y = (y) => T + (H - T - B) * (1 - (y - minY) / (maxY - minY));
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img">`;
  for (let v = minY; v <= maxY; v += (maxY - minY) / 5) s += `<line class="grid" x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${L - 6}" y="${Y(v) + 4}" text-anchor="end">${Math.round(v)}</text>`;
  xs.forEach((x) => { s += `<text class="lbl" x="${X(x)}" y="${H - 8}" text-anchor="middle">${Math.round(x * 100)}%</text>`; });
  series.forEach((se) => {
    s += `<polyline fill="none" stroke="${se.color}" stroke-width="2.5" points="${se.points.map(([x, y]) => `${X(x)},${Y(y)}`).join(" ")}"/>`;
    se.points.forEach(([x, y]) => { s += `<circle cx="${X(x)}" cy="${Y(y)}" r="4" fill="${se.color}"><title>${esc(se.name)} @ ${Math.round(x * 100)}%: EM ${fmt(y)}</title></circle>`; });
  });
  return s + "</svg>";
}

function legend(items) { return `<div class="legend">${items.map(([n, c]) => `<span><i style="background:${c}"></i>${esc(n)}</span>`).join("")}</div>`; }

function lossSpark(hist) {
  if (!hist?.length) return "";
  const W = 300, H = 70, xs = hist.map((h) => h.epoch), ys = hist.map((h) => +h.loss);
  const mn = Math.min(...ys), mx = Math.max(...ys);
  const pts = hist.map((h, i) => `${4 + (W - 8) * (i / (hist.length - 1 || 1))},${6 + (H - 12) * (1 - (ys[i] - mn) / (mx - mn || 1))}`).join(" ");
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Training loss curve"><polyline fill="none" stroke="${PALETTE[0]}" stroke-width="2" points="${pts}"/></svg>`;
}

async function renderResults() {
  const page = $("#resultsPage");
  let data;
  try { data = await api("/api/results"); } catch (err) { page.innerHTML = `<div class="card error-card">${esc(err.message)}</div>`; return; }
  const { baseline: b, advanced: a, training: t } = data;
  if (!b) { page.innerHTML = `<div class="card">No results yet. Run <code>python -m qa.evaluate</code>.</div>`; return; }

  const find = (rows, key, setting) => rows.find((r) => r.key === key && r.setting.startsWith(setting));
  const baseOpen = find(b.readers, "finetuned", "retrieved");
  const best = a ? [...a.readers].sort((x, y) => y.f1 - x.f1)[0] : null;
  const dense = a?.retrieval["Dense (MiniLM + FAISS)"], hyb = a?.retrieval["Hybrid + cross-encoder re-rank (top 20)"];

  let html = `<h1>Measured results</h1><div class="sub">SQuAD v1.1 validation · ${esc(b.settings.machine)} · CPU only. Every number is produced by the evaluation scripts in this repo.</div>`;
  html += `<div class="kpis">
    <div class="kpi"><div class="kpi-label">Recall@3 · dense retriever</div><div class="kpi-value">${fmt((dense || b.retrieval)["recall@3"])}%</div><div class="kpi-note">${dense ? `${a.settings.retrieval_samples.toLocaleString()} questions` : `${b.retrieval.n.toLocaleString()} questions`}</div></div>
    ${hyb ? `<div class="kpi"><div class="kpi-label">Recall@3 · hybrid + re-rank</div><div class="kpi-value">${fmt(hyb["recall@3"])}%</div><div class="kpi-delta ${hyb["recall@3"] >= dense["recall@3"] ? "up" : "down"}">${hyb["recall@3"] >= dense["recall@3"] ? "+" : ""}${fmt(hyb["recall@3"] - dense["recall@3"])} pts vs dense (same ${a.settings.retrieval_samples} q)</div></div>` : ""}
    ${baseOpen ? `<div class="kpi"><div class="kpi-label">Pipeline F1 · baseline</div><div class="kpi-value">${fmt(baseOpen.f1)}</div><div class="kpi-note">dense top-3 → DistilBERT (ours)</div></div>` : ""}
    ${best ? `<div class="kpi"><div class="kpi-label">Pipeline F1 · best</div><div class="kpi-value">${fmt(best.f1)}</div><div class="kpi-delta up">${esc(best.reader)}</div></div>` : ""}
    ${a ? `<div class="kpi"><div class="kpi-label">EM when readers agree</div><div class="kpi-value">${fmt(a.agreement.agree_em_ext)}</div><div class="kpi-note">${fmt(a.agreement.agree_rate)}% of questions · vs ${fmt(a.agreement.disagree_em_ext)} when they disagree</div></div>` : ""}
  </div>`;

  if (a) {
    const names = Object.keys(a.retrieval);
    html += `<h2>Retrieval: does hybrid search + re-ranking find the right passage more often?</h2>
      <div class="chart-card">${legend(names.map((n, i) => [n, PALETTE[i]]))}
      ${groupedBars({ groups: ["Recall@1", "Recall@3", "Recall@5", "Recall@10"], series: names.map((n, i) => ({ name: n, color: PALETTE[i], values: [1, 3, 5, 10].map((k) => a.retrieval[n][`recall@${k}`]) })), unit: "" })}</div>`;
    html += `<h2>Full pipeline: retrieve top-3 → read</h2><div class="chart-card">${legend([["F1", PALETTE[1]], ["Exact match", PALETTE[0]]])}
      ${hBars(a.readers.map((r) => ({ label: r.reader.replace(" (fine-tuned by us)", ""), sub: `${r.retrieval}${r.scoring !== "-" ? " · " + r.scoring : ""}`, em: r.exact_match, f1: r.f1 })))}</div>`;
    const sel = a.selective;
    const xs = sel.extractive_calibrated.map((p) => p.coverage);
    const ys = [...sel.extractive_calibrated, ...sel.generative].map((p) => p.exact_match);
    html += `<div class="two-col"><div><h2>Selective answering</h2><div class="chart-card">${legend([["DistilBERT (calibrated)", PALETTE[0]], ["Flan-T5", PALETTE[2]]])}
      ${lineChart([{ name: "DistilBERT", color: PALETTE[0], points: sel.extractive_calibrated.map((p) => [p.coverage, p.exact_match]) },
                   { name: "Flan-T5", color: PALETTE[2], points: sel.generative.map((p) => [p.coverage, p.exact_match]) }],
                  { xs, minY: Math.floor(Math.min(...ys) / 10) * 10, maxY: 100 })}
      <div class="kpi-note">Exact match when only the most-confident X% of questions are answered.</div></div></div>
      <div><h2>Reader agreement</h2><div class="table-wrap"><table class="data"><thead><tr><th>Subset</th><th class="num">Questions</th><th class="num">DistilBERT EM</th><th class="num">Flan-T5 EM</th></tr></thead><tbody>
      <tr><td>Readers agree</td><td class="num">${a.agreement.agree_n}</td><td class="num">${fmt(a.agreement.agree_em_ext)}</td><td class="num">${fmt(a.agreement.agree_em_gen)}</td></tr>
      <tr><td>Readers disagree</td><td class="num">${a.agreement.disagree_n}</td><td class="num">${fmt(a.agreement.disagree_em_ext)}</td><td class="num">${fmt(a.agreement.disagree_em_gen)}</td></tr>
      <tr class="best"><td>Ensemble</td><td class="num">${a.settings.reader_samples}</td><td class="num" colspan="2">EM ${fmt(a.agreement.ensemble.exact_match)} · F1 ${fmt(a.agreement.ensemble.f1)}</td></tr>
      </tbody></table></div></div></div>`;
  }

  html += `<h2>Baseline (first version)</h2><div class="table-wrap"><table class="data"><thead><tr><th>Reader</th><th>Setting</th><th class="num">Questions</th><th class="num">EM</th><th class="num">F1</th><th class="num">ms / q</th></tr></thead><tbody>
    ${b.readers.map((r) => `<tr><td>${esc(r.reader)}</td><td>${esc(r.setting)}</td><td class="num">${r.n.toLocaleString()}</td><td class="num">${fmt(r.exact_match, 2)}</td><td class="num">${fmt(r.f1, 2)}</td><td class="num">${Math.round(r.ms_per_question)}</td></tr>`).join("")}
    </tbody></table></div>`;

  if (t) {
    html += `<h2>Fine-tuning run</h2><div class="two-col"><div class="kpis">
      <div class="kpi"><div class="kpi-label">Training examples</div><div class="kpi-value">${t.train_examples.toLocaleString()}</div><div class="kpi-note">${t.epochs} epoch · ${t.global_steps} steps · lr ${t.learning_rate}</div></div>
      <div class="kpi"><div class="kpi-label">Time on CPU</div><div class="kpi-value">${fmt(t.train_minutes)} min</div><div class="kpi-note">${esc(t.cpu)}</div></div>
      <div class="kpi"><div class="kpi-label">Validation loss</div><div class="kpi-value">${fmt(t.eval_loss_after, 3)}</div><div class="kpi-note">before: ${fmt(t.eval_loss_before, 3)}</div></div>
      </div><div class="chart-card"><div class="kpi-label">Training loss (every 20 steps)</div>${lossSpark(t.loss_history)}</div></div>`;
  }
  page.innerHTML = html;
}

/* ---------- Views, theme, mobile ---------- */
let resultsLoaded = false;
function showView(name) {
  $$(".tab").forEach((t) => t.setAttribute("aria-selected", String(t.dataset.view === name)));
  $$(".view").forEach((v) => (v.hidden = v.id !== `view-${name}`));
  if (name === "results" && !resultsLoaded) { resultsLoaded = true; renderResults(); }
  if (name === "ask") $("#question").focus();
}

function initChrome() {
  $$(".tab").forEach((t) => t.addEventListener("click", () => showView(t.dataset.view)));
  $("#themeToggle").addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("qa-theme", document.documentElement.dataset.theme); } catch { /* ignore */ }
  });
  const sb = $("#sidebar"), scrim = $("#scrim");
  const open = (o) => { sb.classList.toggle("open", o); scrim.hidden = !o; };
  $("#openSidebar").addEventListener("click", () => open(true));
  $("#closeSidebar").addEventListener("click", () => open(false));
  scrim.addEventListener("click", () => open(false));
}

initSettings();
initDocs();
initComposer();
initChrome();
renderHistory();
pollHealth();
