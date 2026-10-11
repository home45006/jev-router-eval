"use strict";
// ---------------------------------------------------------------- 工具
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtMs = (ms) => (ms == null ? "-" : ms >= 1000 ? (ms / 1000).toFixed(2) + " s" : Math.round(ms) + " ms");
const pct = (a, b) => (b ? ((a / b) * 100).toFixed(1) + "%" : "-");
const quant = (arr, q) => {
  if (!arr.length) return null;
  const s = [...arr].sort((a, b) => a - b);
  const i = (s.length - 1) * q, lo = Math.floor(i), hi = Math.ceil(i);
  return s[lo] + (s[hi] - s[lo]) * (i - lo);
};
const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* 无痕模式等 */ } },
};
const KIND = {
  jev: { name: "Jev", color: "var(--k-jev)" },
  llm: { name: "大模型（DeepSeek）", color: "var(--k-llm)" },
  agent: { name: "能力（子 Agent）", color: "var(--k-agent)" },
  fast: { name: "快思考", color: "var(--k-fast)" },
  code: { name: "代码", color: "var(--k-code)" },
};

async function streamPost(url, body, onLine) {
  const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, i).trim();
      buf = buf.slice(i + 1);
      if (line) onLine(JSON.parse(line));
    }
  }
  if (buf.trim()) onLine(JSON.parse(buf));
}

// ---------------------------------------------------------------- 全局状态
const S = {
  config: null,
  settings: null,
  sid: null,
  runs: [],          // 所有完成的结果，用于耗时统计
  samples: null,
  busy: false,
};
const newSid = () => Math.random().toString(36).slice(2, 10);

// ---------------------------------------------------------------- 提示框
const tip = $("#tip");
function showTip(e, text) {
  tip.textContent = text;
  tip.hidden = false;
  const x = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
  const y = Math.min(e.clientY + 14, window.innerHeight - tip.offsetHeight - 8);
  tip.style.left = x + "px";
  tip.style.top = y + "px";
}
const hideTip = () => (tip.hidden = true);
function bindTips(root) {
  $$("[data-tip]", root).forEach((n) => {
    n.addEventListener("mousemove", (e) => showTip(e, n.dataset.tip));
    n.addEventListener("mouseleave", hideTip);
  });
}

// ---------------------------------------------------------------- 瀑布图
function niceStep(max) {
  const raw = max / 5, p = Math.pow(10, Math.floor(Math.log10(raw || 1)));
  return [1, 2, 2.5, 5, 10].map((m) => m * p).find((s) => s >= raw) || raw;
}
function waterfall(steps) {
  const done = steps.filter((s) => s.start != null);
  const total = Math.max(1, ...done.map((s) => s.end ?? s.start + 1));
  const W = 900, L = 180, R = 76, row = 26, top = 22;
  const H = top + done.length * row + 6;
  const x = (t) => L + (t / total) * (W - L - R);
  const step = niceStep(total);
  let ticks = "";
  for (let t = 0; t <= total + 1e-6; t += step) {
    ticks += `<line x1="${x(t)}" x2="${x(t)}" y1="${top - 4}" y2="${H}"/><text x="${x(t)}" y="12" text-anchor="middle">${fmtMs(t)}</text>`;
  }
  const bars = done.map((s, i) => {
    const y = top + i * row;
    const end = s.end ?? s.start;
    const w = Math.max(3, x(end) - x(s.start));
    const dur = s.end != null ? s.end - s.start : null;
    const tipText = `${s.name}\n${fmtMs(s.start)} → ${s.end != null ? fmtMs(s.end) : "进行中"}，耗时 ${fmtMs(dur)}\n${s.summary || ""}`;
    return `<g data-tip="${esc(tipText)}">
      <rect x="0" y="${y}" width="${W}" height="${row}" fill="transparent"/>
      <text x="${L - 10}" y="${y + 17}" text-anchor="end">${esc(s.name)}</text>
      <rect class="bar ${s.status === "running" ? "running" : ""} ${s.status === "error" ? "err" : ""}" x="${x(s.start)}" y="${y + 6}" width="${w}" height="${row - 12}" rx="4" fill="${KIND[s.kind]?.color || "var(--k-code)"}"/>
      <text class="ms" x="${W - R + 8}" y="${y + 17}">${s.end != null ? fmtMs(dur) : "…"}</text>
    </g>`;
  }).join("");
  return `<svg class="wf" viewBox="0 0 ${W} ${H}" role="img" aria-label="各步骤耗时瀑布图"><g class="axis">${ticks}</g>${bars}</svg>`;
}
function legend(kinds) {
  return `<div class="legend">${kinds.map((k) => `<span><i style="background:${KIND[k].color}"></i>${KIND[k].name}</span>`).join("")}</div>`;
}

// ---------------------------------------------------------------- 单句：一轮的渲染
function flowNodes(steps, result) {
  const nodes = steps.map((s) => {
    const cls = s.status === "running" ? "running" : s.status === "error" ? "err" : "on";
    return `<span class="node ${cls}">${esc(s.name)}</span>`;
  });
  if (result) {
    nodes.push(`<span class="node on">${result.destination === "快思考" ? "执行" : "播报"}</span>`);
  }
  return `<div class="flow">${nodes.join('<span class="arrow">→</span>')}</div>`;
}

function probRow(name, p, thr, tag) {
  const over = p >= thr;
  return `<div class="prob" data-tip="${esc(`${name}：${p.toFixed(3)}，阈值 ${thr}`)}">
    <span class="name">${esc(name)}${tag ? ` <span class="tag">${esc(tag)}</span>` : ""}</span>
    <span class="track"><span class="fill ${over ? "over" : "under"}" style="width:${(p * 100).toFixed(1)}%"></span><span class="thr" style="left:calc(${thr * 100}% - 1px)"></span></span>
    <span class="val">${p.toFixed(2)}</span></div>`;
}

function stepOf(steps, name) { return steps.find((s) => s.name === name); }

function panels(result) {
  const st = result.steps, set = S.settings, out = [];
  const jev = stepOf(st, "Jev 第一级");
  if (jev && jev.detail.probabilities) {
    const p = jev.detail.probabilities, d = jev.detail;
    let rows = probRow("is_fast", p.is_fast, set.is_fast_t, "交给快思考") + probRow("needs_rewrite", p.needs_rewrite, set.rewrite_t, "模糊指代");
    for (const k of Object.keys(p)) {
      if (k === "is_fast" || k === "needs_rewrite") continue;
      const c = S.config.corrections[k];
      rows += probRow(k, p[k], set.correction_t, c ? `${set.corrections[k]} → ${c.route}` : "");
    }
    out.push(`<div class="panel"><h4>Jev 第一级 <span class="ms">${fmtMs(jev.end - jev.start)}${d.input_tokens ? ` · ${d.input_tokens} token` : ""}</span></h4>
      ${rows}<p class="hint">一次请求同时回答 ${Object.keys(p).length} 个问题；竖线是阈值。模型：${esc(d.model)}</p></div>`);
  }
  const dec = stepOf(st, "分流决策");
  if (dec) {
    out.push(`<div class="panel"><h4>分流决策</h4><ul>${(dec.detail.reasons || []).map((r) => `<li>${esc(r)}</li>`).join("")}</ul>
      <p class="hint">路径：${esc((result.path || []).join(" → "))}</p></div>`);
  }
  const rw = stepOf(st, "大模型改写");
  if (rw) {
    out.push(`<div class="panel"><h4>大模型改写 <span class="ms">${fmtMs(rw.end - rw.start)}</span></h4>
      <dl class="kv"><dt>原句</dt><dd>${esc(rw.detail.before)}</dd><dt>改写后</dt><dd><b>${esc(rw.detail.after)}</b></dd>
      <dt>模型</dt><dd>${esc(rw.detail.model)}</dd><dt>token</dt><dd>输入 ${rw.detail.prompt_tokens ?? "-"} · 输出 ${rw.detail.completion_tokens ?? "-"} · 推理 ${rw.detail.reasoning_tokens ?? "-"}</dd></dl></div>`);
  }
  const fm = stepOf(st, "快思考模型");
  if (fm) {
    out.push(`<div class="panel"><h4>快思考模型 <span class="ms">${fmtMs(fm.end - fm.start)}</span></h4>
      <dl class="kv"><dt>输入</dt><dd>${esc(fm.detail.input)}</dd><dt>结果</dt><dd>${fm.detail.rejected ? "拒识 → 转 Agent 大脑" : "理解成功，交给执行链路"}</dd></dl>
      <pre>${esc(JSON.stringify(fm.detail.response, null, 2))}</pre></div>`);
  }
  if (result.brain) {
    const b = result.brain;
    const calls = st.filter((s) => s.kind === "llm" && s.name.startsWith("Agent 大脑"));
    const tools = (b.tools_called || []).map((t) => `<div class="tool"><span class="pill label">${esc(t.label)}</span> <span class="t">任务：</span>${esc(t.task)}<br><span class="t">结果：</span>${esc(t.result)}</div>`).join("");
    const callRows = calls.map((c) => `<li>${esc(c.name)}：${fmtMs(c.end - c.start)} · 输入 ${c.detail.prompt_tokens ?? "-"} / 输出 ${c.detail.completion_tokens ?? "-"} / 推理 ${c.detail.reasoning_tokens ?? "-"} token</li>`).join("");
    out.push(`<div class="panel" style="grid-column:1/-1"><h4>Agent 大脑 <span class="pill">${b.mode === "orchestrate" ? "编排模式" : "标签模式"}</span><span class="ms">会话状态：${esc(b.status)}</span></h4>
      <dl class="kv"><dt>能力标签</dt><dd>${(b.labels || []).map((l) => `<span class="pill label">${esc(l)}</span>`).join(" ") || "无"}</dd>
      <dt>计划</dt><dd>${esc(b.plan)}</dd></dl>
      ${tools ? `<h4 style="margin-top:10px">调用的能力</h4>${tools}` : ""}
      <h4 style="margin-top:10px">大模型调用</h4><ul>${callRows}</ul>
      ${b.raw ? `<details><summary class="hint">大脑原始输出</summary><pre>${esc(b.raw)}</pre></details>` : ""}</div>`);
  }
  return `<div class="details">${out.join("")}</div>`;
}

function renderTurn(card, steps, result, error) {
  const head = card.querySelector(".turn-head");
  const body = card.querySelector(".turn-body");
  const kinds = [...new Set(steps.map((s) => s.kind))];
  let headHtml = `<span class="q">${esc(card.dataset.q)}</span>`;
  if (result) {
    const fast = result.destination === "快思考";
    headHtml += `<span class="pill ${fast ? "dest-fast" : "dest-brain"}">${fast ? "快思考" : "Agent 大脑"}</span>`;
    if (!fast) headHtml += (result.labels || []).map((l) => `<span class="pill label">${esc(l)}</span>`).join("");
    if (result.rewritten) headHtml += `<span class="pill">改写：${esc(result.rewritten)}</span>`;
    if (result.correction) headHtml += `<span class="pill busy">纠错题 ${esc(result.correction)}</span>`;
    if (result.suspended) headHtml += `<span class="pill busy">挂起进行中的任务</span>`;
    const t = result.timing;
    headHtml += `<span class="ms" data-tip="${esc(`Jev ${fmtMs(t.jev_ms)} · 大模型 ${fmtMs(t.llm_ms)} · 能力 ${fmtMs(t.agent_ms)} · 快思考 ${fmtMs(t.fast_ms)} · 大模型调用 ${t.llm_calls} 次`)}">端到端 ${fmtMs(t.total_ms)}</span>`;
  } else if (error) {
    headHtml += `<span class="pill no">出错</span>`;
  } else {
    headHtml += `<span class="pill busy">处理中…</span>`;
  }
  head.innerHTML = headHtml;
  let html = flowNodes(steps, result);
  const speech = result?.brain?.speech;
  if (speech) html += `<div class="speech"><b>播报</b>${esc(speech)}</div>`;
  if (error) html += `<div class="speech" style="color:var(--bad)">${esc(error)}</div>`;
  html += `<div><div class="scroll">${waterfall(steps)}</div>${legend(kinds)}</div>`;
  if (result) html += panels(result);
  body.innerHTML = html;
  bindTips(card);
}

function updateOccupy(sess) {
  const o = $("#occupy");
  if (sess && sess.active) {
    o.className = "pill busy";
    o.textContent = `大脑占用会话：${(sess.labels || []).join("、") || "进行中"} · 第 ${sess.turns} 轮`;
  } else {
    o.className = "pill muted";
    o.textContent = sess ? `没有进行中的会话 · 已 ${sess.turns} 轮` : "没有进行中的会话";
  }
}

async function ask(query) {
  query = query.trim();
  if (!query || S.busy) return;
  S.busy = true;
  $("#askBtn").disabled = true;
  $("#emptyHint").hidden = true;
  const card = document.createElement("div");
  card.className = "turn";
  card.dataset.q = query;
  card.innerHTML = `<div class="turn-head"></div><div class="turn-body"></div>`;
  $("#turns").prepend(card);
  const steps = [];
  renderTurn(card, steps, null);
  try {
    await streamPost("/api/run", { query, session_id: S.sid, settings: S.settings }, (ev) => {
      if (ev.type === "step") {
        const i = steps.findIndex((s) => s.id === ev.step.id);
        if (i >= 0) steps[i] = ev.step; else steps.push(ev.step);
        renderTurn(card, steps, null);
      } else if (ev.type === "result") {
        renderTurn(card, ev.result.steps, ev.result);
        updateOccupy(ev.result.session);
        S.runs.push(ev.result);
      } else if (ev.type === "error") {
        renderTurn(card, steps, null, ev.error);
      }
    });
  } catch (e) {
    renderTurn(card, steps, null, String(e));
  } finally {
    S.busy = false;
    $("#askBtn").disabled = false;
  }
}

async function playDialog(d) {
  await resetSession();
  for (const t of d.turns) {
    $("#q").value = t;
    await ask(t);
  }
  $("#q").value = "";
}

async function resetSession() {
  if (S.sid) await fetch("/api/reset", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session_id: S.sid }) });
  S.sid = newSid();
  $("#sid").textContent = S.sid;
  $("#turns").innerHTML = "";
  $("#emptyHint").hidden = false;
  updateOccupy(null);
}

// ---------------------------------------------------------------- 批量验证
const B = { rows: [], items: [] };

function batchItems() {
  const src = $("#batchSource").value;
  if (src === "builtin") return S.samples.builtin;
  if (src === "eval") return S.samples.eval_set;
  return $("#batchCustom").value.split("\n").map((l) => l.trim()).filter(Boolean).map((l) => {
    const [query, dest, labels] = l.split("\t");
    return { query, dest: dest || null, labels: labels ? labels.split(/[,，]/).map((x) => x.trim()).filter(Boolean) : null, source: "自定义" };
  });
}

function mark(v) { return v === true ? '<span class="mark-ok">✓</span>' : v === false ? '<span class="mark-no">✗</span>' : '<span class="mark-na">–</span>'; }

function renderBatch() {
  const rows = B.rows.filter(Boolean);
  const n = rows.length;
  const destJ = rows.filter((r) => r.correct && r.correct.dest != null);
  const labJ = rows.filter((r) => r.correct && r.correct.label != null);
  const okD = destJ.filter((r) => r.correct.dest).length, okL = labJ.filter((r) => r.correct.label).length;
  const okA = labJ.filter((r) => r.correct.label_any).length;
  const jevMs = [], e2e = [], brainMs = [];
  rows.forEach((r) => {
    if (!r.result) return;
    r.result.steps.forEach((s) => {
      if (s.kind === "jev") jevMs.push(s.end - s.start);
      if (s.kind === "llm" && s.name.startsWith("Agent 大脑")) brainMs.push(s.end - s.start);
    });
    e2e.push(r.result.timing.total_ms);
  });
  const errs = rows.filter((r) => r.error).length;
  $("#batchTiles").innerHTML = [
    ["已完成", `${n} / ${B.items.length}`, errs ? `${errs} 条出错` : "　"],
    ["去向准确率（Jev）", pct(okD, destJ.length), `${okD} / ${destJ.length}`],
    ["大脑首选标签准确率", pct(okL, labJ.length), `${okL} / ${labJ.length}；标签里包含正确答案 ${pct(okA, labJ.length)}`],
    ["Jev 耗时 p50 / p90", `${fmtMs(quant(jevMs, 0.5))}`, `p90 ${fmtMs(quant(jevMs, 0.9))}`],
    ["大脑单次推理 p50", fmtMs(quant(brainMs, 0.5)), `p90 ${fmtMs(quant(brainMs, 0.9))}`],
    ["端到端 p50", fmtMs(quant(e2e, 0.5)), `p90 ${fmtMs(quant(e2e, 0.9))}`],
  ].map(([k, v, s]) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s}</div></div>`).join("");

  // 按应有标签
  const per = {};
  rows.forEach((r) => {
    const g = (r.item.labels || ["（无）"]).join("、");
    per[g] ??= { n: 0, d: 0, dn: 0, l: 0, ln: 0 };
    per[g].n++;
    if (r.correct?.dest != null) { per[g].dn++; per[g].d += r.correct.dest ? 1 : 0; }
    if (r.correct?.label != null) { per[g].ln++; per[g].l += r.correct.label ? 1 : 0; }
  });
  $("#perLabelCard").hidden = !n;
  $("#perLabel").innerHTML = `<thead><tr><th>应有标签</th><th class="num">样本</th><th class="num">去向正确</th><th class="num">标签正确</th></tr></thead><tbody>` +
    Object.entries(per).sort((a, b) => b[1].n - a[1].n).map(([g, v]) =>
      `<tr><td>${esc(g)}</td><td class="num">${v.n}</td><td class="num">${pct(v.d, v.dn)}</td><td class="num">${pct(v.l, v.ln)}</td></tr>`).join("") + "</tbody>";

  const only = $("#onlyWrong").checked;
  const body = B.rows.map((r, i) => {
    if (!r) return "";
    if (only && !(r.error || r.correct?.dest === false || r.correct?.label === false)) return "";
    const res = r.result || {};
    return `<tr data-i="${i}"><td class="num">${i + 1}</td><td>${esc(r.item.query)}<div class="hint">${esc(r.item.source || "")}</div></td>
      <td>${esc(r.item.dest || "-")}</td><td>${r.error ? `<span class="mark-no">${esc(r.error)}</span>` : esc(res.destination)} ${mark(r.correct?.dest)}</td>
      <td>${esc((r.item.labels || []).join("、"))}</td><td>${esc((res.labels || []).join("、"))} ${mark(r.correct?.label)}</td>
      <td class="num">${res.jev ? res.jev.is_fast.toFixed(2) : "-"}</td><td class="num">${fmtMs(res.timing?.total_ms)}</td></tr>`;
  }).join("");
  $("#batchTable").innerHTML = `<thead><tr><th class="num">#</th><th>句子</th><th>应去</th><th>实际去向</th><th>应有标签</th><th>大脑标签</th><th class="num">is_fast</th><th class="num">端到端</th></tr></thead><tbody>${body}</tbody>`;
  $$("#batchTable tbody tr[data-i]").forEach((tr) => tr.addEventListener("click", () => toggleBatchRow(tr)));
}

function toggleBatchRow(tr) {
  const next = tr.nextElementSibling;
  if (next && next.classList.contains("expand")) { next.remove(); return; }
  const r = B.rows[+tr.dataset.i];
  if (!r?.result) return;
  const res = r.result;
  const x = document.createElement("tr");
  x.className = "expand";
  x.innerHTML = `<td></td><td colspan="7">
    <div class="hint">路径：${esc((res.path || []).join(" → "))}${res.rewritten ? ` · 改写：${esc(res.rewritten)}` : ""}${res.brain?.plan ? ` · 计划：${esc(res.brain.plan)}` : ""}</div>
    <div class="hint">Jev：${Object.entries(res.jev || {}).map(([k, v]) => `${k} ${v.toFixed(2)}`).join(" · ")}</div>
    <div class="scroll">${waterfall(res.steps)}</div>${legend([...new Set(res.steps.map((s) => s.kind))])}</td>`;
  tr.after(x);
  bindTips(x);
}

async function runBatch() {
  const items = batchItems();
  if (!items.length) return;
  B.items = items;
  B.rows = new Array(items.length).fill(null);
  $("#batchBtn").disabled = true;
  let done = 0;
  renderBatch();
  try {
    await streamPost("/api/batch", { items, settings: S.settings, concurrency: +$("#batchConc").value || 4 }, (ev) => {
      if (ev.type !== "item") return;
      B.rows[ev.index] = ev;
      if (ev.result) S.runs.push({ ...ev.result, query: ev.item.query });
      done++;
      $("#batchBar").style.width = `${(done / items.length) * 100}%`;
      renderBatch();
    });
  } catch (e) {
    $("#batchNote").textContent = String(e);
  } finally {
    $("#batchBtn").disabled = false;
  }
}

// ---------------------------------------------------------------- 耗时统计
function latencySeries() {
  const rows = [
    { key: "Jev 第一级", kind: "jev", vals: [] },
    { key: "大模型改写", kind: "llm", vals: [] },
    { key: "大脑单次推理", kind: "llm", vals: [] },
    { key: "能力（子 Agent）", kind: "agent", vals: [] },
    { key: "端到端 · 快思考路径", kind: "fast", vals: [] },
    { key: "端到端 · 大脑路径", kind: "code", vals: [] },
  ];
  const by = Object.fromEntries(rows.map((r) => [r.key, r]));
  for (const r of S.runs) {
    const q = r.query;
    for (const s of r.steps || []) {
      if (s.end == null) continue;
      const v = { ms: s.end - s.start, q };
      if (s.kind === "jev") by["Jev 第一级"].vals.push(v);
      else if (s.name === "大模型改写") by["大模型改写"].vals.push(v);
      else if (s.kind === "llm" && s.name.startsWith("Agent 大脑")) by["大脑单次推理"].vals.push(v);
      else if (s.kind === "agent") by["能力（子 Agent）"].vals.push(v);
    }
    if (r.timing) by[r.destination === "快思考" ? "端到端 · 快思考路径" : "端到端 · 大脑路径"].vals.push({ ms: r.timing.total_ms, q });
  }
  return rows;
}

function renderLatency() {
  const rows = latencySeries();
  const all = rows.flatMap((r) => r.vals.map((v) => v.ms));
  if (!all.length) {
    $("#strip").innerHTML = '<p class="empty">还没有数据。先在“单句与多轮”或“批量验证”里跑几句。</p>';
    $("#latTable").innerHTML = "";
    return;
  }
  const max = Math.max(...all) * 1.05;
  const W = 900, L = 170, R = 20, row = 38, top = 24, H = top + rows.length * row + 4;
  const x = (t) => L + (t / max) * (W - L - R);
  const step = niceStep(max);
  let ax = "";
  for (let t = 0; t <= max; t += step) ax += `<line class="grid" x1="${x(t)}" x2="${x(t)}" y1="${top - 6}" y2="${H}"/><text x="${x(t)}" y="12" text-anchor="middle">${fmtMs(t)}</text>`;
  const body = rows.map((r, i) => {
    const y = top + i * row + row / 2;
    const rnd = (k) => (Math.sin(k * 999.13) * 0.5) * 14;     // 固定的纵向抖动
    const dots = r.vals.map((v, k) => `<circle cx="${x(v.ms)}" cy="${y + rnd(k + i * 31)}" r="4.5" fill="${KIND[r.kind].color}" fill-opacity=".75" stroke="var(--surface)" stroke-width="1.5" data-tip="${esc(`${r.key}：${fmtMs(v.ms)}\n${v.q || ""}`)}"/>`).join("");
    const med = quant(r.vals.map((v) => v.ms), 0.5);
    return `<text x="${L - 10}" y="${y + 4}" text-anchor="end">${esc(r.key)}</text>${dots}` +
      (med != null ? `<line class="med" x1="${x(med)}" x2="${x(med)}" y1="${y - 13}" y2="${y + 13}"/>` : "");
  }).join("");
  $("#strip").innerHTML = `<svg class="strip wf" viewBox="0 0 ${W} ${H}" role="img" aria-label="各环节耗时分布"><g class="axis">${ax}</g>${body}</svg>`;
  bindTips($("#strip"));
  $("#latTable").innerHTML = `<thead><tr><th>环节</th><th class="num">次数</th><th class="num">最快</th><th class="num">p50</th><th class="num">p90</th><th class="num">最慢</th><th class="num">平均</th></tr></thead><tbody>` +
    rows.map((r) => {
      const v = r.vals.map((x) => x.ms);
      const mean = v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
      return `<tr><td>${esc(r.key)}</td><td class="num">${v.length}</td><td class="num">${fmtMs(v.length ? Math.min(...v) : null)}</td><td class="num">${fmtMs(quant(v, 0.5))}</td><td class="num">${fmtMs(quant(v, 0.9))}</td><td class="num">${fmtMs(v.length ? Math.max(...v) : null)}</td><td class="num">${fmtMs(mean)}</td></tr>`;
    }).join("") + "</tbody>";
}

// ---------------------------------------------------------------- 设置
function saveSettings() { store.set("cockpit-demo-settings", S.settings); }

function renderSettings() {
  const s = S.settings, c = S.config;
  const sliders = [["is_fast_t", "is_fast 阈值"], ["rewrite_t", "needs_rewrite 阈值"], ["correction_t", "纠错题命中阈值"]];
  $("#thresholds").innerHTML = sliders.map(([k, n]) => `<label class="slider"><span>${n}</span><input type="range" min="0.05" max="0.95" step="0.05" data-k="${k}" value="${s[k]}"><output>${s[k]}</output></label>`).join("");
  $$("#thresholds input").forEach((inp) => inp.addEventListener("input", () => {
    s[inp.dataset.k] = +inp.value;
    inp.nextElementSibling.textContent = inp.value;
    saveSettings();
  }));
  $("#corrSettings").innerHTML = Object.entries(c.corrections).map(([k, v]) => `<div class="corr">
    <div class="row"><code>${esc(k)}</code><span class="spacer"></span>
      <select data-k="${k}">${["off", "shadow", "on"].map((o) => `<option ${s.corrections[k] === o ? "selected" : ""}>${o}</option>`).join("")}</select></div>
    <div>${esc(v.instructions)}</div><div class="why">命中后：${esc(v.route)}${v.hint ? `（提示 ${esc(v.hint)}）` : ""}。${esc(v.why)}</div></div>`).join("");
  $$("#corrSettings select").forEach((sel) => sel.addEventListener("change", () => { s.corrections[sel.dataset.k] = sel.value; saveSettings(); }));
  $("#simReject").checked = s.simulate_fast_reject;
  $("#maxIters").value = s.max_iters;
  $$("input[name=mode]").forEach((r) => (r.checked = r.value === s.brain_mode));
}

function renderInfo() {
  const c = S.config, b = c.backends;
  const pill = (kind, real, mock, absent) => kind === "real" ? `<span class="pill real">${real}</span>` : kind === "mock" ? `<span class="pill mock">${mock}</span>` : `<span class="pill">${absent}</span>`;
  $("#backends").innerHTML = pill(b.jev, "Jev 真实", "Jev 模拟") + pill(b.llm, "DeepSeek 真实", "DeepSeek 模拟") + pill(b.fast, "快思考已接入", "", "快思考未接入");
  $("#backendInfo").innerHTML = `<dl class="kv">
    <dt>Jev</dt><dd>${esc(b.jev_model)}${b.jev === "mock" ? "（未设置 TYPESAFE_API_KEY，结果由关键词规则产生）" : ""}</dd>
    <dt>大模型</dt><dd>${esc(b.llm_model)}${b.llm === "mock" ? "（未设置 DEEPSEEK_API_KEY，结果由关键词规则产生）" : "，非思考模式"}</dd>
    <dt>快思考</dt><dd>${b.fast === "real" ? "已接入 FAST_MODEL_URL" : "未接入：只记录交给快思考，可在左侧模拟拒识"}</dd>
    <dt>判据</dt><dd>${esc(c.criteria_version)}</dd></dl>`;
  $("#stage1Info").innerHTML = Object.entries(c.stage1).map(([k, v]) => `<p><code>${esc(k)}</code> ${esc(v)}</p>`).join("");
  $("#abilityInfo").innerHTML = `<table><tbody>${c.abilities.map((a) => `<tr><td><span class="pill label">${esc(a.label)}</span></td><td>${esc(a.desc)}</td></tr>`).join("")}</tbody></table>`;
}

function renderChips() {
  const items = S.samples.builtin;
  const fast = items.filter((x) => x.dest === "快思考"), brain = items.filter((x) => x.dest !== "快思考");
  const chip = (x) => `<button type="button" data-q="${esc(x.query)}" title="应去：${esc(x.dest)}；标签：${esc((x.labels || []).join("、"))}">${esc(x.query)}</button>`;
  $("#chips").innerHTML = `<span class="grp">快思考</span>${fast.map(chip).join("")}<span class="grp">大脑</span>${brain.map(chip).join("")}`;
  $$("#chips button").forEach((b) => b.addEventListener("click", () => { $("#q").value = b.dataset.q; ask(b.dataset.q); }));
  $("#dialogs").innerHTML = `<span class="grp">多轮脚本</span>` + S.samples.dialogs.map((d, i) => `<button type="button" data-i="${i}">${esc(d.name)}</button>`).join("");
  $$("#dialogs button").forEach((b) => b.addEventListener("click", () => playDialog(S.samples.dialogs[+b.dataset.i])));
  const src = $("#batchSource");
  src.innerHTML = `<option value="builtin">内置样例（${items.length} 句）</option>` +
    (S.samples.eval_set.length ? `<option value="eval">评测集 dev 抽样（${S.samples.eval_set.length} 句）</option>` : "") +
    `<option value="custom">自定义粘贴</option>`;
  $("#batchNote").textContent = S.samples.eval_set.length ? "" : "提示：在仓库根目录用 build_dataset.py 生成 eval_set.jsonl 后，这里会多出“评测集 dev 抽样”。";
}

// ---------------------------------------------------------------- 启动
async function init() {
  const theme = store.get("cockpit-demo-theme", null);
  if (theme) document.documentElement.dataset.theme = theme;
  $("#themeBtn").addEventListener("click", () => {
    const cur = document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = cur === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    store.set("cockpit-demo-theme", next);
  });
  $$(".tabs button").forEach((b) => b.addEventListener("click", () => {
    $$(".tabs button").forEach((x) => x.classList.toggle("on", x === b));
    $$(".tab").forEach((t) => t.classList.toggle("on", t.id === "tab-" + b.dataset.tab));
    if (b.dataset.tab === "latency") renderLatency();
  }));

  S.config = await (await fetch("/api/config")).json();
  S.samples = await (await fetch("/api/samples")).json();
  const saved = store.get("cockpit-demo-settings", null);
  S.settings = { ...structuredClone(S.config.defaults), ...(saved || {}) };
  S.settings.corrections = { ...S.config.defaults.corrections, ...(saved?.corrections || {}) };
  renderInfo();
  renderSettings();
  renderChips();

  $("#askForm").addEventListener("submit", (e) => { e.preventDefault(); ask($("#q").value); });
  $("#resetBtn").addEventListener("click", resetSession);
  $$("input[name=mode]").forEach((r) => r.addEventListener("change", () => { S.settings.brain_mode = r.value; saveSettings(); }));
  $("#simReject").addEventListener("change", (e) => { S.settings.simulate_fast_reject = e.target.checked; saveSettings(); });
  $("#maxIters").addEventListener("change", (e) => { S.settings.max_iters = +e.target.value || 6; saveSettings(); });
  $("#restoreBtn").addEventListener("click", () => { S.settings = structuredClone(S.config.defaults); saveSettings(); renderSettings(); });
  $("#batchSource").addEventListener("change", (e) => $("#batchCustom").classList.toggle("hidden", e.target.value !== "custom"));
  $("#batchBtn").addEventListener("click", runBatch);
  $("#onlyWrong").addEventListener("change", renderBatch);
  await resetSession();
}
init();
