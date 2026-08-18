/* Belief ROC Studio — frontend logic.
   Analysis functions mirror generic_RoC/generic_roc.py exactly. */
"use strict";

const $ = (id) => document.getElementById(id);

const state = {
  registry: null,
  datasetId: null,
  columns: [],
  preview: [],
  nRows: 0,
  contextCols: [],
  outcomeCol: null,
  positiveValues: [],
  jobId: null,
  pollTimer: null,
  results: null,      // [{index, label, belief, error, context_preview}]
  roc: null,          // {points, auc, P, N}
  bfu: null,
  decideSetup: null,
};

/* ================= Analysis (ports of generic_roc.py) ================= */

function rocCurve(beliefs, labels) {
  const P = labels.filter((y) => y === 1).length;
  const N = labels.length - P;
  if (P === 0 || N === 0) return null;
  const order = beliefs.map((b, i) => i).sort((a, b) => beliefs[b] - beliefs[a]);
  const points = [{ thr: Infinity, tpr: 0, fpr: 0, tp: 0, fp: 0 }];
  let tp = 0, fp = 0;
  for (let k = 0; k < order.length; k++) {
    const i = order[k];
    if (labels[i] === 1) tp++; else fp++;
    const thr = beliefs[i];
    // Merge ties: one ROC point per distinct belief value.
    if (k + 1 < order.length && beliefs[order[k + 1]] === thr) continue;
    points.push({ thr, tpr: tp / P, fpr: fp / N, tp, fp });
  }
  let auc = 0;
  for (let k = 1; k < points.length; k++) {
    auc += (points[k].fpr - points[k - 1].fpr) * (points[k].tpr + points[k - 1].tpr) / 2;
  }
  return { points, auc, P, N };
}

function bestFixedUtility(beliefs, labels, costRatio) {
  const n = labels.length;
  const P = Math.max(labels.filter((y) => y === 1).length, 1);
  const N = Math.max(n - labels.filter((y) => y === 1).length, 1);
  const candidates = Array.from(new Set([0, ...beliefs, 1 + 1e-9])).sort((a, b) => a - b);
  let best = null, bestCost = Infinity;
  for (const thr of candidates) {
    let tp = 0, fp = 0, tn = 0, fn = 0;
    for (let i = 0; i < n; i++) {
      const act = beliefs[i] >= thr;
      if (act && labels[i] === 1) tp++;
      else if (act) fp++;
      else if (labels[i] === 1) fn++;
      else tn++;
    }
    const cost = fp + costRatio * fn;
    if (cost < bestCost - 1e-12) {   // strict: keeps the smallest threshold among ties
      bestCost = cost;
      best = {
        costRatio, threshold: thr,
        utilityRatio: thr > 0 ? (1 - thr) / thr : Infinity,
        tpr: tp / P, fpr: fp / N, accuracy: (tp + tn) / n,
        totalCost: cost, tp, fp, tn, fn,
      };
    }
  }
  return best;
}

function fmtRatio(r) {
  if (!isFinite(r)) return "act on every case";
  const sig = (x) => (x >= 100 ? Math.round(x) : x >= 10 ? x.toFixed(1) : x.toFixed(2))
    .toString().replace(/\.0+$/, "");
  return r >= 1 ? `${sig(r)} : 1` : `1 : ${sig(1 / r)}`;
}
const pct = (x) => `${(100 * x).toFixed(0)}%`;

/* ========================= Small UI helpers ========================= */

function notice(el, kind, text) {
  el.hidden = false;
  el.className = `notice ${kind}`;
  el.textContent = text;
}
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const c of children) {
    node.append(c);
  }
  return node;
}

/* ============================== Tabs ============================== */

const pages = { setup: $("page-setup"), decide: $("page-decide"), help: $("page-help") };
function showTab(name) {
  for (const [key, page] of Object.entries(pages)) page.hidden = key !== name;
  for (const [key, tab] of Object.entries({ setup: $("tab-setup"), decide: $("tab-decide"), help: $("tab-help") })) {
    tab.classList.toggle("active", key === name);
    tab.setAttribute("aria-selected", String(key === name));
  }
  if (name === "decide") renderDecideTab();
}
$("tab-setup").addEventListener("click", () => showTab("setup"));
$("tab-decide").addEventListener("click", () => showTab("decide"));
$("tab-help").addEventListener("click", () => showTab("help"));

/* ======================= Step 1 — dataset ======================= */

const dropzone = $("dropzone");
dropzone.addEventListener("click", () => $("file-input").click());
dropzone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") $("file-input").click(); });
dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("drag"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("drag");
  if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]);
});
$("file-input").addEventListener("change", (e) => { if (e.target.files[0]) loadFile(e.target.files[0]); });

async function loadFile(file) {
  const status = $("data-status");
  notice(status, "busy", `Reading “${file.name}”…`);
  try {
    const text = await file.text();
    await loadCsvText(text, file.name);
  } catch (err) {
    notice(status, "err", `Could not read the file: ${err.message}`);
  }
}

async function loadCsvText(text, name) {
  const status = $("data-status");
  const resp = await api("/api/dataset", { csv_text: text });
  state.datasetId = resp.dataset_id;
  state.columns = resp.columns;
  state.preview = resp.preview;
  state.nRows = resp.n_rows;
  state.contextCols = [];
  state.outcomeCol = null;
  state.positiveValues = [];
  notice(status, "ok", `Loaded ${resp.n_rows} patients from “${name}”. Now tell the tool what each column means.`);
  renderColumnConfig();
  updateRunReadiness();
}

function renderColumnConfig() {
  $("column-config").hidden = false;

  const table = $("preview-table");
  table.replaceChildren(
    el("tr", {}, ...state.columns.map((c) => el("th", {}, c))),
    ...state.preview.map((row) =>
      el("tr", {}, ...state.columns.map((c) => el("td", { title: row[c] || "" }, row[c] || "")))),
  );

  const ctxList = $("context-cols");
  ctxList.replaceChildren(...state.columns.map((c) => {
    const cb = el("input", { type: "checkbox", "data-col": c });
    cb.addEventListener("change", () => {
      state.contextCols = state.contextCols.filter((x) => x !== c);
      if (cb.checked) state.contextCols.push(c);
      updatePromptPreview();
      updateRunReadiness();
    });
    return el("label", {}, cb, c);
  }));

  const sel = $("outcome-col");
  sel.replaceChildren(
    el("option", { value: "" }, "— choose a column —"),
    ...state.columns.map((c) => el("option", { value: c }, c)),
  );
  sel.value = "";
  sel.onchange = () => {
    state.outcomeCol = sel.value || null;
    state.positiveValues = [];
    if (state.outcomeCol) loadOutcomeValues();
    else $("positive-picker").hidden = true;
    updateRunReadiness();
  };
  $("positive-picker").hidden = true;
}

async function loadOutcomeValues() {
  const resp = await api("/api/dataset/values", { dataset_id: state.datasetId, column: state.outcomeCol });
  const box = $("positive-values");
  const suggested = new Set(resp.suggested_positive || []);
  box.replaceChildren(...resp.values.map(({ value, count }) => {
    const cb = el("input", { type: "checkbox" });
    cb.checked = suggested.has(value);
    if (cb.checked) state.positiveValues = [...state.positiveValues, value];
    cb.addEventListener("change", () => {
      state.positiveValues = state.positiveValues.filter((v) => v !== value);
      if (cb.checked) state.positiveValues.push(value);
      updateRunReadiness();
    });
    return el("label", {}, cb, value === "" ? "(empty)" : value,
      el("span", { class: "count" }, `${count} patients`));
  }));
  $("positive-picker").hidden = false;
  if (resp.n_distinct > 10) {
    box.prepend(el("div", { class: "hint" },
      "This column has many different values — are you sure it is the outcome column?"));
  }
  updateRunReadiness();
}

/* ---- Built-in example (also used by the ?selftest=1 smoke test) ---- */

async function loadExample() {
  const status = $("data-status");
  try {
    const text = await fetch("/api/sample_dataset").then((r) => {
      if (!r.ok) throw new Error("sample not found");
      return r.text();
    });
    await loadCsvText(text, "built-in example");
    const cb = document.querySelector('#context-cols input[data-col="patient_description"]');
    if (cb) { cb.checked = true; state.contextCols = ["patient_description"]; }
    $("outcome-col").value = "needed_emergency_care";
    state.outcomeCol = "needed_emergency_care";
    state.positiveValues = [];
    await loadOutcomeValues();
    $("decision-question").value = "Should this patient be sent to emergency care now?";
    $("belief-question").value = "Based only on the information above, what is the probability that this patient genuinely needs emergency-level care now?";
    $("provider-select").value = "demo";
    renderModels();
    updatePromptPreview();
    updateCostMeanings();
    updateRunReadiness();
    notice(status, "ok",
      "Example loaded: 48 made-up patients, practice mode selected. Scroll down and press “Analyze all patients”.");
  } catch (err) {
    notice(status, "err", `Could not load the example: ${err.message}`);
  }
}
$("load-example").addEventListener("click", loadExample);

/* ======================= Step 2 — questions ======================= */

const FORMAT_INSTRUCTION =
  "Answer in exactly this format, with no other text:\n\nPROBABILITY: <a single number between 0 and 1>";

function currentContext(row) {
  if (state.contextCols.length === 1) return row[state.contextCols[0]] || "";
  return state.contextCols.map((c) => (row[c] ? `${c}: ${row[c]}` : "")).filter(Boolean).join("\n");
}

function updatePromptPreview() {
  const q = $("belief-question").value.trim() || "(your probability question)";
  const ctx = state.preview.length && state.contextCols.length
    ? currentContext(state.preview[0])
    : "(the patient information from your file)";
  $("prompt-preview").textContent = `${ctx}\n\n${q}\n\n${FORMAT_INSTRUCTION}`;
}
$("belief-question").addEventListener("input", () => { updatePromptPreview(); persistPrefs(); updateRunReadiness(); });
$("decision-question").addEventListener("input", () => { persistPrefs(); updateRunReadiness(); updateCostMeanings(); });

function updateCostMeanings() {
  const dq = $("decision-question").value.trim();
  $("fn-meaning").textContent = dq
    ? `deciding “no” when the true answer was “yes”`
    : "a false negative";
  $("fp-meaning").textContent = dq
    ? `deciding “yes” when the true answer was “no”`
    : "a false positive";
}

/* ======================= Step 3 — AI connection ======================= */

async function loadRegistry() {
  const resp = await fetch("/api/providers").then((r) => r.json());
  state.registry = resp.providers;
  const sel = $("provider-select");
  sel.replaceChildren(...Object.entries(state.registry).map(([id, p]) =>
    el("option", { value: id }, p.label)));
  sel.onchange = () => { renderModels(); persistPrefs(); };
  renderModels();
  restorePrefs();
}

function currentProvider() { return $("provider-select").value; }
function currentModelSpec() {
  const p = state.registry[currentProvider()];
  return p.models.find((m) => m.id === $("model-select").value) || p.models[0];
}

function renderModels() {
  const p = state.registry[currentProvider()];
  const sel = $("model-select");
  sel.replaceChildren(...p.models.map((m) =>
    el("option", { value: m.id }, m.note ? `${m.label} — ${m.note}` : m.label)));
  sel.onchange = () => { renderEfforts(); persistPrefs(); };
  $("key-hint").textContent = p.key_hint;
  const isDemo = currentProvider() === "demo";
  $("api-key").disabled = isDemo;
  $("api-key").placeholder = isDemo ? "Not needed in practice mode" : "Paste your API key";
  renderEfforts();
}

function renderEfforts() {
  const m = currentModelSpec();
  const sel = $("effort-select");
  sel.replaceChildren(...m.efforts.map((e) => el("option", { value: e.id }, e.label)));
  sel.value = m.default_effort;
  sel.onchange = () => persistPrefs();
}

$("toggle-key").addEventListener("click", () => {
  const inp = $("api-key");
  inp.type = inp.type === "password" ? "text" : "password";
  $("toggle-key").textContent = inp.type === "password" ? "Show" : "Hide";
});

$("test-connection").addEventListener("click", async () => {
  const status = $("connection-status");
  notice(status, "busy", "Contacting the AI service…");
  try {
    await api("/api/test_connection", {
      provider: currentProvider(), model: $("model-select").value,
      effort: $("effort-select").value, api_key: $("api-key").value,
    });
    notice(status, "ok", "Connected! The AI answered a test question successfully.");
  } catch (err) {
    notice(status, "err", err.message);
  }
});

/* ======================= Step 4 — run ======================= */

function setupComplete() {
  return state.datasetId && state.contextCols.length && state.outcomeCol &&
    state.positiveValues.length && $("belief-question").value.trim() &&
    $("decision-question").value.trim();
}

function updateRunReadiness() {
  const ready = Boolean(setupComplete());
  $("run-all").disabled = !ready;
  $("run-test").disabled = !ready;
  $("run-summary").textContent = ready
    ? `Ready: ${state.nRows} patients · the AI will be asked one probability question per patient.`
    : "Complete steps 1–3 first (file, columns, both questions).";
}

async function startRun(limit) {
  const status = $("run-status");
  status.hidden = true;
  if (currentProvider() !== "demo" && !$("api-key").value.trim()) {
    notice(status, "err", "Paste your API key in step 3 first (or switch to practice mode).");
    return;
  }
  try {
    const resp = await api("/api/run", {
      dataset_id: state.datasetId,
      context_cols: state.contextCols,
      outcome_col: state.outcomeCol,
      positive_values: state.positiveValues,
      belief_question: $("belief-question").value.trim(),
      decision_question: $("decision-question").value.trim(),
      provider: currentProvider(), model: $("model-select").value,
      effort: $("effort-select").value, api_key: $("api-key").value,
      parallel: 4, limit: limit || null,
    });
    state.jobId = resp.job_id;
    $("run-cancel").hidden = false;
    $("run-all").disabled = true;
    $("run-test").disabled = true;
    $("progress-wrap").hidden = false;
    poll();
  } catch (err) {
    notice(status, "err", err.message);
  }
}
$("run-all").addEventListener("click", () => startRun(null));
$("run-test").addEventListener("click", () => startRun(10));
$("run-cancel").addEventListener("click", async () => {
  if (state.jobId) await api(`/api/job/${state.jobId}/cancel`, {});
});

function poll() {
  clearTimeout(state.pollTimer);
  state.pollTimer = setTimeout(async () => {
    try {
      const job = await fetch(`/api/job/${state.jobId}`).then((r) => r.json());
      const fillPct = job.total ? (100 * job.done / job.total) : 0;
      $("progress-fill").style.width = `${fillPct}%`;
      $("progress-text").textContent =
        `${job.done} of ${job.total} patients asked` +
        (job.failed ? ` · ${job.failed} failed` : "") +
        (job.status === "running" ? "…" : "");
      if (job.status === "running") { poll(); return; }
      $("run-cancel").hidden = true;
      updateRunReadiness();
      finishRun(job);
    } catch {
      poll();
    }
  }, 900);
}

function finishRun(job) {
  const status = $("run-status");
  state.results = job.results || [];
  const usable = state.results.filter((r) => r.belief !== null);
  if (job.status === "cancelled") notice(status, "err", "Stopped. Partial results are shown below if enough cases finished.");
  else if (job.failed) notice(status, "busy", `Done, but ${job.failed} of ${job.total} cases failed — their errors are listed in the results table.`);
  else notice(status, "ok", "Done! Scroll down for the results.");
  if (usable.length >= 4) renderResults(usable);
  else notice(status, "err", "Too few cases produced a usable probability to build a curve (need at least 4).");
}

/* ======================= Step 5 — results ======================= */

function renderResults(usable) {
  const beliefs = usable.map((r) => r.belief);
  const labels = usable.map((r) => r.label);
  const roc = rocCurve(beliefs, labels);
  const status = $("run-status");
  if (!roc) {
    notice(status, "err",
      "The outcome column has only one class in the analyzed rows (all yes or all no) — a curve needs both. Check the value you ticked in step 1.");
    return;
  }
  state.roc = roc;
  state.beliefs = beliefs;
  state.labels = labels;
  $("step-results").hidden = false;

  // Stat tiles
  const failures = state.results.length - usable.length;
  $("stat-row").replaceChildren(
    statTile(roc.auc.toFixed(2), "Ranking quality (AUROC)", aucVerdict(roc.auc)),
    statTile(String(usable.length), "Patients analyzed",
      `${roc.P} with the outcome · ${roc.N} without`),
    statTile(String(failures), "Cases skipped", failures ? "see the table below" : "none — all cases answered"),
  );

  recomputeBfu();
  renderResultsTable();
  $("download-beliefs").href = `/api/job/${state.jobId}/beliefs.csv`;
  $("step-results").scrollIntoView({ behavior: "smooth", block: "start" });
  if (state.selftestMode === "decide") {
    showTab("decide");
    $("decide-context").value =
      "A 60-year-old man. Symptoms: crushing chest pain radiating to the left arm for 40 minutes, sweating.";
    $("decide-run").click();
  }
}

function statTile(val, lbl, sub) {
  return el("div", { class: "stat-tile" },
    el("div", { class: "val" }, val),
    el("div", { class: "lbl" }, lbl),
    el("div", { class: "sub" }, sub));
}
function aucVerdict(auc) {
  if (auc >= 0.9) return "excellent separation of the two groups";
  if (auc >= 0.8) return "good separation of the two groups";
  if (auc >= 0.7) return "fair separation — interpret with care";
  return "weak separation — this AI may not be reliable here";
}

function targetRatio() {
  const fn = Math.max(parseFloat($("fn-cost").value) || 1, 0.001);
  const fp = Math.max(parseFloat($("fp-cost").value) || 1, 0.001);
  return fn / fp;
}

function recomputeBfu() {
  if (!state.roc) return;
  const r = targetRatio();
  state.bfu = bestFixedUtility(state.beliefs, state.labels, r);
  drawRocChart();
  renderBfuReadout();
  renderExplainBox();
  updateChips();
  renderDecideTab();
}
$("fn-cost").addEventListener("input", recomputeBfu);
$("fp-cost").addEventListener("input", recomputeBfu);
document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    $("fn-cost").value = chip.dataset.fn;
    $("fp-cost").value = chip.dataset.fp;
    recomputeBfu();
  });
});
function updateChips() {
  const r = targetRatio();
  document.querySelectorAll(".chip").forEach((chip) => {
    chip.classList.toggle("active",
      Math.abs((chip.dataset.fn / chip.dataset.fp) - r) < 1e-9);
  });
}

function renderBfuReadout() {
  const b = state.bfu;
  const box = $("bfu-readout");
  const per100 = (x) => Math.round(100 * x);
  box.replaceChildren(
    el("div", { class: "bfu-headline" },
      el("div", { class: "lbl hint tight" }, "Best rule for your priorities:"),
      el("div", { class: "big" },
        b.threshold <= 0 ? "Act on every case" :
          `Act when the AI's probability ≥ ${b.threshold.toFixed(2)}`),
      el("div", { class: "hint tight" },
        isFinite(b.utilityRatio)
          ? `Equivalent prompt weighting — missed case : unnecessary action = ${fmtRatio(b.utilityRatio)}`
          : "Equivalent prompt weighting: always act"),
    ),
    el("div", { class: "bfu-stats" },
      el("span", {}, "True cases caught"),
      el("span", { class: "v" }, `${b.tp} of ${b.tp + b.fn} (${pct(b.tpr)})`),
      el("span", {}, "Unnecessary actions"),
      el("span", { class: "v" }, `${b.fp} of ${b.fp + b.tn} (${pct(b.fpr)})`),
      el("span", {}, "Missed cases"),
      el("span", { class: "v" }, `${b.fn}`),
      el("span", {}, "Overall accuracy"),
      el("span", { class: "v" }, pct(b.accuracy)),
      el("span", {}, "Out of 100 future cases like these,"),
      el("span", { class: "v" },
        `≈ ${per100((b.tp + b.fp) / state.labels.length)} would be acted on`),
    ),
  );
}

function renderExplainBox() {
  const b = state.bfu;
  const fnCost = parseFloat($("fn-cost").value) || 1;
  const fpCost = parseFloat($("fp-cost").value) || 1;
  const r = targetRatio();
  const dq = $("decision-question").value.trim() || "your decision question";
  const ratioText = r >= 1
    ? `one missed case is ${fmtRatio(r)} times as costly as one unnecessary action`
    : `one unnecessary action is ${fmtRatio(1 / r)} times as costly as one missed case`;
  const calNote = isFinite(b.utilityRatio) && Math.abs(Math.log(b.utilityRatio / r)) > Math.log(1.5)
    ? ` Note: this differs from your stated ${fmtRatio(r)} priority — that is expected, and it means the AI's raw probabilities are systematically too high or too low (mis-calibrated). The cut-off found here corrects for that automatically, which is exactly why it is chosen from your own data rather than assumed.`
    : "";
  $("explain-box").replaceChildren(
    el("h3", {}, "What this means, in plain language"),
    el("p", {},
      `You told the tool that ${ratioText} (${fnCost} vs ${fpCost}). ` +
      `Scanning every possible cut-off on the curve, the one that best serves that priority on your ${state.labels.length} reviewed cases is: `,
      el("strong", {}, b.threshold <= 0 ? "always act." :
        `answer “yes” to “${dq}” whenever the AI's probability is at least ${b.threshold.toFixed(2)}.`)),
    el("p", {},
      isFinite(b.utilityRatio)
        ? `If you prefer to instruct the AI with costs instead of a cut-off, tell it a missed case costs ${fmtRatio(b.utilityRatio)} as much as an unnecessary action — that weighting implies the same cut-off.${calNote}`
        : `The best rule on this data is to act on every case.${calNote}`),
  );
}

/* ------------------------- ROC chart (SVG) ------------------------- */

const SVGNS = "http://www.w3.org/2000/svg";
function svgEl(tag, attrs = {}) {
  const node = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  return node;
}

const CHART = { w: 560, h: 520, left: 62, right: 18, top: 16, bottom: 64 };
const X = (fpr) => CHART.left + fpr * (CHART.w - CHART.left - CHART.right);
const Y = (tpr) => CHART.h - CHART.bottom - tpr * (CHART.h - CHART.top - CHART.bottom);

function drawRocChart() {
  const wrap = $("roc-chart");
  wrap.replaceChildren();
  const svg = svgEl("svg", { viewBox: `0 0 ${CHART.w} ${CHART.h}`, role: "img",
    "aria-label": "Curve showing the trade-off between catching true cases and unnecessary actions" });

  // Gridlines + ticks every 20%
  for (let i = 0; i <= 5; i++) {
    const v = i / 5;
    svg.append(
      svgEl("line", { x1: X(v), y1: Y(0), x2: X(v), y2: Y(1), stroke: "var(--grid)", "stroke-width": 1 }),
      svgEl("line", { x1: X(0), y1: Y(v), x2: X(1), y2: Y(v), stroke: "var(--grid)", "stroke-width": 1 }),
    );
    const xt = svgEl("text", { x: X(v), y: Y(0) + 20, "text-anchor": "middle",
      fill: "var(--muted)", "font-size": 12 });
    xt.textContent = pct(v);
    const yt = svgEl("text", { x: X(0) - 10, y: Y(v) + 4, "text-anchor": "end",
      fill: "var(--muted)", "font-size": 12 });
    yt.textContent = pct(v);
    svg.append(xt, yt);
  }
  // Axis titles (plain language)
  const xTitle = svgEl("text", { x: (X(0) + X(1)) / 2, y: CHART.h - 26, "text-anchor": "middle",
    fill: "var(--text-secondary)", "font-size": 13.5 });
  xTitle.textContent = "Patients WITHOUT the outcome acted on unnecessarily →";
  const xTitle2 = svgEl("text", { x: (X(0) + X(1)) / 2, y: CHART.h - 10, "text-anchor": "middle",
    fill: "var(--muted)", "font-size": 12 });
  xTitle2.textContent = "(false-positive rate)";
  const yTitle = svgEl("text", { x: 16, y: (Y(0) + Y(1)) / 2, "text-anchor": "middle",
    fill: "var(--text-secondary)", "font-size": 13.5,
    transform: `rotate(-90 16 ${(Y(0) + Y(1)) / 2})` });
  yTitle.textContent = "True cases caught → (sensitivity)";
  svg.append(xTitle, xTitle2, yTitle);

  // Chance diagonal
  svg.append(svgEl("line", { x1: X(0), y1: Y(0), x2: X(1), y2: Y(1),
    stroke: "var(--muted)", "stroke-width": 1.5, "stroke-dasharray": "2 5" }));

  // ROC curve
  const pts = state.roc.points;
  const d = pts.map((p, i) => `${i ? "L" : "M"}${X(p.fpr).toFixed(1)},${Y(p.tpr).toFixed(1)}`).join("");
  svg.append(svgEl("path", { d, fill: "none", stroke: "var(--series-1)",
    "stroke-width": 2.5, "stroke-linejoin": "round" }));

  // Best-fixed-utility operating point (2px surface ring via stroke)
  const b = state.bfu;
  svg.append(svgEl("rect", {
    x: X(b.fpr) - 6.5, y: Y(b.tpr) - 6.5, width: 13, height: 13, rx: 2.5,
    fill: "var(--series-2)", stroke: "var(--surface-1)", "stroke-width": 2,
  }));

  // Hover layer: highlight dot + transparent capture rect
  const hoverDot = svgEl("circle", { r: 5.5, fill: "var(--series-1)",
    stroke: "var(--surface-1)", "stroke-width": 2, visibility: "hidden" });
  svg.append(hoverDot);
  const capture = svgEl("rect", { x: 0, y: 0, width: CHART.w, height: CHART.h, fill: "transparent" });
  svg.append(capture);

  const tooltip = el("div", { class: "roc-tooltip" });
  tooltip.style.display = "none";
  wrap.append(svg, tooltip);

  const toLocal = (evt) => {
    const r = svg.getBoundingClientRect();
    return { x: (evt.clientX - r.left) * CHART.w / r.width,
             y: (evt.clientY - r.top) * CHART.h / r.height, rect: r };
  };
  capture.addEventListener("pointermove", (evt) => {
    const { x, y, rect } = toLocal(evt);
    let best = null, bestD = Infinity;
    for (const p of pts) {
      const dx = X(p.fpr) - x, dy = Y(p.tpr) - y;
      const dist = dx * dx + dy * dy;
      if (dist < bestD) { bestD = dist; best = p; }
    }
    if (!best) return;
    hoverDot.setAttribute("cx", X(best.fpr));
    hoverDot.setAttribute("cy", Y(best.tpr));
    hoverDot.setAttribute("visibility", "visible");
    renderTooltip(tooltip, best);
    const px = X(best.fpr) * rect.width / CHART.w;
    const py = Y(best.tpr) * rect.height / CHART.h;
    tooltip.style.display = "block";
    tooltip.style.left = `${Math.min(px + 14, rect.width - 200)}px`;
    tooltip.style.top = `${Math.max(py - 10, 0)}px`;
  });
  capture.addEventListener("pointerleave", () => {
    hoverDot.setAttribute("visibility", "hidden");
    tooltip.style.display = "none";
  });

  $("roc-legend").replaceChildren(
    el("span", { class: "key" }, el("span", { class: "stroke" }), "AI probabilities (every possible cut-off)"),
    el("span", { class: "key" }, el("span", { class: "sq" }), "Best cut-off for your priorities"),
    el("span", { class: "key" }, el("span", { class: "stroke dashed" }), "Guessing at random"),
  );
}

function renderTooltip(tooltip, p) {
  const { P, N } = state.roc;
  tooltip.replaceChildren(
    el("div", { class: "big" },
      p.thr === Infinity ? "Cut-off: act on no one" : `Cut-off: probability ≥ ${p.thr.toFixed(2)}`),
    el("div", {}, el("span", { class: "lbl" }, "True cases caught: "),
      el("strong", {}, `${p.tp} of ${P} (${pct(p.tpr)})`)),
    el("div", {}, el("span", { class: "lbl" }, "Unnecessary actions: "),
      el("strong", {}, `${p.fp} of ${N} (${pct(p.fpr)})`)),
  );
}

/* ------------------------- results table ------------------------- */

function renderResultsTable() {
  const table = $("results-table");
  table.replaceChildren(
    el("tr", {},
      el("th", { class: "num" }, "Row"), el("th", {}, "Patient (start of description)"),
      el("th", { class: "num" }, "AI probability"), el("th", {}, "True outcome"), el("th", {}, "Problem")),
    ...state.results.map((r) => el("tr", {},
      el("td", { class: "num" }, String(r.index + 1)),
      el("td", { title: r.context_preview }, r.context_preview),
      el("td", { class: "num" }, r.belief === null ? "—" : r.belief.toFixed(2)),
      el("td", {}, r.label === 1 ? "yes" : "no"),
      el("td", {}, r.error || ""))),
  );
}

/* ------------------------- downloads ------------------------- */

$("download-summary").addEventListener("click", () => {
  const b = state.bfu;
  const summary = {
    tool: "Belief ROC Studio",
    generated_from: "holdout dataset analyzed in the browser session",
    provider: currentProvider(), model: $("model-select").value, effort: $("effort-select").value,
    decision_question: $("decision-question").value.trim(),
    belief_question: $("belief-question").value.trim(),
    n_cases: state.labels.length,
    n_positive: state.roc.P, n_negative: state.roc.N,
    auroc: state.roc.auc,
    target_cost_ratio_fn_fp: targetRatio(),
    best_fixed_utility: {
      threshold: b.threshold, utility_ratio_fn_fp: b.utilityRatio,
      tpr: b.tpr, fpr: b.fpr, accuracy: b.accuracy,
      tp: b.tp, fp: b.fp, tn: b.tn, fn: b.fn,
    },
  };
  const blob = new Blob([JSON.stringify(summary, null, 2)], { type: "application/json" });
  const a = el("a", { href: URL.createObjectURL(blob), download: "roc_summary.json" });
  a.click();
  URL.revokeObjectURL(a.href);
});

/* ======================= Decide tab ======================= */

$("goto-decide").addEventListener("click", () => showTab("decide"));

function deployPrompt() {
  const b = state.bfu;
  const dq = $("decision-question").value.trim();
  const thr = b.threshold.toFixed(2);
  return `[PASTE THE PATIENT INFORMATION HERE]

${dq}

Consider the following probability threshold when you decide:
- First, judge the probability that the answer is truly yes.
- Decide YES if that probability is at least ${thr}
  - the threshold at or above which action is warranted.
- Decide NO if that probability is below ${thr}
  - the threshold below which action is not warranted.
You have decision analysis training: Please apply this probability threshold considering the probabilities involved.

Answer in exactly this format, with no other text:

DECISION: <YES or NO>`;
}

function renderDecideTab() {
  const ready = state.bfu && setupComplete();
  $("decide-ready").hidden = !ready;
  const noticeBox = $("decide-setup-notice");
  if (!ready) {
    notice(noticeBox, "busy",
      "First run an analysis in tab 1 — the cut-off found there is what this tab applies to new cases.");
    return;
  }
  noticeBox.hidden = true;
  const b = state.bfu;
  const m = currentModelSpec();
  $("decide-summary").replaceChildren(
    el("div", {}, el("strong", {}, "Question: "), $("decision-question").value.trim()),
    el("div", {}, el("strong", {}, "AI: "),
      `${state.registry[currentProvider()].label} · ${m.label} · reasoning ${$("effort-select").value}`),
    el("div", {}, el("strong", {}, "Rule: "),
      b.threshold <= 0 ? "act on every case" :
        `act when the AI's probability ≥ ${b.threshold.toFixed(2)} (chosen from your ${state.labels.length}-case review, ` +
        `priorities ${fmtRatio(targetRatio())})`),
  );
  $("deploy-prompt").textContent = deployPrompt();
}

$("copy-deploy").addEventListener("click", async () => {
  await navigator.clipboard.writeText($("deploy-prompt").textContent);
  $("copy-deploy").textContent = "Copied ✓";
  setTimeout(() => { $("copy-deploy").textContent = "Copy the prompt"; }, 1600);
});

$("decide-run").addEventListener("click", async () => {
  const status = $("decide-status");
  const context = $("decide-context").value.trim();
  if (!context) { notice(status, "err", "Describe the patient first."); return; }
  notice(status, "busy", "Asking the AI for its probability…");
  $("decide-result").hidden = true;
  try {
    const resp = await api("/api/decide", {
      provider: currentProvider(), model: $("model-select").value,
      effort: $("effort-select").value, api_key: $("api-key").value,
      belief_question: $("belief-question").value.trim(),
      context, threshold: state.bfu.threshold,
    });
    status.hidden = true;
    const yes = resp.decision;
    const dq = $("decision-question").value.trim();
    const box = $("decide-result");
    box.hidden = false;
    box.replaceChildren(el("div", { class: `decide-verdict ${yes ? "yes" : "no"}` },
      el("div", { class: "head" }, yes ? "YES — act on this case" : "NO — action not indicated"),
      el("p", {},
        `The AI put the probability at ${resp.belief.toFixed(2)}. Your rule acts at ` +
        `${state.bfu.threshold.toFixed(2)} or higher, so for “${dq}” the answer is ${yes ? "YES" : "NO"}.`),
      el("p", { class: "hint" },
        "This is decision support, not a clinical order — confirm with your own judgment and protocols."),
    ));
  } catch (err) {
    notice(status, "err", err.message);
  }
});

/* ======================= plumbing ======================= */

async function api(path, body) {
  const resp = await fetch(path, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(data.error || `Request failed (${resp.status}).`);
  return data;
}

const PREF_KEY = "belief-roc-studio-prefs";
function persistPrefs() {
  try {
    localStorage.setItem(PREF_KEY, JSON.stringify({
      decision_question: $("decision-question").value,
      belief_question: $("belief-question").value,
      provider: currentProvider(), model: $("model-select").value,
      effort: $("effort-select").value,
    }));
  } catch { /* private browsing */ }
}
function restorePrefs() {
  try {
    const p = JSON.parse(localStorage.getItem(PREF_KEY) || "{}");
    if (p.decision_question) $("decision-question").value = p.decision_question;
    if (p.belief_question) $("belief-question").value = p.belief_question;
    if (p.provider && state.registry[p.provider]) {
      $("provider-select").value = p.provider;
      renderModels();
      if (p.model && state.registry[p.provider].models.some((m) => m.id === p.model)) {
        $("model-select").value = p.model;
        renderEfforts();
      }
      if (p.effort && [...$("effort-select").options].some((o) => o.value === p.effort)) {
        $("effort-select").value = p.effort;
      }
    }
  } catch { /* ignore */ }
  updatePromptPreview();
  updateCostMeanings();
  updateRunReadiness();
}

loadRegistry().then(() => {
  const mode = new URLSearchParams(location.search).get("selftest");
  if (mode) {
    state.selftestMode = mode;
    loadExample().then(() => startRun(null));
  }
});
