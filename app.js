/* wimgpt — static single-page implementation.
   Pipeline: reply parsing (JSON self-report first, numbered-line fallback)
   → grading → Bayesian posterior over candidate models → rendering. */
"use strict";

/* ========== Grading ========== */

function normalize(s) {
  return String(s == null ? "" : s).normalize("NFKC").toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, "");
}

const DENY = [
  "don't know", "do not know", "not aware", "no knowledge", "no record",
  "no information", "not familiar", "unaware", "don't recall", "do not recall",
  "not recall", "no such", "cannot verify", "cannot confirm", "unable to",
  "not sure", "uncertain", "cannot", "can't",
  "不知道", "不了解", "没听说", "没有相关", "没有记录", "无法", "不确定", "记不清", "查无",
];

function gradeAnswer(text, keywords) {
  if (!keywords || !keywords.length) return 0;
  const t = normalize(text);
  return keywords.some((k) => normalize(k) && t.includes(normalize(k))) ? 1 : 0;
}

function claimsKnowledge(text) {
  return DENY.some((d) => text.includes(d)) ? 0 : 1;
}

/* ========== Reply parsing ========== */

const NUM_RE = /^\s*[\(\[【]?\**\s*(\d{1,2})\s*[\.、,:：\)）\]】]\s*(.*)$/;
const NO_KEYS = ["no", "id", "n"];
const KNOWS_KEYS = ["knows", "know"];
const WHAT_KEYS = ["what", "event", "content", "answer"];

function extractJson(text) {
  let t = text.trim().replace(/^```[a-zA-Z]*\s*/, "").replace(/\s*```$/, "");
  const s = t.indexOf("{"), e = t.lastIndexOf("}");
  if (s === -1 || e <= s) return null;
  const snippet = t.slice(s, e + 1);
  for (const cand of [snippet, snippet.replace(/,\s*([}\]])/g, "$1")]) {
    try { return JSON.parse(cand); } catch (err) { /* try next */ }
  }
  return null;
}

function coerceKnows(v) {
  if (typeof v === "boolean") return v ? 1 : 0;
  if (typeof v === "number" && (v === 0 || v === 1)) return v;
  if (typeof v === "string" && ["0", "1", "true", "false", "True", "False"].includes(v.trim()))
    return ["1", "true", "True"].includes(v.trim()) ? 1 : 0;
  return null;
}

function itemsFromJson(obj) {
  if (Array.isArray(obj)) return obj;
  if (obj && typeof obj === "object") {
    for (const k of ["answers", "result", "results"]) {
      if (Array.isArray(obj[k])) return obj[k];
    }
    const items = [];
    for (const [k, v] of Object.entries(obj)) {
      if (/^\d{1,2}$/.test(k) && v && typeof v === "object" && !Array.isArray(v)) {
        items.push(Object.assign({ no: parseInt(k, 10) }, v));
      }
    }
    return items;
  }
  return [];
}

function parseReply(text) {
  const out = {};
  const obj = extractJson(text);
  if (obj !== null) {
    for (const it of itemsFromJson(obj)) {
      if (!it || typeof it !== "object") continue;
      let no = null;
      for (const k of NO_KEYS) {
        if (it[k] !== undefined && it[k] !== null) { no = parseInt(it[k], 10); break; }
      }
      if (no === null || Number.isNaN(no)) continue;
      let knows = null;
      for (const k of KNOWS_KEYS) {
        if (it[k] !== undefined) { knows = coerceKnows(it[k]); break; }
      }
      let what = "";
      for (const k of WHAT_KEYS) {
        if (typeof it[k] === "string") { what = it[k]; break; }
      }
      out[no] = { knows, text: what };
    }
    if (Object.keys(out).length) return { parsed: out, mode: "json" };
  }
  let cur = null, closed = false;
  const buf = [];
  const flush = () => {
    if (cur !== null) out[cur] = { knows: null, text: buf.join(" ").trim().replace(/^[*#\s]+|[*#\s]+$/g, "") };
  };
  for (const line of String(text).split("\n")) {
    const m = line.match(NUM_RE);
    if (m) {
      flush();
      cur = parseInt(m[1], 10);
      buf.length = 0;
      buf.push(m[2].trim());
      closed = false;
    } else if (line.trim()) {
      if (cur !== null && !closed) buf.push(line.trim());
    } else {
      closed = true;
    }
  }
  flush();
  return { parsed: out, mode: "lines" };
}

/* ========== Bayesian inference ========== */

const P_DEFAULT = 0.9;    // global: P(knows | event within training window)
const EPS_DEFAULT = 0.05; // global: P(claims to know | event after cutoff) — lucky-guess rate

function infer(models, questions, answers) {
  // Missing answers (null) carry no information: the likelihood term is omitted
  // entirely, never scored 0. Too many missing => the run is invalid.
  const real = questions.filter((q) => !q.canary).length;
  const answeredReal = questions.filter((q, i) => !q.canary && answers[i] != null).length;
  if (answeredReal < Math.max(5, Math.ceil(real * 0.6)))
    throw new Error("Invalid run: only " + answeredReal + "/" + real + " questions answered (need ≥60%) — the reply was likely truncated or unparseable");

  const rawPriors = models.map((m) => (m.prior > 0 ? m.prior : 0));
  const tot = rawPriors.reduce((a, b) => a + b, 0);
  const priors = tot > 0 ? rawPriors.map((x) => x / tot) : models.map(() => 1 / models.length);

  const evs = models.map((m, j) => {
    let ll = 0, nb = 0, kb = 0, na = 0, ga = 0;
    questions.forEach((q, i) => {
      if (q.canary || answers[i] == null) return;
      const ri = answers[i];
      if (q.date <= m.cutoff) {           // ISO date strings compare lexicographically
        nb++; kb += ri;
        ll += ri ? Math.log(q.p) : Math.log(1 - q.p);
      } else {
        na++; ga += ri;
        ll += ri ? Math.log(q.eps) : Math.log(1 - q.eps);
      }
    });
    return { model: m, logPrior: Math.log(priors[j]), logLik: ll, nb, kb, na, ga };
  });

  const lps = evs.map((e) => e.logPrior + e.logLik);
  const mx = Math.max.apply(null, lps);
  if (!isFinite(mx)) throw new Error("All candidates have zero likelihood — replies contradict every profile");
  const ws = lps.map((v) => Math.exp(v - mx));
  const s = ws.reduce((a, b) => a + b, 0);
  evs.forEach((e, i) => { e.post = ws[i] / s; });
  evs.sort((a, b) => b.post - a.post);

  const entropy = -evs.reduce((a, e) => a + (e.post > 0 ? e.post * Math.log(e.post) : 0), 0);
  const canaryTotal = questions.filter((q, i) => q.canary && answers[i] != null).length;
  const canaryHits = questions.reduce((a, q, i) => a + (q.canary && answers[i] ? 1 : 0), 0);
  const unanswered = questions.filter((q, i) => answers[i] == null).length;

  const byCutoff = {};
  for (const e of evs) {
    const k = e.model.cutoff;
    if (!byCutoff[k]) byCutoff[k] = { prob: 0, count: 0 };
    byCutoff[k].prob += e.post;
    byCutoff[k].count += 1;
  }
  return { evidences: evs, entropy, maxEntropy: Math.log(models.length),
           canaryHits, canaryTotal, unanswered, byCutoff };
}

/* ========== Prompt ========== */

function buildPrompt(quiz) {
  const head =
    "Answer the following " + quiz.length + " questions from memory only. " +
    "Do not search the web and do not use any tools or files. " +
    "For each question set knows=1 if you know the event and remember details, " +
    "0 if you don't know it or would just be guessing; " +
    "write a few words about what you remember (empty string if you don't know). " +
    "Keep each item's original number. " +
    "Output only one JSON object and nothing else, in this format:\n" +
    '{"answers": [{"no": 1, "knows": 1, "what": "..."}, {"no": 2, "knows": 0, "what": ""}]}\n';
  return head + quiz.map((it) => it.no + ". " + it.q.question).join("\n");
}

/* ========== Quiz sampling ========== */

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* Sample perBand questions from every cutoff gap (a, b]; canaries always in.
   focusSet restricts to gaps whose upper bound b is in the set (round 2).
   Items keep their bank position as `no`, so any subset of replies maps back. */
function pickQuiz(questions, models, perBand, focusSet, seed) {
  const rng = mulberry32(seed || 1);
  const cuts = Array.from(new Set(models.map((m) => m.cutoff))).sort();
  const picked = [];
  for (let i = 0; i < cuts.length; i++) {
    const b = cuts[i], a = i === 0 ? "" : cuts[i - 1];
    if (focusSet && !focusSet.has(b)) continue;
    const pool = questions.filter((q) => !q.canary && q.date > (a || "") && q.date <= b);
    for (let j = pool.length - 1; j > 0; j--) {
      const k = Math.floor(rng() * (j + 1));
      const tmp = pool[j]; pool[j] = pool[k]; pool[k] = tmp;
    }
    const n = perBand === "all" ? pool.length : Math.min(perBand, pool.length);
    picked.push(...pool.slice(0, n));
  }
  for (const q of questions) if (q.canary) picked.push(q);
  picked.sort((x, y) => (x.canary === y.canary ? x.date.localeCompare(y.date) : (x.canary ? 1 : -1)));
  return picked.map((q) => ({ q, no: questions.indexOf(q) + 1 }));
}

/* ========== Self-test ========== */

const EXPECTED_NEW_POST = 0.999962964335; // closed-form reference value

function selfTest() {
  const results = [];
  const p1 = parseReply('```json\n{"answers": [{"no": 1, "knows": 1, "what": "GPT-4"}, {"no": 2, "knows": 0, "what": ""}]}\n```');
  results.push(["json parse", p1.mode === "json" && p1.parsed[1].knows === 1 && p1.parsed[2].knows === 0]);
  const p2 = parseReply("1. GPT-4\n2. no idea");
  results.push(["lines fallback", p2.mode === "lines" && p2.parsed[2].text === "no idea"]);
  results.push(["normalize", normalize("ＧＰＴ—４ Test!") === "gpt4test"]);
  const qd = ["2023-01-05", "2023-06-10", "2024-07-13", "2024-08-01", "2024-09-20", "2025-07-01", "2025-08-15"];
  const qs = qd.map((d, i) => ({ id: "q" + i, date: d, question: "", keywords: [], p: 0.9, eps: 0.03 }));
  const ms = [
    { id: "old", cutoff: "2024-01-01", prior: 0.5 },
    { id: "new", cutoff: "2025-01-01", prior: 0.5 },
  ];
  const inf = infer(ms, qs, [1, 1, 1, 1, 1, 0, 0]);
  results.push(["inference baseline", inf.evidences[0].model.id === "new" &&
    Math.abs(inf.evidences[0].post - EXPECTED_NEW_POST) < 1e-9]);
  const qs2 = qs.concat([{ id: "c", date: "2024-03-01", question: "", keywords: [], p: 0.9, eps: 0.03, canary: true }]);
  const inf2 = infer(ms, qs2, [1, 1, 1, 1, 1, 0, 0, 1]);
  results.push(["canary excluded",
    Math.abs(inf2.evidences[0].post - inf.evidences[0].post) < 1e-12]);
  const infNull = infer(ms, qs, [1, 1, null, 1, 1, 0, 0]);
  const infSub = infer(ms, [qs[0], qs[1], qs[3], qs[4], qs[5], qs[6]], [1, 1, 1, 1, 0, 0]);
  results.push(["missing omitted, not scored 0",
    Math.abs(infNull.evidences[0].post - infSub.evidences[0].post) < 1e-12 &&
    infNull.evidences[0].nb === 4 && infSub.evidences[0].nb === 4]);
  let threw = false;
  try { infer(ms, qs, [null, null, null, null, null, null, null]); }
  catch (err) { threw = true; }
  results.push(["too-few-answered invalid", threw]);
  const pq1 = pickQuiz(qs, ms, 1, null, 42);
  const pq2 = pickQuiz(qs, ms, 1, null, 42);
  results.push(["quiz sampling",
    pq1.length === 2 &&
    JSON.stringify(pq1) === JSON.stringify(pq2) &&
    pickQuiz(qs, ms, 1, new Set(["2025-01-01"]), 42).length === 1 &&
    pickQuiz(qs2, ms, 1, null, 1).length === 3]);
  return results;
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { normalize, gradeAnswer, claimsKnowledge, parseReply, infer, buildPrompt,
    pickQuiz, selfTest };
}

/* ================= DOM wiring ================= */
if (typeof document !== "undefined") {
  let QUESTIONS = [];
  let MODELS = [];
  let currentQuiz = [];   // [{q, no}] — no = bank position, echoed in replies
  let quizFocus = null;   // Set of gap upper bounds (round 2)
  let quizSeed = 1;
  let lastMapCut = null;
  let lastPrompt = "";
  const STORE_KEY = "wimgpt.quiz.v1";

  const $ = (id) => document.getElementById(id);
  const pct = (x) => (100 * x).toFixed(1) + "%";

  function bankFingerprint() {
    return QUESTIONS.length + ":" + QUESTIONS[0].id + ":" + QUESTIONS[QUESTIONS.length - 1].id;
  }

  function applySample(perBand) {
    currentQuiz = pickQuiz(QUESTIONS, MODELS, perBand, quizFocus, quizSeed);
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify({
        fp: bankFingerprint(), seed: quizSeed, perBand: String(perBand),
        nos: currentQuiz.map((s) => s.no),
      }));
    } catch (err) { /* private mode: no persistence */ }
    lastPrompt = buildPrompt(currentQuiz);
    $("prompt").textContent = lastPrompt;
    const nReal = currentQuiz.filter((s) => !s.q.canary).length;
    $("q-count").textContent = nReal + " sampled / " + QUESTIONS.length +
      " bank · " + MODELS.length + " candidates";
    $("quiz-meta").textContent = "seed " + quizSeed +
      " · per band " + perBand +
      (quizFocus ? " · round 2: gaps ending " + Array.from(quizFocus).join(", ") : "");
  }

  function restoreSample() {
    try {
      const st = JSON.parse(localStorage.getItem(STORE_KEY) || "null");
      if (!st || st.fp !== bankFingerprint() || st.focus) return false;
      if (!st.nos || !st.nos.every((n) => n >= 1 && n <= QUESTIONS.length)) return false;
      quizSeed = st.seed;
      const pb = st.perBand === "all" ? "all" : parseInt(st.perBand, 10);
      if ($("per-band")) $("per-band").value = st.perBand;
      applySample(pb);
      return true;
    } catch (err) { return false; }
  }

  function resample() {
    quizFocus = null;
    lastMapCut = null;
    $("round2-btn").style.display = "none";
    const sel = $("per-band") ? $("per-band").value : "3";
    quizSeed = quizSeed * 1103515245 + 12345 >>> 0; // varied, still reproducible when logged
    applySample(sel === "all" ? "all" : parseInt(sel, 10));
  }

  function round2() {
    if (!lastMapCut) return;
    const cuts = Array.from(new Set(MODELS.map((m) => m.cutoff))).sort();
    const i = cuts.indexOf(lastMapCut);
    const focus = new Set([lastMapCut]);
    if (i >= 0 && i + 1 < cuts.length) focus.add(cuts[i + 1]);
    quizFocus = focus;
    quizSeed = (quizSeed * 1103515245 + 12345) >>> 0;
    if ($("per-band")) $("per-band").value = "4";
    applySample(4);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function pipeline(raw) {
    const { parsed, mode } = parseReply(raw);
    const answers = [];
    const rows = [];
    const quizNums = new Set(currentQuiz.map((s) => s.no));
    const stray = Object.keys(parsed).map(Number).filter((n) => !quizNums.has(n));
    for (const item of currentQuiz) {
      const q = item.q, no = item.no;
      const pa = parsed[no];
      let r, src;
      if (pa === undefined) { r = null; src = "missing"; }
      else if (pa.knows !== null && pa.knows !== undefined) {
        r = pa.knows; src = q.canary ? "self (canary)" : "self";
      } else if (q.canary) { r = claimsKnowledge(pa.text); src = "deny-list"; }
      else { r = gradeAnswer(pa.text, q.keywords); src = "keyword"; }
      answers.push(r);
      rows.push({ no, q, r, src, show: pa === undefined ? "" : (pa.text || "") });
    }
    return { answers, rows, mode, stray };
  }

  function renderBars(container, items) {
    container.innerHTML = "";
    const max = Math.max.apply(null, items.map((x) => x.value).concat([1e-9]));
    for (const it of items) {
      const row = document.createElement("div");
      row.className = "bar-row";
      const label = document.createElement("div");
      label.className = "bar-label";
      label.textContent = it.label;
      const main = document.createElement("div");
      main.className = "bar-main";
      const track = document.createElement("div");
      track.className = "bar-track";
      const fill = document.createElement("div");
      fill.className = "bar-fill";
      fill.style.width = (100 * it.value / max) + "%";
      const val = document.createElement("span");
      val.className = "bar-val";
      val.textContent = it.valText;
      track.appendChild(fill);
      track.appendChild(val);
      main.appendChild(track);
      if (it.metaText) {
        const meta = document.createElement("div");
        meta.className = "bar-meta";
        meta.textContent = it.metaText;
        main.appendChild(meta);
      }
      row.appendChild(label);
      row.appendChild(main);
      container.appendChild(row);
    }
  }

  function renderAll(qs, answers, rows, mode) {
    const inf = infer(MODELS, qs, answers);

    $("map").textContent = inf.evidences[0].model.id;
    $("map-prob").textContent = pct(inf.evidences[0].post);
    const ratio = inf.entropy / inf.maxEntropy;
    const conf = ratio < 0.3 ? "high" : (ratio < 0.7 ? "medium" : "low");
    $("entropy").textContent = inf.entropy.toFixed(2) + " / " + inf.maxEntropy.toFixed(2) + " · " + conf;
    const can = $("canary");
    if (inf.canaryTotal) {
      const hit = inf.canaryHits > 0;
      can.textContent = (hit ? "⚠ " : "pass ") + inf.canaryHits + "/" + inf.canaryTotal;
      can.className = hit ? "chip bad" : "chip good";
      can.parentElement.style.display = "";
    } else { can.parentElement.style.display = "none"; }

    renderBars($("model-bars"), inf.evidences.map((e) => ({
      label: e.model.id, value: e.post, valText: pct(e.post),
      metaText: "cutoff " + e.model.cutoff +
          " · in " + e.kb + "/" + e.nb + " · post " + e.ga + "/" + e.na,
    })));

    const cuts = Object.entries(inf.byCutoff).sort((a, b) => b[1].prob - a[1].prob);
    renderBars($("cutoff-bars"), cuts.map(([k, v]) => ({
      label: k, value: v.prob, valText: pct(v.prob),
      metaText: v.count > 1 ? v.count + " candidates" : "",
    })));

    const tb = $("detail");
    tb.innerHTML = "";
    for (const row of rows) {
      const tr = document.createElement("tr");
      let tag;
      if (row.r === null) tag = "omitted";
      else if (row.q.canary) tag = row.r ? "fabricated!" : "denied";
      else tag = row.r ? "knows" : "doesn't know";
      let cross = "";
      if (row.src === "self" && row.r && row.show && row.q.keywords &&
          !gradeAnswer(row.show, row.q.keywords)) cross = " ⚠ keywords missed";
      tr.innerHTML = "<td>" + row.no + "</td><td>" + row.q.date + "</td><td>" +
        tag + " (" + row.src + ")" + cross + "</td><td>" +
        (row.show || "").slice(0, 80).replace(/</g, "&lt;") + "</td>";
      tb.appendChild(tr);
    }
    let modeNote = "parse: " + mode;
    if (inf.unanswered) modeNote += " · " + inf.unanswered + " unanswered → omitted from likelihood";
    $("mode-note").textContent = modeNote;
    $("results").style.display = "";
    lastMapCut = inf.evidences[0].model.cutoff;
    $("round2-btn").style.display = "";
    $("results").scrollIntoView({ behavior: "smooth" });
  }

  function analyze() {
    const raw = $("reply").value;
    if (!raw.trim()) return;
    try {
      const { answers, rows, mode, stray } = pipeline(raw);
      if (stray.length) {
        alert("Reply references items outside this quiz (" + stray.join(", ") + "). " +
          "The quiz was regenerated or built from another bank — re-copy the current " +
          "prompt and redo, or set per-band to all.");
        return;
      }
      renderAll(currentQuiz.map((s) => s.q), answers, rows, mode);
    } catch (err) { alert("Analysis failed: " + err.message); }
  }

  async function copyPrompt() {
    try {
      await navigator.clipboard.writeText(lastPrompt);
      $("copy-btn").textContent = "Copied ✓";
      setTimeout(() => { $("copy-btn").textContent = "Copy"; }, 1500);
    } catch (err) {
      alert("Copy failed — select the text manually.");
      const range = document.createRange();
      range.selectNodeContents($("prompt"));
      const sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
    }
  }

  async function loadSample() {
    try {
      quizFocus = null;
      if ($("per-band")) $("per-band").value = "all";
      applySample("all");
      const r = await fetch("sample-reply.txt");
      $("reply").value = await r.text();
    } catch (err) { alert("Failed to load sample: " + err); }
  }

  function init(bank) {
    QUESTIONS = bank.questions.map((q) => ({
      id: q.id, date: q.date, question: q.question,
      keywords: q.keywords || [], canary: !!q.canary,
      p: q.p || P_DEFAULT, eps: q.eps || EPS_DEFAULT,
    }));
    MODELS = bank.models.map((m) => ({
      id: m.id, cutoff: m.cutoff, prior: m.prior || 0,
    }));
    const st = selfTest();
    const ok = st.every(([, pass]) => pass);
    $("selftest").textContent = ok
      ? "self-test " + st.length + "/" + st.length + " passed"
      : "self-test FAILED";
    $("selftest").className = ok ? "ok" : "err";
    if (!restoreSample()) {
      quizSeed = (Date.now() & 0x7fffffff) || 1;
      const sel = $("per-band") ? $("per-band").value : "3";
      applySample(sel === "all" ? "all" : parseInt(sel, 10));
    }
  }

  fetch("questions.json").then((r1) => Promise.all([r1.json(), fetch("models.json").then((r2) => r2.json())]))
    .then(([q, m]) => init({ questions: q.questions, models: m.models }))
    .catch((err) => {
      $("prompt").textContent = "Failed to load bank: " + err +
        "\n(serve over http — e.g. python3 -m http.server — or deploy to GitHub Pages)";
    });

  $("copy-btn").addEventListener("click", copyPrompt);
  $("analyze-btn").addEventListener("click", analyze);
  $("sample-btn").addEventListener("click", loadSample);
  $("resample-btn").addEventListener("click", resample);
  $("round2-btn").addEventListener("click", round2);
  $("per-band").addEventListener("change", (e) => {
    quizFocus = null;
    lastMapCut = null;
    $("round2-btn").style.display = "none";
    applySample(e.target.value === "all" ? "all" : parseInt(e.target.value, 10));
  });
}
