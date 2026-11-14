import { GraphView, typeColor, sweepChart } from "./viz.js";
import { ManifoldCanvas } from "./manifold.js";
import { pieceSummary, samePhrase, surfaceType, toMusicXML } from "./core.js";
import { Player } from "./audio.js";

const $ = (s) => document.querySelector(s);
const load = (p) => fetch(p).then((r) => { if (!r.ok) throw new Error(`${p}: ${r.status}`); return r.json(); });
const pct = (v) => `${(100 * v).toFixed(1)}%`;

const [piece, gallery, sweep] = await Promise.all([load("data/piece.json"), load("data/gallery.json"), load("data/sweep.json")]);
const medoids = piece.medoid_phrases.map((i) => piece.phrases[i]);
const lambdaLabel = (m) => (m.lambda === 0 ? "λ = 0 (구조 손실 없음)" : `λ = ${m.lambda}`);

// ------------------------------------------------------------------ section 1: structure
const gStruct = new GraphView($("#graph-structure"), piece);
const mStruct = new ManifoldCanvas($("#M-structure"), piece);
mStruct.setPiece(piece.phrases, piece.labels);
mStruct.draw(0, { complete: true });
$("#layout-fidelity").textContent = mStruct.fidelity.toFixed(2);
gStruct.onHover((c) => { gStruct.highlightType(c); mStruct.focus = c; mStruct.draw(0, { complete: true });
$("#layout-fidelity").textContent = mStruct.fidelity.toFixed(2); });

// ------------------------------------------------------------------ explorer
const gView = new GraphView($("#graph-explore"), piece);
const mView = new ManifoldCanvas($("#M-explore"), piece);
gView.onHover((c) => { gView.highlightType(c); mView.focus = c; mView.draw(pos, drawOpts()); });

const player = new Player();
let current = null; // {title, meta, phrases, walk, rows, kind}
let pos = 0, playing = false, raf = 0, shownMeasure = -2;
const drawOpts = () => ({ ghost: $("#ghost").checked });
const starts = () => current.phrases.reduce((acc, ev) => [...acc, acc[acc.length - 1] + ev.reduce((x, [, d]) => x + d, 0)], [0]);
const measureAt = (b) => { const st = starts(); let m = 0; while (m + 1 < st.length - 1 && st[m + 1] <= b + 1e-9) m++; return m; };
const sameAs = (p) => piece.phrases.map((o, i) => (samePhrase(o, p) ? i + 1 : 0)).filter(Boolean);

function show(item) {
  stop();
  current = item;
  pos = 0;
  $("#viewer-title").textContent = item.title;
  $("#viewer-meta").textContent = item.meta;
  mView.setPiece(item.phrases, item.walk);
  const surface = item.phrases.map((p) => surfaceType(p, medoids));
  const copy = item.phrases.map((p) => item.kind !== "original" && sameAs(p).length > 0);
  drawWalk(item.walk, surface, copy);
  $("#metrics").innerHTML = item.rows.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("");
  shownMeasure = -2;
  update();
  renderScore();
}

function drawWalk(walk, surface, copy) {
  const row = $("#rowPhr"), chap = $("#rowChap");
  row.replaceChildren(); chap.replaceChildren();
  const firsts = new Set(piece.chapters.map((c) => c.measures[0] - 1));
  current.phrases.forEach((ev, m) => {
    const d = document.createElement("div");
    d.className = "cell" + (copy[m] ? " copy" : "") + (surface[m] !== walk[m] ? " off" : "") + (m && firsts.has(m) ? " ch-start" : "");
    d.style.flex = `${ev.reduce((a, [, x]) => a + x, 0)} 0 0`;
    d.style.setProperty("--tc", typeColor(walk[m]));
    d.textContent = piece.type_names[walk[m]];
    d.title = `${m + 1}마디 · 조건 유형 ${piece.type_names[walk[m]]} · 표면 유형 ${piece.type_names[surface[m]]}${copy[m] ? " · 원곡 마디와 같음" : ""}`;
    d.dataset.m = m;
    row.appendChild(d);
  });
  const st = starts();
  piece.chapters.forEach((c) => {
    const s = document.createElement("span");
    s.style.flex = `${st[c.measures[1]] - st[c.measures[0] - 1]} 0 0`;
    s.textContent = `제${c.chapter}장`;
    chap.appendChild(s);
  });
}
$("#strip").addEventListener("click", (e) => {
  const cell = e.target.closest(".cell");
  if (cell) return play(Number(cell.dataset.m));
  const r = $("#rowPhr").getBoundingClientRect();
  play(measureAt(((e.clientX - r.left) / r.width) * mView.total));
});
$("#strip").addEventListener("keydown", (e) => {
  const m = Math.max(0, measureAt(pos));
  if (e.key === "ArrowRight") play(Math.min(current.phrases.length - 1, m + 1));
  else if (e.key === "ArrowLeft") play(Math.max(0, m - 1));
  else return;
  e.preventDefault();
});

// the line under the figure: which lobe, which visit, a repeat or a variation
function nowText(m) {
  if (m < 0) return `재생하면 곡선 <span class="m">γ</span>가 엽 <span class="m">M<sub>${piece.type_names[current.walk[0]]}</sub></span>에서 출발합니다.`;
  const c = current.walk[m], ph = current.phrases[m];
  let visit = 0, repeat = false;
  for (let i = 0; i <= m; i++) if (current.walk[i] === c) { visit++; if (i < m && samePhrase(current.phrases[i], ph)) repeat = true; }
  const ret = piece.returns.find(([, t]) => t === m);
  let how = visit === 1 ? "처음 그리는 길" : repeat ? "<b>전에 지난 길을 그대로 다시 지나는 중</b>" : "<b>전에 지난 길 옆을 지나는 중</b> (변주)";
  if (ret) how += ` · 환입: ${ret[0] + 1}마디와 같은 자리`;
  const orig = current.kind !== "original" ? sameAs(ph) : [];
  if (orig.length) how += ` · 원곡 ${orig.slice(0, 3).join(", ")}마디와 같은 마디`;
  return `지금: ${m + 1}마디 · 엽 <span class="m">M<sub>${piece.type_names[c]}</sub></span> · 이 엽에 <b>${visit}번째</b> 들어옴 · ${how}`;
}

function update() {
  const playingNow = playing && pos > 0;
  const m = playing || pos > 0 ? measureAt(pos) : -1;
  if (m !== shownMeasure) {
    shownMeasure = m;
    $("#strip").classList.toggle("idle", m < 0);
    document.querySelectorAll("#rowPhr .cell").forEach((c) => {
      const k = Number(c.dataset.m);
      c.classList.toggle("done", k <= m);
      c.classList.toggle("current", k === m);
    });
    gView.setWalk(current.walk, m > 0 ? `${current.walk[m - 1]},${current.walk[m]}` : null, m >= 0 ? current.walk[m] : null);
    const box = $("#score");
    (box.measures ?? []).forEach((g, i) => g.classList.toggle("current", i === m));
    const g = box.measures?.[m];
    if (g && playing) {
      const r = g.getBoundingClientRect(), b = box.getBoundingClientRect();
      if (r.top < b.top || r.bottom > b.bottom) box.scrollTop += r.top - b.top - 40;
    }
    $("#now").innerHTML = nowText(m) + '<br><span class="cn" id="cn"></span>';
  }
  if (m >= 0) $("#cn").innerHTML = mView.readout(pos, piece.yulmyung);
  const spb = 60 / tempo(), sec = (b) => { const t = b * spb; return `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`; };
  $("#clock").textContent = `${sec(pos)} / ${sec(mView.total)}`;
  const row = $("#rowPhr");
  $("#head").style.left = `${row.offsetLeft + (row.offsetWidth * (m < 0 ? 0 : pos)) / mView.total - 1}px`;
  $("#strip").setAttribute("aria-valuenow", String(Math.round((100 * (m < 0 ? 0 : pos)) / mView.total)));
  mView.draw(m < 0 ? 0 : pos, drawOpts());
}

// ------------------------------------------------------------------ sources
const originalItem = {
  kind: "original", title: "원곡 〈양청도드리〉 피리", meta: "천년만세 중 · 7장 72마디 · 1–4마디 = 25–28마디(환입) · 유형 9개 · 전이의 55%가 목을 지남",
  phrases: piece.phrases, walk: piece.labels,
  rows: [["마디", "72"], ["유형", "9 (A–I)"], ["서로 다른 마디", String(new Set(piece.phrases.map((p) => JSON.stringify(p))).size)],
    ["그래프의 전이 종류", String(piece.transition_counts.flat().filter((n) => n > 0).length)], ["환입", "1–4 ↔ 25–28마디"]],
};

const models = gallery.models;
for (const sel of [$("#gallery-model"), $("#gen-model")]) {
  for (const m of models) sel.add(new Option(lambdaLabel(m), m.arm));
  sel.value = models.find((m) => m.lambda === 0.1)?.arm ?? models[0].arm;
}
function fillGalleryPieces() {
  const sel = $("#gallery-piece");
  sel.replaceChildren(...gallery.pieces[$("#gallery-model").value].map((p, i) => new Option(`${i + 1}번 (생성 Seed ${p.seed})`, i)));
}
function galleryItem() {
  const arm = $("#gallery-model").value, model = models.find((m) => m.arm === arm);
  const p = gallery.pieces[arm][Number($("#gallery-piece").value)];
  const mt = p.metrics;
  return {
    kind: "gallery", title: `논문 생성곡 · ${lambdaLabel(model)}`, meta: `생성 Seed ${p.seed} · 원곡과 다른 유형 자리 ${pct(mt.walk_positions_differing)} · 원곡 마디와 같은 마디 ${pct(mt.original_phrase_rate)}`,
    phrases: p.phrases, walk: p.walk,
    rows: [["재생성 전 유형 소속", pct(mt.first_pass_membership)], ["유형 소속(최종)", pct(mt.membership)],
      ["원곡 마디와 같은 마디", pct(mt.original_phrase_rate)], ["표면 전이가 G에 있음", pct(mt.surface_edge_validity)],
      ["환입 일치", mt.returns_exact ? "예" : "아니요"], ["원곡과 다른 유형 자리", pct(mt.walk_positions_differing)],
      ["새 4음 연쇄", pct(mt.note_4gram_novelty)]],
  };
}
$("#gallery-model").addEventListener("change", () => { const k = $("#gallery-piece").value; fillGalleryPieces(); $("#gallery-piece").value = k; show(galleryItem()); });
$("#gallery-piece").addEventListener("change", () => show(galleryItem()));
fillGalleryPieces();

document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".tabs button").forEach((x) => x.setAttribute("aria-selected", String(x === b)));
  const src = b.dataset.source;
  document.querySelectorAll(".controls").forEach((c) => { c.hidden = c.dataset.for !== src; });
  if (src === "original") show(originalItem);
  if (src === "gallery") show(galleryItem());
  if (src === "generate" && lastGenerated) show(lastGenerated);
}));

// ------------------------------------------------------------------ generation in a worker
const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
worker.postMessage({ type: "init", piece });
let jobId = 0, lastGenerated = null;
$("#gen-dice").addEventListener("click", () => { $("#gen-seed").value = 1 + Math.floor(Math.random() * 999998); });
$("#gen-run").addEventListener("click", () => {
  const arm = $("#gen-model").value, model = models.find((m) => m.arm === arm);
  const seed = Number($("#gen-seed").value) || 1;
  const options = { steps: Number($("#gen-steps").value), temperature: Number($("#gen-temp").value), attempts: Number($("#gen-attempts").value) };
  const id = ++jobId;
  $("#gen-run").disabled = true;
  const prog = $(".progress");
  prog.hidden = false;
  prog.style.setProperty("--p", "0%");
  prog.querySelector(".ptext").textContent = "모형을 불러오는 중…";
  worker.onmessage = ({ data }) => {
    if (data.id !== id) return;
    if (data.type === "progress") {
      prog.style.setProperty("--p", `${(100 * data.step) / data.steps}%`);
      prog.querySelector(".ptext").textContent = data.round ? `재생성 ${data.round}회차 · ${data.step}/${data.steps}` : `확산 ${data.step}/${data.steps} 단계`;
    } else if (data.type === "done") {
      const r = data.result;
      prog.querySelector(".ptext").textContent = "다양체 위의 곡선 계산 중…";
      setTimeout(() => {
        const s = pieceSummary(piece, r.phrases, r.walk);
        lastGenerated = {
          kind: "generated", title: `새로 생성 · ${lambdaLabel(model)}`, meta: `Seed ${seed} · 원곡과 다른 유형 자리 ${pct(s.walk_positions_differing)} · 원곡 마디와 같은 마디 ${pct(s.original_phrase_rate)} · ${(data.ms / 1000).toFixed(1)}초`,
          phrases: r.phrases, walk: r.walk,
          rows: [["재생성 전 유형 소속", pct(r.firstPass)], ["유형 소속(최종)", pct(s.membership)], ["재생성 횟수", String(r.rounds)],
            ["원곡 마디와 같은 마디", pct(s.original_phrase_rate)], ["표면 전이가 G에 있음", pct(s.surface_edge_validity)],
            ["환입 일치", s.returns_exact ? "예" : "아니요"], ["원곡과 다른 유형 자리", pct(s.walk_positions_differing)],
            ["걸린 시간", `${(data.ms / 1000).toFixed(1)}초`]],
        };
        show(lastGenerated);
        prog.hidden = true;
        $("#gen-run").disabled = false;
      }, 20);
    } else if (data.type === "error") {
      prog.querySelector(".ptext").textContent = `오류: ${data.message}`;
      $("#gen-run").disabled = false;
    }
  };
  worker.postMessage({ type: "generate", id, seed, options, modelUrl: new URL(model.file, location.href).href });
});

// ------------------------------------------------------------------ score (Verovio)
let toolkit = null;
const verovioReady = new Promise((resolve) => {
  const wait = () => {
    const v = window.verovio;
    if (!v?.module) return setTimeout(wait, 100);
    if (v.module.calledRun) resolve(new v.toolkit());
    else v.module.onRuntimeInitialized = () => resolve(new v.toolkit());
  };
  wait();
});
async function renderScore() {
  if (!current) return;
  const box = $("#score");
  toolkit ??= await verovioReady;
  const width = Math.max(box.clientWidth, 320);
  const scale = width < 600 ? 30 : 38;
  toolkit.setOptions({ pageWidth: Math.round((width * 100) / scale), scale, adjustPageHeight: true, pageHeight: 60000,
    breaks: "auto", header: "none", footer: "none", lyricSize: 4, spacingSystem: 10 });
  toolkit.loadData(toMusicXML(piece, current.phrases, current.walk, { yulmyung: $("#show-yul").checked, title: current.title }));
  const pages = toolkit.getPageCount();
  box.innerHTML = Array.from({ length: pages }, (_, i) => toolkit.renderToSVG(i + 1)).join("");
  box.measures = [...box.querySelectorAll("g.measure")];
  box.measures.forEach((g, m) => { g.style.cursor = "pointer"; g.addEventListener("click", () => play(m)); });
  shownMeasure = -2;
  update();
}
$("#show-yul").addEventListener("change", renderScore);
let resizeTimer;
new ResizeObserver(() => { clearTimeout(resizeTimer); resizeTimer = setTimeout(renderScore, 250); }).observe($("#score"));

// ------------------------------------------------------------------ playback
const tempo = () => Number($("#tempo").value);
$("#tempo").addEventListener("change", () => { if (playing) play(measureAt(pos)); else update(); });
$("#ghost").addEventListener("change", () => mView.draw(pos, drawOpts()));
function play(from = 0) {
  if (!current) return;
  player.play(current.phrases, tempo(), from);
  playing = true;
  pos = starts()[from];
  $("#play").textContent = "멈춤";
  cancelAnimationFrame(raf);
  const tick = () => {
    const b = player.beat();
    if (b === null) { stop(); pos = mView.total; update(); return; }
    pos = b;
    update();
    raf = requestAnimationFrame(tick);
  };
  raf = requestAnimationFrame(tick);
}
function stop() {
  player.stop();
  cancelAnimationFrame(raf);
  playing = false;
  $("#play").textContent = "재생";
}
$("#play").addEventListener("click", () => { if (playing) { stop(); update(); } else play(pos > 0 && pos < mView.total - 1e-6 ? measureAt(pos) : 0); });
$("#restart").addEventListener("click", () => { stop(); pos = 0; shownMeasure = -2; update(); });
$("#download").addEventListener("click", () => {
  if (!current) return;
  const xml = toMusicXML(piece, current.phrases, current.walk, { title: current.title });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([xml], { type: "application/vnd.recordare.musicxml+xml" }));
  a.download = `${current.title.replace(/[^\p{L}\p{N}.=]+/gu, "_")}.musicxml`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});

// ------------------------------------------------------------------ λ sweep
sweepChart($("#sweep"), sweep, [
  { key: "first_pass_membership", title: "유형을 따르는 정도", note: "재생성 전 마디가 조건 유형에 속하는 비율 ↑", scale: 100, fmt: (v) => `${v.toFixed(0)}%` },
  { key: "z_novel", title: "복제 검정 Z (새 마디)", note: "−2 아래면 학습 마디에 지나치게 가까움. 녹색 = 기준 안", scale: 1, band: [-2, 2], fmt: (v) => v.toFixed(1) },
  { key: "exact_copy_rate", title: "원곡 마디를 그대로 낸 비율", note: "학습 마디와 똑같은 생성 마디 ↓", scale: 100, fmt: (v) => `${Number.isInteger(+v.toFixed(6)) ? v.toFixed(0) : v.toFixed(1)}%` },
]);

show(originalItem);
