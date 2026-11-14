// Pure logic shared by the page, the sampling worker and the Node parity test.
// Each function mirrors a Python function in src/gugakgen (named in its comment).

// ---------------------------------------------------------------------------------------------
// Seeded random numbers (the browser stream differs from NumPy/PyTorch, so pieces generated here
// are new draws from the same model, not re-runs of the paper's pieces).

export function rng(seed) {
  let a = (seed >>> 0) || 1;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function choice(random, weights) {
  let total = 0;
  for (const w of weights) total += w;
  let u = random() * total;
  for (let i = 0; i < weights.length; i++) {
    u -= weights[i];
    if (u < 0) return i;
  }
  for (let i = weights.length - 1; i >= 0; i--) if (weights[i] > 0) return i;
  return weights.length - 1;
}

// ---------------------------------------------------------------------------------------------
// Edit distance (gugakgen.chart.edit_distance.align): events are [pitch|null, duration].

const REST_PITCH_COST = 6.0, INDEL_WEIGHT = 3.0, FRAGMENT_SURCHARGE = 0.5, MAX_FRAGMENTS = 3;

function pitchCost(a, b) {
  if (a === null && b === null) return 0.0;
  if (a === null || b === null) return REST_PITCH_COST;
  return Math.abs(a - b);
}
function substitution(x, y) {
  return pitchCost(x[0], y[0]) * (x[1] + y[1]) / 2.0 + Math.abs(x[1] - y[1]);
}
function fragmentation(one, many) {
  let total = 0, cost = 0;
  for (const [, d] of many) total += d;
  for (const [p, d] of many) cost += pitchCost(one[0], p) * d;
  return cost + Math.abs(one[1] - total) + FRAGMENT_SURCHARGE;
}

// Returns {value, steps}; steps are {kind, i0, i1, j0, j1} as in Python's Step (same tie order).
export function align(left, right) {
  const R = left.length, C = right.length;
  const table = Array.from({ length: R + 1 }, () => new Float64Array(C + 1).fill(Infinity));
  const back = Array.from({ length: R + 1 }, () => new Array(C + 1).fill(null));
  table[0][0] = 0;
  const relax = (i, j, kind, i0, j0, cost) => {
    const v = table[i0][j0] + cost;
    if (v < table[i][j] - 1e-12) { table[i][j] = v; back[i][j] = { kind, i0, i1: i, j0, j1: j }; }
  };
  for (let i = 0; i <= R; i++) {
    for (let j = 0; j <= C; j++) {
      if (i > 0 && j > 0) relax(i, j, "sub", i - 1, j - 1, substitution(left[i - 1], right[j - 1]));
      if (i > 0) relax(i, j, "del", i - 1, j, INDEL_WEIGHT * left[i - 1][1]);
      if (j > 0) relax(i, j, "ins", i, j - 1, INDEL_WEIGHT * right[j - 1][1]);
      for (let k = 2; k <= MAX_FRAGMENTS; k++) {
        if (i > 0 && j >= k) relax(i, j, "frag", i - 1, j - k, fragmentation(left[i - 1], right.slice(j - k, j)));
        if (j > 0 && i >= k) relax(i, j, "cons", i - k, j - 1, fragmentation(right[j - 1], left.slice(i - k, i)));
      }
    }
  }
  const steps = [];
  for (let i = R, j = C; i || j;) { const st = back[i][j]; steps.push(st); i = st.i0; j = st.j0; }
  steps.reverse();
  let sumLeft = 0, sumRight = 0; // summed separately, as in Python, so ties break identically
  for (const [, d] of left) sumLeft += d;
  for (const [, d] of right) sumRight += d;
  return { value: table[R][C] / (sumLeft + sumRight), steps };
}

export function editDistance(left, right) {
  return align(left, right).value;
}

// gugakgen.chart.coords.to_chart: each event's medoid slot, whether it was inserted, and its
// deviation eta from the medoid note in steps of the piece's scale (null for a rest).
export function toChart(phrase, medoid, scale) {
  const { steps } = align(phrase, medoid);
  const pitched = medoid.map(([p], i) => [p, i]).filter(([p]) => p !== null).map(([, i]) => i);
  const refs = medoid.map((_, j) => scale.indexOf(medoid[pitched.reduce((b, i) => (Math.abs(i - j) < Math.abs(b - j) ? i : b), pitched[0])][0]));
  const slots = [];
  let last = 0;
  for (const st of steps) {
    if (st.kind === "ins") continue;
    const inserted = st.kind === "del";
    if (!inserted) last = st.j0;
    for (let n = st.i0; n < st.i1; n++) slots.push([last, inserted]);
  }
  return phrase.map(([p], i) => ({ slot: slots[i][0], inserted: slots[i][1], eta: p === null ? null : scale.indexOf(p) - refs[slots[i][0]] }));
}

export function samePhrase(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i][0] !== b[i][0] || Math.abs(a[i][1] - b[i][1]) > 1e-9) return false;
  return true;
}

// Typology.is_member: no farther from its own medoid than from any other (eq. 9).
export function medoidDistances(phrase, medoids) {
  return medoids.map((m) => editDistance(phrase, m));
}
export function isMember(phrase, type, medoids) {
  const d = medoidDistances(phrase, medoids);
  return d[type] <= Math.min(...d) + 1e-9;
}
export function surfaceType(phrase, medoids) {
  const d = medoidDistances(phrase, medoids);
  let best = 0;
  for (let i = 1; i < d.length; i++) if (d[i] < d[best]) best = i;
  return best;
}

// ---------------------------------------------------------------------------------------------
// Manifold projection (scripts/export_web.py project)

export function project(phrase, originals, coords, k = 8, iterations = 100) {
  const d = originals.map((o) => editDistance(phrase, o));
  let nearest = 0;
  for (let i = 1; i < d.length; i++) if (d[i] < d[nearest]) nearest = i;
  if (d[nearest] < 1e-9) return coords[nearest].slice();
  // the k nearest distinct positions (repeated measures share one)
  const idx = [];
  for (const [, i] of d.map((v, n) => [v, n]).sort((a, b) => a[0] - b[0] || a[1] - b[1])) {
    if (idx.every((j) => Math.max(Math.abs(coords[i][0] - coords[j][0]), Math.abs(coords[i][1] - coords[j][1])) > 1e-9)) idx.push(i);
    if (idx.length === k) break;
  }
  const w = idx.map((i) => 1 / (d[i] * d[i]));
  const W = w.reduce((a, b) => a + b, 0);
  let y = [0, 0];
  idx.forEach((i, n) => { y[0] += w[n] * coords[i][0]; y[1] += w[n] * coords[i][1]; });
  y = [y[0] / W, y[1] / W];
  for (let it = 0; it < iterations; it++) {
    const next = [0, 0];
    idx.forEach((i, n) => {
      const dx = y[0] - coords[i][0], dy = y[1] - coords[i][1];
      const norm = Math.max(Math.hypot(dx, dy), 1e-12);
      next[0] += w[n] * (coords[i][0] + d[i] * dx / norm);
      next[1] += w[n] * (coords[i][1] + d[i] * dy / norm);
    });
    y = [next[0] / W, next[1] / W];
  }
  return y;
}

// ---------------------------------------------------------------------------------------------
// Type walks on the phrase graph (gugakgen.chart.walk.TypeWalk.sample)

export class TypeWalk {
  constructor(spec) {
    Object.assign(this, spec); // P, length, start, end, block, blocks, block_weights
    const K = this.P.length;
    const eye = Array.from({ length: K }, (_, i) => Array.from({ length: K }, (_, j) => (i === j ? 1 : 0)));
    this.powers = [eye];
    for (let n = 0; n < this.length; n++) this.powers.push(matmul(this.powers[n], this.P));
  }
  bridge(random, left, right, a, b) {
    const out = [];
    let current = a;
    for (let pos = left + 1; pos < right; pos++) {
      const weights = this.P[current].map((p, c) => p * this.powers[right - pos][c][b]);
      current = choice(random, weights);
      out.push(current);
    }
    return out;
  }
  sample(random) {
    const walk = new Array(this.length).fill(-1);
    const fixed = new Map([[0, this.start], [this.length - 1, this.end]]);
    if (this.block) {
      const [source, target] = this.block;
      const block = this.blocks[choice(random, this.block_weights)];
      block.forEach((v, i) => { fixed.set(source + i, v); fixed.set(target + i, v); });
    }
    const anchors = [...fixed.keys()].sort((a, b) => a - b);
    for (const p of anchors) walk[p] = fixed.get(p);
    for (let n = 0; n + 1 < anchors.length; n++) {
      const l = anchors[n], r = anchors[n + 1];
      this.bridge(random, l, r, fixed.get(l), fixed.get(r)).forEach((v, i) => { walk[l + 1 + i] = v; });
    }
    return walk;
  }
}

function matmul(A, B) {
  return A.map((row) => B[0].map((_, j) => row.reduce((s, a, k) => s + a * B[k][j], 0)));
}

// ---------------------------------------------------------------------------------------------
// Grid tokens (gugakgen.chart.grid.Grid)

export function decode(grid, tokens) {
  const events = [];
  for (const t of tokens) {
    if (t === grid.hold) {
      if (!events.length) throw new Error("A phrase cannot start with hold");
      events[events.length - 1][1] += 1;
    } else if (t === grid.on_rest) events.push([null, 1]);
    else if (t >= 0 && t < grid.pitches.length) events.push([grid.pitches[t], 1]);
    else throw new Error(`Token ${t} is not a decoded token`);
  }
  return events.map(([p, n]) => [p, n / grid.cells_per_beat]);
}

function sounding(grid, tokens) {
  const out = [];
  let current = null;
  for (const t of tokens) {
    if (t === grid.mask) current = null;
    else if (t === grid.on_rest) current = grid.pitches.length;
    else if (t !== grid.hold) current = t;
    out.push(current);
  }
  return out;
}

// gugakgen.chart.diffusion.make_rows, as flat BigInt64 arrays for onnxruntime
export function makeRows(piece, tokens, types) {
  const { grid, layout } = piece;
  const count = types.length, size = layout.max_cells;
  const rows = {
    tokens: new BigInt64Array(count * size).fill(BigInt(grid.mask)),
    length: new BigInt64Array(count), meter: new BigInt64Array(count), type: new BigInt64Array(count),
    prev_type: new BigInt64Array(count), next_type: new BigInt64Array(count),
    prev_b: new BigInt64Array(count), next_b: new BigInt64Array(count),
  };
  const boundary = (t, last) => {
    if (t < 0 || t >= count) return layout.boundary_none;
    const s = sounding(grid, tokens[t]);
    const v = last ? s[s.length - 1] : s[0];
    return v === null ? layout.boundary_mask : v;
  };
  for (let t = 0; t < count; t++) {
    tokens[t].forEach((v, j) => { rows.tokens[t * size + j] = BigInt(v); });
    rows.length[t] = BigInt(tokens[t].length);
    rows.meter[t] = BigInt(layout.meters.indexOf(tokens[t].length));
    rows.type[t] = BigInt(types[t]);
    rows.prev_type[t] = BigInt(t ? types[t - 1] : layout.type_start);
    rows.next_type[t] = BigInt(t < count - 1 ? types[t + 1] : layout.type_end);
    rows.prev_b[t] = BigInt(boundary(t - 1, true));
    rows.next_b[t] = BigInt(boundary(t + 1, false));
  }
  return rows;
}

export function pickRows(rows, index, size) {
  const out = {};
  for (const [k, v] of Object.entries(rows)) {
    const width = k === "tokens" ? size : 1;
    const a = new BigInt64Array(index.length * width);
    index.forEach((r, n) => a.set(v.subarray(r * width, (r + 1) * width), n * width));
    out[k] = a;
  }
  return out;
}

// ---------------------------------------------------------------------------------------------
// Reverse process (gugakgen.chart.diffusion.reverse_sample) with an async model call
//   predict(rows, count) -> Float32Array of log-probabilities (count, size, outputs)

export async function reverseSample(piece, predict, types, random, opts = {}) {
  const { grid, layout, lengths } = piece;
  const steps = opts.steps ?? 24, temperature = opts.temperature ?? 1.0;
  const returns = opts.returns ?? [], given = opts.given ?? new Map();
  const count = types.length, size = layout.max_cells, K = grid.num_outputs;
  const targetOf = new Map(returns.map(([s, t]) => [t, s]));
  const state = Array.from({ length: count }, (_, t) => new Array(lengths[t]).fill(grid.mask));
  for (const [t, phrase] of given) state[t] = phrase.slice();
  for (const [t, s] of targetOf) state[t] = state[s].slice();
  for (let k = 0; k < steps; k++) {
    const r = 1 - k / steps, s = 1 - (k + 1) / steps;
    const rows = makeRows(piece, state, types);
    // Only rows with a masked cell need predictions (and both rows of a return pair).
    const need = new Set();
    for (let t = 0; t < count; t++) if (state[t].includes(grid.mask)) need.add(t);
    for (const [t, src] of targetOf) if (need.has(src) || need.has(t)) { need.add(src); need.add(t); }
    if (!need.size) break;
    const index = [...need].sort((a, b) => a - b);
    const out = await predict(pickRows(rows, index, size), index.length);
    const at = new Map(index.map((t, n) => [t, n]));
    const lp = (t, j, c) => out[(at.get(t) * size + j) * K + c] / temperature;
    for (const t of index) {
      if (targetOf.has(t)) continue; // targets are written from their source
      const partner = [...targetOf].find(([, src]) => src === t)?.[0];
      for (let j = 0; j < lengths[t]; j++) {
        if (state[t][j] !== grid.mask) continue;
        if (random() >= (r - s) / r) continue;
        const logits = new Float64Array(K);
        for (let c = 0; c < K; c++) logits[c] = partner === undefined ? lp(t, j, c) : 0.5 * (lp(t, j, c) + lp(partner, j, c));
        const m = Math.max(...logits);
        const p = Array.from(logits, (x) => Math.exp(x - m));
        state[t][j] = choice(random, p);
      }
    }
    for (const [t, src] of targetOf) state[t] = state[src].slice();
    opts.onStep?.(k + 1, steps);
  }
  return state;
}

// gugakgen.workflows.chart_diffusion_experiment.generate_piece
export async function generatePiece(piece, predict, seed, opts = {}) {
  const random = rng(seed);
  const walk = opts.walk ?? new TypeWalk(piece.walk).sample(random);
  const medoids = piece.medoid_phrases.map((i) => piece.phrases[i]);
  const returns = piece.returns;
  const targets = new Set(returns.map(([, t]) => t));
  const free = walk.map((_, t) => t).filter((t) => !targets.has(t));
  const kw = { steps: opts.steps ?? piece.sampling.steps, temperature: opts.temperature ?? piece.sampling.temperature, returns };
  const report = (round, k, n) => opts.onProgress?.({ round, step: k, steps: n });
  let tokens = await reverseSample(piece, predict, walk, random, { ...kw, onStep: (k, n) => report(0, k, n) });
  const rejected = (st) => free.filter((t) => !isMember(decode(piece.grid, st[t]), walk[t], medoids));
  let bad = rejected(tokens);
  const firstPass = 1 - bad.length / free.length;
  const attempts = opts.attempts ?? piece.sampling.rejection_attempts;
  let rounds = 0;
  for (let a = 0; a < attempts && bad.length; a++) {
    const redo = new Set(bad);
    for (const [s, t] of returns) if (bad.includes(s)) redo.add(t);
    const given = new Map();
    walk.forEach((_, t) => { if (!redo.has(t)) given.set(t, tokens[t]); });
    tokens = await reverseSample(piece, predict, walk, random, { ...kw, given, onStep: (k, n) => report(a + 1, k, n) });
    bad = rejected(tokens);
    rounds = a + 1;
  }
  const phrases = tokens.map((t) => decode(piece.grid, t));
  return { seed, walk, phrases, firstPass, rounds, stillRejected: bad.length };
}

// Per-piece summary in the spirit of piece_metrics (subset that needs no other generator).
export function pieceSummary(piece, phrases, walk) {
  const medoids = piece.medoid_phrases.map((i) => piece.phrases[i]);
  const targets = new Set(piece.returns.map(([, t]) => t));
  const free = walk.map((_, t) => t).filter((t) => !targets.has(t));
  const originals = piece.phrases;
  const copies = free.filter((t) => originals.some((o) => samePhrase(o, phrases[t]))).length;
  const member = free.filter((t) => isMember(phrases[t], walk[t], medoids)).length;
  const surface = phrases.map((p) => surfaceType(p, medoids));
  const edges = piece.transition_counts;
  let valid = 0;
  for (let t = 0; t + 1 < surface.length; t++) if (edges[surface[t]][surface[t + 1]] > 0) valid++;
  return {
    membership: member / free.length,
    original_phrase_rate: copies / free.length,
    surface_edge_validity: valid / (surface.length - 1),
    returns_exact: piece.returns.every(([s, t]) => samePhrase(phrases[s], phrases[t])) ? 1 : 0,
    walk_positions_differing: walk.filter((c, t) => c !== piece.labels[t]).length / walk.length,
    surface,
  };
}

// ---------------------------------------------------------------------------------------------
// MusicXML (for Verovio and for download)

const SPELL = { 58: ["B", -1, 3], 60: ["C", 0, 4], 63: ["E", -1, 4], 65: ["F", 0, 4], 68: ["A", -1, 4],
  70: ["B", -1, 4], 72: ["C", 0, 5], 75: ["E", -1, 5], 77: ["F", 0, 5] };
const NOTE_TYPE = { 1: ["16th", 0], 2: ["eighth", 0], 3: ["eighth", 1], 4: ["quarter", 0], 6: ["quarter", 1],
  8: ["half", 0], 12: ["half", 1], 16: ["whole", 0], 24: ["whole", 1] };

function spellMidi(m) {
  if (SPELL[m]) return SPELL[m];
  const names = [["C", 0], ["C", 1], ["D", 0], ["E", -1], ["E", 0], ["F", 0], ["F", 1], ["G", 0], ["A", -1], ["A", 0], ["B", -1], ["B", 0]];
  const [s, a] = names[m % 12];
  return [s, a, Math.floor(m / 12) - 1];
}

// split a duration (in cells) starting at cell `start` into notatable pieces that do not cross beats
// unless they start on a beat
function chunks(start, cells, perBeat) {
  const out = [];
  let pos = start, left = cells;
  while (left > 0) {
    const onBeat = pos % perBeat === 0;
    let size;
    if (onBeat) size = [24, 16, 12, 8, 6, 4].find((v) => v <= left && NOTE_TYPE[v]) ?? Math.min(left, perBeat - (pos % perBeat));
    else size = Math.min(left, perBeat - (pos % perBeat));
    if (!NOTE_TYPE[size]) size = [3, 2, 1].find((v) => v <= size);
    out.push(size);
    pos += size; left -= size;
  }
  return out;
}

const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");

export function toMusicXML(piece, phrases, walk, opts = {}) {
  const perBeat = piece.grid.cells_per_beat;
  const names = piece.type_names, yul = piece.yulmyung;
  const chapterStart = new Map(piece.chapters.map((c) => [c.measures[0] - 1, c.chapter]));
  const parts = [];
  let meter = null;
  phrases.forEach((events, t) => {
    const cells = Math.round(events.reduce((a, [, d]) => a + d, 0) * perBeat);
    const beats = cells / perBeat;
    const x = [`<measure number="${t + 1}">`];
    if (t === 0 || beats !== meter) {
      x.push(`<attributes>${t === 0 ? `<divisions>${perBeat}</divisions><key><fifths>-3</fifths></key>` : ""}` +
        `<time><beats>${beats}</beats><beat-type>4</beat-type></time>${t === 0 ? "<clef><sign>G</sign><line>2</line></clef>" : ""}</attributes>`);
      meter = beats;
    }
    const label = [chapterStart.has(t) ? `제${chapterStart.get(t)}장` : null, walk ? names[walk[t]] : null].filter(Boolean).join("  ");
    if (label) x.push(`<direction placement="above"><direction-type><words font-weight="bold">${esc(label)}</words></direction-type></direction>`);
    let pos = 0;
    for (const [pitch, dur] of events) {
      const n = Math.round(dur * perBeat);
      const sizes = chunks(pos, n, perBeat);
      sizes.forEach((size, i) => {
        const [type, dots] = NOTE_TYPE[size];
        const tie = sizes.length > 1 ? (i === 0 ? ["start"] : i === sizes.length - 1 ? ["stop"] : ["stop", "start"]) : [];
        let note = "<note>";
        if (pitch === null) note += "<rest/>";
        else {
          const [step, alter, octave] = spellMidi(pitch);
          note += `<pitch><step>${step}</step>${alter ? `<alter>${alter}</alter>` : ""}<octave>${octave}</octave></pitch>`;
        }
        note += `<duration>${size}</duration>` + tie.map((k) => `<tie type="${k}"/>`).join("");
        note += `<type>${type}</type>` + "<dot/>".repeat(dots);
        if (pitch !== null && tie.length) note += `<notations>${tie.map((k) => `<tied type="${k}"/>`).join("")}</notations>`;
        if (pitch !== null && i === 0 && opts.yulmyung !== false && yul[pitch]) note += `<lyric><text>${esc(yul[pitch])}</text></lyric>`;
        note += "</note>";
        x.push(note);
      });
      pos += n;
    }
    x.push("</measure>");
    parts.push(x.join(""));
  });
  const title = esc(opts.title ?? piece.title.ko);
  return `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="4.0"><work><work-title>${title}</work-title></work>
<identification><encoding><software>DiffDodree web</software></encoding></identification>
<part-list><score-part id="P1"><part-name>${esc(piece.title.part)}</part-name></score-part></part-list>
<part id="P1">${parts.join("\n")}</part></score-partwise>`;
}
