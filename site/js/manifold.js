// The phrase manifold M drawn from data (paper fig. 4): one lobe M_c per type, necks between
// types that are each other's nearest medoids for some measure, and the piece as a curve γ(t)
// that moves note by note inside each lobe's local chart (τ, η) around the medoid.
//   τ: where on the medoid the note sits (edit alignment; angle around the lobe)
//   η: scale steps away from the medoid's note (radius; the dashed circle is η = 0, the medoid)
import { toChart } from "./core.js";

const DT = 1 / 16, TRAVEL = 0.12, R0 = 0.56, KETA = 0.09;
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
function bez(p0, p1, p2, p3, u) {
  const v = 1 - u;
  return [v * v * v * p0[0] + 3 * v * v * u * p1[0] + 3 * v * u * u * p2[0] + u * u * u * p3[0],
    v * v * v * p0[1] + 3 * v * v * u * p1[1] + 3 * v * u * u * p2[1] + u * u * u * p3[1]];
}
function tokens() {
  const cs = getComputedStyle(document.documentElement), g = (n) => cs.getPropertyValue(n).trim();
  return { fg: g("--ink"), muted: g("--muted"), surf: g("--surf"), edge: g("--surf-edge"), curve: g("--curve"),
    faint: g("--faint"), plate: g("--surface"), acc: g("--accent"), t: Array.from({ length: 9 }, (_, c) => g(`--t${c}`)) };
}

export class ManifoldCanvas {
  constructor(canvas, piece, opts = {}) {
    this.cv = canvas;
    this.ctx = canvas.getContext("2d");
    this.piece = piece;
    this.W = opts.width ?? 900;
    this.H = opts.height ?? 560;
    this.names = piece.type_names;
    this.medoids = piece.medoid_phrases.map((i) => piece.phrases[i]);
    this.focus = null;
    this.layout();
    const redraw = () => this.draw(this.last ?? 0, this.lastOpts ?? {});
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redraw);
    new MutationObserver(redraw).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    document.fonts?.ready.then(redraw);
  }

  // ------------------------------------------------------------------ lobes and necks
  layout() {
    const { sites, contact, visits } = this.piece.lobes, K = sites.length, W = this.W, H = this.H;
    const vmax = Math.max(...visits);
    // Layout (drawing only; the analysis never reads it). Metric MDS of the medoid edit distances
    // where each pair's target distance is at least the room two lobes and a neck need:
    //   minimise Σ (‖x_i − x_j‖ − T_ij)²,  T_ij = max(s·D_ij, R_i + R_j + GAP)
    // started from the exported SMACOF sites, by Guttman updates.
    const D = this.piece.lobes.distance, GAP = 40;
    const sx = sites.map((p) => p[0]), sy = sites.map((p) => p[1]);
    const s0 = Math.min((W - 160) / (Math.max(...sx) - Math.min(...sx)), (H - 140) / (Math.max(...sy) - Math.min(...sy)));
    const L = sites.map(([x, y], c) => ({ c, x: x * s0, y: -y * s0, visits: visits[c], R: 40 + 34 * Math.sqrt(visits[c] / vmax) }));
    const T = D.map((row, i) => row.map((d, j) => (i === j ? 0 : Math.max(s0 * d, L[i].R + L[j].R + GAP))));
    for (let it = 0; it < 500; it++) {
      const next = L.map((li, i) => {
        let x = 0, y = 0;
        L.forEach((lj, j) => {
          if (i === j) return;
          const dx = li.x - lj.x, dy = li.y - lj.y, e = Math.hypot(dx, dy) || 1e-6;
          x += lj.x + (T[i][j] * dx) / e; y += lj.y + (T[i][j] * dy) / e;
        });
        return [x / (K - 1), y / (K - 1)];
      });
      next.forEach(([x, y], i) => { L[i].x = x; L[i].y = y; });
    }
    for (let it = 0; it < 300; it++) { // safety net: no pair closer than its room
      let moved = false;
      for (let i = 0; i < K; i++) for (let j = i + 1; j < K; j++) {
        const A = L[i], B = L[j], dx = B.x - A.x, dy = B.y - A.y, d = Math.hypot(dx, dy) || 1e-6, need = A.R + B.R + GAP * 0.8;
        if (d < need) { const push = ((need - d) / 2) * 0.5; A.x -= (dx / d) * push; A.y -= (dy / d) * push; B.x += (dx / d) * push; B.y += (dy / d) * push; moved = true; }
      }
      if (!moved) break;
    }
    // how faithful the drawing is: correlation of drawn and edit distances between medoids
    const pairs = [];
    for (let i = 0; i < K; i++) for (let j = i + 1; j < K; j++) pairs.push([Math.hypot(L[i].x - L[j].x, L[i].y - L[j].y), D[i][j]]);
    const mean = (k) => pairs.reduce((a, p) => a + p[k], 0) / pairs.length, m0 = mean(0), m1 = mean(1);
    const cov = pairs.reduce((a, p) => a + (p[0] - m0) * (p[1] - m1), 0);
    this.fidelity = cov / Math.sqrt(pairs.reduce((a, p) => a + (p[0] - m0) ** 2, 0) * pairs.reduce((a, p) => a + (p[1] - m1) ** 2, 0));
    // rotation/reflection keep every distance: take the one that fills the frame best,
    // with type A on the left (the piece starts there)
    const fit = (pts) => {
      const minx = Math.min(...pts.map((q, i) => q[0] - L[i].R)), maxx = Math.max(...pts.map((q, i) => q[0] + L[i].R));
      const miny = Math.min(...pts.map((q, i) => q[1] - L[i].R)), maxy = Math.max(...pts.map((q, i) => q[1] + L[i].R));
      return { s: Math.min((W - 120) / (maxx - minx), (H - 96) / (maxy - miny)), minx, maxx, miny, maxy };
    };
    let best = null;
    for (const flip of [1, -1]) for (let k = 0; k < 72; k++) {
      const a = (k * Math.PI) / 36, c = Math.cos(a), sn = Math.sin(a);
      const pts = L.map((l) => [c * l.x - sn * l.y * flip, sn * l.x + c * l.y * flip]);
      const f = fit(pts), leftA = pts[0][0] <= Math.min(...pts.map((q) => q[0])) + 0.25 * (f.maxx - f.minx);
      const score = f.s * (leftA ? 1 : 0.9);
      if (!best || score > best.score + 1e-9) best = { score, pts, f };
    }
    const { pts, f } = best, s = Math.min(f.s, 1.6);
    L.forEach((l, i) => { l.x = W / 2 + (pts[i][0] - (f.minx + f.maxx) / 2) * s; l.y = H / 2 + 4 + (pts[i][1] - (f.miny + f.maxy) / 2) * s; l.R *= s; });
    const necks = [];
    for (let a = 0; a < K; a++) for (let b = a + 1; b < K; b++) if (contact[a][b] > 0) necks.push({ a, b, n: contact[a][b] });
    necks.forEach((nk) => { // bend a neck around any lobe it would cross
      const A = L[nk.a], B = L[nk.b], dx = B.x - A.x, dy = B.y - A.y, d = Math.hypot(dx, dy);
      let bend = 0;
      L.forEach((o) => {
        if (o === A || o === B) return;
        const t = ((o.x - A.x) * dx + (o.y - A.y) * dy) / (d * d);
        if (t <= 0 || t >= 1) return;
        const dist = Math.hypot(o.x - (A.x + t * dx), o.y - (A.y + t * dy));
        if (dist < o.R + 14) {
          const side = ((o.x - A.x) * dy - (o.y - A.y) * dx) > 0 ? -1 : 1;
          bend = Math.max(Math.abs(bend), o.R + 26 - dist) * side;
        }
      });
      nk.mx = (A.x + B.x) / 2 - (dy / d) * bend * 1.7;
      nk.my = (A.y + B.y) / 2 + (dx / d) * bend * 1.7;
      nk.w = Math.min(A.R, B.R) * (0.30 + 0.05 * Math.min(nk.n, 4)); // width: drawing only
    });
    L.forEach((l) => { // the chart's τ = 0 points to where the necks leave
      let vx = 0, vy = 0;
      necks.forEach((nk) => { if (nk.a === l.c || nk.b === l.c) { vx += (nk.mx - l.x) * nk.n; vy += (nk.my - l.y) * nk.n; } });
      l.port = vx || vy ? Math.atan2(vy, vx) : -Math.PI / 2;
    });
    // medoid note positions: ticks on the η = 0 circle
    L.forEach((l) => {
      const md = this.medoids[l.c], len = md.reduce((a, [, d]) => a + d, 0);
      let on = 0;
      l.ticks = [];
      md.forEach(([p, d]) => { if (p !== null) l.ticks.push(on / len); on += d; });
    });
    // label spots: around each lobe, the angle farthest from other lobes, necks and labels
    const neckPts = necks.flatMap((nk) => {
      const A = L[nk.a], B = L[nk.b], cx = 2 * nk.mx - (A.x + B.x) / 2, cy = 2 * nk.my - (A.y + B.y) / 2;
      return Array.from({ length: 13 }, (_, k) => { const u = k / 12, v = 1 - u;
        return [v * v * A.x + 2 * v * u * cx + u * u * B.x, v * v * A.y + 2 * v * u * cy + u * u * B.y, nk.w / 2]; });
    });
    const placed = [];
    L.forEach((l) => {
      let best = null;
      for (let k = 0; k < 16; k++) {
        const a = l.port + Math.PI * 0.75 + (k % 2 ? 1 : -1) * Math.ceil(k / 2) * (Math.PI / 8);
        const x = l.x + (l.R + 15) * Math.cos(a), y = l.y + (l.R + 15) * Math.sin(a);
        let clear = Math.min(x - 14, W - 14 - x, y - 40, H - 30 - y) * 2;
        L.forEach((o) => { if (o !== l) clear = Math.min(clear, Math.hypot(x - o.x, y - o.y) - o.R - 12); });
        neckPts.forEach(([nx, ny, r]) => { clear = Math.min(clear, Math.hypot(x - nx, y - ny) - r - 10); });
        placed.forEach(([px, py]) => { clear = Math.min(clear, Math.hypot(x - px, y - py) - 26); });
        const score = Math.min(clear, 24) - k * 0.15; // prefer the default angle when there is room
        if (!best || score > best.score) best = { score, x, y };
      }
      l.label = [best.x, best.y];
      placed.push(l.label);
    });
    this.L = L;
    this.necks = necks;
    this.neckOf = (a, b) => necks.find((z) => (z.a === a && z.b === b) || (z.a === b && z.b === a));
  }

  lobePoint(l, tau, eta) {
    const r = l.R * (R0 + KETA * clamp(eta, -4, 4)), th = l.port + 2 * Math.PI * tau;
    return [l.x + r * Math.cos(th), l.y + r * Math.sin(th)];
  }

  // ------------------------------------------------------------------ a piece as a curve
  setPiece(phrases, walk) {
    const scale = this.piece.grid.pitches;
    let a = 0;
    this.phrases = phrases.map((events, t) => {
      const c = walk[t], md = this.medoids[c];
      const mdLen = md.reduce((x, [, d]) => x + d, 0), len = events.reduce((x, [, d]) => x + d, 0);
      const mdOn = []; let o = 0;
      md.forEach(([, d]) => { mdOn.push(o / mdLen); o += d; });
      const chart = toChart(events, md, scale);
      const notes = []; let on = 0;
      events.forEach(([p, d]) => { notes.push([on, p, d]); on += d; });
      // τ: the slot's place on the medoid, plus how far into the slot this note starts
      const tau = [], eta = [];
      let slotStart = null, prevSlot = -1, lastEta = 0;
      chart.forEach((e, i) => {
        if (e.slot !== prevSlot) { slotStart = notes[i][0]; prevSlot = e.slot; }
        const next = e.slot + 1 < mdOn.length ? mdOn[e.slot + 1] : 1;
        let v = mdOn[e.slot] + (notes[i][0] - slotStart) / len;
        v = Math.min(v, next - 0.004);
        if (tau.length) v = Math.max(v, tau[tau.length - 1] + 0.002);
        tau.push(v);
        if (e.eta !== null) lastEta = e.eta;
        eta.push(lastEta);
      });
      const ph = { a, b: a + len, c, notes, tau, eta, rawEta: chart.map((e) => e.eta), slots: chart.map((e) => e.slot), events };
      a += len;
      return ph;
    });
    this.total = a;
    this.walk = walk;
    this.buildTrack();
  }

  phraseAt(t) {
    const ph = this.phrases;
    let lo = 0, hi = ph.length - 1;
    while (lo < hi) { const m = (lo + hi + 1) >> 1; if (ph[m].a <= t + 1e-9) lo = m; else hi = m - 1; }
    return lo;
  }

  noteIdx(ph, rel) { let j = 0; while (j + 1 < ph.notes.length && ph.notes[j + 1][0] <= rel + 1e-9) j++; return j; }

  chartAt(ph, t) {
    const rel = t - ph.a, j = this.noteIdx(ph, rel), n = ph.notes[j];
    const u = clamp((rel - n[0]) / n[2], 0, 1);
    const t1 = j + 1 < ph.tau.length ? ph.tau[j + 1] : 1, tau = ph.tau[j] + u * (t1 - ph.tau[j]);
    const e1 = j + 1 < ph.eta.length ? ph.eta[j + 1] : ph.eta[j], w = u < 0.8 ? 0 : (u - 0.8) / 0.2;
    return [tau, ph.eta[j] * (1 - w) + e1 * w];
  }

  buildTrack() {
    const L = this.L, track = [], dots = [];
    const n = Math.ceil(this.total / DT);
    for (let i = 0; i <= n; i++) {
      const t = Math.min(this.total - 1e-6, i * DT), pi = this.phraseAt(t), ph = this.phrases[pi];
      const l = L[ph.c], len = ph.b - ph.a, s = (t - ph.a) / len;
      const nx = this.phrases[pi + 1], nl = nx ? L[nx.c] : null;
      if (nl && nl !== l && s > 1 - TRAVEL) {
        const u = (s - (1 - TRAVEL)) / TRAVEL;
        const p0 = this.lobePoint(l, ...this.chartAt(ph, ph.a + (1 - TRAVEL) * len)), p3 = this.lobePoint(nl, nx.tau[0], nx.eta[0]);
        const nk = this.neckOf(l.c, nl.c);
        if (!nk) { // no neck: a jump over the surface
          const d = Math.hypot(p3[0] - p0[0], p3[1] - p0[1]);
          const lift = Math.min(36 + 0.16 * d, Math.max(8, Math.min(p0[1], p3[1]) - 18) / 0.75);
          track.push({ p: bez(p0, [p0[0], p0[1] - lift], [p3[0], p3[1] - lift], p3, u), c: l.c, trans: true, jump: true });
          continue;
        }
        const m = [nk.mx, nk.my];
        const pA = [l.x + l.R * 1.05 * Math.cos(l.port), l.y + l.R * 1.05 * Math.sin(l.port)];
        const pB = [nl.x + nl.R * 1.05 * Math.cos(nl.port), nl.y + nl.R * 1.05 * Math.sin(nl.port)];
        const pt = u < 0.5 ? bez(p0, pA, [(pA[0] + m[0]) / 2, (pA[1] + m[1]) / 2], m, u * 2)
          : bez(m, [(pB[0] + m[0]) / 2, (pB[1] + m[1]) / 2], pB, p3, u * 2 - 1);
        track.push({ p: pt, c: l.c, trans: true });
        continue;
      }
      track.push({ p: this.lobePoint(l, ...this.chartAt(ph, t)), c: l.c });
    }
    this.phrases.forEach((ph) => ph.notes.forEach((nt, j) => {
      if (nt[1] !== null) dots.push({ t: ph.a + nt[0], p: this.lobePoint(L[ph.c], ph.tau[j], ph.eta[j]) });
    }));
    this.track = track;
    this.dots = dots;
  }

  // ------------------------------------------------------------------ drawing
  surface(curC) {
    const g = this.ctx, t = this.tok, L = this.L;
    const lobes = (grow) => { g.beginPath(); L.forEach((l) => { g.moveTo(l.x + l.R + grow, l.y); g.arc(l.x, l.y, l.R + grow, 0, 2 * Math.PI); }); };
    const necks = (extra) => this.necks.forEach((nk) => {
      const A = L[nk.a], B = L[nk.b];
      g.lineWidth = nk.w + extra;
      g.beginPath(); g.moveTo(A.x, A.y);
      g.quadraticCurveTo(2 * nk.mx - (A.x + B.x) / 2, 2 * nk.my - (A.y + B.y) / 2, B.x, B.y); g.stroke();
    });
    g.save(); g.lineCap = "round";
    g.fillStyle = t.edge; g.strokeStyle = t.edge; lobes(1.4); g.fill(); necks(2.8);
    g.fillStyle = t.surf; g.strokeStyle = t.surf; lobes(0); g.fill(); necks(0);
    L.forEach((l) => { // the type's colour, stronger where the curve is now or on hover
      const on = l.c === curC || l.c === this.focus;
      g.globalAlpha = on ? 0.26 : this.focus !== null ? 0.05 : 0.12;
      g.fillStyle = t.t[l.c]; g.beginPath(); g.arc(l.x, l.y, l.R, 0, 2 * Math.PI); g.fill();
    });
    g.globalAlpha = 0.5; g.strokeStyle = t.edge; g.lineWidth = 0.6; // neck sections and lobe equators
    this.necks.forEach((nk) => {
      const A = L[nk.a], B = L[nk.b], dir = Math.atan2(B.y - A.y, B.x - A.x);
      g.beginPath(); g.ellipse(nk.mx, nk.my, (nk.w / 2) * 0.28, nk.w / 2, dir, 0, 2 * Math.PI); g.stroke();
    });
    L.forEach((l) => {
      g.beginPath(); g.ellipse(l.x, l.y + l.R * 0.02, l.R, l.R * 0.28, 0, 0, Math.PI); g.stroke();
      g.setLineDash([3, 3]); g.beginPath(); g.ellipse(l.x, l.y + l.R * 0.02, l.R, l.R * 0.28, 0, Math.PI, 2 * Math.PI); g.stroke(); g.setLineDash([]);
    });
    g.globalAlpha = 0.55; g.strokeStyle = t.muted; g.lineWidth = 0.8; g.setLineDash([2, 3]); // chart: η = 0 and medoid notes
    L.forEach((l) => { g.beginPath(); g.arc(l.x, l.y, l.R * R0, 0, 2 * Math.PI); g.stroke(); });
    g.setLineDash([]);
    L.forEach((l) => l.ticks.forEach((tv) => {
      const th = l.port + 2 * Math.PI * tv, c = Math.cos(th), s = Math.sin(th);
      g.beginPath(); g.moveTo(l.x + l.R * (R0 - 0.07) * c, l.y + l.R * (R0 - 0.07) * s);
      g.lineTo(l.x + l.R * (R0 + 0.07) * c, l.y + l.R * (R0 + 0.07) * s); g.stroke();
    }));
    g.restore();
  }

  strokeTrack(from, to, width, alpha, colorOf) {
    const g = this.ctx, track = this.track;
    g.lineCap = "round"; g.lineJoin = "round"; g.lineWidth = width; g.globalAlpha = alpha;
    let cur = null;
    g.beginPath();
    for (let i = from; i < to; i++) {
      const a = track[i], b = track[i + 1];
      const key = colorOf(a) + (a.jump ? "|j" : "");
      if (key !== cur) {
        if (cur) g.stroke();
        g.strokeStyle = key.split("|")[0]; g.setLineDash(a.jump ? [5, 4] : []);
        g.beginPath(); g.moveTo(a.p[0], a.p[1]); cur = key;
      }
      g.lineTo(b.p[0], b.p[1]);
    }
    if (cur) g.stroke();
    g.setLineDash([]); g.globalAlpha = 1;
  }

  // pos: beats from the start. opts: {ghost, complete (whole curve as past, no marker)}
  draw(pos, opts = {}) {
    this.last = pos; this.lastOpts = opts;
    if (!this.track) return;
    const g = this.ctx, W = this.W, H = this.H, dpr = Math.min(2, window.devicePixelRatio || 1);
    this.tok = tokens();
    const t = this.tok;
    if (this.cv.width !== W * dpr) { this.cv.width = W * dpr; this.cv.height = H * dpr; }
    g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
    const end = this.track.length - 1;
    const i = opts.complete ? end : Math.min(end, Math.floor(pos / DT)), cur = opts.complete ? null : this.track[i];
    this.surface(cur ? cur.c : -1);
    if (opts.ghost !== false && !opts.complete) this.strokeTrack(0, end, 1, 0.55, () => t.faint);
    if (opts.complete) { // the whole original, quietly: paths inside lobes, then necks, then jumps
      this.strokeTrack(0, end, 1.2, 0.8, (a) => (a.jump ? t.faint : a.trans ? t.muted : t.t[a.c]));
    } else {
      this.strokeTrack(0, i, 1.9, 0.95, (a) => (a.trans ? t.muted : t.t[a.c]));
    }
    for (const d of this.dots) {
      const past = opts.complete || d.t <= pos + 1e-9;
      if (!past && opts.ghost === false) continue;
      g.globalAlpha = past ? 0.9 : 0.45; g.fillStyle = past ? t.curve : t.faint;
      g.beginPath(); g.arc(d.p[0], d.p[1], past ? (opts.complete ? 1.6 : 2.1) : 1.6, 0, 2 * Math.PI); g.fill();
    }
    g.globalAlpha = 1;
    // lobe names M_A … M_I, centred on their spots
    g.textBaseline = "middle"; g.textAlign = "left";
    this.L.forEach((l) => {
      const [x, y] = l.label;
      g.font = 'italic 17px "STIX Two Text", "Times New Roman", serif';
      const w = g.measureText("M").width;
      g.font = 'italic 12px "STIX Two Text", serif';
      const wl = g.measureText(this.names[l.c]).width, x0 = x - (w + wl) / 2;
      g.fillStyle = t.fg;
      g.font = 'italic 17px "STIX Two Text", "Times New Roman", serif'; g.fillText("M", x0, y);
      g.font = 'italic 12px "STIX Two Text", serif'; g.fillStyle = t.t[l.c]; g.fillText(this.names[l.c], x0 + w + 0.5, y + 5);
    });
    g.textBaseline = "alphabetic";
    if (cur) {
      g.fillStyle = t.plate; g.strokeStyle = t.curve; g.lineWidth = 2;
      g.beginPath(); g.arc(cur.p[0], cur.p[1], 7, 0, 2 * Math.PI); g.fill(); g.stroke();
      g.fillStyle = cur.trans ? t.curve : t.t[cur.c];
      g.beginPath(); g.arc(cur.p[0], cur.p[1], 4, 0, 2 * Math.PI); g.fill();
      g.font = 'italic 15px "STIX Two Text", serif'; g.fillStyle = t.fg; g.textAlign = "left"; g.fillText("γ(t)", cur.p[0] + 11, cur.p[1] - 9);
    }
    g.font = 'italic 22px "STIX Two Text", serif'; g.fillStyle = t.fg; g.textAlign = "left"; g.fillText("M", 18, 34);
    g.font = '12px "Noto Sans KR", sans-serif'; g.fillStyle = t.muted;
    const y0 = H - 12;
    g.strokeStyle = t.muted; g.lineWidth = 1.6; g.setLineDash([5, 4]);
    g.beginPath(); g.moveTo(18, y0 - 2); g.quadraticCurveTo(30, y0 - 14, 42, y0 - 2); g.stroke(); g.setLineDash([]);
    g.fillText("도약: 목이 없는(닮지 않은) 유형으로의 전이", 48, y0);
  }

  // what the curve is doing at pos, as HTML
  readout(pos, yul) {
    const tr = this.track[Math.min(this.track.length - 1, Math.floor(pos / DT))];
    if (tr?.jump) return "<b>도약</b>: 닮지 않은 유형으로 건너뛰는 중 (두 엽 사이에 목이 없음)";
    if (tr?.trans) return "<b>목을 지나는 중</b>: 닮은 이웃 유형으로";
    const ph = this.phrases[this.phraseAt(Math.min(pos, this.total - 1e-6))];
    const j = this.noteIdx(ph, pos - ph.a), nt = ph.notes[j];
    if (nt[1] === null) return "쉼표";
    const e = ph.rawEta[j], sg = e > 0 ? `+${e}` : e < 0 ? `−${-e}` : "0";
    return `현재 음 <b>${yul[nt[1]] ?? nt[1]}</b> → 엽 <span class="m">M<sub>${this.names[ph.c]}</sub></span>의 차트에서 ` +
      `<span class="m">τ</span> = ${ph.tau[j].toFixed(2)} (대표 마디 ${ph.slots[j] + 1}번째 음 자리), <span class="m">η</span> = ${sg}음계 단계`;
  }
}
