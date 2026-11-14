// SVG views: the phrase graph G (the quotient of the manifold) and the λ sweep charts.
// The manifold itself is drawn on a canvas (manifold.js).
const NS = "http://www.w3.org/2000/svg";
export const typeColor = (c) => `var(--t${c})`;

export function el(name, attrs = {}, parent = null) {
  const node = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) if (v !== undefined && v !== null) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}

const uid = (() => { let n = 0; return (p) => `${p}${++n}`; })();

function arc(p, q, bend, r0, r1) {
  const dx = q[0] - p[0], dy = q[1] - p[1], L = Math.hypot(dx, dy) || 1;
  const ux = dx / L, uy = dy / L;
  const a = [p[0] + ux * r0, p[1] + uy * r0], b = [q[0] - ux * r1, q[1] - uy * r1];
  const mx = (a[0] + b[0]) / 2 - uy * L * bend, my = (a[1] + b[1]) / 2 + ux * L * bend;
  return `M${a[0].toFixed(1)},${a[1].toFixed(1)} Q${mx.toFixed(1)},${my.toFixed(1)} ${b[0].toFixed(1)},${b[1].toFixed(1)}`;
}

// ---------------------------------------------------------------------------------------------
// G = π(M): one node per type on a circle (order of first appearance), edges = transitions of the
// original (width = count). Dashed edges join types without a neck in M (jumps).
export class GraphView {
  constructor(container, piece, opts = {}) {
    this.piece = piece;
    const S = opts.size ?? 240, cx = S / 2, cy = S / 2 + 4, R = S / 2 - 32, r = 13;
    this.svg = el("svg", { viewBox: `0 0 ${S} ${S + 8}`, class: "graph", role: "img",
      "aria-label": "프레이즈 그래프 G: 아홉 유형과 원곡의 전이" }, container);
    const K = piece.type_names.length, contact = piece.lobes.contact;
    this.pos = Array.from({ length: K }, (_, c) => {
      const a = -Math.PI / 2 + (2 * Math.PI * c) / K;
      return [cx + R * Math.cos(a), cy + R * Math.sin(a)];
    });
    const ids = [uid("ga"), uid("gb")];
    const defs = el("defs", {}, this.svg);
    ids.forEach((id, k) => {
      const m = el("marker", { id, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 5, markerHeight: 5, orient: "auto-start-reverse" }, defs);
      el("path", { d: "M1 1L9 5L1 9", class: k ? "chev dark" : "chev" }, m);
    });
    this.markers = ids.map((id) => `url(#${id})`);
    this.gEdges = el("g", {}, this.svg);
    this.edges = new Map();
    piece.transition_counts.forEach((row, a) => row.forEach((n, b) => {
      if (!n) return;
      let d;
      if (a === b) {
        const [x, y] = this.pos[a], ox = x - cx, oy = y - cy, l0 = Math.hypot(ox, oy) || 1, ux = ox / l0, uy = oy / l0;
        d = `M${x + ux * r - uy * 5} ${y + uy * r + ux * 5}C${x + ux * (r + 22) - uy * 14} ${y + uy * (r + 22) + ux * 14} ${x + ux * (r + 22) + uy * 14} ${y + uy * (r + 22) - ux * 14} ${x + ux * r + uy * 5} ${y + uy * r - ux * 5}`;
      } else {
        const [x1, y1] = this.pos[a], [x2, y2] = this.pos[b], dx = x2 - x1, dy = y2 - y1, D = Math.hypot(dx, dy);
        const ux = dx / D, uy = dy / D, bend = piece.transition_counts[b][a] ? 10 : 4;
        const sx = x1 + ux * r, sy = y1 + uy * r, ex = x2 - ux * (r + 2), ey = y2 - uy * (r + 2);
        d = `M${sx} ${sy}Q${(sx + ex) / 2 - uy * bend} ${(sy + ey) / 2 + ux * bend} ${ex} ${ey}`;
      }
      const jump = a !== b && !(contact[a][b] > 0);
      const path = el("path", { d, class: `edge${jump ? " jump" : ""}`, "stroke-width": 0.7 + 0.5 * Math.min(n, 4), "marker-end": this.markers[0] }, this.gEdges);
      el("title", {}, path).textContent = `${piece.type_names[a]} → ${piece.type_names[b]}: 원곡에서 ${n}번${jump ? " (도약)" : ""}`;
      this.edges.set(`${a},${b}`, path);
    }));
    const sizes = piece.lobes.visits;
    this.nodes = this.pos.map(([x, y], c) => {
      const g = el("g", { class: "node", tabindex: 0, style: `--tc: var(--t${c})` }, this.svg);
      el("circle", { cx: x, cy: y, r }, g);
      el("text", { x, y: y + 4.5, "text-anchor": "middle", class: "lab" }, g).textContent = piece.type_names[c];
      el("title", {}, g).textContent = `유형 ${piece.type_names[c]}: 원곡 ${sizes[c]}마디`;
      return g;
    });
    el("text", { x: 6, y: 20, class: "gname" }, this.svg).textContent = "G";
  }
  onHover(fn) {
    this.nodes.forEach((g, c) => {
      g.addEventListener("pointerenter", () => fn(c));
      g.addEventListener("pointerleave", () => fn(null));
      g.addEventListener("focus", () => fn(c));
      g.addEventListener("blur", () => fn(null));
    });
  }
  highlightType(c) {
    this.svg.classList.toggle("focus", c !== null);
    this.nodes.forEach((g, k) => g.classList.toggle("on", k === c));
    this.edges.forEach((p, key) => {
      const [a, b] = key.split(",").map(Number);
      p.classList.toggle("on", a === c || b === c);
    });
  }
  // walk: the piece's type sequence (edges it uses); current: "a,b" of the transition just taken
  setWalk(walk, current = null, here = null) {
    const used = new Set();
    this.svg.classList.toggle("has-walk", Boolean(walk));
    if (walk) for (let i = 0; i + 1 < walk.length; i++) used.add(`${walk[i]},${walk[i + 1]}`);
    this.edges.forEach((p, key) => {
      p.classList.toggle("used", walk ? used.has(key) : false);
      const on = current === key;
      p.classList.toggle("current", on);
      p.setAttribute("marker-end", this.markers[on ? 1 : 0]);
    });
    this.nodes.forEach((g, c) => g.classList.toggle("current", here === c));
  }
}

// ---------------------------------------------------------------------------------------------
// λ sweep: small multiples of mean ± 95% CI over 15 training seeds
export function sweepChart(container, sweep, panels) {
  const W = 300, H = 210, m = { l: 46, r: 12, t: 30, b: 38 };
  for (const panel of panels) {
    const data = sweep.metrics[panel.key];
    const box = document.createElement("figure");
    box.className = "sweep-panel";
    container.appendChild(box);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": panel.title }, box);
    const n = data.lambda.length;
    const lo = data.mean.map((v, i) => (v - data.ci95_half[i]) * panel.scale);
    const hi = data.mean.map((v, i) => (v + data.ci95_half[i]) * panel.scale);
    const extra = panel.band ? panel.band : [];
    let y0 = Math.min(...lo, ...extra), y1 = Math.max(...hi, ...extra);
    const padY = (y1 - y0) * 0.12; y0 -= padY; y1 += padY;
    const X = (i) => m.l + (i * (W - m.l - m.r)) / (n - 1);
    const Y = (v) => H - m.b - ((v - y0) / (y1 - y0)) * (H - m.t - m.b);
    if (panel.band) el("rect", { x: m.l, width: W - m.l - m.r, y: Y(Math.min(y1, panel.band[1])), height: Y(Math.max(y0, panel.band[0])) - Y(Math.min(y1, panel.band[1])), class: "band" }, svg);
    const ticks = niceTicks(y0, y1, 4);
    for (const t of ticks) {
      el("line", { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), class: "grid" }, svg);
      const lab = el("text", { x: m.l - 6, y: Y(t) + 3.5, "text-anchor": "end", class: "tick" }, svg);
      lab.textContent = panel.fmt(t);
    }
    data.lambda.forEach((l, i) => {
      const lab = el("text", { x: X(i), y: H - m.b + 15, "text-anchor": "middle", class: "tick" }, svg);
      lab.textContent = l === 0 ? "0" : String(l);
    });
    const xl = el("text", { x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle", class: "axis-label" }, svg);
    xl.textContent = "구조 손실 가중치 λ";
    el("path", { d: data.mean.map((v, i) => `${i ? "L" : "M"}${X(i)},${Y(v * panel.scale)}`).join(""), class: "line" }, svg);
    data.mean.forEach((v, i) => {
      el("line", { x1: X(i), x2: X(i), y1: Y(lo[i]), y2: Y(hi[i]), class: "ci" }, svg);
      const dot = el("circle", { cx: X(i), cy: Y(v * panel.scale), r: 3.4, class: "pt" }, svg);
      const title = el("title", {}, dot);
      title.textContent = `λ = ${data.lambda[i]}: ${panel.fmt(v * panel.scale)} (95% 구간 ${panel.fmt(lo[i])}–${panel.fmt(hi[i])})`;
    });
    const cap = document.createElement("figcaption");
    cap.innerHTML = `<strong>${panel.title}</strong><span>${panel.note}</span>`;
    box.prepend(cap);
  }
}

function niceTicks(a, b, count) {
  const span = b - a, step0 = span / count, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((s) => s * mag).find((s) => span / s <= count + 1);
  const out = [];
  for (let v = Math.ceil(a / step) * step; v <= b + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}
