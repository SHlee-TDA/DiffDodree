// Run the web demo's ManifoldCanvas layout in Node and dump lobes, necks and curve tracks
// for the original piece and one gallery piece, so the video draws the same manifold.
// Usage: node video/extract_manifold.mjs <W> <H> <arm> <pieceIndex>
import { readFileSync, writeFileSync } from "node:fs";
const ROOT = new URL("../", import.meta.url);
const stubCtx = new Proxy({}, { get: () => () => ({ width: 10 }) });
globalThis.window = { matchMedia: () => ({ addEventListener() {} }), devicePixelRatio: 1 };
globalThis.document = { documentElement: {}, fonts: null };
globalThis.MutationObserver = class { observe() {} };
globalThis.getComputedStyle = () => ({ getPropertyValue: () => "" });
const { ManifoldCanvas } = await import(new URL("site/js/manifold.js", ROOT));
const piece = JSON.parse(readFileSync(new URL("site/data/piece.json", ROOT)));
const gallery = JSON.parse(readFileSync(new URL("site/data/gallery.json", ROOT)));
const [W, H, arm, idx] = [Number(process.argv[2] ?? 1000), Number(process.argv[3] ?? 1000), process.argv[4] ?? "diffusion_graph_010", Number(process.argv[5] ?? 0)];
const canvas = { getContext: () => stubCtx, width: 0, height: 0 };
const mc = new ManifoldCanvas(canvas, piece, { width: W, height: H });
const dump = (phrases, walk) => {
  mc.setPiece(phrases, walk);
  return { total: mc.total, track: mc.track.map((t) => [t.p[0], t.p[1], t.c, t.trans ? 1 : 0, t.jump ? 1 : 0]),
    dots: mc.dots.map((d) => [d.t, d.p[0], d.p[1]]),
    measures: mc.phrases.map((p) => ({ a: p.a, b: p.b, c: p.c })),
    // chart coordinates (measure index, τ, η) for each track sample and note dot, so the video can
    // place the same curve on its own layout of lobes
    chart: mc.track.map((_, i) => { const t = Math.min(mc.total - 1e-6, i * (1 / 16)), pi = mc.phraseAt(t);
      return [pi, ...mc.chartAt(mc.phrases[pi], t)]; }),
    dchart: mc.phrases.flatMap((ph, pi) => ph.notes.flatMap((nt, j) => (nt[1] !== null ? [[pi, ph.tau[j], ph.eta[j]]] : []))) };
};
const orig = dump(piece.phrases, piece.labels);
const g = gallery.pieces[arm][idx];
const gen = dump(g.phrases, g.walk);
const out = { W, H, DT: 1 / 16, R0: 0.56,
  lobes: mc.L.map((l) => ({ c: l.c, x: l.x, y: l.y, R: l.R, port: l.port, label: l.label, ticks: l.ticks })),
  necks: mc.necks.map((n) => ({ a: n.a, b: n.b, n: n.n, mx: n.mx, my: n.my, w: n.w })),
  fidelity: mc.fidelity, original: orig, generated: { arm, index: idx, seed: g.seed, walk: g.walk, phrases: g.phrases, metrics: g.metrics, ...gen } };
writeFileSync(new URL("video/data/manifold.json", ROOT), JSON.stringify(out));
console.log("lobes", out.lobes.length, "necks", out.necks.length, "fidelity", mc.fidelity.toFixed(3),
  "orig track", orig.track.length, "gen track", gen.track.length, "gen seed", g.seed);
