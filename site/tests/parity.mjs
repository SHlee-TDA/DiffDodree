// node site/tests/parity.mjs — the JavaScript port reproduces the Python numbers in data/parity.json.
import { readFileSync } from "node:fs";
import { editDistance, project, TypeWalk, rng, toMusicXML, decode, isMember, toChart } from "../js/core.js";

const here = new URL("..", import.meta.url);
const read = (p) => JSON.parse(readFileSync(new URL(p, here), "utf8"));
const parity = read("data/parity.json"), piece = read("data/piece.json"), gallery = read("data/gallery.json");
let failed = 0;
const check = (ok, msg) => { if (!ok) { failed++; console.error("FAIL", msg); } };

for (const { a, b, distance } of parity.pairs) {
  const got = editDistance(a, b);
  check(Math.abs(got - distance) < 1e-12, `edit distance ${got} vs ${distance}`);
}
for (const { phrase, coords } of parity.projections) {
  const got = project(phrase, piece.phrases, piece.manifold.coords, piece.manifold.k_nearest, piece.manifold.iterations);
  check(Math.hypot(got[0] - coords[0], got[1] - coords[1]) < 1e-6, `projection ${got} vs ${coords}`);
}
for (const [arm, pieces] of Object.entries(gallery.pieces)) {
  for (const p of pieces.slice(0, 2)) {
    p.phrases.forEach((ph, t) => {
      const c = project(ph, piece.phrases, piece.manifold.coords, piece.manifold.k_nearest, piece.manifold.iterations);
      check(Math.hypot(c[0] - p.coords[t][0], c[1] - p.coords[t][1]) < 1e-4, `${arm} gallery coords ${t}`);
    });
  }
}
// local coordinates around the medoid (coords.to_chart)
const medoidsAll = piece.medoid_phrases.map((i) => piece.phrases[i]);
for (const { phrase, type, chart } of parity.charts ?? []) {
  const got = toChart(phrase, medoidsAll[type], piece.grid.pitches).map((e) => [e.slot, e.inserted ? 1 : 0, e.eta]);
  check(JSON.stringify(got) === JSON.stringify(chart), `chart ${JSON.stringify(got)} vs ${JSON.stringify(chart)}`);
}
check((parity.charts ?? []).length > 100, "chart cases present");
// medoids are members of their own type; every original measure belongs to its own type
const medoids = piece.medoid_phrases.map((i) => piece.phrases[i]);
piece.phrases.forEach((ph, t) => check(isMember(ph, piece.labels[t], medoids), `original ${t + 1} membership`));
// walks: right length, start/end, return block, only observed transitions
const walk = new TypeWalk(piece.walk);
const random = rng(7);
for (let n = 0; n < 200; n++) {
  const w = walk.sample(random);
  check(w.length === 72 && w[0] === piece.walk.start && w[71] === piece.walk.end, "walk ends");
  for (const [s, t] of piece.returns) check(w[s] === w[t], "walk return");
  for (let i = 0; i < 71; i++) check(piece.transition_counts[w[i]][w[i + 1]] > 0, `walk edge ${w[i]}->${w[i + 1]}`);
}
// grid round trip and MusicXML durations
const xml = toMusicXML(piece, piece.phrases, piece.labels);
check((xml.match(/<measure /g) || []).length === 72, "musicxml measures");
const durations = [...xml.matchAll(/<measure number="(\d+)">(.*?)<\/measure>/gs)].map(([, , body]) =>
  [...body.matchAll(/<duration>(\d+)<\/duration>/g)].reduce((s, m) => s + Number(m[1]), 0));
durations.forEach((d, t) => check(d === piece.lengths[t], `measure ${t + 1} has ${d} cells, expected ${piece.lengths[t]}`));
check(JSON.stringify(decode(piece.grid, [0, piece.grid.hold, piece.grid.on_rest])) === JSON.stringify([[piece.grid.pitches[0], 0.5], [null, 0.25]]), "decode");

console.log(failed ? `${failed} checks failed` : "parity: all checks passed");
process.exit(failed ? 1 : 0);
