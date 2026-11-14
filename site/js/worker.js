// Module worker: runs the denoiser with onnxruntime-web and the reverse process from core.js.
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.wasm.min.mjs";
import { generatePiece } from "./core.js";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
ort.env.wasm.numThreads = 1; // GitHub Pages is not cross-origin isolated

let piece = null;
const sessions = new Map();

async function session(url) {
  if (!sessions.has(url)) sessions.set(url, ort.InferenceSession.create(url, { executionProviders: ["wasm"] }));
  return sessions.get(url);
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "init") {
      piece = data.piece;
      return;
    }
    if (data.type === "generate") {
      const s = await session(data.modelUrl);
      const size = piece.layout.max_cells;
      let calls = 0;
      const predict = async (rows, count) => {
        const feeds = {};
        for (const [k, v] of Object.entries(rows)) feeds[k] = new ort.Tensor("int64", v, k === "tokens" ? [count, size] : [count]);
        calls++;
        return (await s.run(feeds)).log_probs.data;
      };
      const t0 = performance.now();
      const result = await generatePiece(piece, predict, data.seed, {
        ...data.options,
        onProgress: (p) => self.postMessage({ type: "progress", id: data.id, ...p }),
      });
      self.postMessage({ type: "done", id: data.id, result, ms: performance.now() - t0, calls });
    }
  } catch (err) {
    self.postMessage({ type: "error", id: data.id, message: String(err?.message ?? err) });
  }
};
