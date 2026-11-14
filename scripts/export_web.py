"""Export the trained models and the piece data for the GitHub Pages site (``site/``).

    python scripts/export_web.py            # writes site/models/*.onnx and site/data/*.json

* ``site/models/<arm>.onnx``: the denoiser of every checkpoint in
  ``results/checkpoints/20261113_1653_cpu_lambda_sweep`` (same-length λ sweep, training seed
  260807), with the medoid reference tables baked in. Checked against PyTorch on this machine.
* ``site/data/piece.json``: the 72 measures, types A–I, the phrase graph, the type walk
  (transition matrix and the return block), the grid alphabet and the manifold coordinates.
* ``site/data/gallery.json``: the first pieces each model generated in the paper run, with
  their per-piece metrics and manifold coordinates.
* ``site/data/sweep.json``: the λ sweep summary of figure 12 (15 training seeds).
* ``site/data/parity.json``: edit distances and projections computed here, which the site's
  JavaScript must reproduce (``node site/tests/parity.mjs``).

The manifold coordinates are a metric MDS (SMACOF, started from classical MDS) of the
duration-weighted edit distance between the 72 measures. A new measure is placed by minimising
the stress against its ``K_NEAREST`` nearest original measures (``project``); the site uses the
same rule, so generated pieces can be drawn on the same map.
"""

from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gugakgen.chart.diffusion import GridDenoiser, Tables  # noqa: E402
from gugakgen.chart.coords import Scale, to_chart  # noqa: E402
from gugakgen.chart.edit_distance import distance, distance_matrix  # noqa: E402
from gugakgen.chart.evaluation import as_phrase  # noqa: E402
from gugakgen.workflows import chart_diffusion_experiment as v1  # noqa: E402

RUN = "20261113_1653_cpu_lambda_sweep"
CONFIG = ROOT / "configs/chart_diffusion_v1_lambda_sweep.yaml"
CHECKPOINTS = ROOT / "results/checkpoints" / RUN
SITE = ROOT / "site"
KEYS = ["tokens", "length", "meter", "type", "prev_type", "next_type", "prev_b", "next_b"]
TYPE_NAMES = "ABCDEFGHI"
K_NEAREST = 8
GALLERY_PIECES = 12


# ----------------------------------------------------------------------------------------------
# Manifold coordinates


def smacof(D: np.ndarray, iterations: int = 1000, tol: float = 1e-10) -> np.ndarray:
    n = len(D)
    J = np.eye(n) - 1.0 / n
    w, V = np.linalg.eigh(-0.5 * J @ (D**2) @ J)
    order = np.argsort(w)[::-1][:2]
    X = V[:, order] * np.sqrt(np.maximum(w[order], 0))
    previous = np.inf
    for _ in range(iterations):
        E = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
        with np.errstate(divide="ignore", invalid="ignore"):
            B = np.where(E > 1e-12, -D / E, 0.0)
        np.fill_diagonal(B, 0.0)
        np.fill_diagonal(B, -B.sum(1))
        X = B @ X / n
        stress = float(((np.triu(E - D, 1)) ** 2).sum())
        if previous - stress < tol:
            break
        previous = stress
    X = X - X.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)  # principal axis horizontal
    X = X @ Vt.T
    if X[0, 0] > 0:  # measure 1 on the left
        X[:, 0] *= -1
    if X[:, 1].max() < -X[:, 1].min():
        X[:, 1] *= -1
    return X


def project(d: np.ndarray, X: np.ndarray, iterations: int = 100) -> list[float]:
    """Place a new measure with distances ``d`` to the originals (local stress, Guttman updates).

    The anchors are the ``K_NEAREST`` nearest *distinct positions*: repeated measures share a
    position, and anchors on only two positions would leave two mirror-image solutions.
    """

    nearest = int(np.argmin(d))
    if d[nearest] < 1e-9:
        return [float(X[nearest, 0]), float(X[nearest, 1])]
    idx: list[int] = []
    for i in np.argsort(d, kind="stable"):
        if all(np.abs(X[i] - X[j]).max() > 1e-9 for j in idx):
            idx.append(int(i))
        if len(idx) == K_NEAREST:
            break
    w = 1.0 / d[idx] ** 2
    P, r = X[idx], d[idx]
    y = (w[:, None] * P).sum(0) / w.sum()
    for _ in range(iterations):
        diff = y - P
        norm = np.sqrt((diff**2).sum(1))
        norm = np.where(norm < 1e-12, 1e-12, norm)
        y = (w[:, None] * (P + r[:, None] * diff / norm[:, None])).sum(0) / w.sum()
    return [float(y[0]), float(y[1])]


def lobes(typology) -> dict:
    """The phrase manifold drawn as lobes and necks (site figure; paper fig. 4 made from data).

    * ``sites``: SMACOF MDS of the distances between the nine medoids, so similar types sit close;
    * ``contact[a][b]``: how many of the 72 measures have medoids ``a`` and ``b`` as their two
      nearest medoids (``Typology.contact_pairs`` counted per measure). A neck joins ``a`` and
      ``b`` when this is positive; transitions between types without a neck are jumps;
    * ``visits``: measures of each type in the original (lobe size);
    * ``distance``: the 9 × 9 edit distances between medoids, from which the site lays the lobes out.

    These numbers only draw the figure; the analysis uses the types, the transition counts and the
    set of contact pairs, not the positions or the per-pair counts.
    """

    K = typology.k
    medoids = [tuple(m) for m in typology.medoids]
    contact = np.zeros((K, K), dtype=int)
    for phrase in typology.phrases:
        d = np.array(typology.medoid_distances(phrase))
        order = np.argsort(d, kind="stable")
        for other in order[1:]:
            if d[other] > d[order[1]] + 1e-9:
                break
            a, b = sorted((int(order[0]), int(other)))
            contact[a, b] += 1
            contact[b, a] += 1
    D = distance_matrix(medoids)
    return {"sites": np.round(smacof(D), 6).tolist(), "distance": np.round(D, 6).tolist(), "contact": contact.tolist(),
            "visits": np.bincount(typology.labels, minlength=K).tolist()}


def chart(phrase, medoid, scale: Scale) -> list[list]:
    """Local coordinates of each event around the medoid (``coords.to_chart``): [slot, inserted, eta]."""

    return [[e.slot, int(e.inserted), e.eta] for e in to_chart(tuple(phrase), tuple(medoid), scale)]


# ----------------------------------------------------------------------------------------------
# Models


class Exported(torch.nn.Module):
    """GridDenoiser with the medoid tables as buffers, taking the batch as positional tensors."""

    def __init__(self, model: GridDenoiser, tables: Tables) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("ref", tables.ref)
        self.register_buffer("onset", tables.onset)
        self.register_buffer("cost", tables.cost)

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        return self.model(dict(zip(KEYS, tensors, strict=True)), self)  # self has .ref/.onset/.cost


def export_models(setup: dict) -> list[dict]:
    import onnxruntime as ort

    layout, config = setup["layout"], setup["config"]
    rows = {k: torch.as_tensor(v) for k, v in setup["rows"].items()}
    generator = torch.Generator().manual_seed(0)
    masked = torch.rand(rows["tokens"].shape, generator=generator) < 0.5
    rows["tokens"] = torch.where(masked, torch.full_like(rows["tokens"], setup["grid"].mask), rows["tokens"])
    args = tuple(rows[k] for k in KEYS)
    lambdas = {arm: float(spec["lambda_graph"]) for arm, spec in config["arms"].items()}
    out = []
    for path in sorted(CHECKPOINTS.glob("full_*_s*.pt")):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        model = GridDenoiser(layout, **config["model"])
        model.load_state_dict(saved["state_dict"])
        model.eval()
        wrapped = Exported(model, Tables(setup["typology"], layout, torch.device("cpu"))).eval()
        target = SITE / "models" / f"{saved['arm']}.onnx"
        with torch.no_grad():
            reference = wrapped(*args).numpy()
        # Exported with autograd on: under no_grad, nn.TransformerEncoderLayer takes a fused fast
        # path (aten::_transformer_encoder_layer_fwd) that has no ONNX symbolic.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(
                wrapped, args, str(target), input_names=KEYS, output_names=["log_probs"],
                dynamic_axes={**{k: {0: "batch"} for k in KEYS}, "log_probs": {0: "batch"}},
                opset_version=17, dynamo=False,
            )
        session = ort.InferenceSession(str(target))
        got = session.run(None, {k: rows[k].numpy().astype(np.int64) for k in KEYS})[0]
        # compare probabilities: log-probabilities of tokens with p < 1e-30 differ more in float32
        error = float(np.abs(np.exp(got) - np.exp(reference)).max())
        if error > 1e-5:
            raise RuntimeError(f"ONNX output differs from PyTorch by {error} for {path.name}")
        out.append({
            "arm": saved["arm"], "lambda": lambdas[saved["arm"]], "seed": int(saved["seed"]),
            "steps": int(saved["steps"]), "file": f"models/{target.name}",
            "checkpoint": path.name, "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "onnx_max_prob_error": error,
        })
        print(f"{target.relative_to(ROOT)}  λ={lambdas[saved['arm']]:g}  max|Δ|={error:.1e}")
    return sorted(out, key=lambda m: m["lambda"])


def check_reproduction(setup: dict, report: dict) -> None:
    """The λ = 0 checkpoint regenerates the run's first piece exactly (CPU)."""

    unit = next(u for u in report["full"] if u["arm"] == "diffusion" and u["seed"] == 260807)
    saved = torch.load(CHECKPOINTS / unit["checkpoint"], map_location="cpu", weights_only=False)
    model = GridDenoiser(setup["layout"], **setup["config"]["model"])
    model.load_state_dict(saved["state_dict"])
    model.eval()
    tables = Tables(setup["typology"], setup["layout"], torch.device("cpu"))
    piece = unit["generation"]["pieces"][0]
    phrases, walk, _ = v1.generate_piece(model, tables, setup, piece["seed"])
    same = walk == piece["walk"] and [as_phrase(p) for p in piece["phrases"]] == [tuple(p) for p in phrases]
    print("checkpoint regenerates the paper's piece", piece["seed"], ":", "yes" if same else "NO (CPU/library differences)")


# ----------------------------------------------------------------------------------------------


def dump(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else list(o))


def events(phrase) -> list[list]:
    return [[None if p is None else int(p), float(d)] for p, d in phrase]


def main() -> None:
    setup = v1.load_setup(CONFIG, ROOT)
    piece_config = v1.load_yaml(ROOT / setup["config"]["piece_config"])
    typology, grid, layout, walk = setup["typology"], setup["grid"], setup["layout"], setup["walk"]
    phrases = [tuple(p) for p in typology.phrases]
    D = distance_matrix(phrases)
    X = np.round(smacof(D), 6)  # the site gets these rounded values; project with the same ones
    report = json.loads((ROOT / "results/runs" / RUN / "report_slim.json").read_text(encoding="utf-8"))

    (SITE / "models").mkdir(parents=True, exist_ok=True)
    (SITE / "data").mkdir(parents=True, exist_ok=True)
    models = export_models(setup)
    check_reproduction(setup, report)

    chapters = piece_config["chapters"]
    piece = {
        "title": {"ko": piece_config["piece"]["title_ko"], "en": piece_config["piece"]["title_en"],
                  "part": piece_config["piece"]["part"]},
        "yulmyung": {str(k): v for k, v in piece_config["pitch"]["midi_to_yulmyung"].items()},
        "chapters": [{"chapter": c["chapter"], "measures": c["measures"], "meter": c["meter"]} for c in chapters],
        "type_names": TYPE_NAMES[: typology.k],
        "phrases": [events(p) for p in phrases],
        "labels": list(typology.labels),
        "measure_chapters": list(typology.chapters),
        "medoid_phrases": list(typology.medoid_phrases),
        "transition_counts": typology.transition_counts().tolist(),
        "returns": [list(pair) for pair in setup["returns"]],
        "lengths": setup["lengths"],
        "grid": {"pitches": list(grid.pitches), "cells_per_beat": grid.cells_per_beat, "on_rest": grid.on_rest,
                 "hold": grid.hold, "mask": grid.mask, "num_outputs": grid.num_outputs, "num_states": grid.num_states},
        "layout": {"num_types": layout.num_types, "meters": list(layout.meters), "max_cells": layout.max_cells,
                   "type_start": layout.type_start, "type_end": layout.type_end, "type_null": layout.type_null,
                   "boundary_mask": layout.boundary_mask, "boundary_none": layout.boundary_none,
                   "boundary_null": layout.boundary_null},
        "walk": {"P": walk.P.tolist(), "length": walk.length, "start": walk.start, "end": walk.end,
                 "block": list(walk.block) if walk.block else None,
                 "blocks": [list(b) for b in walk.blocks], "block_weights": walk.block_weights.tolist()},
        "sampling": {k: setup["config"]["sampling"][k] for k in ("steps", "temperature", "rejection_attempts")},
        "manifold": {"coords": X.tolist(), "k_nearest": K_NEAREST, "iterations": 100,
                     "method": "SMACOF metric MDS of the edit distance; new measures by local stress against the k nearest distinct positions"},
        "lobes": lobes(typology),
        "edit_distance": {"rest_pitch_cost": 6.0, "indel_weight": 3.0, "fragment_surcharge": 0.5, "max_fragments": 3},
    }
    (SITE / "data/piece.json").write_text(dump(piece), encoding="utf-8")

    gallery = {"run": RUN, "seed": 260807, "models": models, "pieces": {}}
    for unit in report["full"]:
        if unit["seed"] != 260807:
            continue
        rows = []
        for item, metrics in list(zip(unit["generation"]["pieces"], unit["generation"]["per_piece"], strict=True))[:GALLERY_PIECES]:
            ph = [as_phrase(p) for p in item["phrases"]]
            coords = [project(np.array([distance(p, o) for o in phrases]), X) for p in ph]
            rows.append({"seed": item["seed"], "walk": item["walk"], "phrases": [events(p) for p in ph],
                         "coords": np.round(coords, 5).tolist(),
                         "metrics": {k: round(float(v), 4) for k, v in metrics.items()}})
        gallery["pieces"][unit["arm"]] = rows
    (SITE / "data/gallery.json").write_text(dump(gallery), encoding="utf-8")

    stats = json.loads((ROOT / "figures/output/f15_lambda_sweep_stats.json").read_text(encoding="utf-8"))
    (SITE / "data/sweep.json").write_text(dump(stats), encoding="utf-8")

    sample = gallery["pieces"][models[-1]["arm"]][0]
    scale = Scale.from_phrases(phrases)
    tests = [as_phrase(p) for p in sample["phrases"][:6]]
    parity = {
        "pairs": [{"a": events(a), "b": events(b), "distance": distance(a, b)}
                  for a, b in [(phrases[0], phrases[5]), (phrases[10], phrases[60]), (tests[2], phrases[30]),
                               (phrases[55], phrases[70]), (tests[4], tests[5])]],
        "projections": [{"phrase": events(p), "coords": project(np.array([distance(p, o) for o in phrases]), X)}
                        for p in tests],
        "charts": [{"phrase": events(p), "type": int(c), "chart": chart(p, typology.medoids[c], scale)}
                   for p, c in [*zip(phrases, typology.labels),
                                *zip([as_phrase(q) for q in sample["phrases"]], sample["walk"])]],
    }
    (SITE / "data/parity.json").write_text(dump(parity), encoding="utf-8")
    for name in ("piece", "gallery", "sweep", "parity"):
        print(f"site/data/{name}.json  {(SITE / f'data/{name}.json').stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
