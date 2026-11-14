"""Collect the numbers and arrays the manuscript figures need (run 20261112_2112 + the original piece).

    python figures/extract_data.py [RUN_DIR]   # default: results/runs/20261112_2112

Writes figures/figure_data.json, read by check_tda_order.py and make_figures.py.
"""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "figures"))
from _style import RUNS, read_report
sys.path.insert(0, str(ROOT / "src"))
from gugakgen.workflows import chart_diffusion_experiment as v1
from gugakgen.chart.evaluation import as_phrase
from gugakgen.chart.edit_distance import distance

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else RUNS / "20261112_2112"
OUT = Path(__file__).resolve().parent / "figure_data.json"
setup = v1.load_setup(ROOT / "configs/chart_diffusion_v1.yaml", ROOT)
typ = setup["typology"]; ref = v1.build_reference(setup)
phr = [tuple(p) for p in typ.phrases]
med = [tuple(m) for m in typ.medoids]
labels = list(typ.labels)
data = {"labels": labels, "chapters": list(typ.chapters),
        "phrases": [[[p, float(d)] for p, d in ph] for ph in phr],
        "medoid_measures": [phr.index(m) + 1 for m in med]}
data["counts"] = typ.transition_counts().tolist()
# contact pairs (C2) and transitions crossing contacts
contact = sorted(tuple(int(x) for x in c) for c in typ.contact_pairs())
data["contact_pairs"] = contact
data["contact_summary"] = typ.contact_summary()
summary = typ.contact_summary()
data["transitions_crossing"] = [summary["transitions_across_contact"], summary["transitions"]]
data["medoid_distance"] = [[distance(a, b) for b in med] for a in med]
# sampled walks
rng = np.random.default_rng(20261113)
data["walks"] = [[int(x) for x in setup["walk"].sample(rng)] for _ in range(3)]
# run results
ev = json.loads((RUN / "evaluation.json").read_text()); rp = read_report(RUN)
res = defaultdict(dict)
for m in ev["models"]:
    res[m["arm"]][m["seed"]] = {k: m["per_piece"][k]["mean"] for k in m["per_piece"]} | {k: m["set"][k]["value"] for k in m["set"]}
    res[m["arm"]][m["seed"]]["first_pass_ci"] = m["per_piece"]["first_pass_membership"]["ci95"]
data["results"] = res
data["z_novel"] = ev["copy_z_novel_combined"]; data["z_all"] = ev["copy_z_combined"]
data["reference"] = ev["reference"]
data["selected_steps"] = {a: c["selected"] for a, c in rp["selected_steps"].items()}
data["heldout_curve"] = {a: c["heldout_elbo_by_step"] for a, c in rp["selected_steps"].items()}
data["heldout_baseline"] = rp["heldout_baseline"]
train = defaultdict(lambda: defaultdict(list))
for u in rp["heldout"]:
    for p in u["curve"]:
        train[u["arm"]][p["step"]].append(p["train_elbo"])
data["train_curve"] = {a: {s: float(np.mean(v)) for s, v in d.items()} for a, d in train.items()}
# example pieces
for u in rp["full"]:
    if u["seed"] == 260807:
        data[f"example_{u['arm']}"] = {"walk": u["generation"]["pieces"][0]["walk"],
                                       "phrases": u["generation"]["pieces"][0]["phrases"]}
        data[f"pieces_{u['arm']}"] = [{"seed": pc["seed"], "walk": pc["walk"], "phrases": pc["phrases"]}
                                      for pc in u["generation"]["pieces"]]
# nearest-training distances for the copy test (generated novel vs held-out), per arm, seed 260807
content = set(phr)
near = {}
for arm in ("diffusion", "diffusion_graph"):
    gen, held_d = [], []
    for u in rp["copytest"]:
        if u["arm"] != arm or u["seed"] != 260807: continue
        held = {m - 1 for m in u["heldout_measures"]}
        train_set = sorted({phr[t] for t in range(72) if t not in held}, key=repr)
        cache = {}
        for piece in u["pieces"]:
            for t in ref.free:
                g = as_phrase(piece["phrases"][t])
                if g not in cache: cache[g] = min(distance(g, x) for x in train_set)
                gen.append((cache[g], g in set(train_set)))
        held_d += [min(distance(phr[t], x) for x in train_set) for t in sorted(held)]
    near[arm] = {"generated": [d for d, _ in gen], "generated_is_train": [bool(c) for _, c in gen], "heldout": held_d}
data["copy_distances"] = near
# copy at return positions
data["return_copy"] = {}
for arm in ("diffusion", "diffusion_graph"):
    a = defaultdict(list)
    for u in rp["full"]:
        if u["arm"] != arm: continue
        for piece in u["generation"]["pieces"]:
            for t in ref.free:
                a["ret" if t < 4 else "other"].append(as_phrase(piece["phrases"][t]) in content)
    data["return_copy"][arm] = {k: float(np.mean(v)) for k, v in a.items()}
# surface types read back from the generated notes (same walk for both arms, sample seed 260807)
# pick the piece (training seed 1) whose read-back transitions disagree most with its walk at λ = 0
best = None
for k, (a, b) in enumerate(zip(data["pieces_diffusion"], data["pieces_diffusion_graph"])):
    w = a["walk"]
    ra = ref.surface_types([as_phrase(m) for m in a["phrases"]]); rb = ref.surface_types([as_phrase(m) for m in b["phrases"]])
    ta = sum((ra[i], ra[i + 1]) != (w[i], w[i + 1]) for i in range(71)); tb = sum((rb[i], rb[i + 1]) != (w[i], w[i + 1]) for i in range(71))
    if best is None or (ta, -tb) > best[0]:
        best = ((ta, -tb), {"walk": w, "diffusion": ra, "diffusion_graph": rb, "piece_index": k, "seed": a["seed"]})
data["surface_example"] = best[1]
cnt = Counter(phr)
data["repeat_rate"] = float(np.mean([cnt[p] > 1 for p in phr]))
data["dist_to_own_medoid"] = {"original": float(np.mean([distance(phr[t], med[labels[t]]) for t in ref.free]))}
data["equal_own_medoid"] = {"original": float(np.mean([phr[t] == med[labels[t]] for t in ref.free]))}
for arm in ("diffusion", "diffusion_graph"):
    ds, eq, cache = [], [], {}
    for u in rp["full"]:
        if u["arm"] != arm: continue
        for piece in u["generation"]["pieces"]:
            for t in ref.free:
                g, y = as_phrase(piece["phrases"][t]), piece["walk"][t]
                if (g, y) not in cache: cache[(g, y)] = distance(g, med[y])
                ds.append(cache[(g, y)]); eq.append(g == med[y])
    data["dist_to_own_medoid"][arm] = float(np.mean(ds)); data["equal_own_medoid"][arm] = float(np.mean(eq))
OUT.write_text(json.dumps(data, ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
print("ok", OUT, data["transitions_crossing"], len(contact), data["selected_steps"], data["return_copy"])
