"""Re-evaluate 1장–4장 similarity with the project's DTW measure distance."""
import json, sys
from pathlib import Path
import numpy as np, torch
from gugakgen.structure.measure_distance import measure_events as events_from_tokens
from gugakgen.structure.measure_distance import passage_distance as _passage_distance


def passage_distance(a, b, rel):
    return _passage_distance(a, b, relative_pitch=rel)
from gugakgen.device import resolve_device
from gugakgen.workflows.generate import load_generation_bundle
torch.set_num_threads(2)
bd = load_generation_bundle(sys.argv[1], config_path=sys.argv[2], project_root=".", runtime=resolve_device("cpu"))
s, ctx, voc = bd.sampler, bd.context, bd.context.vocabulary
out = Path(sys.argv[3])
if out.exists(): raise FileExistsError(out)
CH1, CH4 = (1, 2, 3, 4), (25, 26, 27, 28)
WINDOWS = [k for k in range(5, 53) if not set(range(k, k + 4)) & set(CH4)]  # 4/4 measures 5–55 only
ev = lambda t: [events_from_tokens(t, voc, i) for i in range(72)]
orig = ev(ctx.tokens)

# Rhythm check that actually changes something: swap two different durations in each measure.
def swap_diff(m):
    m = list(m)
    for i in range(len(m)):
        for j in range(i + 1, len(m)):
            if m[i][1] != m[j][1]:
                m[i], m[j] = (m[i][0], m[j][1]), (m[j][0], m[i][1]); return m
    return m
ch1 = [orig[n - 1] for n in CH1]
print("calibration, rhythm: swap two different durations per measure -> abs",
      round(passage_distance(ch1, [swap_diff(m) for m in ch1], False), 3), "rel", round(passage_distance(ch1, [swap_diff(m) for m in ch1], True), 3), flush=True)

def stats(e):
    four = [e[n - 1] for n in CH4]
    res = {}
    for rel, key in ((False, "abs"), (True, "rel")):
        d1 = passage_distance(four, [e[n - 1] for n in CH1], rel)
        others = np.array([passage_distance(four, [e[k + i - 1] for i in range(4)], rel) for k in WINDOWS])
        res[key] = {"d_ch1": d1, "d_others_mean": float(others.mean()),
                    "ch1_is_nearest": bool(d1 < others.min()),
                    "share_of_windows_farther": float((others > d1).mean())}
    return res

fixed2 = tuple(n for n in range(1, 73) if n not in CH4)
results = {"A_two_stage": {"true": [], "none": []}, "B_single_pass": {"true": [], "none": []}}
for seed in range(260807, 260867):
    st1 = s.sample(seed=seed, config=bd.sampling_config, score_anchors=tuple(range(5, 73)),
                   family_anchors=bd.family_anchors, null_condition_measures=CH1, record_trace=False).tokens
    for name, rel in (("true", ctx.hwanip_relations), ("none", None)):
        saved = s.hwanip_relations; s.hwanip_relations = rel
        try:
            a = s.sample(seed=seed + 100, config=bd.sampling_config, score_anchors=fixed2, family_anchors=bd.family_anchors,
                         initial=st1, null_condition_measures=CH1 + CH4, record_trace=False).tokens
            b = s.sample(seed=seed, config=bd.sampling_config, family_anchors=bd.family_anchors,
                         null_condition_measures=tuple(range(1, 73)), record_trace=False).tokens
        finally:
            s.hwanip_relations = saved
        results["A_two_stage"][name].append({"seed": seed, **stats(ev(a))})
        results["B_single_pass"][name].append({"seed": seed, **stats(ev(b))})

orig_stats = stats(orig)
print("original score:", json.dumps({k: {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in orig_stats.items()}))
summary = {}
for setup, by in results.items():
    for key in ("abs", "rel"):
        t = by["true"]; n = by["none"]
        dt = np.array([r[key]["d_ch1"] for r in t]); dn = np.array([r[key]["d_ch1"] for r in n])
        row = {
            "d(4장,1장) relation on": round(float(dt.mean()), 3),
            "d(4장,1장) relation off": round(float(dn.mean()), 3),
            "d(4장,other windows) relation on": round(float(np.mean([r[key]["d_others_mean"] for r in t])), 3),
            "seeds closer with relation (paired)": f"{int((dt < dn).sum())}/60",
            "1장 is nearest passage to 4장, on": f"{sum(r[key]['ch1_is_nearest'] for r in t)}/60",
            "1장 is nearest passage to 4장, off": f"{sum(r[key]['ch1_is_nearest'] for r in n)}/60",
            "share of windows farther than 1장, on": round(float(np.mean([r[key]["share_of_windows_farther"] for r in t])), 3),
            "share of windows farther than 1장, off": round(float(np.mean([r[key]["share_of_windows_farther"] for r in n])), 3),
        }
        summary[f"{setup}/{key}"] = row
        print(f"\n[{setup} / {key}]"); [print(f"  {k}: {v}") for k, v in row.items()]
out.write_text(json.dumps({"summary": summary, "original": orig_stats, "results": results}, indent=1, ensure_ascii=False))
