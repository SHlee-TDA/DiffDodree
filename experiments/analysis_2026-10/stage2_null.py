"""Two-stage 환입 test on the conddrop checkpoint: null ch4 only vs null ch1+ch4 in stage 2."""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
from gugakgen.device import resolve_device
from gugakgen.structure.hwanip_relations import HwanipRelation, HwanipRelations
from gugakgen.workflows.generate import load_generation_bundle

torch.set_num_threads(2)
ckpt, cfg, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
bd = load_generation_bundle(ckpt, config_path=cfg, project_root=".", runtime=resolve_device("cpu"))
s, ctx = bd.sampler, bd.context
print("checkpoint", bd.checkpoint.sha256[:12], "null_conditions", bd.model.config.null_conditions, flush=True)
sig = lambda t: [(tuple(t.pitch[i].tolist()), tuple(t.duration[i].tolist())) for i in range(72)]
orig = sig(ctx.tokens); oset = set(orig)
CH1, CH4 = (1, 2, 3, 4), (25, 26, 27, 28)
first, fourth = [n - 1 for n in CH1], [n - 1 for n in CH4]
relations = {
    "true": ctx.hwanip_relations,
    "none": None,
    "negative": HwanipRelations(relations=(HwanipRelation(relation_id="neg", source_measures=(5, 8),
                                target_measures=(25, 28), relation_type="return_of"),), num_measures=72),
}
fixed2 = tuple(n for n in range(1, 73) if n not in CH4)
SEEDS = range(260807, 260867)

# Stage 1 is shared by every condition (same as the Colab protocol).
stage1 = {}
for seed in SEEDS:
    stage1[seed] = s.sample(seed=seed, config=bd.sampling_config, score_anchors=tuple(range(5, 73)),
                            family_anchors=bd.family_anchors, null_condition_measures=CH1, record_trace=False).tokens
novel1 = np.mean([sum(sig(t)[i] not in oset for i in first) for t in stage1.values()])
print(f"stage 1: ch1 novel measures mean {novel1:.2f}/4", flush=True)

results = {"checkpoint_sha256": bd.checkpoint.sha256, "stage_one_ch1_novel_mean": float(novel1), "conditions": {}}
for variant, nulls in (("null_ch4_only", CH4), ("null_ch1_and_ch4", CH1 + CH4)):
    for rel_name, rel in relations.items():
        t0 = time.time(); rows = []
        for seed in SEEDS:
            g1 = sig(stage1[seed])
            saved = s.hwanip_relations; s.hwanip_relations = rel
            try:
                t = s.sample(seed=seed + 100, config=bd.sampling_config, score_anchors=fixed2,
                             family_anchors=bd.family_anchors, initial=stage1[seed],
                             null_condition_measures=nulls, record_trace=False).tokens
            finally:
                s.hwanip_relations = saved
            g = sig(t)
            for a, b in zip(first, fourth):
                same_pitch = float(np.mean(np.array(g[b][0]) == np.array(g1[a][0])))
                rows.append((g[b] == g1[a], g[b] == orig[b], g[b] not in oset, same_pitch))
        arr = np.array(rows, dtype=float)
        res = {"ch4_equals_new_ch1": arr[:, 0].mean(), "ch4_equals_original": arr[:, 1].mean(),
               "ch4_novel": arr[:, 2].mean(), "slot_pitch_match_with_new_ch1": arr[:, 3].mean()}
        res = {k: round(float(v), 4) for k, v in res.items()}
        results["conditions"][f"{variant}/{rel_name}"] = res
        print(f"[{variant} / {rel_name}] {res}  ({time.time()-t0:.0f}s)", flush=True)
out.write_text(json.dumps(results, indent=1))
