"""Compare chapter 1/4 behaviour with and without the family-only anchor flag fix."""
import json, sys, time
from pathlib import Path
import numpy as np
from gugakgen.device import resolve_device
from gugakgen.sampling.maskgit import MaskGITSampler
from gugakgen.workflows.generate import load_generation_bundle

root, ckpt, n, out = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
bundle = load_generation_bundle(ckpt, project_root=root, runtime=resolve_device("cpu"))
s, tmpl = bundle.sampler, bundle.context.tokens
fixed = MaskGITSampler._anchor_arrays

def old_anchor(self, sa, fa):
    a, b, c, _ = fixed(self, sa, fa)
    return a, b, c, a | b

def sig(t, i):
    return (tuple(t.pitch[i].tolist()), tuple(t.duration[i].tolist()))

orig = [sig(tmpl, i) for i in range(72)]
origset = set(orig)
C1, C4 = range(0, 4), range(24, 28)
conds = {
    "A_old_flag": ((), True),
    "B_fixed": ((), False),
    "C_score_1_72": ((1, 72), False),
}
res = {}
for name, (sa, old) in conds.items():
    MaskGITSampler._anchor_arrays = old_anchor if old else fixed
    rows, t0 = [], time.time()
    for seed in range(260807, 260807 + n):
        tok = s.sample(seed=seed, config=bundle.sampling_config, score_anchors=sa,
                       family_anchors=bundle.family_anchors, record_trace=False).tokens
        g = [sig(tok, i) for i in range(72)]
        rows.append({
            "seed": seed,
            "m1_orig": g[0] == orig[0],
            "m72_orig": g[71] == orig[71],
            "g1_eq_g4": sum(g[a] == g[b] for a, b in zip(C1, C4)),
            "g1_eq_o1": sum(g[a] == orig[a] for a in C1),
            "g4_eq_o4": sum(g[b] == orig[b] for b in C4),
            "novel_return": sum(g[a] == g[b] and g[a] not in origset for a, b in zip(C1, C4)),
            "copy_rate": float(np.mean([g[i] == orig[i] for i in range(72)])),
        })
    res[name] = rows
    r = rows
    print(f"{name}: n={n} {time.time()-t0:.0f}s | m1=orig {np.mean([x['m1_orig'] for x in r]):.3f}"
          f" | m72=orig {np.mean([x['m72_orig'] for x in r]):.3f}"
          f" | 1장=4장 4/4 {sum(x['g1_eq_g4']==4 for x in r)}"
          f" | 1장=원곡1장 4/4 {sum(x['g1_eq_o1']==4 for x in r)}"
          f" | 4장=원곡4장 4/4 {sum(x['g4_eq_o4']==4 for x in r)}"
          f" | 1장=4장 mean {np.mean([x['g1_eq_g4'] for x in r]):.2f}/4"
          f" | novel return {sum(x['novel_return'] for x in r)}"
          f" | copy rate {np.mean([x['copy_rate'] for x in r]):.3f}", flush=True)
MaskGITSampler._anchor_arrays = fixed
out.write_text(json.dumps({"checkpoint": str(ckpt), "results": res}, indent=1))
