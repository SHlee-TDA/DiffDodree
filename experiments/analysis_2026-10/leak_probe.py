"""Probe how measure-level conditions let the denoiser recall original measures."""
import sys, collections
from pathlib import Path
import numpy as np
from gugakgen.device import resolve_device
from gugakgen.workflows.generate import load_generation_bundle

root = Path(sys.argv[1])
for ck in sys.argv[2:]:
    bd = load_generation_bundle(Path(ck), project_root=root, runtime=resolve_device("cpu"))
    s, v, t = bd.sampler, bd.context.vocabulary, bd.context.tokens
    T, K = t.num_measures, t.pitch.shape[1]
    key = [(int(t.chapter[i]), int(t.meter[i]), int(t.position[i])) for i in range(T)]
    cnt = collections.Counter(key)
    sigs = [(tuple(t.pitch[i]), tuple(t.duration[i])) for i in range(T)]
    # how many distinct original measures share each key
    distinct = {k: len({sigs[i] for i in range(T) if key[i] == k}) for k in cnt}
    uniq = [distinct[key[i]] == 1 for i in range(T)]
    fam_anchor = np.full(T, v.family_mask_id)
    for m, f in bd.family_anchors.items():
        fam_anchor[int(m) - 1] = v.encode_family(f)

    def recall(chapter=None, position=None, family=None, anchor=None):
        P = np.full_like(t.pitch, v.pitch_mask_id); D = np.full_like(t.duration, v.duration_mask_id)
        ch_saved, pos_saved = s.template.chapter.copy(), s.template.position.copy()
        if chapter is not None: s.template.chapter[:] = chapter
        if position is not None: s.template.position[:] = position
        pl, dl, _ = s._predict(P, D, fam_anchor if family is None else family,
                               np.zeros(T, bool) if anchor is None else anchor)
        s.template.chapter[:] = ch_saved; s.template.position[:] = pos_saved
        pred = [(tuple(pl[i].argmax(-1)), tuple(dl[i].argmax(-1))) for i in range(T)]
        lp = pl - np.logaddexp.reduce(pl, -1, keepdims=True); ld = dl - np.logaddexp.reduce(dl, -1, keepdims=True)
        prob = [float(np.exp(np.mean([lp[i, j, t.pitch[i, j]] + ld[i, j, t.duration[i, j]] for j in range(K)]))) for i in range(T)]
        exact = np.array([pred[i] == sigs[i] for i in range(T)])
        return exact, np.array(prob)

    print(f"\n### {Path(ck).name}  keys={len(cnt)}  measures whose (chapter,meter,pos) key maps to one original measure: {sum(uniq)}/{T}")
    print("  ch1 keys:", [key[i] for i in range(4)], " ch4 keys:", [key[i] for i in range(24, 28)])
    e, p = recall()
    u = np.array(uniq)
    print(f"  all-masked argmax == original: unique-key {e[u].mean():.3f} (n={u.sum()}) | shared-key {e[~u].mean():.3f} (n={(~u).sum()})")
    print(f"  ch1 per-slot p(orig) {np.round(p[:4],3).tolist()}  ch4 {np.round(p[24:28],3).tolist()}")
    # ablations on chapter 1
    ch = t.chapter.copy(); ch[:4] = 2  # pretend chapter 1 measures are chapter 3
    e2, p2 = recall(chapter=ch); print(f"  ch1 relabelled as chapter 3 -> ch1 exact {e2[:4].mean():.2f}, p(orig) {np.round(p2[:4],3).tolist()}")
    pos = t.position.copy(); pos[:4] = pos[:4][::-1]
    e3, p3 = recall(position=pos); print(f"  ch1 position buckets reversed -> ch1 exact {e3[:4].mean():.2f}, p(orig) {np.round(p3[:4],3).tolist()}")
    fam = np.full(T, v.family_mask_id)
    e4, p4 = recall(family=fam); print(f"  no family labels at all -> ch1 exact {e4[:4].mean():.2f}, all measures exact {e4.mean():.3f}")
