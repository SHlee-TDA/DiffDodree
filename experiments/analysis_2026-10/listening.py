"""Two-stage samples from the conddrop checkpoint, similarity stats, and audio for listening."""
import io, json, subprocess, sys, wave
from pathlib import Path
import numpy as np
import torch
from gugakgen.data.musicxml import tokens_to_score, write_musicxml
from gugakgen.device import resolve_device
from gugakgen.workflows.generate import load_generation_bundle

torch.set_num_threads(2)
ckpt, cfg, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=False)  # never overwrite an earlier listening set
bd = load_generation_bundle(ckpt, config_path=cfg, project_root=".", runtime=resolve_device("cpu"))
s, ctx = bd.sampler, bd.context
CH1, CH4 = (1, 2, 3, 4), (25, 26, 27, 28)
fixed2 = tuple(n for n in range(1, 73) if n not in CH4)

def score_of(tokens):
    return tokens_to_score(tokens, ctx.vocabulary, piece_id=ctx.score.piece_id, part=ctx.score.part,
                           meter_id_to_value=ctx.meter_id_to_value)

def pitches(score, numbers):
    return [e.pitch if e.pitch is not None else -1 for n in numbers for e in score.measures[n - 1].events]

def lev(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[-1]

sim = lambda a, b: 1 - lev(a, b) / max(len(a), len(b))
windows = [tuple(range(k, k + 4)) for k in range(5, 54) if not set(range(k, k + 4)) & set(CH4)]  # 4-measure, 4/4 only

rows, kept = [], {}
for seed in range(260807, 260867):
    st1 = s.sample(seed=seed, config=bd.sampling_config, score_anchors=tuple(range(5, 73)),
                   family_anchors=bd.family_anchors, null_condition_measures=CH1, record_trace=False).tokens
    versions = {}
    for name, rel in (("true", ctx.hwanip_relations), ("none", None)):
        saved = s.hwanip_relations; s.hwanip_relations = rel
        try:
            versions[name] = s.sample(seed=seed + 100, config=bd.sampling_config, score_anchors=fixed2,
                                      family_anchors=bd.family_anchors, initial=st1,
                                      null_condition_measures=CH1 + CH4, record_trace=False).tokens
        finally:
            s.hwanip_relations = saved
    sc = {k: score_of(v) for k, v in versions.items()}
    new1 = pitches(sc["true"], CH1)
    r = {"seed": seed,
         "sim_true": sim(pitches(sc["true"], CH4), new1),
         "sim_none": sim(pitches(sc["none"], CH4), new1),
         "sim_other_windows": float(np.mean([sim(pitches(sc["true"], CH4), pitches(sc["true"], w)) for w in windows]))}
    rows.append(r); kept[seed] = sc

sims = {k: np.array([r[k] for r in rows]) for k in ("sim_true", "sim_none", "sim_other_windows")}
diff = sims["sim_true"] - sims["sim_none"]
summary = {k: round(float(v.mean()), 3) for k, v in sims.items()}
summary["true_minus_none_mean"] = round(float(diff.mean()), 3)
summary["seeds_true_gt_none"] = int((diff > 0).sum()); summary["seeds_true_lt_none"] = int((diff < 0).sum())
summary["seeds_true_gt_other_windows"] = int((sims["sim_true"] > sims["sim_other_windows"]).sum())
print(json.dumps(summary, ensure_ascii=False))

# Pick the median and the highest seed by sim_true (labelled as such, not cherry-picked silently).
order = sorted(rows, key=lambda r: r["sim_true"])
picks = {"중간": order[len(order) // 2], "가장비슷": order[-1]}

SR, BPM = 22050, 120  # simple synthesis, not the real tempo or timbre (same as earlier listening files)
def tone(freq, dur):
    n = max(1, int(dur * SR)); t = np.arange(n) / SR
    ph = 2 * np.pi * freq * np.cumsum(1 + 0.004 * np.sin(2 * np.pi * 5.2 * t)) / SR
    x = np.sin(ph) + .42 * np.sin(2 * ph) + .24 * np.sin(3 * ph) + .10 * np.sin(4 * ph)
    a, r_ = min(int(.018 * SR), n // 3 + 1), min(int(.055 * SR), n // 3 + 1)
    env = np.ones(n); env[:a] = np.linspace(0, 1, a); env[n - r_:] = np.linspace(1, 0, r_)
    return .5 * x * env / 1.8
def render(score, numbers, path):
    chunks = []
    for n in numbers:
        for e in score.measures[n - 1].events:
            dur = float(e.duration_q) * 60 / BPM
            if e.pitch is None: chunks.append(np.zeros(int(dur * SR)))
            else: chunks += [tone(440 * 2 ** ((e.pitch - 69) / 12), max(dur - .012, .04)), np.zeros(int(.012 * SR))]
    x = np.concatenate(chunks); pcm = (np.clip(x / (np.abs(x).max() or 1) * .92, -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
    subprocess.run(["lame", "--quiet", "-V4", "-", str(path)], input=buf.getvalue(), check=True)

orig = ctx.score
render(orig, CH1, out / "00_원곡_제1장.mp3")
for label, r in picks.items():
    sc = kept[r["seed"]]; tag = f"{label}_seed{r['seed']}"
    render(sc["true"], CH1, out / f"{tag}_1_새제1장.mp3")
    render(sc["true"], CH4, out / f"{tag}_2_제4장_환입관계있음.mp3")
    render(sc["none"], CH4, out / f"{tag}_3_제4장_환입관계없음.mp3")
    render(sc["true"], range(1, 73), out / f"{tag}_4_전곡_환입관계있음.mp3")
    write_musicxml(sc["true"], out / f"{tag}_전곡_환입관계있음.musicxml")
    print(label, r)
(out / "similarity.json").write_text(json.dumps({"summary": summary, "picks": picks, "rows": rows}, indent=1, ensure_ascii=False))
