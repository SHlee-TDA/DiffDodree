"""Figure 12 of the paper (file f15_lambda_sweep.png): metrics against the weight of the structure loss, pooled over runs that share
code and settings except the training seeds. Mean ± 95% CI over training seeds, paired tests vs λ = 0.

Usage: python figures/make_sweep_figure.py [<out png> <out stats json> <run dir> ...]  (default: the paper runs)
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import OUTPUT, RUNS, korean_font, read_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": korean_font(), "font.size": 9, "axes.unicode_minus": False})
if len(sys.argv) > 1:
    OUT, STATS, RUNS = Path(sys.argv[1]), Path(sys.argv[2]), [Path(p) for p in sys.argv[3:]]
else:  # the paper: the 3-seed and 12-seed same-length sweeps pooled
    OUT, STATS = OUTPUT / "f15_lambda_sweep.png", OUTPUT / "f15_lambda_sweep_stats.json"
    RUNS = [RUNS / "20261113_1653_cpu_lambda_sweep", RUNS / "20261113_2206_cpu_lambda_sweep_seeds12"]
INK, POINT, BAND, REF, GOOD = "#1d2433", "#9aa3b2", "#c9d3e6", "#7a8291", "#dcefe0"

lam, per = {}, defaultdict(lambda: defaultdict(dict))  # per[metric][arm][seed]
ref_vals, base = {}, []
for run in RUNS:
    ev = json.loads((run / "evaluation.json").read_text(encoding="utf-8"))
    rep = read_report(run)
    lam.update(ev["lambda"]); ref_vals = ev["reference"]; base += list(rep["heldout_baseline"].values())
    for m in ev["models"]:
        for key in ("first_pass_membership", "exact_copy_rate", "ph_cycles", "chapter_cadence_ok", "interval_jsd", "authenticity"):
            v = m["per_piece"].get(key, m["set"].get(key))
            per[key][m["arm"]][int(m["seed"])] = v["mean"] if "mean" in v else v["value"]
    for arm, d in ev["copy_z_novel_combined"].items():
        for seed, z in d.items(): per["z_novel"][arm][int(seed)] = z
    folds = defaultdict(list)
    for u in rep["heldout"]:
        step = rep["selected_steps"][u["arm"]]["selected"]
        pt = [p for p in u["curve"] if p["step"] == step]
        if pt: folds[(u["arm"], int(u["seed"]))].append(pt[0]["heldout_elbo"])
    for (arm, seed), v in folds.items(): per["elbo"][arm][seed] = float(np.mean(v))

arms = sorted(lam, key=lam.get); zero = arms[0]
seeds = sorted(set.intersection(*[set(per["first_pass_membership"][a]) for a in arms]))
n = len(seeds)


def summary(key):
    out = []
    for a in arms:
        v = np.array([per[key][a][s] for s in seeds])
        half = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        out.append((v.mean(), half, v))
    # paired comparison with λ = 0 (Wilcoxon signed-rank), Holm correction over the other weights
    base_v = np.array([per[key][zero][s] for s in seeds])
    raw = []
    for a in arms[1:]:
        v = np.array([per[key][a][s] for s in seeds])
        try: raw.append(stats.wilcoxon(v, base_v).pvalue)
        except ValueError: raw.append(1.0)
    order = np.argsort(raw); holm = np.ones(len(raw)); run_max = 0
    for rank, i in enumerate(order):
        run_max = max(run_max, min(1.0, raw[i] * (len(raw) - rank))); holm[i] = run_max
    xs = np.repeat(np.array([lam[a] for a in arms]), n); ys = np.concatenate([o[2] for o in out])
    rho, p_trend = stats.spearmanr(xs, ys)
    return out, [1.0] + list(holm), rho, p_trend


PANELS = [
    ("first_pass_membership", "(가) 주어진 유형을 따르는 마디의 비율", "높을수록 유형 구조를 잘 지킴", "비율", None, None, None),
    ("z_novel", "(나) 복제 검정 Z", "0에 가까울수록 새 선율, -2보다 작으면 복제로 판정", "Z", 0, None, (-2, 2)),
    ("exact_copy_rate", "(다) 원곡과 음까지 똑같은 마디의 비율", "낮을수록 원곡을 덜 되풀이함", "비율", None, None, None),
    ("ph_cycles", "(라) 음표 연결 순환의 수", "원곡(9개)에 가까울수록 선행 연구의 기하에 가까움", "개수", None, None, None),
    ("chapter_cadence_ok", "(마) 장 끝음이 원곡의 장 끝음에 드는 비율", "높을수록 종지를 잘 재현함", "비율", None, None, None),
    ("elbo", "(바) 학습에 쓰지 않은 마디를 설명하는 정도", "낮을수록 좋음(보류 ELBO)", "ELBO", None, None, None),
]
refs = {"ph_cycles": (ref_vals["barcode_bars"], "원곡 9개"), "exact_copy_rate": (ref_vals["repeat_rate_free"], "원곡 자체의 반복률 0.43"),
        "elbo": (float(np.mean(base)), "문맥 없는 기준값")}

fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.9))
x = np.arange(len(arms))
table = {}
for ax, (key, title, sub, ylab, zline, _, band) in zip(axes.ravel(), PANELS):
    out, holm, rho, p_trend = summary(key)
    means = np.array([o[0] for o in out]); halves = np.array([o[1] for o in out])
    if band: ax.axhspan(*band, color=GOOD, zorder=0)
    rng = np.random.default_rng(0)
    for i, o in enumerate(out):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(o[2])), o[2], s=9, color=POINT, zorder=2, linewidths=0)
    ax.fill_between(x, means - halves, means + halves, color=BAND, alpha=0.8, zorder=1, linewidth=0)
    ax.plot(x, means, color=INK, lw=1.5, marker="o", ms=3.5, zorder=4)
    if key in refs:
        val, lab = refs[key]; ax.axhline(val, color=REF, ls="--", lw=1, zorder=1)
        ax.annotate(lab, (x[-1], val), xytext=(0, 3), textcoords="offset points", ha="right", va="bottom", fontsize=7.5, color=REF)
    if zline is not None: ax.axhline(zline, color=REF, ls=":", lw=0.8)
    top = max(np.max(o[2]) for o in out)
    span = top - min(np.min(o[2]) for o in out)
    for i, p in enumerate(holm):
        if i and p < 0.05: ax.text(i, means[i] + halves[i] + 0.06 * span, "*", ha="center", va="bottom", fontsize=11, color=INK)
    ax.set_title(f"{title}\n{sub}", fontsize=8.8, loc="left", fontweight="bold", linespacing=1.35)
    ax.set_xticks(x); ax.set_xticklabels([("0\n(없음)" if lam[a] == 0 else f"{lam[a]:g}") for a in arms], fontsize=8)
    ax.set_ylabel(ylab, fontsize=8); ax.grid(axis="y", alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel(f"구조 손실의 가중치   (가중치와의 순위상관 ρ = {rho:+.2f}, p {'< 0.001' if p_trend < 0.001 else '= ' + format(p_trend, '.3f')})",
                  fontsize=7.8)
    table[key] = {"lambda": [lam[a] for a in arms], "mean": means.round(4).tolist(), "ci95_half": halves.round(4).tolist(),
                  "p_vs_zero_holm": [round(float(p), 4) for p in holm], "spearman_rho": round(float(rho), 3), "spearman_p": float(p_trend)}
fig.suptitle(f"회색 점: 학습 시드 하나의 값(시드마다 60곡)   검은 선: 시드 {n}개의 평균   푸른 띠: 95% 신뢰구간   "
             f"*: 가중치 0과 유의하게 다름(p < 0.05)   점선: 원곡 또는 기준값", fontsize=8, color=INK, y=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(OUT, dpi=220, bbox_inches="tight")
pass_c4 = {f"{lam[a]:g}": round(float(np.mean([abs(per['z_novel'][a][s]) < 2 for s in seeds])), 3) for a in arms}
STATS.write_text(json.dumps({"seeds": seeds, "n": n, "metrics": table, "share_of_seeds_passing_copy_test": pass_c4}, ensure_ascii=False, indent=1), encoding="utf-8")
print("seeds", n, "wrote", OUT, pass_c4)
