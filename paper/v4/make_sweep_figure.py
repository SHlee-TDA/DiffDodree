"""Figure 15: how each metric changes with the weight of the structure loss (one run, every λ).

Usage: python make_sweep_figure.py <run dir> <output png>
Points are training seeds, the line is their mean; reference values of the original piece are dashed.
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "Apple SD Gothic Neo", "font.size": 8, "axes.unicode_minus": False})
RUN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
ev = json.loads((RUN / "evaluation.json").read_text(encoding="utf-8"))
rep = json.loads((RUN / "report.json").read_text(encoding="utf-8"))
lam = ev["lambda"]
arms = sorted(lam, key=lam.get)
xs = {a: i for i, a in enumerate(arms)}
INK, ACC, GREY, GOOD = "#1d2433", "#2c4a8a", "#8a919e", "#dcefe0"


def per_seed(key):
    out = defaultdict(dict)
    for m in ev["models"]:
        v = m["per_piece"].get(key, m["set"].get(key))
        out[m["arm"]][m["seed"]] = v["mean"] if "mean" in v else v["value"]
    return out


def series(ax, data, title, ylabel=None, ref=None, ref_label=None, band=None):
    if band:
        ax.axhspan(*band, color=GOOD, zorder=0)
    for a in arms:
        vals = list(data[a].values())
        ax.scatter([xs[a]] * len(vals), vals, s=10, color=ACC, alpha=0.55, zorder=3, linewidths=0)
    ax.plot([xs[a] for a in arms], [np.mean(list(data[a].values())) for a in arms], color=INK, lw=1.3, zorder=4)
    if ref is not None:
        ax.axhline(ref, color=GREY, ls="--", lw=1)
        if ref_label: ax.text(len(arms) - 1, ref, ref_label, ha="right", va="bottom", fontsize=7, color=GREY)
    ax.set_title(title, fontsize=8.5, fontweight="bold", loc="left")
    ax.set_xticks(range(len(arms))); ax.set_xticklabels([f"{lam[a]:g}" for a in arms], fontsize=7)
    if ylabel: ax.set_ylabel(ylabel, fontsize=7.5)
    ax.grid(axis="y", alpha=0.25); ax.spines[["top", "right"]].set_visible(False)


elbo = {a: {s: None for s in []} for a in arms}
for u in rep["heldout"]:
    pass
elbo_seed = defaultdict(lambda: defaultdict(list))
for u in rep["heldout"]:
    step = rep["selected_steps"][u["arm"]]["selected"]
    pt = [p for p in u["curve"] if p["step"] == step]
    if pt: elbo_seed[u["arm"]][u["seed"]].append(pt[0]["heldout_elbo"])
elbo = {a: {s: float(np.mean(v)) for s, v in elbo_seed[a].items()} for a in arms}
base = float(np.mean(list(rep["heldout_baseline"].values())))
znovel = {a: {int(s): z for s, z in ev["copy_z_novel_combined"][a].items()} for a in arms}

panels = [
    (per_seed("first_pass_membership"), "(가) 주어진 유형에 속한 마디의 비율", None, None, None, None),
    (znovel, "(나) 복제 검정 Z (원곡에 없는 마디)", None, 0, None, (-2, 2)),
    (per_seed("exact_copy_rate"), "(다) 원곡과 똑같은 마디의 비율", None, ev["reference"]["repeat_rate_free"], "원곡 자체의 반복률", None),
    (per_seed("authenticity"), "(라) 복제가 아닌 마디의 비율", None, None, None, None),
    (per_seed("ph_cycles"), "(마) 음표 연결 순환의 수", None, ev["reference"]["barcode_bars"], "원곡", None),
    (per_seed("ph_bottleneck"), "(바) 순환 구조의 차이", None, None, None, None),
    (per_seed("interval_jsd"), "(사) 음정 분포의 차이", None, None, None, None),
    (elbo, "(아) 학습에 쓰지 않은 마디의 설명", "보류 ELBO (낮을수록 좋음)", base, "문맥 없는 기준", None),
]
fig, axes = plt.subplots(2, 4, figsize=(9.6, 4.6))
for ax, (data, title, yl, ref, rl, band) in zip(axes.ravel(), panels):
    series(ax, data, title, yl, ref, rl, band)
for ax in axes[1]: ax.set_xlabel("구조 손실의 가중치", fontsize=7.5)
fig.tight_layout()
fig.savefig(OUT, dpi=220, bbox_inches="tight")
print("wrote", OUT, [f"{lam[a]:g}" for a in arms])
