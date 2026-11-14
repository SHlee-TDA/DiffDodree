"""Figures for the manuscript (v2). Reads figure_data.json written by extract_data.py."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle, Ellipse, Polygon

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"; OUT.mkdir(exist_ok=True)
D = json.loads((HERE / "figure_data.json").read_text())

plt.rcParams.update({"font.family": "Apple SD Gothic Neo", "font.size": 8.5, "axes.unicode_minus": False,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 220})
INK, GREY, LIGHT, ACCENT = "#222222", "#8a8a8a", "#e9e9e9", "#c0392b"
TYPE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#D9B92E", "#7B6CC9", "#5BA8A4", "#9D755D", "#8C9A3A"]
LET = "ABCDEFGHI"
PL = ["(가)", "(나)", "(다)", "(라)"]
import subprocess, sys
L0, L1 = "#4C78A8", "#E07B2B"   # λ = 0, λ = 1
PN = {58: "B♭3", 60: "C4", 63: "E♭4", 65: "F4", 68: "A♭4", 70: "B♭4", 72: "C5", 75: "E♭5", 77: "F5"}
labels, phrases, chapters = D["labels"], D["phrases"], D["chapters"]
T = len(labels)


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def arrow(ax, a, b, color=GREY, lw=1.2, rad=0.0, ls="-", ms=9, z=1):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls, zorder=z))


def roll(ax, measures, x0, y0, w, h, colors=None, plo=56, phi=79, lw=1.6, bars=True, beats=None):
    """Piano-roll of consecutive measures inside the box (x0, y0, w, h)."""
    total = sum(sum(d for _, d in m) for m in measures) if beats is None else beats
    t = 0.0
    for k, m in enumerate(measures):
        c = INK if colors is None else colors[k]
        if bars and k:
            ax.plot([x0 + w * t / total] * 2, [y0, y0 + h], color="#cccccc", lw=0.6, zorder=0)
        for p, d in m:
            if p is not None:
                y = y0 + h * (p - plo) / (phi - plo)
                ax.plot([x0 + w * t / total, x0 + w * (t + d) / total - w * 0.004], [y, y], color=c, lw=lw,
                        solid_capstyle="butt")
            t += d


def strip(ax, seq, x0, y0, w, h, fs=6, alpha=1.0, letters=True):
    n = len(seq); cw = w / n
    for i, s in enumerate(seq):
        ax.add_patch(Rectangle((x0 + i * cw, y0), cw * 0.94, h, color=TYPE[s], alpha=alpha, lw=0))
        if letters:
            ax.text(x0 + (i + 0.47) * cw, y0 + h / 2, LET[s], ha="center", va="center", color="white", fontsize=fs,
                    fontweight="bold")


def box(ax, x, y, w, h, title=None, fc="#f7f7f7", ec="#cfcfcf", fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.015", fc=fc, ec=ec, lw=0.8))
    if title:
        ax.text(x + 0.012, y + h - 0.03, title, ha="left", va="top", fontsize=fs, fontweight="bold", color=INK)


# ---------------------------------------------------------------- figure 1: overview
def fig1():
    fig = plt.figure(figsize=(7.2, 4.0)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.01, 0.975, "분석: 원곡에서 구조를 읽는다", fontsize=9.5, fontweight="bold", va="top")
    ax.text(0.2, 0.497, "생성: 그 구조 위에 새 선율을 채운다", fontsize=9.5, fontweight="bold", va="top")
    W, H, Y1, Y2 = 0.225, 0.36, 0.56, 0.07
    xs = [0.01, 0.258, 0.506, 0.754]
    titles1 = ["① 악보를 마디로 나눈다", "② 닮은 마디끼리 묶는다", "③ 유형의 순서와 연결", "④ 지킬 구조"]
    titles2 = ["⑤ 새 유형열을 뽑는다", "⑥ 마디를 채운다(국소 확산)", "⑦ 환입 자리와 유형을 지킨다", "⑧ 새 곡"]
    for x, t in zip(xs, titles1): box(ax, x, Y1, W, H, t)
    for x, t in zip(xs, titles2): box(ax, x, Y2, W, H, t)
    for i in range(3):
        arrow(ax, (xs[i] + W + 0.002, Y1 + H / 2), (xs[i + 1] - 0.002, Y1 + H / 2), ms=10)
        arrow(ax, (xs[i] + W + 0.002, Y2 + H / 2), (xs[i + 1] - 0.002, Y2 + H / 2), ms=10)
    xa, xb, yl = xs[3] + W / 2, xs[0] + W / 2, 0.522
    ax.plot([xa, xa, xb], [Y1 - 0.005, yl, yl], color=ACCENT, lw=1.4)
    arrow(ax, (xb, yl), (xb, Y2 + H + 0.004), color=ACCENT, lw=1.4, ms=11)
    ax.text(0.5, yl, "구조를 조건으로 넘긴다", color=ACCENT, ha="center", va="center", fontsize=8,
            bbox=dict(fc="white", ec="none", pad=1.5), zorder=5)
    # ① roll of measures 1-4 with bar lines
    roll(ax, phrases[:4], xs[0] + 0.015, Y1 + 0.05, W - 0.03, 0.22)
    ax.text(xs[0] + W / 2, Y1 + 0.02, "1–4마디", ha="center", color=GREY, fontsize=7.5)
    # ② tiles coloured by type
    for k in range(4):
        x = xs[1] + 0.012 + k * 0.052
        ax.add_patch(Rectangle((x, Y1 + 0.07), 0.046, 0.17, fc="white", ec=TYPE[labels[k]], lw=1.4))
        roll(ax, [phrases[k]], x + 0.003, Y1 + 0.09, 0.04, 0.13, colors=[TYPE[labels[k]]], lw=1.2, bars=False)
        ax.text(x + 0.023, Y1 + 0.04, LET[labels[k]], ha="center", color=TYPE[labels[k]], fontweight="bold")
    ax.text(xs[1] + W / 2, Y1 + 0.005, "72마디 → 9개 유형(a–i)", ha="center", color=GREY, fontsize=7.5)
    # ③ strip + mini graph
    strip(ax, labels[:28], xs[2] + 0.012, Y1 + 0.235, W - 0.024, 0.035, letters=False)
    ax.text(xs[2] + 0.012, Y1 + 0.215, "유형열(1–28마디)", fontsize=7, color=GREY, va="top")
    cx, cy = xs[2] + W / 2, Y1 + 0.095
    pos = {k: (cx + 0.07 * np.cos(a), cy + 0.06 * np.sin(a)) for k, a in zip(range(5), np.linspace(0.5, 0.5 + 2 * np.pi, 6)[:5])}
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0), (2, 4), (4, 1)]:
        arrow(ax, pos[a], pos[b], color="#9a9a9a", lw=0.8, rad=0.2, ms=6)
    for k, (x, y) in pos.items():
        ax.add_patch(Circle((x, y), 0.016, color=TYPE[k], zorder=3))
        ax.text(x, y, LET[k], ha="center", va="center", color="white", fontsize=6.5, zorder=4, fontweight="bold")
    # ④ preserved structure: strip with return arc
    y = Y1 + 0.19
    strip(ax, labels[:28], xs[3] + 0.012, y, W - 0.024, 0.035, letters=False)
    cw = (W - 0.024) / 28
    a1, a2 = xs[3] + 0.012 + 2 * cw, xs[3] + 0.012 + 26 * cw
    ax.add_patch(FancyArrowPatch((a1, y + 0.04), (a2, y + 0.04), arrowstyle="-", connectionstyle="arc3,rad=-0.3",
                                 color=ACCENT, lw=1.4))
    ax.text(xs[3] + W / 2, y - 0.012, "환입: 1–4 = 25–28마디", ha="center", va="top", color=ACCENT, fontsize=7.5)
    ax.text(xs[3] + 0.015, Y1 + 0.125, "· 원곡에 있던 유형 연결\n· 처음과 끝 마디의 유형\n· 마디마다의 박 수", va="top",
            fontsize=7, color=INK, linespacing=1.35)
    # ⑤ original vs new walk
    strip(ax, labels[:28], xs[0] + 0.012, Y2 + 0.2, W - 0.024, 0.032, letters=False)
    strip(ax, D["walks"][0][:28], xs[0] + 0.012, Y2 + 0.1, W - 0.024, 0.032, letters=False)
    ax.text(xs[0] + 0.012, Y2 + 0.255, "원곡", fontsize=7, color=GREY)
    ax.text(xs[0] + 0.012, Y2 + 0.155, "새 유형열", fontsize=7, color=GREY)
    ax.text(xs[0] + W / 2, Y2 + 0.03, "자리의 약 80%가 원곡과 다른 유형", ha="center", fontsize=7, color=GREY)
    # ⑥ grid filling
    gx, gy, cw2 = xs[1] + 0.02, Y2 + 0.08, (W - 0.04) / 16
    rng = np.random.default_rng(3)
    for r, frac in enumerate([0.0, 0.45, 1.0]):
        shown = rng.random(16) < frac
        for j in range(16):
            yy = gy + (2 - r) * 0.075
            ax.add_patch(Rectangle((gx + j * cw2, yy), cw2 * 0.9, 0.05,
                                   fc=(TYPE[5] if shown[j] else "#d9d9d9"), ec="none", alpha=0.85 if shown[j] else 1))
    ax.text(xs[1] + W / 2, Y2 + 0.025, "모두 가린 상태 → 조금씩 채움", ha="center", fontsize=7, color=GREY)
    # ⑦ return tied
    ex = D["example_diffusion_graph"]["phrases"]
    for k, (lab, xx) in enumerate([("1–4마디", xs[2] + 0.02), ("25–28마디", xs[2] + 0.125)]):
        ax.add_patch(Rectangle((xx, Y2 + 0.1), 0.085, 0.14, fc="white", ec=ACCENT, lw=1.2))
        roll(ax, ex[:4], xx + 0.004, Y2 + 0.115, 0.077, 0.11, colors=[ACCENT] * 4, lw=1.0)
        ax.text(xx + 0.0425, Y2 + 0.065, lab, ha="center", fontsize=7, color=ACCENT)
    ax.text(xs[2] + 0.1125, Y2 + 0.17, "=", ha="center", va="center", fontsize=12, color=ACCENT)
    ax.text(xs[2] + W / 2, Y2 + 0.02, "벗어난 마디는 다시 만든다", ha="center", fontsize=7, color=GREY)
    # ⑧ generated piece roll
    roll(ax, ex[:6], xs[3] + 0.012, Y2 + 0.07, W - 0.024, 0.2, colors=[TYPE[s] for s in D["example_diffusion_graph"]["walk"][:6]])
    ax.text(xs[3] + W / 2, Y2 + 0.025, "생성곡 1–6마디", ha="center", fontsize=7, color=GREY)
    save(fig, "f01_overview.png")


# ---------------------------------------------------------------- figure 2: structure of the piece
def fig2():
    fig = plt.figure(figsize=(7.2, 4.3))
    ax = fig.add_axes([0.02, 0.56, 0.96, 0.42]); ax.axis("off"); ax.set_xlim(-0.5, T + 0.5); ax.set_ylim(-3.2, 5.6)
    for t, s in enumerate(labels):
        ax.add_patch(Rectangle((t, 0), 0.94, 1.0, color=TYPE[s], lw=0))
        ax.text(t + 0.47, 0.5, LET[s], ha="center", va="center", color="white", fontsize=6.3, fontweight="bold")
    bounds = [0] + [t for t in range(1, T) if chapters[t] != chapters[t - 1]] + [T]
    for k in range(len(bounds) - 1):
        a, b = bounds[k], bounds[k + 1]
        ax.plot([a - 0.03, a - 0.03], [-1.1, 1.05], color=INK, lw=0.8)
        ax.text((a + b) / 2, -0.75, f"{k + 1}장", ha="center", fontsize=8)
    ax.plot([T - 0.03] * 2, [-1.1, 1.05], color=INK, lw=0.8)
    for m in [1, 10, 20, 30, 40, 50, 56, 60, 70]:
        ax.text(m - 0.5, -1.7, str(m), ha="center", fontsize=6.5, color=GREY)
    ax.text(55.5, -2.5, "7장부터 6/4박", fontsize=6.5, color=GREY)
    ax.add_patch(FancyArrowPatch((2, 1.1), (26, 1.1), arrowstyle="-", connectionstyle="arc3,rad=-0.25", color=ACCENT, lw=2.0))
    ax.plot([0, 3.94], [1.12, 1.12], color=ACCENT, lw=2.2); ax.plot([24, 27.94], [1.12, 1.12], color=ACCENT, lw=2.2)
    ax.text(14, 3.75, "환입: 1–4마디 = 25–28마디", color=ACCENT, ha="center", fontsize=8.5, fontweight="bold")
    ax.text(-0.5, 5.3, "(가) 유형열 (마디 하나가 한 칸)", fontsize=9, fontweight="bold")
    # b graph
    gx = fig.add_axes([0.02, 0.0, 0.46, 0.53]); gx.axis("off"); gx.set_aspect("equal"); gx.set_xlim(-1.45, 1.45); gx.set_ylim(-1.35, 1.4)
    W = np.array(D["counts"]); size = np.bincount(labels, minlength=9)
    ang = np.pi / 2 - np.arange(9) * 2 * np.pi / 9; P = np.c_[np.cos(ang), np.sin(ang)]
    for a in range(9):
        for b in range(9):
            if a != b and W[a, b] > 0:
                gx.add_patch(FancyArrowPatch(P[a], P[b], arrowstyle="-|>", mutation_scale=7, color="#7f7f7f", alpha=0.75,
                                             lw=0.5 + 0.9 * W[a, b], connectionstyle="arc3,rad=0.12",
                                             shrinkA=12, shrinkB=12))
    for k in range(9):
        r = 0.08 + 0.017 * size[k]
        gx.add_patch(Circle(P[k], r, color=TYPE[k], zorder=3))
        gx.text(*P[k], LET[k], ha="center", va="center", color="white", fontweight="bold", zorder=4, fontsize=9)
        if W[k, k]:
            gx.text(*(P[k] * (1 + (r + 0.13) / 1.0)), f"연속 {W[k, k]}", ha="center", va="center", fontsize=6.5, color=GREY)
    gx.text(-1.45, 1.38, "(나) 전이 그래프 (굵기 = 연결 횟수, 연속 = 같은 유형이 이어진 횟수)", fontsize=9, fontweight="bold", va="top")
    # c table-like bars
    bx = fig.add_axes([0.6, 0.06, 0.37, 0.42])
    bx.barh(range(9)[::-1], size, color=TYPE)
    bx.set_yticks(range(9)[::-1]); bx.set_yticklabels([f"{LET[k]}  (대표 {m}마디)" for k, m in enumerate(D["medoid_measures"])], fontsize=7.5)
    bx.set_xlabel("마디 수"); bx.set_title("(다) 유형별 마디 수와 대표 마디", fontsize=9, fontweight="bold", loc="left")
    save(fig, "f03_structure.png")


# ---------------------------------------------------------------- figure 3: variation within type f
def fig3():
    f = 5
    members = [t for t in range(T) if labels[t] == f]
    medm = D["medoid_measures"][f] - 1
    med = phrases[medm]
    def segs(m):
        tot = sum(d for _, d in m); t = 0; out = []
        for p, d in m:
            if p is not None: out.append((t / tot, (t + d) / tot, p))
            t += d
        return out
    def contour(u):
        for a, b, p in segs(med):
            if a <= u < b: return p
        return segs(med)[-1][2]
    dist = D["medoid_distance"]
    fig = plt.figure(figsize=(7.2, 3.3))
    ax = fig.add_axes([0.07, 0.15, 0.36, 0.72])
    for t in members:
        if t == medm: continue
        for a, b, p in segs(phrases[t]):
            ax.plot([a, b - 0.006], [p + np.random.default_rng(t).uniform(-0.25, 0.25)] * 2, color="#9a9a9a", lw=1.0, alpha=0.7)
    for a, b, p in segs(med):
        ax.plot([a, b - 0.006], [p, p], color=TYPE[f], lw=3.2, solid_capstyle="butt")
    ax.set_yticks(sorted(PN)); ax.set_yticklabels([PN[p] for p in sorted(PN)]); ax.set_ylim(62, 74)
    ax.set_xlabel("마디 안의 위치 (마디 길이로 나눔)"); ax.set_xlim(0, 1)
    ax.set_title(f"a  유형 f의 13마디를 겹친 모습\n    (굵은 선: 대표 마디 {medm + 1}마디)", fontsize=9, fontweight="bold", loc="left")
    # b small multiples: medoid + three members (nearest/middle/farthest)
    from_med = sorted((t for t in members if t != medm),
                      key=lambda t: sum(abs(p - contour((a + b) / 2)) * (b - a) for a, b, p in segs(phrases[t])))
    pick = [medm, from_med[0], from_med[len(from_med) // 2], from_med[-1]]
    names = ["대표 마디", "가까운 변주", "중간 변주", "먼 변주"]
    for k, t in enumerate(pick):
        bx = fig.add_axes([0.52 + (k % 2) * 0.24, 0.55 - (k // 2) * 0.42, 0.21, 0.3])
        cuts = sorted({0.0, 1.0} | {a for a, _, _ in segs(med)})
        for a, b, p in segs(phrases[t]):
            pts = [a] + [c for c in cuts if a < c < b] + [b]
            for u, v in zip(pts[:-1], pts[1:]):
                same = abs(p - contour((u + v) / 2)) < 0.5
                bx.plot([u, v - (0.01 if v == b else 0)], [p, p], color=(TYPE[f] if same else ACCENT), lw=2.6, solid_capstyle="butt")
        bx.set_ylim(62, 74); bx.set_xlim(0, 1); bx.set_xticks([]); bx.set_yticks([63, 68, 70, 72]); bx.set_yticklabels(["E♭4", "A♭4", "B♭4", "C5"], fontsize=6.5)
        bx.set_title(f"{t + 1}마디 · {names[k]}", fontsize=8, loc="left")
    fig.text(0.52, 0.93, "b  대표 마디와 다른 음(붉은색)", fontsize=9, fontweight="bold")
    save(fig, "fig03_type_variation.png")


# ---------------------------------------------------------------- figure 4: phrase manifold (concept + piece)
def neck(ax, p, q, r1, r2, color="#b9b0e6", lw=9):
    ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, alpha=0.75, solid_capstyle="round", zorder=1)


def fig4():
    fig = plt.figure(figsize=(7.2, 3.6))
    ax = fig.add_axes([0.0, 0.02, 0.5, 0.9]); ax.axis("off"); ax.set_aspect("equal"); ax.set_xlim(0, 10); ax.set_ylim(0, 8.6)
    C = {"a": (2.0, 5.6), "b": (5.0, 6.1), "c": (8.0, 5.3), "d": (5.4, 2.0)}
    R = {"a": 1.3, "b": 1.15, "c": 1.2, "d": 1.25}
    col = {"a": TYPE[0], "b": TYPE[1], "c": TYPE[2], "d": TYPE[3]}
    neck(ax, C["a"], C["b"], R["a"], R["b"]); neck(ax, C["b"], C["c"], R["b"], R["c"])
    for k in C:
        ax.add_patch(Circle(C[k], R[k], fc=col[k], alpha=0.16, ec=col[k], lw=1.2, zorder=2))
        ax.text(C[k][0], C[k][1] - R[k] + 0.3, f"유형 {k}", ha="center", color=col[k], fontsize=8, fontweight="bold")
    pts = np.array([(1.4, 5.2), (2.0, 6.1), (2.6, 5.4), (3.9, 5.9), (5.0, 6.4), (5.6, 5.7), (5.0, 2.6), (5.8, 1.6), (4.8, 1.4)])
    ax.plot(pts[:6, 0], pts[:6, 1], color=INK, lw=1.3, zorder=4)
    ax.add_patch(FancyArrowPatch(pts[5], pts[6], arrowstyle="-|>", mutation_scale=9, connectionstyle="arc3,rad=-0.35",
                                 color=INK, lw=1.1, linestyle=(0, (3, 2)), zorder=4))
    ax.plot(pts[6:, 0], pts[6:, 1], color=INK, lw=1.3, zorder=4)
    ax.scatter(pts[:, 0], pts[:, 1], s=9, color=INK, zorder=5)
    ax.text(3.5, 7.15, "닮은 유형끼리는 목으로 이어짐", fontsize=7.5, ha="center", color="#5e52b5")
    ax.text(7.4, 3.6, "닮지 않은 유형으로\n건너뜀(점선)", fontsize=7.5, ha="left", color=INK)
    ax.text(0.2, 8.55, "a  개념: 곡은 영역들을 지나는 하나의 흐름", fontsize=9, fontweight="bold", va="top")
    ax.text(0.3, 0.3, "점 = 음, 선 = 시간 순서", fontsize=7, color=GREY)
    # b the piece: MDS of medoid distances
    M = np.array(D["medoid_distance"]); n = len(M); J = np.eye(n) - 1 / n
    B = -0.5 * J @ (M ** 2) @ J; w, V = np.linalg.eigh(B); idx = np.argsort(w)[::-1][:2]
    X = V[:, idx] * np.sqrt(np.maximum(w[idx], 0)); X = (X - X.mean(0)) / np.abs(X).max() * 3.4 + np.array([5, 4])
    size = np.bincount(labels, minlength=9); rad = 0.3 + 0.045 * size
    for _ in range(400):   # push overlapping lobes apart, keeping the MDS layout otherwise
        for i in range(9):
            for j in range(i + 1, 9):
                d = X[j] - X[i]; dist = np.linalg.norm(d); need = rad[i] + rad[j] + 0.35
                if dist < need:
                    X[i] -= d / dist * (need - dist) / 2; X[j] += d / dist * (need - dist) / 2
    X = (X - X.min(0)) / (X.max(0) - X.min(0)).max() * 6.6 + np.array([1.6, 1.3])
    bx = fig.add_axes([0.5, 0.02, 0.5, 0.9]); bx.axis("off"); bx.set_aspect("equal"); bx.set_xlim(0, 10); bx.set_ylim(0, 8.6)
    for a, b in D["contact_pairs"]:
        neck(bx, X[a], X[b], rad[a], rad[b], lw=7)
    for k in range(9):
        bx.add_patch(Circle(X[k], rad[k], fc=TYPE[k], alpha=0.25, ec=TYPE[k], lw=1.2, zorder=2))
        bx.text(*X[k], LET[k], ha="center", va="center", fontsize=11, fontweight="bold", color=TYPE[k], zorder=3)
    c, n_ = D["transitions_crossing"]
    bx.text(0.2, 8.55, "b  양청도드리 피리의 9개 유형", fontsize=9, fontweight="bold", va="top")
    bx.text(0.3, 0.3, f"목 13개 = 닮은 유형 쌍 · 원곡의 유형 바뀜 {n_}번 중 {c}번({100 * c / n_:.0f}%)이 목을 지난다",
            fontsize=7, color=GREY)
    save(fig, "fig04_manifold.png")


# ---------------------------------------------------------------- figure 5: new type sequences
def fig5():
    fig = plt.figure(figsize=(7.2, 2.4)); ax = fig.add_axes([0.08, 0.05, 0.9, 0.9]); ax.axis("off")
    ax.set_xlim(0, T); ax.set_ylim(-0.3, 3.7)
    rows = [("원곡", labels), ("새 유형열 1", D["walks"][0]), ("새 유형열 2", D["walks"][1])]
    for r, (name, seq) in enumerate(rows):
        y = 2.6 - r * 1.2
        strip(ax, seq, 0, y, T, 0.6, fs=5.6)
        ax.text(-0.8, y + 0.3, name, ha="right", va="center", fontsize=8)
        for a, b in [(0, 4), (24, 28)]:
            ax.add_patch(Rectangle((a - 0.05, y - 0.06), b - a, 0.72, fill=False, ec=ACCENT, lw=1.2))
        for t in (0, T - 1):
            ax.add_patch(Rectangle((t - 0.05, y - 0.06), 1.0, 0.72, fill=False, ec=INK, lw=1.2, ls=(0, (2, 1))))
        if r:
            diff = np.mean([s != o for s, o in zip(seq, labels)])
            ax.text(T + 0.4, y + 0.3, f"원곡과 다른 자리 {100 * diff:.0f}%", va="center", fontsize=7, color=GREY)
    ax.text(0, -0.25, "붉은 틀: 같은 유형으로 묶인 환입 자리(1–4, 25–28마디) · 점선 틀: 원곡과 같게 둔 처음과 끝 마디의 유형", fontsize=7, color=GREY)
    save(fig, "f06_new_walks.png")


# ---------------------------------------------------------------- figure 6: local diffusion on the grid
def fig6():
    ex = D["example_diffusion_graph"]
    t = next(i for i in range(4, T) if sum(d for _, d in ex["phrases"][i]) == 4 and len(ex["phrases"][i]) >= 5)
    m = ex["phrases"][t]
    cells = []
    for p, d in m:
        n = int(round(d * 4)); cells += [(PN.get(p, "쉼") if p is not None else "쉼")] + ["—"] * (n - 1)
    cells = cells[:16]
    order = np.random.default_rng(7).permutation(16)
    fig = plt.figure(figsize=(7.2, 3.3)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    box(ax, 0.01, 0.12, 0.2, 0.76, "조건", fc="#f4f2fb", ec="#c9c3ea")
    ax.text(0.025, 0.76, f"· 이 자리의 유형: {LET[ex['walk'][t]]}\n· 대표 마디의 윤곽\n· 앞뒤 마디의 유형\n· 박 수(4박 = 16칸)\n· 이웃 마디와\n  맞닿는 음", va="top", fontsize=7.8, linespacing=1.6)
    steps = [(0, "단계 0: 모두 가림(?)"), (6, "단계 8"), (11, "단계 16"), (16, "단계 24: 완성")]
    x0, cw = 0.335, 0.027
    for r, (k, name) in enumerate(steps):
        y = 0.78 - r * 0.19
        ax.text(x0 - 0.01, y + 0.04, name, ha="right", va="center", fontsize=7.5)
        shown = set(order[:k])
        for j in range(16):
            vis = j in shown
            ax.add_patch(Rectangle((x0 + j * cw, y), cw * 0.92, 0.08, fc=("#ffffff" if vis else "#d6d6d6"),
                                   ec=(TYPE[5] if vis else "#bdbdbd"), lw=0.8))
            ax.text(x0 + (j + 0.46) * cw, y + 0.04, cells[j] if vis else "?", ha="center", va="center",
                    fontsize=(5.6 if vis and len(cells[j]) > 2 else 7), color=(INK if vis else "#8a8a8a"))
        if r < 3:
            arrow(ax, (x0 + 8 * cw, y - 0.005), (x0 + 8 * cw, y - 0.1), ms=7)
    for j in range(0, 17, 4):
        ax.plot([x0 + j * cw - 0.0015] * 2, [0.2, 0.89], color=INK, lw=0.5, alpha=0.35)
    ax.text(x0 + 8 * cw, 0.075, "한 박 = 네 칸 · 칸마다 ‘새 음’, ‘이어짐(—)’, ‘쉼’ 가운데 하나", ha="center", fontsize=7, color=GREY)
    arrow(ax, (0.215, 0.5), (0.235, 0.5), ms=9)
    box(ax, 0.79, 0.3, 0.2, 0.5, "결과 마디", fc="#ffffff")
    roll(ax, [m], 0.805, 0.38, 0.17, 0.3, colors=[TYPE[5]], lw=2.2, plo=60, phi=76)
    arrow(ax, (x0 + 16 * cw + 0.005, 0.24), (0.785, 0.45), ms=9, rad=0.2)
    ax.text(0.89, 0.2, "학습: 원곡 마디의 칸을\n무작위로 가리고 맞히기", ha="center", fontsize=7, color=GREY)
    save(fig, "f07_local_diffusion.png")


# ---------------------------------------------------------------- figure 7: structure loss (PINN analogy)
def fig7():
    fig = plt.figure(figsize=(7.2, 3.0))
    ax = fig.add_axes([0.05, 0.18, 0.36, 0.66])
    x = np.linspace(0, 1, 200); y = np.sin(2 * np.pi * x) * np.exp(-x)
    ax.plot(x, y, color=INK, lw=1.2)
    xd = np.array([0.05, 0.12, 0.2]); ax.scatter(xd, np.sin(2 * np.pi * xd) * np.exp(-xd), s=22, color=L0, zorder=3, label="자료가 있는 점: 자료와 맞춘다")
    xc = np.linspace(0.3, 0.95, 9); yc = np.sin(2 * np.pi * xc) * np.exp(-xc)
    ax.scatter(xc, np.zeros_like(xc) - 0.75, marker="x", s=18, color=ACCENT, label="자료가 없는 점: 법칙을 지키게 한다")
    for a, b in zip(xc, yc): ax.plot([a, a], [-0.72, b], color=ACCENT, lw=0.5, alpha=0.4, ls=":")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(-0.85, 0.85)
    ax.legend(fontsize=6.8, frameon=False, loc="upper right")
    ax.set_title("a  물리 정보 신경망", fontsize=9, fontweight="bold", loc="left")
    bx = fig.add_axes([0.45, 0.0, 0.55, 1]); bx.axis("off"); bx.set_xlim(0, 1); bx.set_ylim(0, 1)
    bx.text(0.0, 0.94, "b  본 연구의 구조 손실", fontsize=9, fontweight="bold")
    box(bx, 0.0, 0.62, 0.3, 0.22, None, fc="#eef3f9", ec="#b9cbe0")
    bx.text(0.15, 0.78, "원곡 72마디", ha="center", fontweight="bold", fontsize=8)
    bx.text(0.15, 0.68, "가린 칸 맞히기\n(확산 손실)", ha="center", fontsize=7.2, color=GREY)
    box(bx, 0.0, 0.12, 0.3, 0.32, None, fc="#fbefee", ec="#e6b5b0")
    bx.text(0.15, 0.37, "새 유형열", ha="center", fontweight="bold", fontsize=8)
    bx.text(0.15, 0.31, "(원곡에 없는 조건)", ha="center", fontsize=7, color=GREY)
    seq = D["walks"][2][:14]; strip(bx, seq, 0.02, 0.17, 0.26, 0.07, letters=False)
    arrow(bx, (0.31, 0.28), (0.4, 0.28), ms=9)
    box(bx, 0.41, 0.12, 0.25, 0.32, None, fc="#ffffff")
    bx.text(0.535, 0.37, "신경망이 만든 마디들을", ha="center", fontsize=7.2)
    bx.text(0.535, 0.31, "다시 유형으로 읽음", ha="center", fontsize=7.2)
    read = list(seq); read[5] = (read[5] + 3) % 9; read[10] = (read[10] + 4) % 9
    strip(bx, read, 0.43, 0.17, 0.21, 0.07, letters=False)
    for i in (5, 10):
        bx.text(0.43 + (i + 0.47) * 0.21 / 14, 0.13, "×", ha="center", color=ACCENT, fontsize=7)
    arrow(bx, (0.67, 0.28), (0.74, 0.28), ms=9)
    box(bx, 0.75, 0.12, 0.24, 0.32, None, fc="#fbefee", ec="#e6b5b0")
    bx.text(0.87, 0.36, "주어진 유형열과의", ha="center", fontsize=7.2)
    bx.text(0.87, 0.3, "어긋남 = 구조 손실", ha="center", fontsize=7.6, fontweight="bold", color=ACCENT)
    bx.text(0.87, 0.2, "(전이 그래프의 차이)", ha="center", fontsize=6.8, color=GREY)
    arrow(bx, (0.31, 0.73), (0.74, 0.73), ms=9)
    bx.text(0.53, 0.76, "자료가 있는 점", ha="center", fontsize=7, color=L0)
    bx.text(0.53, 0.035, "자료가 없는 점에서 구조 규칙(유형 연결)을 지키게 한다", ha="center", fontsize=7, color=ACCENT)
    box(bx, 0.75, 0.62, 0.24, 0.22, None, fc="#eef3f9", ec="#b9cbe0")
    bx.text(0.87, 0.73, "두 손실의 합으로 학습", ha="center", fontsize=7.6, fontweight="bold")
    arrow(bx, (0.87, 0.45), (0.87, 0.61), color=ACCENT, ms=9)
    save(fig, "fig07_structure_loss.png")


# ---------------------------------------------------------------- figure 8: joint decoding of the return
def fig8():
    fig = plt.figure(figsize=(7.2, 2.9)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    widths = [4, 6, 14, 4, 13, 14, 17]; x = 0.03; scale = 0.94 / 72; xs = []
    for k, w in enumerate(widths):
        hl = k in (0, 3)
        ax.add_patch(Rectangle((x, 0.12), w * scale - 0.003, 0.11, fc=(ACCENT if hl else "#d9d9d9"), alpha=(0.85 if hl else 1), lw=0))
        ax.text(x + w * scale / 2, 0.175, f"{k + 1}장", ha="center", va="center", fontsize=7.5, color=("white" if hl else INK))
        xs.append((x, x + w * scale)); x += w * scale
    c1 = (xs[0][0] + xs[0][1]) / 2; c4 = (xs[3][0] + xs[3][1]) / 2
    box(ax, 0.03, 0.55, 0.26, 0.3, None, fc="#ffffff")
    ax.text(0.16, 0.79, "1–4마디 자리의 예측", ha="center", fontsize=8, fontweight="bold")
    ax.text(0.16, 0.67, "문맥: 곡의 시작, 5마디", ha="center", fontsize=7.2, color=GREY)
    box(ax, 0.71, 0.55, 0.26, 0.3, None, fc="#ffffff")
    ax.text(0.84, 0.79, "25–28마디 자리의 예측", ha="center", fontsize=8, fontweight="bold")
    ax.text(0.84, 0.67, "문맥: 24마디, 29마디", ha="center", fontsize=7.2, color=GREY)
    box(ax, 0.38, 0.55, 0.24, 0.3, None, fc="#fbefee", ec="#e6b5b0")
    ax.text(0.5, 0.78, "두 예측을 고르게 합침", ha="center", fontsize=8, fontweight="bold", color=ACCENT)
    ax.text(0.5, 0.66, "한 번 뽑아 두 자리에 같이 씀", ha="center", fontsize=7.2)
    arrow(ax, (0.29, 0.7), (0.375, 0.7), ms=9); arrow(ax, (0.71, 0.7), (0.625, 0.7), ms=9)
    arrow(ax, (0.47, 0.545), (c1, 0.235), color=ACCENT, rad=-0.15, ms=10)
    arrow(ax, (0.53, 0.545), (c4, 0.235), color=ACCENT, rad=0.1, ms=10)
    ax.text(0.5, 0.03, "환입 자리의 두 구간은 언제나 같은 선율이 되고, 그 선율은 두 자리의 앞뒤 문맥에 모두 어울리는 것으로 고른다", ha="center", fontsize=7.2, color=GREY)
    save(fig, "f09_return_joint.png")


# ---------------------------------------------------------------- figure 9: repetition vs copying
def fig9():
    fig = plt.figure(figsize=(7.2, 2.9))
    ax = fig.add_axes([0.06, 0.2, 0.26, 0.62])
    rep = round(D["repeat_rate"] * 72)
    ax.barh([0], [rep], color=TYPE[5]); ax.barh([0], [72 - rep], left=[rep], color="#d0d0d0")
    ax.text(rep / 2, 0, f"{rep}마디\n곡 안에서 다시 나옴", ha="center", va="center", color="white", fontsize=7.5)
    ax.text(rep + (72 - rep) / 2, 0, f"{72 - rep}마디\n한 번만 나옴", ha="center", va="center", fontsize=7.5)
    ax.set_xlim(0, 72); ax.set_yticks([]); ax.set_xlabel("양청도드리 피리 72마디"); ax.spines["left"].set_visible(False)
    ax.set_title("a  원곡 자체의 반복", fontsize=9, fontweight="bold", loc="left")
    bx = fig.add_axes([0.42, 0.2, 0.55, 0.62])
    cd = D["copy_distances"]
    def ecdf(v, **kw):
        v = np.sort(v); bx.step(v, np.arange(1, len(v) + 1) / len(v), where="post", **kw)
    ecdf(cd["diffusion"]["heldout"], color=INK, lw=1.8, label="학습에서 뺀 원곡 마디")
    for arm, c, lab in [("diffusion", L0, "생성 마디, λ = 0"), ("diffusion_graph", L1, "생성 마디, λ = 1")]:
        v = [x for x, s in zip(cd[arm]["generated"], cd[arm]["generated_is_train"]) if not s]
        ecdf(v, color=c, lw=1.4, label=lab)
    bx.set_xlabel("학습한 마디 가운데 가장 가까운 것까지의 거리"); bx.set_ylabel("누적 비율"); bx.set_xlim(0, 1.2)
    bx.legend(fontsize=7, frameon=False, loc="lower right")
    bx.set_title("b  복제 검정: 학습 마디에 체계적으로 더 가까운가", fontsize=9, fontweight="bold", loc="left")
    bx.text(0.02, 0.92, "학습 마디와 똑같은 마디는 빼고 비교", transform=bx.transAxes, fontsize=7, color=GREY)
    save(fig, "fig09_copy_test.png")


# ---------------------------------------------------------------- figure 10: results λ = 0 vs λ = 1
def fig10():
    R = D["results"]; seeds = sorted(R["diffusion"])
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw={"wspace": 0.45})
    for j, (arm, c, lab) in enumerate([("diffusion", L0, "λ = 0"), ("diffusion_graph", L1, "λ = 1")]):
        xs = np.arange(3) + (j - 0.5) * 0.25
        v = [R[arm][s]["first_pass_membership"] for s in seeds]; ci = [R[arm][s]["first_pass_ci"] for s in seeds]
        axs[0].errorbar(xs, v, yerr=[[a - c_[0] for a, c_ in zip(v, ci)], [c_[1] - a for a, c_ in zip(v, ci)]], fmt="o", color=c, ms=4, capsize=2, label=lab)
        z = [D["z_novel"][arm][s] for s in seeds]
        axs[1].plot(xs, z, "o", color=c, ms=5, label=lab)
    axs[0].set_xticks(range(3)); axs[0].set_xticklabels(["시드 1", "시드 2", "시드 3"]); axs[0].set_ylim(0.75, 1.0)
    axs[0].set_title("(가) 주어진 유형에 속한 비율", fontsize=8.5, fontweight="bold", loc="left"); axs[0].legend(fontsize=7, frameon=False)
    axs[1].axhspan(-2, 2, color="#e8f3e8"); axs[1].axhline(0, color=GREY, lw=0.6)
    axs[1].set_xticks(range(3)); axs[1].set_xticklabels(["시드 1", "시드 2", "시드 3"]); axs[1].set_ylim(-4.2, 2.5)
    axs[1].set_title("(나) 복제 검정 Z (초록: 통과)", fontsize=8.5, fontweight="bold", loc="left")
    names = ["원곡", "λ = 0", "λ = 1"]
    dmed = [D["dist_to_own_medoid"]["original"], D["dist_to_own_medoid"]["diffusion"], D["dist_to_own_medoid"]["diffusion_graph"]]
    axs[2].bar(names, dmed, color=["#7d7d7d", L0, L1])
    for i, v in enumerate(dmed): axs[2].text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    axs[2].set_title("(다) 대표 마디까지의 평균 거리", fontsize=8.5, fontweight="bold", loc="left"); axs[2].set_ylim(0, 0.6)
    save(fig, "f11_results.png")


# ---------------------------------------------------------------- figure 11: original vs generated
def fig11():
    content = {tuple(map(tuple, m)) for m in phrases}
    def copies(pc, idx): return sum(tuple(map(tuple, pc["phrases"][t])) in content for t in idx)
    free = [t for t in range(T) if not 24 <= t < 28]
    pieces = D["pieces_diffusion_graph"]
    rates = np.array([copies(pc, free) / len(free) for pc in pieces]); med = np.median(rates)
    ok = [i for i, pc in enumerate(pieces) if copies(pc, range(4)) == 0]
    pick = min(ok, key=lambda i: abs(rates[i] - med))
    ex = pieces[pick]
    print("fig11 example: piece seed", ex["seed"], "copy rate", round(rates[pick], 3), "median", round(med, 3))
    rows = [("원곡", labels[:28], phrases[:28]), ("생성곡 (λ = 1)", ex["walk"][:28], ex["phrases"][:28])]
    fig = plt.figure(figsize=(7.2, 3.6)); ax = fig.add_axes([0.09, 0.04, 0.9, 0.92]); ax.axis("off")
    ax.set_xlim(0, 112); ax.set_ylim(0, 10)
    for r, (name, walk, ph) in enumerate(rows):
        y = 5.4 - r * 5.0
        strip(ax, walk, 0, y + 3.7, 112, 0.55, fs=5.8)
        roll(ax, ph, 0, y, 112, 3.4, colors=[TYPE[s] for s in walk], lw=1.4, plo=57, phi=78)
        ax.text(-1.2, y + 2.0, name, ha="right", va="center", fontsize=8.5, fontweight="bold")
        for a, b in [(0, 16), (96, 112)]:
            ax.add_patch(Rectangle((a, y - 0.15), b - a, 4.5, fill=False, ec=ACCENT, lw=1.2))
        for k, lab in [(8, "1장"), (28, "2장"), (68, "3장"), (104, "4장")]:
            ax.text(k, y - 0.55, lab, ha="center", fontsize=7, color=GREY)
    save(fig, "f12_example.png")


# ---------------------------------------------------------------- appendix figure: held-out curve
def figA1():
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    for arm, c, lab in [("diffusion", L0, "λ = 0"), ("diffusion_graph", L1, "λ = 1")]:
        cur = {int(k): v for k, v in D["heldout_curve"][arm].items()}; tr = {int(k): v for k, v in D["train_curve"][arm].items()}
        s = sorted(cur)
        ax.plot(s, [cur[k] for k in s], "o-", color=c, ms=3, label=f"{lab} · 학습에서 뺀 마디")
        ax.plot(s, [tr[k] for k in s], ":", color=c, label=f"{lab} · 학습 마디")
    ax.axhline(np.mean(list(D["heldout_baseline"].values())), color=GREY, ls="--", lw=1, label="문맥 없는 기준값")
    ax.axvline(125, color=GREY, lw=0.8); ax.text(130, 1.55, "고른 길이 125", fontsize=7, color=GREY)
    ax.set_xscale("log"); ax.set_xlabel("학습 단계"); ax.set_ylabel("칸당 ELBO (nat)"); ax.legend(fontsize=6.5, frameon=False)
    save(fig, "fA1_heldout.png")


# ---------------------------------------------------------------- figure 2: TDA barcode depends on node order
def fig_tda_order():
    T_ = json.loads((HERE / "tda_order_check.json").read_text())
    ex = T_["examples"]
    names = ["음높이 순서로 기록", "무작위 순서로 기록 1", "무작위 순서로 기록 2"]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9), sharex=True, gridspec_kw={"wspace": 0.12})
    xmax = max(d for bc in ex for _, d in bc) * 1.05
    for k, (ax, bc) in enumerate(zip(axs, ex)):
        bc = sorted(bc, key=lambda p: (p[0], p[1]))
        for i, (b, d) in enumerate(bc):
            ax.plot([b, d], [i, i], color=(INK if k == 0 else ACCENT), lw=2.2, solid_capstyle="butt")
        ax.set_ylim(-1, 25); ax.set_xlim(0, xmax); ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_title(f"{PL[k]} {names[k]}\n1차원 막대 {len(bc)}개", fontsize=8.5, fontweight="bold", loc="left")
        ax.set_xlabel("거리 척도")
    save(fig, "f02_tda_order.png")


# ---------------------------------------------------------------- figure 4: the phrase manifold (from the methodology page)
def fig_manifold():
    import pymupdf
    man = HERE / "manifold"
    tmp1, tmp2 = man / "_arrows.svg", man / "_final.svg"
    subprocess.run([sys.executable, str(man / "fix_arrows.py"), str(man / "manifold_source.svg"), str(tmp1)], check=True, capture_output=True)
    subprocess.run([sys.executable, str(man / "inline_svg.py"), str(tmp1), str(tmp2)], check=True, capture_output=True)
    pix = pymupdf.open(str(tmp2))[0].get_pixmap(matrix=pymupdf.Matrix(4, 4), alpha=False)
    pix.save(str(OUT / "f04_manifold.png"))
    tmp1.unlink(); tmp2.unlink()


# ---------------------------------------------------------------- figure 5: variation within a type, on the staff
SPELL = {58: ("B", -1, 3), 60: ("C", 0, 4), 63: ("E", -1, 4), 65: ("F", 0, 4), 68: ("A", -1, 4), 70: ("B", -1, 4),
         72: ("C", 0, 5), 75: ("E", -1, 5), 77: ("F", 0, 5)}
NOTE_TYPES = {0.25: ("16th", 0), 0.5: ("eighth", 0), 0.75: ("eighth", 1), 1.0: ("quarter", 0), 1.5: ("quarter", 1),
              1.75: ("quarter", 2), 2.0: ("half", 0), 3.0: ("half", 1), 4.0: ("whole", 0)}


def musicxml(measure, colors, beats):
    out = ['<?xml version="1.0" encoding="UTF-8"?><score-partwise version="3.1"><part-list><score-part id="P1">'
           '<part-name></part-name></score-part></part-list><part id="P1"><measure number="1">'
           f'<attributes><divisions>4</divisions><key><fifths>-3</fifths></key><time><beats>{beats}</beats>'
           '<beat-type>4</beat-type></time><clef><sign>G</sign><line>2</line></clef></attributes>']
    for (p, d), c in zip(measure, colors):
        typ, dots = NOTE_TYPES[d]; col = f' color="{c}"' if c else ""
        if p is None:
            out.append(f'<note{col}><rest/><duration>{int(d * 4)}</duration><type>{typ}</type>{"<dot/>" * dots}</note>')
        else:
            st, al, oc = SPELL[p]; alter = f"<alter>{al}</alter>" if al else ""
            out.append(f'<note{col}><pitch><step>{st}</step>{alter}<octave>{oc}</octave></pitch><duration>{int(d * 4)}</duration>'
                       f'<type>{typ}</type>{"<dot/>" * dots}</note>')
    out.append("</measure></part></score-partwise>")
    return "".join(out)


def staff_image(measure, colors, beats=4):
    import pymupdf, verovio
    sys.path.insert(0, str(HERE))
    from verovio_svg import clean_verovio_svg
    tk = verovio.toolkit()
    tk.setOptions({"pageWidth": 900, "pageHeight": 400, "adjustPageHeight": True, "scale": 60, "header": "none",
                   "footer": "none", "breaks": "none", "svgViewBox": True, "pageMarginLeft": 20, "pageMarginRight": 20,
                   "pageMarginTop": 10, "pageMarginBottom": 10})
    tk.loadData(musicxml(measure, colors, beats))
    svg = clean_verovio_svg(tk.renderToSVG(1))
    tmp = OUT / "_staff.svg"; tmp.write_text(svg, encoding="utf-8")
    pix = pymupdf.open(str(tmp))[0].get_pixmap(matrix=pymupdf.Matrix(6, 6), alpha=False); tmp.unlink()
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)


def segs_of(m):
    tot = sum(d for _, d in m); t = 0.0; out = []
    for p, d in m:
        out.append((t / tot, (t + d) / tot, p)); t += d
    return out


def fig_type_variation():
    f = 5; medm = D["medoid_measures"][f] - 1; med = phrases[medm]
    def pitch_at(u):
        for a, b, p in segs_of(med):
            if a <= u < b: return p
        return segs_of(med)[-1][2]
    def colors_for(m):
        cuts = sorted({a for a, _, _ in segs_of(med)} | {1.0})
        out = []
        for a, b, p in segs_of(m):
            pts = [a] + [c for c in cuts if a < c < b] + [b]
            out.append(ACCENT if any(pitch_at((u + v) / 2) != p for u, v in zip(pts[:-1], pts[1:])) else None)
        return out
    members = [t for t in range(T) if labels[t] == f and t != medm]
    order = sorted(members, key=lambda t: sum(abs((p or 0) - (pitch_at((a + b) / 2) or 0)) * (b - a) for a, b, p in segs_of(phrases[t])))
    pick = [medm, order[0], order[len(order) // 2], order[-1]]
    names = ["대표 마디", "가까운 변주", "중간 변주", "먼 변주"]
    fig = plt.figure(figsize=(7.2, 3.6))
    for k, t in enumerate(pick):
        ax = fig.add_axes([0.0 + (k % 2) * 0.33, 0.5 - (k // 2) * 0.46, 0.32, 0.4]); ax.axis("off")
        ax.imshow(staff_image(phrases[t], [None] * len(phrases[t]) if k == 0 else colors_for(phrases[t])))
        ax.set_title(f"{t + 1}마디 · {names[k]}", fontsize=8.5, loc="left")
    fig.text(0.0, 0.97, f"{PL[0]} 유형 F의 대표 마디와 변주 (붉은 음: 대표 마디와 다른 음)", fontsize=9, fontweight="bold")
    bx = fig.add_axes([0.73, 0.14, 0.26, 0.7])
    rates = []
    for c in range(9):
        cmed = phrases[D["medoid_measures"][c] - 1]
        def at(u, m=cmed):
            for a, b, p in segs_of(m):
                if a <= u < b: return p
            return segs_of(m)[-1][2]
        n = dif = 0
        for t in range(T):
            if labels[t] != c: continue
            for a, b, p in segs_of(phrases[t]):
                if p is None: continue
                n += 1; dif += at((a + b) / 2) != p
        rates.append(100 * dif / n)
    bx.barh(range(9)[::-1], rates, color=TYPE)
    bx.set_yticks(range(9)[::-1]); bx.set_yticklabels(list(LET)); bx.set_xlabel("대표 마디와 다른 음의 비율(%)")
    bx.set_title(f"{PL[1]} 유형별 변주의 폭", fontsize=9, fontweight="bold", loc="left")
    for i, r in enumerate(rates):
        bx.text(r + 1, 8 - i, f"{r:.0f}", va="center", fontsize=7)
    bx.set_xlim(0, 65)
    save(fig, "f05_type_variation.png")


# ---------------------------------------------------------------- figure 8: graph read back from the generated notes
def fig_graph_compare():
    S = D["surface_example"]; walk = S["walk"]
    def edges(seq):
        e = {}
        for a, b in zip(seq[:-1], seq[1:]):
            if a != b: e[(a, b)] = e.get((a, b), 0) + 1
        return e
    given = edges(walk)
    ang = np.pi / 2 - np.arange(9) * 2 * np.pi / 9; P = np.c_[np.cos(ang), np.sin(ang)]
    panels = [("조건으로 준\n새 유형열의 그래프", walk, None),
              ("구조 손실 없이(λ = 0) 만든\n음표에서 다시 읽은 그래프", S["diffusion"], given),
              ("구조 손실을 써서(λ = 1) 만든\n음표에서 다시 읽은 그래프", S["diffusion_graph"], given)]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.2), gridspec_kw={"wspace": 0.12})
    for k, (ax, (title, seq, ref_e)) in enumerate(zip(axs, panels)):
        ax.axis("off"); ax.set_aspect("equal"); ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35)
        e = edges(seq); bad = 0
        for (a, b), c in e.items():
            wrong = ref_e is not None and (a, b) not in ref_e
            bad += c if wrong else 0
            ax.add_patch(FancyArrowPatch(P[a], P[b], arrowstyle="-|>", mutation_scale=6, lw=0.6 + 0.5 * c,
                                         color=(ACCENT if wrong else "#7f7f7f"), connectionstyle="arc3,rad=0.12",
                                         shrinkA=8, shrinkB=8, linestyle=("--" if wrong else "-"), zorder=2))
        for i in range(9):
            ax.add_patch(Circle(P[i], 0.15, color=TYPE[i], zorder=3))
            ax.text(*P[i], LET[i], ha="center", va="center", color="white", fontweight="bold", fontsize=7.5, zorder=4)
        sub = "\n " if ref_e is None else f"\n조건에 없는 연결 {bad}번(붉은 점선)"
        ax.set_title(f"{PL[k]} {title}{sub}", fontsize=7.6, fontweight="bold", loc="left")
    save(fig, "f08_structure_loss.png")


# ---------------------------------------------------------------- figure 10: repetition vs copying
def fig_copy_test():
    fig = plt.figure(figsize=(7.2, 2.8))
    ax = fig.add_axes([0.03, 0.22, 0.22, 0.58])
    rep_ = round(D["repeat_rate"] * 72)
    ax.barh([0], [rep_], color=TYPE[5]); ax.barh([0], [72 - rep_], left=[rep_], color="#d0d0d0")
    ax.text(rep_ / 2, 0, f"{rep_}마디\n곡 안에서\n다시 나옴", ha="center", va="center", color="white", fontsize=7.5)
    ax.text(rep_ + (72 - rep_) / 2, 0, f"{72 - rep_}마디\n한 번만\n나옴", ha="center", va="center", fontsize=7.5)
    ax.set_xlim(0, 72); ax.set_yticks([]); ax.set_xlabel("양청도드리 피리 72마디"); ax.spines["left"].set_visible(False)
    ax.set_title(f"{PL[0]} 원곡 자체의 반복", fontsize=9, fontweight="bold", loc="left")
    bx = fig.add_axes([0.55, 0.22, 0.43, 0.58])
    cd = D["copy_distances"]
    rows = [("학습에서 뺀\n원곡 마디(기준)", cd["diffusion"]["heldout"], INK),
            ("생성 마디,\n구조 손실 없음(λ = 0)", [x for x, s in zip(cd["diffusion"]["generated"], cd["diffusion"]["generated_is_train"]) if not s], L0),
            ("생성 마디,\n구조 손실 있음(λ = 1)", [x for x, s in zip(cd["diffusion_graph"]["generated"], cd["diffusion_graph"]["generated_is_train"]) if not s], L1)]
    z = D["z_novel"]
    for i, (name, v, c) in enumerate(rows):
        y = 2 - i; q1, med, q3 = np.percentile(v, [25, 50, 75])
        bx.plot([q1, q3], [y, y], color=c, lw=5, alpha=0.35, solid_capstyle="butt")
        bx.plot([med], [y], "o", color=c, ms=7)
        bx.text(-0.02, y, name, ha="right", va="center", fontsize=7.5, transform=bx.get_yaxis_transform())
        if i: bx.text(q3 + 0.03, y, f"Z = {z['diffusion' if i == 1 else 'diffusion_graph']['260807']:.1f}", va="center", fontsize=7.5, color=c)
    bx.axvline(np.median(rows[0][1]), color=INK, lw=0.8, ls=":")
    bx.set_yticks([]); bx.set_ylim(-0.6, 2.6); bx.set_xlim(0, 0.8); bx.spines["left"].set_visible(False)
    bx.set_xlabel("가장 가까운 학습 마디까지의 거리\n(점: 중앙값, 막대: 가운데 50%)")
    bx.annotate("", xy=(0.02, -0.45), xytext=(0.3, -0.45), arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.2))
    bx.text(0.32, -0.45, "왼쪽일수록 학습 마디에 가깝다", va="center", fontsize=7, color=ACCENT)
    bx.set_title(f"{PL[1]} 생성 마디가 학습 마디에\n지나치게 가까운가 (학습 시드 1)", fontsize=9, fontweight="bold", loc="left")
    save(fig, "f10_copy_test.png")


for f in (fig1, fig_tda_order, fig2, fig_manifold, fig_type_variation, fig5, fig6, fig_graph_compare, fig8, fig_copy_test, fig10, fig11, figA1):
    f(); print("drew", f.__name__)
