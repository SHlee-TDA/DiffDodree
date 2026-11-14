"""Score images for manuscript v4: full scores (appendix) and excerpts (main text).

Usage: python render_scores.py <project root with src/ and the CPU run> <output dir>
The pieces come from run 20261113_1331_cpu (λ = 0, 0.25, 0.5, 1), training seed 260807,
sample seed 260807: all four share one new type sequence, so they differ only in λ.
"""
import json, math, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pymupdf, verovio

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from verovio_svg import clean_verovio_svg

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))
from gugakgen.workflows import chart_diffusion_experiment as v1
from gugakgen.chart.evaluation import as_phrase

plt.rcParams.update({"font.family": "Apple SD Gothic Neo", "font.size": 9, "axes.unicode_minus": False})
BLUE, RED, INK, GREY = "#1F5FBF", "#C2412D", "#1d2433", "#6b7280"
LET = "abcdefghi"
RUN = "20261113_1331_cpu"
ARMS = [("diffusion", "λ = 0"), ("diffusion_graph_025", "λ = 0.25"), ("diffusion_graph_050", "λ = 0.5"), ("diffusion_graph", "λ = 1")]
SPELL = {58: ("B", -1, 3), 60: ("C", 0, 4), 63: ("E", -1, 4), 65: ("F", 0, 4), 68: ("A", -1, 4), 70: ("B", -1, 4),
         72: ("C", 0, 5), 75: ("E", -1, 5), 77: ("F", 0, 5)}
TYPE = {0.25: ("16th", 0), 0.5: ("eighth", 0), 0.75: ("eighth", 1), 1.0: ("quarter", 0), 2.0: ("half", 0),
        3.0: ("half", 1), 4.0: ("whole", 0)}

setup = v1.load_setup(ROOT / "configs/chart_diffusion_v1.yaml", ROOT)
typ = setup["typology"]; ref = v1.build_reference(setup)
ORIG = [tuple(p) for p in typ.phrases]
WHERE = defaultdict(list)
for i, p in enumerate(ORIG): WHERE[p].append(i + 1)
CHAPTER_START = {1: 1, 5: 2, 11: 3, 25: 4, 29: 5, 42: 6, 56: 7}

report = json.loads((ROOT / f"artifacts/chart_diffusion_runs/{RUN}/report.json").read_text(encoding="utf-8"))
PIECES, WALK = {}, None
for u in report["full"]:
    if u["seed"] == 260807:
        p = u["generation"]["pieces"][0]
        PIECES[u["arm"]] = [as_phrase(x) for x in p["phrases"]]
        WALK = p["walk"]


def split(start, dur):
    """Notated pieces of a note starting at ``start`` (beats into the measure): beat-aligned, tied."""
    out, t, end = [], start, start + dur
    while t < end - 1e-9:
        if abs(t - round(t)) > 1e-9:
            seg = min(end, math.floor(t) + 1) - t
        else:
            whole = math.floor(end - t + 1e-9)
            seg = next((v for v in (4, 3, 2, 1) if v <= whole), end - t)
        out.append(round(seg, 4)); t += seg
    return out


def measure_xml(no, notes, beats, *, first=False, meter_change=False, words_above=(), words_below=(), color=None):
    x = [f'<measure number="{no}">']
    if first or meter_change:
        x.append("<attributes>")
        if first: x.append("<divisions>4</divisions><key><fifths>-3</fifths></key>")
        x.append(f"<time><beats>{beats}</beats><beat-type>4</beat-type></time>")
        if first: x.append("<clef><sign>G</sign><line>2</line></clef>")
        x.append("</attributes>")
    for text, col, size, kind in words_above:
        c = f' color="{col}"' if col else ""
        weight = ' font-weight="bold"' if kind == "chapter" else ""
        tag = f'<words font-size="{size}"{weight}{c}>{text}</words>'
        x.append(f'<direction placement="above"><direction-type>{tag}</direction-type></direction>')
    for text, col, size in words_below:
        x.append(f'<direction placement="below"><direction-type><words font-size="{size}" color="{col}">{text}</words></direction-type></direction>')
    pos = 0.0
    nc = f' color="{color}"' if color else ""
    for p, d in notes:
        segs = split(pos, d)
        for k, s in enumerate(segs):
            name, dots = TYPE[s]
            tie = ""
            if p is not None and len(segs) > 1:
                ties = (["stop"] if k > 0 else []) + (["start"] if k < len(segs) - 1 else [])
                tie = "".join(f'<tie type="{t}"/>' for t in ties)
                tied = "".join(f'<tied type="{t}"/>' for t in ties)
            if p is None:
                x.append(f'<note{nc}><rest/><duration>{int(s * 4)}</duration><type>{name}</type>{"<dot/>" * dots}</note>')
            else:
                st, al, oc = SPELL[p]
                alt = f"<alter>{al}</alter>" if al else ""
                notations = f"<notations>{tied}</notations>" if tie else ""
                x.append(f'<note{nc}><pitch><step>{st}</step>{alt}<octave>{oc}</octave></pitch><duration>{int(s * 4)}</duration>'
                         f'{tie}<type>{name}</type>{"<dot/>" * dots}{notations}</note>')
        pos += d
    x.append("</measure>")
    return "".join(x)


def piece_xml(phrases, labels, measures, *, original=False, chapter_marks=True, letter_size=9):
    surf = ref.surface_types(phrases)
    body, prev_beats = [], None
    for k, t in enumerate(measures):
        notes = phrases[t]; beats = int(round(sum(d for _, d in notes)))
        above = []
        if chapter_marks and (t + 1) in CHAPTER_START:
            ch = CHAPTER_START[t + 1]
            above.append((f"제{ch}장" + (" · 환입" if t + 1 in (1, 25) else ""), None, 11, "chapter"))
        lab = LET[labels[t]]
        if surf[t] != labels[t]:
            above.append((f"{lab}→{LET[surf[t]]}", RED, letter_size, "words"))
        else:
            above.append((lab, None, letter_size, "words"))
        below, color = [], None
        src = [m for m in WHERE.get(tuple(notes), []) if not (original and m == t + 1)]
        if src and not (original and not src):
            color = BLUE
            below.append(("= " + ",".join(map(str, src[:3])), BLUE, 7))
        body.append(measure_xml(t + 1, notes, beats, first=(k == 0), meter_change=(k > 0 and beats != prev_beats),
                                words_above=above, words_below=below, color=color))
        prev_beats = beats
    return ('<?xml version="1.0" encoding="UTF-8"?><score-partwise version="3.1"><part-list><score-part id="P1">'
            '<part-name></part-name></score-part></part-list><part id="P1">' + "".join(body) + "</part></score-partwise>")


def toolkit(page_width, breaks, scale):
    tk = verovio.toolkit()
    tk.setOptions({"pageWidth": page_width, "pageHeight": 2970, "adjustPageHeight": True, "scale": scale,
                   "header": "none", "footer": "none", "breaks": breaks, "svgViewBox": True, "mnumInterval": 1,
                   "pageMarginLeft": 40, "pageMarginRight": 40, "pageMarginTop": 30, "pageMarginBottom": 30,
                   "spacingSystem": 14, "justifyVertically": False})
    return tk


def raster(svg, zoom, path=None):
    tmp = OUT / "_tmp.svg"; tmp.write_text(clean_verovio_svg(svg), encoding="utf-8")
    pix = pymupdf.open(str(tmp))[0].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False); tmp.unlink()
    if path: pix.save(str(path))
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)


def full_score(name, phrases, labels, original=False):
    tk = toolkit(2100, "auto", 38)
    tk.loadData(piece_xml(phrases, labels, range(72), original=original))
    paths = []
    for page in range(1, tk.getPageCount() + 1):
        path = OUT / f"{name}_p{page}.png"; raster(tk.renderToSVG(page), 2.2, path); paths.append(path.name)
    return paths


def staff(phrases, labels, measures, original=False):
    tk = toolkit(2400, "none", 40)
    tk.loadData(piece_xml(phrases, labels, measures, original=original, chapter_marks=False, letter_size=10))
    return raster(tk.renderToSVG(1), 3.0)


def excerpt_lambda():
    rows = [("원곡", ORIG, typ.labels, True)] + [(lab, PIECES[a], WALK, False) for a, lab in ARMS]
    ms = range(18, 24)  # measures 19–24
    imgs = [staff(ph, lb, ms, original) for _, ph, lb, original in rows]
    fig = plt.figure(figsize=(7.2, 1.15 * len(rows)))
    for i, ((lab, *_), img) in enumerate(zip(rows, imgs)):
        ax = fig.add_axes([0.13, 1 - (i + 1) / len(rows) + 0.01, 0.86, 1 / len(rows) - 0.02])
        ax.imshow(img, aspect="auto" if False else "equal"); ax.axis("off")
        fig.text(0.005, 1 - (i + 0.55) / len(rows), lab, fontsize=9, fontweight="bold", color=INK, va="center")
        if i == 1:
            fig.text(0.005, 1 - (i + 0.85) / len(rows), "(이하 같은\n새 유형열)", fontsize=7, color=GREY, va="center")
    fig.savefig(OUT.parent / "figures/f13_excerpt_lambda.png", dpi=220, bbox_inches="tight"); plt.close(fig)


def excerpt_return():
    ph = PIECES["diffusion_graph_025"]
    rows = [("원곡 1–4마디", ORIG, typ.labels, range(0, 4), True), ("생성곡 1–4마디", ph, WALK, range(0, 4), False),
            ("생성곡 25–28마디", ph, WALK, range(24, 28), False)]
    imgs = [staff(p, lb, ms, o) for _, p, lb, ms, o in rows]
    fig = plt.figure(figsize=(7.2, 3.3))
    for i, ((lab, *_), img) in enumerate(zip(rows, imgs)):
        ax = fig.add_axes([0.17, 1 - (i + 1) / 3 + 0.02, 0.82, 1 / 3 - 0.04]); ax.imshow(img); ax.axis("off")
        fig.text(0.005, 1 - (i + 0.5) / 3, lab, fontsize=9, fontweight="bold", color=INK, va="center")
    fig.savefig(OUT.parent / "figures/f14_excerpt_return.png", dpi=220, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    manifest = {"original": full_score("score_original", ORIG, typ.labels, original=True)}
    for arm, lab in ARMS:
        manifest[arm] = full_score(f"score_{arm}", PIECES[arm], WALK)
    excerpt_lambda(); excerpt_return()
    manifest["walk"] = "".join(LET[w] for w in WALK)
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))
