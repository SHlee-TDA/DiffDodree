"""Staff notation for the video: measures → ABC → Verovio SVG → Manim, with 율명 under each note.

The piece sits in E♭ (three flats), so the 율명 are spelled without accidentals:
㑣 B♭3, 㑲 C4, 黃 E♭4, 太 F4, 仲 A♭4, 林 B♭4, 南 C5, 潢 E♭5, 汰 F5 (MIDI numbers from piece.json).
"""
import hashlib, re
import xml.etree.ElementTree as ET
from pathlib import Path
import verovio
from manim import SVGMobject, VGroup, Text, Line, DOWN, UP

ABC = {58: "B,", 60: "C", 63: "E", 65: "F", 68: "A", 70: "B", 72: "c", 75: "e", 77: "f"}
HANGUL = {58: "하임", 60: "하남", 63: "황", 65: "태", 68: "중", 70: "임", 72: "남", 75: "청황", 77: "청태"}
STROKE = 1.5   # line weight relative to the engraving
CACHE = Path.home() / ".cache/diffdodree_video_media/scores"


def _split(pos, u, beat=4):
    """Split a duration (in 16ths) at beat lines unless it is whole beats starting on a beat."""
    out = []
    while u > 0:
        if pos % beat == 0 and u % beat == 0 and u in (4, 8, 12, 16):
            out.append(u); break
        if pos % beat == 0 and u > beat and u % beat != 0:
            k = (u // beat) * beat
            k = k if k in (4, 8, 12) else 4
            out.append(k); pos += k; u -= k; continue
        n = min(u, beat - pos % beat)
        out.append(n); pos += n; u -= n
    return out


def to_abc(measures, meter=4):
    """Return ABC text and, per emitted note/rest token, (measure, event, first_of_tie, midi, onset in beats)."""
    toks, info = [], []
    t0 = 0.0
    for mi, ev in enumerate(measures):
        pos, bar = 0, []
        for ei, (p, d) in enumerate(ev):
            u = int(round(d * 4))
            parts = _split(pos, u)
            for k, n in enumerate(parts):
                if pos and pos % 4 == 0:
                    bar.append(" ")
                name = "z" if p is None else ABC[p]
                tie = "-" if (p is not None and k < len(parts) - 1) else ""
                bar.append(f"{name}{n if n != 1 else ''}{tie}")
                info.append((mi, ei, k == 0, p, t0 + pos / 4))
                pos += n
        toks.append("".join(bar))
        t0 += pos / 4
    m = int(round(sum(d for _, d in measures[0])))
    # placeholder lyrics make Verovio leave room for the 율명 (stripped again before Manim sees them)
    syl = ["*" if (p is None or not first) else ("WWW" if len(HANGUL[p]) > 1 else "WW") for _, _, first, p, _ in info
           if p is not None]
    abc = (f"X:1\nT:\nL:1/16\nM:{m}/4\nK:Eb\n" + " | ".join(toks) + " |]\n"
           + "w: " + " ".join(syl) + "\n")
    return abc, info


def _clean(svg):
    """Drop what Manim would draw wrongly: hidden placeholders, lyrics; give strokes an explicit colour."""
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    root = ET.fromstring(svg)
    parent = {c: p for p in root.iter() for c in p}
    for el in list(root.iter()):
        cls = el.get("class", "")
        if any(k in cls.split() for k in ("placeholder", "verse", "pageMilestoneEnd")):
            parent[el].remove(el)
        elif el.get("stroke-width") is not None:
            el.set("stroke", "#000000")
    return ET.tostring(root, encoding="unicode")


def render_svg(measures, width_px=2400, spacing=0.9):
    abc, info = to_abc(measures)
    CACHE.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha1(f"{abc}|{width_px}|{spacing}".encode()).hexdigest()[:12]
    path = CACHE / f"{h}.svg"
    if not path.exists():
        tk = verovio.toolkit()
        tk.setOptions({"inputFrom": "abc", "pageWidth": width_px, "pageHeight": 2000, "scale": 100,
                       "adjustPageHeight": True, "header": "none", "footer": "none", "breaks": "none",
                       "svgViewBox": True, "xmlIdSeed": 1, "svgRemoveXlink": True,
                       "spacingLinear": 0.25, "spacingNonLinear": 0.6, "justifyVertically": False,
                       "pageMarginLeft": 20, "pageMarginRight": 20, "pageMarginTop": 0, "pageMarginBottom": 0})
        tk.loadData(abc)
        path.write_text(_clean(tk.renderToSVG(1)))
    ids = re.findall(r'<g id="([^"]+)" class="(note|rest)"', path.read_text())
    return path, ids, info


def staff(measures, width, color="#1F1D2B", labels=True, font="Apple SD Gothic Neo", lsize=22,
          lcolor=None, colors=None, drop=0.45):
    """A Manim group: the engraved staff, plus 율명 labels under the first note of every tie chain.

    Returns (group, notes, labs) where notes[i] is the i-th note/rest VGroup (token order of to_abc)
    and labs is a list of (token index, Text). `colors` optionally maps measure index → note colour.
    """
    path, ids, info = render_svg(measures)
    svg = SVGMobject(str(path))
    sw0 = [m.get_stroke_width() for m in svg.family_members_with_points()]
    sc = width / svg.width * STROKE      # Manim does not scale stroke widths with the shape
    svg.scale_to_fit_width(width)
    for m, w in zip(svg.family_members_with_points(), sw0):
        m.set_fill(color, opacity=m.get_fill_opacity())
        m.set_stroke(color, width=w * sc if w > 0 else 0, opacity=1)
    d = svg.id_to_vgroup_dict
    notes = [d.get(i, VGroup()) for i, _ in ids]
    if colors is not None:
        for (mi, *_), n in zip(info, notes):
            c = colors(mi) if callable(colors) else colors[mi]
            n.set_fill(c).set_stroke(c)
    labs = VGroup()
    pairs = []
    if labels:
        y = svg.get_bottom()[1] - drop
        for k, ((mi, ei, first, p, _), n) in enumerate(zip(info, notes)):
            if p is None or not first or len(n) == 0:
                continue
            c = lcolor if lcolor else (colors(mi) if callable(colors) else colors[mi]) if colors is not None else color
            t = Text(HANGUL[p], font=font, font_size=lsize, color=c)
            t.move_to([n.get_center()[0], y, 0])
            labs.add(t); pairs.append((k, t))
    return VGroup(svg, labs), notes, pairs, info


def spans(notes, info, nm, pad=0.32):
    """Horizontal extent of each measure on an engraved staff, from its notes."""
    lo, hi = [9e9] * nm, [-9e9] * nm
    for (mi, *_), n in zip(info, notes):
        if len(n):
            lo[mi] = min(lo[mi], n.get_left()[0]); hi[mi] = max(hi[mi], n.get_right()[0])
    b = [lo[0] - pad] + [(hi[m] + lo[m + 1]) / 2 for m in range(nm - 1)] + [hi[-1] + pad]
    return [(b[m], b[m + 1]) for m in range(nm)]


def time_to_x(notes, info, total, right):
    """Piecewise-linear map from beat time to x, through each note's onset."""
    ts, xs = [], []
    for (_, _, _, _, on), n in zip(info, notes):
        if len(n) and (not ts or on > ts[-1]):
            ts.append(on); xs.append(n.get_center()[0])
    ts.append(total); xs.append(right)
    import numpy as _np
    return lambda t: float(_np.interp(t, ts, xs))
