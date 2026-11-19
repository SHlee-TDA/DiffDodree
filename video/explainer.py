"""DiffDodree explainer (16:9, about six and a half minutes), built with Manim Community + Supertonic narration.

Render one scene:  manim -r 1920,1080 --fps 30 video/explainer.py S07_Manifold
All scenes:        bash video/build.sh
Data: site/data/piece.json (the 72 measures, types, transitions), video/data/manifold.json
(the web demo's manifold layout and curve tracks, from video/extract_manifold.mjs),
site/data/sweep.json (the λ sweep), video/audio/*.wav (narration, from video/make_audio.py).
Music is engraved with Verovio (video/score.py): staff notation with 율명 under each note.
"""
import json, random, sys
from pathlib import Path
import numpy as np
from manim import *

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent
PIECE = json.loads((ROOT / "site/data/piece.json").read_text())
MAN = json.loads((HERE / "data/manifold.json").read_text())
SWEEP = json.loads((ROOT / "site/data/sweep.json").read_text())
TIM = json.loads((HERE / "audio/timings.json").read_text())
from narration import SCENES
from score import staff, spans, time_to_x, HANGUL

config.frame_width = 16
config.frame_height = 9
BG, INK, MUTED, FAINT = "#F7F4ED", "#1F1D2B", "#5D5A6E", "#B9B6CC"
ACC, RED, SURF, EDGE, CARD, LINE = "#534AB7", "#B5382B", "#EDEBF8", "#2C2C2A", "#FFFFFF", "#E3E1EC"
TC = ["#534AB7", "#C2571F", "#16866A", "#B07C00", "#B8427A", "#3E7CB1", "#6E8B20", "#8A5A44", "#5F6B7A"]
config.background_color = BG
FONT = "Apple SD Gothic Neo"
NAMES = "ABCDEFGHI"
SCALE = PIECE["grid"]["pitches"]
LABELS = PIECE["labels"]
PHR = PIECE["phrases"]
MED = PIECE["medoid_phrases"]          # 0-based measure index of each type's representative
GEN = MAN["generated"]
GAP = 0.5           # silence after each narration line
FADE = 0.8          # scene-out fade
SPB = 60 / 90       # ♩ = 90, the web demo's default


def T(s, size=30, color=INK, weight=NORMAL, **kw):
    return Text(s, font=FONT, font_size=size, color=color, weight=weight, **kw)


def chip(text, color=INK, size=24, fill=CARD, weight=MEDIUM):
    t = T(text, size, color, weight=weight)
    box = RoundedRectangle(corner_radius=0.14, width=t.width + 0.5, height=t.height + 0.32, fill_color=fill,
                           fill_opacity=1, stroke_color=color, stroke_width=2)
    return VGroup(box, t.move_to(box))


def tokens(events):
    """Grid cells of a measure: 율명 where a note starts, — where it is held, 쉼 for a rest."""
    out = []
    for p, d in events:
        n = int(round(d * 4))
        out += [HANGUL[p] if p is not None else "쉼"] + ["—"] * (n - 1)
    return out


def tok_text(tok, color, size=22):
    if tok == "—":
        return T("—", size, MUTED)
    if tok == "?":
        return T("?", size + 4, MUTED, weight=BOLD)
    if tok == "쉼":
        return T("쉼", size - 2, MUTED)
    return T(tok, size if len(tok) == 1 else size - 4, color, weight=BOLD)


def cell_row(n, cs=0.7, color=TC[2], gap=0.14):
    cells = VGroup(*[Square(cs * 0.94, fill_color=CARD, fill_opacity=1, stroke_color=color, stroke_width=2.5)
                     .move_to([i * cs + (i // 4) * gap, 0, 0]) for i in range(n)])
    return cells.center()


def contour(events, w, h, color, sw=3, opacity=1.0):
    """A measure's pitch contour: a step line, x = time, y = scale degree."""
    total = sum(d for _, d in events); t = 0; pts = []
    for p, d in events:
        if p is not None:
            y = -h / 2 + h * SCALE.index(p) / (len(SCALE) - 1)
            pts += [[-w / 2 + w * t / total, y, 0], [-w / 2 + w * (t + d) / total, y, 0]]
        t += d
    return VMobject().set_points_as_corners(pts).set_stroke(color, sw, opacity)


class Base(Scene):
    KEY = ""
    TITLE = ""

    def setup(self):
        self.lines = {k: (sp, cap) for k, sp, cap in SCENES[self.KEY]}
        self.brand = T("DiffDodree", 22, MUTED, weight=BOLD).to_corner(UR, buff=0.32)
        self.add(self.brand)
        self.title = None
        if self.TITLE:
            self.title = T(self.TITLE, 34, INK, weight=BOLD).to_corner(UL, buff=0.38)
            self.add(self.title)
        self.cap = None
        self.line_end = 0.0

    def say(self, key):
        info = TIM[key]
        self.add_sound(str(HERE / "audio" / f"{key}.wav"))
        self.line_end = self.time + info["dur"] + GAP
        parts = info["caption"].split("|")
        txt = VGroup(*[T(p, 30, INK, weight=MEDIUM) for p in parts]).arrange(DOWN, buff=0.12)
        if txt.width > 14.2:
            txt.scale_to_fit_width(14.2)
        box = RoundedRectangle(corner_radius=0.18, width=txt.width + 0.6, height=txt.height + 0.36,
                               fill_color=CARD, fill_opacity=0.94, stroke_color=LINE, stroke_width=1.5)
        cap = VGroup(box, txt.move_to(box)).move_to([0, -3.82, 0]).set_z_index(20)
        if self.cap is not None:
            self.remove(self.cap)
        self.add(cap)
        self.cap = cap
        return info["dur"]

    def left(self):
        return max(0.0, self.line_end - self.time)

    def hold(self, extra=0.0):
        r = self.left() + extra
        if r > 1e-3:
            self.wait(r)

    def fill(self, frac=1.0, floor=0.3):
        """A run time that uses `frac` of what is left of the current line."""
        return max(floor, self.left() * frac)

    def clear_stage(self, run_time=0.7):
        keep = {id(self.brand), id(self.title), id(self.cap)}
        out = [m for m in self.mobjects if id(m) not in keep]
        if out:
            self.play(FadeOut(Group(*out)), run_time=run_time)

    def end_scene(self, extra=0.3):
        self.hold(extra)
        self.play(FadeOut(Group(*self.mobjects)), run_time=FADE)
        self.wait(0.25)


# ---------------------------------------------------------------- manifold drawing helpers
# The web demo's manifold (site/js/manifold.js, dumped by extract_manifold.mjs): one lobe per type,
# necks only between types whose measures touch, turned a quarter turn so the tall layout fits 16:9.
# Inside a lobe the curve uses the demo's chart (τ = place along the representative, η = distance
# from it); between lobes it runs along the neck's centre line. A change of type with no neck is
# not drawn: the curve stops in one lobe and starts again in the next.
R0, KETA, TRAVEL = MAN["R0"], 0.09, 0.12
NECK = {(min(n["a"], n["b"]), max(n["a"], n["b"])): n for n in MAN["necks"]}
LOBE = {l["c"]: l for l in MAN["lobes"]}


def mpt(x, y, s, c):
    """Canvas px → scene coordinates, turned a quarter turn."""
    return np.array([c[0] + (y - MAN["H"] / 2) * s, c[1] + (x - MAN["W"] / 2) * s, 0.0])


def fit_manifold(cx, cy, max_w, max_h):
    pts = np.array([mpt(l["x"] + dx, l["y"] + dy, 1.0, (0, 0)) for l in MAN["lobes"]
                    for dx, dy in ((l["R"], 0), (-l["R"], 0), (0, l["R"]), (0, -l["R"]))])
    lo, hi = pts.min(0), pts.max(0)
    s = min(max_w / (hi[0] - lo[0]), max_h / (hi[1] - lo[1]))
    return s, (cx - (lo[0] + hi[0]) / 2 * s, cy - (lo[1] + hi[1]) / 2 * s)


class Mani:
    def __init__(self, cx, cy, max_w, max_h):
        self.s, self.c = fit_manifold(cx, cy, max_w, max_h)
        self.cen = [self.p(LOBE[k]["x"], LOBE[k]["y"]) for k in range(9)]
        self.r = [LOBE[k]["R"] * self.s for k in range(9)]

    def p(self, x, y):
        return mpt(x, y, self.s, self.c)

    def neck_px(self, a, b, k=48):
        """Centre line of the neck from lobe a to lobe b, in canvas px."""
        nk = NECK[(min(a, b), max(a, b))]
        pa, pb = np.array([LOBE[a]["x"], LOBE[a]["y"]]), np.array([LOBE[b]["x"], LOBE[b]["y"]])
        ctrl = 2 * np.array([nk["mx"], nk["my"]]) - (pa + pb) / 2
        return [(1 - u) ** 2 * pa + 2 * (1 - u) * u * ctrl + u * u * pb for u in np.linspace(0, 1, k)]

    def neck_poly(self, nk, extra, color):
        pts = self.neck_px(nk["a"], nk["b"], 32)
        hw = (nk["w"] + extra) / 2
        L, R = [], []
        for i, q in enumerate(pts):
            d = pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]
            n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
            L.append(self.p(*(q + n * hw))); R.append(self.p(*(q - n * hw)))
        return Polygon(*(L + R[::-1]), fill_color=color, fill_opacity=1, stroke_width=0)

    def surface(self, tint=0.13):
        """Layers: 0 lobe rims, 1 neck rims, 2 lobes, 3 necks, 4 tints, 5 rings, 6 labels (index = type)."""
        s = self.s
        edge_l = VGroup(*[Circle(radius=(LOBE[k]["R"] + 1.6) * s, fill_color=EDGE, fill_opacity=1, stroke_width=0)
                          .move_to(self.cen[k]) for k in range(9)])
        edge_n = VGroup(*[self.neck_poly(nk, 3.2, EDGE) for nk in MAN["necks"]])
        surf_l = VGroup(*[Circle(radius=self.r[k], fill_color=SURF, fill_opacity=1, stroke_width=0).move_to(self.cen[k])
                          for k in range(9)])
        surf_n = VGroup(*[self.neck_poly(nk, 0, SURF) for nk in MAN["necks"]])
        tints = VGroup(*[Circle(radius=self.r[k], fill_color=TC[k], fill_opacity=tint, stroke_width=0).move_to(self.cen[k])
                         for k in range(9)])
        rings = VGroup(*[DashedVMobject(Circle(radius=self.r[k] * R0, stroke_color=MUTED, stroke_width=1.6, stroke_opacity=0.5),
                                        num_dashes=28).move_to(self.cen[k]) for k in range(9)])
        labels = VGroup(*[T(NAMES[k], 28, TC[k], weight=BOLD).move_to(self.p(*LOBE[k]["label"])) for k in range(9)])
        layers = VGroup(edge_l, edge_n, surf_l, surf_n, tints, rings, labels)
        for z, g in enumerate(layers):
            g.set_z_index(z - 10)       # under everything drawn on the surface (curves, dots, glows)
        return layers

    def lobe_px(self, c, tau, eta):
        l = LOBE[c]
        r = l["R"] * (R0 + KETA * max(-4, min(4, eta))); th = l["port"] + 2 * PI * tau
        return np.array([l["x"] + r * np.cos(th), l["y"] + r * np.sin(th)])

    def track(self, piece):
        """One entry per 1/16 beat: (scene point, type, kind) with kind 'lobe', 'neck' or 'gap' (no neck: not drawn)."""
        ch, me, DT = piece["chart"], piece["measures"], MAN["DT"]
        out = []
        for i, (pi, tau, eta) in enumerate(ch):
            ph = me[pi]; c = ph["c"]; t = min(piece["total"] - 1e-6, i * DT)
            s = (t - ph["a"]) / (ph["b"] - ph["a"])
            nx = me[pi + 1] if pi + 1 < len(me) else None
            if nx is not None and nx["c"] != c and s > 1 - TRAVEL:
                i3 = int(round(nx["a"] / DT))
                if (min(c, nx["c"]), max(c, nx["c"])) not in NECK:
                    out.append((self.p(*self.lobe_px(nx["c"], *ch[i3][1:])), nx["c"], "gap"))
                    continue
                i0 = int(np.ceil((ph["a"] + (1 - TRAVEL) * (ph["b"] - ph["a"])) / DT)) - 1
                path = self._travel(c, nx["c"], self.lobe_px(c, *ch[i0][1:]), self.lobe_px(nx["c"], *ch[i3][1:]))
                u = (s - (1 - TRAVEL)) / TRAVEL
                out.append((self.p(*self._along(path, u)), c, "neck"))
                continue
            out.append((self.p(*self.lobe_px(c, tau, eta)), c, "lobe"))
        return out

    def dots(self, piece):
        return [self.p(*self.lobe_px(piece["measures"][pi]["c"], tau, eta)) for pi, tau, eta in piece["dchart"]]

    def _travel(self, a, b, p0, p3):
        """From a point in lobe a out along the neck's centre line into lobe b, in canvas px."""
        ctr = self.neck_px(a, b, 80)
        A, B = np.array([LOBE[a]["x"], LOBE[a]["y"]]), np.array([LOBE[b]["x"], LOBE[b]["y"]])
        ia = next(i for i, q in enumerate(ctr) if np.linalg.norm(q - A) > LOBE[a]["R"] * 0.9)
        ib = max(i for i, q in enumerate(ctr) if np.linalg.norm(q - B) > LOBE[b]["R"] * 0.9)
        ea, eb = ctr[ia], ctr[ib]
        ca, cb = A + (ea - A) * 0.45, B + (eb - B) * 0.45
        q1 = [(1 - u) ** 2 * p0 + 2 * (1 - u) * u * ca + u * u * ea for u in np.linspace(0, 1, 16)]
        q3 = [(1 - u) ** 2 * eb + 2 * (1 - u) * u * cb + u * u * p3 for u in np.linspace(0, 1, 16)]
        return q1 + ctr[ia + 1:ib] + q3

    @staticmethod
    def _along(path, u):
        seg = [np.linalg.norm(path[k + 1] - path[k]) for k in range(len(path) - 1)]
        tot = sum(seg); want = u * tot; acc = 0.0
        for k, d in enumerate(seg):
            if acc + d >= want:
                return path[k] + (path[k + 1] - path[k]) * ((want - acc) / (d + 1e-12))
            acc += d
        return path[-1]


def track_segments(track, i0, i1, width=4.0):
    """Polylines for track[i0:i1+1], split where the colour (type / neck) changes; gaps are left out."""
    segs = VGroup()
    if i1 <= i0:
        return segs
    key = None; pts = []
    def flush():
        if len(pts) >= 2 and key[1] != "gap":
            cc, kind = key
            segs.add(VMobject(stroke_color=MUTED if kind == "neck" else TC[cc], stroke_width=width).set_points_as_corners(pts))
    for i in range(i0, min(i1, len(track) - 1) + 1):
        p, cc, kind = track[i]
        k = (cc, kind)
        if k != key:
            if pts and kind != "gap" and key[1] != "gap":
                pts.append(p)
            flush()
            key = k; pts = [p]
        else:
            pts.append(p)
    flush()
    return segs


# ---------------------------------------------------------------- transition graph (circle layout)
def circle_pos(center=(0.0, 0.0), R=2.5):
    return [np.array([center[0] + R * np.cos(PI / 2 - 2 * PI * k / 9), center[1] + R * np.sin(PI / 2 - 2 * PI * k / 9), 0])
            for k in range(9)]


def graph_mobs(pos, rad=0.36, size=30):
    Wt = PIECE["transition_counts"]
    nodes = VGroup()
    for c in range(9):
        circ = Circle(radius=rad, fill_color=CARD, fill_opacity=1, stroke_color=TC[c], stroke_width=5).move_to(pos[c])
        nodes.add(VGroup(circ, T(NAMES[c], size, TC[c], weight=BOLD).move_to(pos[c])))
    nodes.set_z_index(3)
    centre = np.mean(pos, axis=0)
    edges = {}
    for a in range(9):
        for b in range(9):
            n = Wt[a][b]
            if not n:
                continue
            sw = min(1.6 + 1.1 * n, 7.5)
            if a == b:
                out = pos[a] - centre; out = out / (np.linalg.norm(out) + 1e-9)
                ang = np.arctan2(out[1], out[0])
                p1 = pos[a] + rad * np.array([np.cos(ang - 0.6), np.sin(ang - 0.6), 0])
                p2 = pos[a] + rad * np.array([np.cos(ang + 0.6), np.sin(ang + 0.6), 0])
                e = CurvedArrow(p1, p2, angle=PI * 1.5, color=MUTED, stroke_width=sw, tip_length=0.13)
            else:
                d = pos[b] - pos[a]; u = d / np.linalg.norm(d)
                bend = 0.32 if Wt[b][a] else 0.0
                p0, p1 = pos[a] + u * rad * 1.08, pos[b] - u * rad * 1.12
                e = (CurvedArrow(p0, p1, angle=-bend, color=MUTED, stroke_width=sw, tip_length=0.16) if bend
                     else Arrow(p0, p1, buff=0, color=MUTED, stroke_width=sw, tip_length=0.16,
                                max_stroke_width_to_length_ratio=20, max_tip_length_to_length_ratio=0.2))
            e.set_stroke(opacity=0.5); e.set_z_index(1)
            edges[(a, b)] = e
    return nodes, edges


def strip(seq, cs=0.56, size=24, gap=0.08, fill=0.16):
    cells = VGroup()
    for cc in seq:
        sq = RoundedRectangle(corner_radius=0.07, width=cs, height=cs, fill_color=TC[cc], fill_opacity=fill,
                              stroke_color=TC[cc], stroke_width=2.5)
        cells.add(VGroup(sq, T(NAMES[cc], size, TC[cc], weight=BOLD).move_to(sq)))
    return cells.arrange(RIGHT, buff=gap)


# ================================================================ scenes
class S00_Title(Base):
    KEY = "S00_Title"

    def construct(self):
        logo = T("DiffDodree", 96, ACC, weight=BOLD).move_to([0, 1.2, 0])
        sub = T("한 곡에서 배워, 구조는 지키고 선율은 새롭게", 40, INK, weight=BOLD).next_to(logo, DOWN, buff=0.45)
        sm = T("〈양청도드리〉 피리 선율  ·  선율형 다양체  ·  국소 확산 모델", 26, MUTED).next_to(sub, DOWN, buff=0.4)
        self.say("t0a")
        self.play(FadeIn(logo, shift=DOWN * 0.3), run_time=1.2)
        self.play(FadeIn(sub, shift=UP * 0.2), run_time=1.0)
        self.play(FadeIn(sm), run_time=0.8)
        self.end_scene(1.0)


class S01_Data(Base):
    KEY = "S01_Data"

    def construct(self):
        rng = random.Random(3)
        cards = VGroup()
        for i in range(7):
            for j in range(8):
                col = rng.choice(TC)
                card = VGroup(RoundedRectangle(corner_radius=0.05, width=0.7, height=0.5, fill_color=CARD,
                                               fill_opacity=1, stroke_color=col, stroke_width=2))
                for k in range(3):
                    card.add(Line([-0.25, 0.12 - 0.12 * k, 0], [0.25 - rng.random() * 0.2, 0.12 - 0.12 * k, 0],
                                  stroke_color=col, stroke_width=2.5))
                cards.add(card.move_to([-6.9 + j * 0.85, 2.4 - i * 0.68, 0]))
        lab = T("수많은 곡", 28, MUTED, weight=BOLD).next_to(cards, DOWN, buff=0.3)
        box = RoundedRectangle(corner_radius=0.3, width=3.4, height=1.5, fill_color=ACC, fill_opacity=1,
                               stroke_width=0).move_to([1.2, 0.4, 0])
        ai = VGroup(box, T("생성 AI", 40, WHITE, weight=BOLD).move_to(box))
        pat = VGroup(*[FunctionGraph(lambda x, a=a: 0.25 * np.sin(2.2 * x + a), x_range=[-1.4, 1.4],
                                     color=TC[a], stroke_width=4).shift([5.2, 1.0 - 0.6 * a, 0]) for a in range(3)])
        plab = T("공통된 패턴 → 새 곡", 28, INK, weight=BOLD).next_to(pat, DOWN, buff=0.4)
        arr = Arrow([2.95, 0.4, 0], [3.7, 0.4, 0], color=MUTED, buff=0)
        self.say("s1a")
        self.play(LaggedStart(*[FadeIn(c, scale=0.6) for c in cards], lag_ratio=0.02), FadeIn(lab), run_time=2.0)
        self.play(FadeIn(ai, scale=0.9), run_time=0.8)
        self.play(LaggedStart(*[c.animate.scale(0.2).move_to(box.get_center()).set_opacity(0) for c in cards],
                              lag_ratio=0.01), FadeOut(lab), run_time=1.8)
        self.play(GrowArrow(arr), Create(pat), FadeIn(plab), run_time=1.2)
        self.end_scene()


class S02_Gugak(Base):
    KEY = "S02_Gugak"

    def construct(self):
        def stack(n, col, x, label):
            g = VGroup(*[RoundedRectangle(corner_radius=0.04, width=1.8, height=0.1, fill_color=col,
                                          fill_opacity=0.85, stroke_color=CARD, stroke_width=1).move_to([x, -1.9 + 0.12 * i, 0])
                         for i in range(n)])
            return VGroup(g, T(label, 28, INK, weight=BOLD).next_to(g, DOWN, buff=0.3).set_y(-2.5))
        many = stack(36, MUTED, -1.6, "대중음악 자료")
        few = stack(3, RED, 1.6, "국악 자료")
        note = T("(개념도)", 20, MUTED).move_to([0, 2.9, 0])
        self.say("s2a")
        self.play(LaggedStart(*[FadeIn(r, shift=UP * 0.2) for r in many[0]], lag_ratio=0.03), FadeIn(many[1]),
                  FadeIn(note), run_time=1.6)
        self.play(LaggedStart(*[FadeIn(r, shift=UP * 0.2) for r in few[0]], lag_ratio=0.2), FadeIn(few[1]), run_time=0.9)
        self.hold()
        self.play(FadeOut(VGroup(many, few, note)), run_time=0.7)

        # two pieces, each with its own purpose and form (한국민족문화대백과사전: 종묘제례악, 밀양아리랑, 세마치장단)
        def block(text, col, w=0.95, fill=0.18):
            r = RoundedRectangle(corner_radius=0.08, width=w, height=0.5, fill_color=col, fill_opacity=fill,
                                 stroke_color=col, stroke_width=2.5)
            return VGroup(r, T(text, 20, col, weight=BOLD).move_to(r))

        def card(x, col, name, kind, rows, form, form_lab):
            head = VGroup(T(name, 38, col, weight=BOLD), T(kind, 24, MUTED)).arrange(RIGHT, buff=0.25, aligned_edge=DOWN)
            body = VGroup()
            for k, v in rows:
                kk = T(k, 20, CARD, weight=BOLD)
                kb = RoundedRectangle(corner_radius=0.1, width=0.95, height=0.42, fill_color=col, fill_opacity=1, stroke_width=0)
                body.add(VGroup(VGroup(kb, kk.move_to(kb)), T(v, 25, INK)).arrange(RIGHT, buff=0.25))
            body.arrange(DOWN, aligned_edge=LEFT, buff=0.22)
            fl = T(form_lab, 22, MUTED)
            inner = VGroup(head, body, VGroup(fl, form).arrange(DOWN, buff=0.15)).arrange(DOWN, buff=0.35)
            inner[1].align_to(inner, LEFT); inner[0].align_to(inner, LEFT)
            bg = RoundedRectangle(corner_radius=0.25, width=7.1, height=inner.height + 0.7, fill_color=CARD, fill_opacity=1,
                                  stroke_color=col, stroke_width=3)
            return VGroup(bg, inner.move_to(bg)).move_to([x, 0.45, 0])

        C1, C2 = "#8A5A44", TC[1]
        f1 = VGroup(*[block(t, C1) for t in ["1장", "2장", "…", "7장"]]).arrange(RIGHT, buff=0.18)
        back = CurvedArrow(f1[-1].get_bottom() + DOWN * 0.04, f1[0].get_bottom() + DOWN * 0.04, angle=-PI / 4,
                           color=C1, stroke_width=3, tip_length=0.16)
        backt = T("끝에서 돌아든다", 18, C1, weight=BOLD).next_to(back, DOWN, buff=0.05)
        f1g = VGroup(f1, back, backt)
        c1 = card(-3.75, C1, "수연장지곡", "밑도드리",
                  [("자리", "궁중 잔치 음악 〈보허자〉에서 나옴"), ("뜻", "壽延長: 목숨이 길게 이어지기를"),
                   ("장단", "6박 한 장단, 일곱 장"), ("이름", "도드리 = 끝에서 돌아든다")],
                  f1g, "짜임: 장을 지나 끝에서 다시 돌아듦")
        f2 = VGroup(*[block(t, C2, w=1.05, fill=0.42 if t == "후렴" else 0.12) for t in ["절", "후렴", "절", "후렴"]]).arrange(RIGHT, buff=0.12)
        beats = VGroup(*[Dot(radius=0.07 if k % 3 == 0 else 0.045, color=C2 if k % 3 == 0 else FAINT) for k in range(12)])
        for k, d in enumerate(beats):
            d.move_to([k * 0.26 + (k // 3) * 0.16, 0, 0])
        f2g = VGroup(f2, VGroup(beats, T("세마치 3박", 18, C2)).arrange(RIGHT, buff=0.25)).arrange(DOWN, buff=0.15)
        c2 = card(3.75, C2, "〈밀양아리랑〉", "경상도 민요",
                  [("자리", "사람들이 함께 부르는 노래"), ("뜻", "삶의 정서를 함께 나눔"),
                   ("소리", "세마치장단, 비교적 빠르게"), ("느낌", "씩씩하고 경쾌하게")],
                  f2g, "짜임: 절과 후렴이 번갈아 되풀이")
        c2.align_to(c1, UP)
        src = T("(내용: 한국민족문화대백과사전 「밑도드리」 「보허사」 「밀양아리랑」 「세마치장단」)", 16, MUTED).move_to([0, 3.55, 0])
        self.say("s2b")
        self.play(FadeIn(VGroup(c1[0], c1[1][0], c1[1][1], c1[1][2][0]), shift=UP * 0.2), FadeIn(src), run_time=1.2)
        rows1 = c1[1][1]
        for r in rows1[:3]:
            self.play(Indicate(r[1], color=C1, scale_factor=1.04), run_time=self.fill(0.25, 0.6))
        self.hold()
        self.say("s2e")
        self.play(Indicate(rows1[3][1], color=C1, scale_factor=1.04), run_time=0.8)
        self.play(LaggedStart(*[FadeIn(b, shift=RIGHT * 0.1) for b in f1], lag_ratio=0.25), run_time=1.0)
        self.play(Create(back), FadeIn(backt), run_time=1.0)
        tag = T("→ 〈양청도드리〉도 도드리 계열", 22, ACC, weight=BOLD).next_to(c1, DOWN, buff=0.2)
        self.play(FadeIn(tag, shift=UP * 0.1), run_time=0.7)
        self.hold()
        self.play(FadeOut(tag), run_time=0.4)
        self.say("s2c")
        self.play(FadeIn(c2, shift=UP * 0.2), run_time=1.2)
        self.play(LaggedStart(*[d.animate(rate_func=there_and_back).scale(1.6) for d in beats], lag_ratio=0.12), run_time=2.0)
        self.play(*[Indicate(f2[k], color=C2, scale_factor=1.12) for k in (1, 3)], run_time=1.0)
        self.hold()
        # mix: the forms and the moods of both pieces shuffled into one strip
        self.say("s2d")
        pieces = list(f1) + list(f2)
        order = [4, 0, 6, 2, 5, 3, 7, 1]
        slots = VGroup(*[block("", MUTED, w=1.05) for _ in order]).arrange(RIGHT, buff=0.14).move_to([0, -1.55, 0])
        moods = VGroup(T("장수의 축원", 26, C1, weight=BOLD), T("+", 30, MUTED), T("씩씩하고 경쾌", 26, C2, weight=BOLD),
                       T("→ ?", 30, RED, weight=BOLD)).arrange(RIGHT, buff=0.25).move_to([0, 0.1, 0])
        keep = VGroup(*pieces)
        self.remove(*pieces)
        self.add(keep)
        self.play(FadeOut(VGroup(c1[0], c1[1][0], c1[1][1], c1[1][2][0], back, backt, c2[0], c2[1][0], c2[1][1], c2[1][2][0],
                                 f2g[1])), run_time=0.8)
        self.play(*[pieces[i].animate.move_to(slots[k]) for k, i in enumerate(order)], run_time=1.6)
        self.play(FadeIn(moods, shift=UP * 0.2), run_time=0.8)
        lost = T("뜻도 짜임도 흐려짐", 34, RED, weight=BOLD).next_to(slots, DOWN, buff=0.45)
        self.play(keep.animate.set_opacity(0.45), FadeIn(lost, shift=UP * 0.2), run_time=0.8)
        self.end_scene()


class S03_Question(Base):
    KEY = "S03_Question"

    def construct(self):
        a = T("단 한 곡만으로", 60, INK, weight=BOLD)
        b = T("구조는 지키고", 60, ACC, weight=BOLD)
        c = T("선율은 새롭게?", 60, RED, weight=BOLD)
        g = VGroup(a, b, c).arrange(DOWN, buff=0.45).move_to([0, 0.5, 0])
        self.say("s3a")
        self.play(FadeIn(a, shift=UP * 0.3), run_time=0.9)
        self.wait(0.9)
        self.play(FadeIn(b, shift=UP * 0.3), run_time=0.9)
        self.wait(0.7)
        self.play(FadeIn(c, shift=UP * 0.3), run_time=0.9)
        self.end_scene(0.8)


class S04_Motivation(Base):
    KEY = "S04_Motivation"
    TITLE = "연주자는 가락의 흐름으로 곡을 이해한다"

    def construct(self):
        rows = []
        for r, (a, y) in enumerate([(0, 1.35), (4, -1.45)]):
            g, notes, labs, info = staff(PHR[a:a + 4], 14.2, lsize=22)
            g.move_to([0, y, 0])
            num = T(f"{a + 1}마디", 20, MUTED).next_to(g, UP, buff=0.1).align_to(g, LEFT)
            rows.append((g, notes, labs, info, a, num))
        self.say("s4a")
        for g, notes, labs, info, a, num in rows:
            self.play(FadeIn(g[0]), FadeIn(num), run_time=1.0)
            self.play(LaggedStart(*[FadeIn(t, shift=UP * 0.1) for _, t in labs], lag_ratio=0.05), run_time=1.2)
        self.hold()
        self.say("s4b")
        boxes, letters = VGroup(), VGroup()
        anims = []
        for g, notes, labs, info, a, num in rows:
            sp = spans(notes, info, 4)
            top = g.get_top()[1] + 0.12; bot = g.get_bottom()[1] - 0.12
            for m, (x0, x1) in enumerate(sp):
                c = TC[LABELS[a + m]]
                bx = RoundedRectangle(corner_radius=0.1, width=x1 - x0 - 0.06, height=top - bot, fill_color=c,
                                      fill_opacity=0.09, stroke_color=c, stroke_width=2.5).move_to([(x0 + x1) / 2, (top + bot) / 2, 0])
                bx.set_z_index(-1)
                boxes.add(bx)
                letters.add(T(NAMES[LABELS[a + m]], 34, c, weight=BOLD).next_to(bx, UP, buff=0.08))
                col = [n for (mi, *_), n in zip(info, notes) if mi == m]
                lt = [t for k, t in labs if info[k][0] == m]
                anims.append((LABELS[a + m], bx, col, lt, c))
        # colour by type, one type at a time (pairs that look alike light up together)
        for c in [2, 3, 1, 0, 4]:
            grp = [x for x in anims if x[0] == c]
            self.play(*[FadeIn(bx) for _, bx, _, _, _ in grp],
                      *[n.animate.set_fill(col).set_stroke(col) for _, _, ns, _, col in grp for n in ns],
                      *[t.animate.set_color(col) for _, _, _, ts, col in grp for t in ts], run_time=0.9)
            self.wait(0.35)
        self.hold()
        self.say("s4c")
        self.play(LaggedStart(*[FadeIn(l, shift=DOWN * 0.15) for l in letters], lag_ratio=0.1), run_time=1.6)
        word = T("= 선율형", 34, INK, weight=BOLD).move_to([6.6, 3.2, 0])
        self.play(FadeIn(word), run_time=0.6)
        self.end_scene()


class S05_Extract(Base):
    KEY = "S05_Extract"
    TITLE = "① 선율형 찾기"

    def construct(self):
        ch = PIECE["chapters"]
        cw, chh, step = 0.66, 0.46, 0.76
        tiles, home, rowlabs = VGroup(), [], VGroup()
        for r, cinfo in enumerate(ch):
            a, b = cinfo["measures"]
            y = 2.75 - r * 0.72
            rowlabs.add(T(f"제{cinfo['chapter']}장", 20, MUTED).move_to([-7.15, y, 0]))
            for k, m in enumerate(range(a, b + 1)):
                x = -6.35 + k * step
                box = RoundedRectangle(corner_radius=0.06, width=cw, height=chh, fill_color=CARD, fill_opacity=1,
                                       stroke_color="#CFCBDC", stroke_width=1.5)
                t = VGroup(box, T(str(m), 18, MUTED).move_to(box)).move_to([x, y, 0])
                tiles.add(t); home.append(np.array([x, y, 0]))
        head = T("〈양청도드리〉 피리 · 72마디 (7장)", 26, INK, weight=BOLD).move_to([3.6, 3.82, 0])
        self.say("s5a")
        self.play(FadeIn(head), FadeIn(rowlabs), LaggedStart(*[FadeIn(t, scale=0.7) for t in tiles], lag_ratio=0.012),
                  run_time=3.0)
        self.hold()
        # edit distance on a real pair: type F's representative (m43) and a far variant (m30)
        self.say("s5b")
        self.play(FadeOut(tiles), FadeOut(rowlabs), FadeOut(head), run_time=0.7)
        gA, nA, lA, iA = staff([PHR[42]], 5.2, lsize=26)
        gB, nB, lB, iB = staff([PHR[29]], 5.2, lsize=26)
        gA.move_to([-3.2, 1.35, 0]); gB.move_to([-3.2, -1.45, 0])
        tA = T("43마디", 26, TC[5], weight=BOLD).next_to(gA, LEFT, buff=0.35)
        tB = T("30마디", 26, INK, weight=BOLD).next_to(gB, LEFT, buff=0.35)
        self.play(FadeIn(gA), FadeIn(tA), run_time=0.8)
        self.play(FadeIn(gB), FadeIn(tB), run_time=0.8)
        # three edits: 仲 仲 → 仲(2박) merge, 林 → 仲, 仲 → 南
        edits = [((0, 1), 0, "① 두 음 합치기"), ((2,), 1, "② 높이 바꾸기"), ((3,), 2, "③ 높이 바꾸기")]
        notes_info = []
        for k, (src, dst, word) in enumerate(edits):
            a = VGroup(*[nA[i] for i in src]); b = nB[dst]
            ra = SurroundingRectangle(a, color=RED, buff=0.1, corner_radius=0.06, stroke_width=3)
            rb = SurroundingRectangle(b, color=RED, buff=0.1, corner_radius=0.06, stroke_width=3)
            ln = DashedLine(ra.get_bottom(), rb.get_top(), color=RED, stroke_width=2.5, dash_length=0.08)
            lab = T(word, 26, RED, weight=BOLD).move_to([3.3, 1.0 - k * 0.75, 0])
            notes_info.append(VGroup(ra, rb, ln, lab))
        for g in notes_info:
            self.play(Create(g[0]), Create(g[1]), Create(g[2]), FadeIn(g[3], shift=LEFT * 0.2), run_time=0.9)
        cost = T("고친 횟수가 적을수록 가까운 두 마디", 28, INK, weight=BOLD).move_to([3.3, -1.6, 0])
        self.play(FadeIn(cost, shift=UP * 0.15), run_time=0.7)
        self.hold()
        self.say("s5c")
        self.play(FadeOut(VGroup(gA, gB, tA, tB, cost, *notes_info)), run_time=0.7)
        for i, t in enumerate(tiles):
            c = TC[LABELS[i]]
            t[0].set_fill(c, opacity=0.18).set_stroke(c, width=2)
            t[1].set_color(c)
        self.play(FadeIn(tiles), FadeIn(rowlabs), run_time=0.8)
        cnt = [0] * 9; grp = []
        for i in range(72):
            c = LABELS[i]; grp.append(np.array([-5.6 + cnt[c] * step, 2.95 - c * 0.66, 0])); cnt[c] += 1
        tlabs = VGroup(*[T(NAMES[c], 30, TC[c], weight=BOLD).move_to([-6.45, 2.95 - c * 0.66, 0]) for c in range(9)])
        self.play(FadeOut(rowlabs), *[t.animate.move_to(grp[i]) for i, t in enumerate(tiles)], run_time=2.0)
        self.play(FadeIn(tlabs), run_time=0.6)
        meds = [tiles[MED[c]] for c in range(9)]
        rings = VGroup(*[SurroundingRectangle(m, color=TC[c], buff=0.05, corner_radius=0.08, stroke_width=4.5)
                         for c, m in enumerate(meds)])
        mtag = VGroup(T("■", 20, ACC), T("= 대표 마디 (무리의 한가운데)", 22, INK)).arrange(RIGHT, buff=0.15).move_to([4.6, -2.75, 0])
        self.wait(max(0.0, self.left() - 2.6))
        self.play(*[m[0].animate.set_fill(TC[c], opacity=0.85) for c, m in enumerate(meds)],
                  *[m[1].animate.set_color(WHITE) for m in meds], Create(rings), FadeIn(mtag), run_time=1.0)
        self.hold()
        self.say("s5d")
        self.play(FadeOut(tlabs), FadeOut(rings), FadeOut(mtag),
                  *[t.animate.move_to(home[i]) for i, t in enumerate(tiles)], run_time=1.6)
        for c, m in enumerate(meds):
            m[0].set_fill(TC[c], opacity=0.18)
        letters = VGroup(*[T(NAMES[LABELS[i]], 22, TC[LABELS[i]], weight=BOLD).move_to(t) for i, t in enumerate(tiles)])
        self.play(FadeIn(rowlabs), *[FadeOut(t[1]) for t in tiles], FadeIn(letters), run_time=1.0)
        r1 = SurroundingRectangle(VGroup(*tiles[0:4]), color=RED, buff=0.06, stroke_width=4.5)
        r4 = SurroundingRectangle(VGroup(*tiles[24:28]), color=RED, buff=0.06, stroke_width=4.5)
        link = CurvedArrow(r1.get_left() + LEFT * 0.05, r4.get_left() + LEFT * 0.05, angle=PI * 0.5, color=RED,
                           stroke_width=4, tip_length=0.18)
        lab = T("환입: 제1장(1~4마디) = 제4장(25~28마디)", 26, RED, weight=BOLD).move_to([3.8, 2.75, 0])
        others = [k for k in range(72) if not (k < 4 or 24 <= k < 28)]
        self.wait(max(0.0, self.left() - 3.4))
        self.play(*[tiles[k].animate.set_opacity(0.3) for k in others], *[letters[k].animate.set_opacity(0.35) for k in others],
                  *[rowlabs[r].animate.set_opacity(0.25) for r in (1, 2, 4, 5, 6)], Create(r1), Create(r4), run_time=1.0)
        self.play(Create(link), FadeIn(lab), run_time=1.0)
        self.end_scene(0.6)


class S06_Graph(Base):
    KEY = "S06_Graph"
    TITLE = "② 선율형 전이 그래프"

    def construct(self):
        pos = circle_pos((-3.0, 0.05), 2.35)
        nodes, edges = graph_mobs(pos)
        evg = VGroup(*edges.values())
        self.say("s6a")
        self.play(LaggedStart(*[GrowFromCenter(n) for n in nodes], lag_ratio=0.1), run_time=1.6)
        self.play(LaggedStart(*[Create(e) for e in edges.values()], lag_ratio=0.03), run_time=3.2)
        note = VGroup(T("점 = 선율형", 26, INK, weight=BOLD), T("화살표 = 원곡에서 이어진 관계", 26, INK, weight=BOLD),
                      T("굵을수록 자주 이어짐", 22, MUTED), T("원곡 71번의 이어짐 → 화살표 41가지", 22, MUTED)
                      ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to([4.2, 1.4, 0])
        self.play(FadeIn(note, shift=LEFT * 0.2), run_time=0.8)
        self.hold()
        self.say("s6b")
        self.play(evg.animate.set_stroke(opacity=0.15), run_time=0.5)
        outs = [edges[(0, b)] for b in range(9) if (0, b) in edges]
        after = T("A 다음에 올 수 있는 것: B · F · G", 28, TC[0], weight=BOLD).move_to([4.2, -1.0, 0])
        self.play(*[e.animate.set_stroke(TC[0], opacity=1) for e in outs], nodes[0][0].animate.set_stroke(width=10),
                  FadeIn(after), run_time=1.2)
        self.hold()
        self.say("s6c")
        self.play(*[e.animate.set_stroke(MUTED, opacity=0.15) for e in outs], nodes[0][0].animate.set_stroke(width=5),
                  FadeOut(after), FadeOut(note), run_time=0.6)
        walk = GEN["walk"][:10]
        orig = LABELS[:10]
        rowO = strip(orig, 0.52, 22); rowN = strip(walk, 0.52, 22)
        lo = T("원곡", 24, MUTED, weight=BOLD); ln = T("새 진행", 24, ACC, weight=BOLD)
        VGroup(rowO, rowN).arrange(DOWN, buff=0.3).move_to([4.0, 0.6, 0])
        lo.next_to(rowO, LEFT, buff=0.25); ln.next_to(rowN, LEFT, buff=0.25)
        self.play(FadeIn(lo), FadeIn(rowO), FadeIn(ln), run_time=0.6)
        tok = Dot(pos[walk[0]], radius=0.15, color=RED).set_z_index(5)
        self.play(FadeIn(tok), FadeIn(rowN[0]), run_time=0.4)
        st = max(0.35, (self.left() - 1.2) / 9)
        for k in range(1, 10):
            a, b = walk[k - 1], walk[k]
            self.play(edges[(a, b)].animate.set_stroke(RED, opacity=1), tok.animate.move_to(pos[b]), FadeIn(rowN[k]),
                      run_time=st)
        diff = VGroup(*[SurroundingRectangle(rowN[k], color=RED, buff=0.05, stroke_width=3) for k in range(10) if walk[k] != orig[k]])
        dl = T("붉은 틀 = 원곡과 다른 자리", 22, RED).next_to(rowN, DOWN, buff=0.3)
        self.play(Create(diff), FadeIn(dl), run_time=0.8)
        self.end_scene(0.6)


class S07_Manifold(Base):
    KEY = "S07_Manifold"
    TITLE = "③ 선율형 다양체: 그래프에 살을 붙이기"

    def construct(self):
        M = Mani(-2.5, 0.0, 9.6, 5.9)
        gpos = circle_pos((-4.3, 0.05), 2.3)
        nodes, edges = graph_mobs(gpos)
        evg = VGroup(*edges.values())
        self.play(FadeIn(evg), FadeIn(nodes), run_time=0.8)
        # (a) a point is only a name
        self.say("s7a")
        q = VGroup(T("C", 40, TC[2], weight=BOLD), T("= 이름표", 30, INK, weight=BOLD)).arrange(RIGHT, buff=0.2)
        qq = T("실제 음은?", 30, RED, weight=BOLD)
        VGroup(q, qq).arrange(DOWN, buff=0.25).move_to([3.0, 1.2, 0])
        self.play(evg.animate.set_stroke(opacity=0.15), nodes[2].animate.scale(1.25), FadeIn(q), run_time=0.9)
        self.play(FadeIn(qq), run_time=0.5)
        self.hold()
        # (b) the graph alone can only paste representatives
        self.say("s7b")
        seq = GEN["walk"][:4]
        g, notes, labs, info = staff([PHR[MED[c]] for c in seq], 13.6, colors=[TC[c] for c in seq], lsize=22)
        g.move_to([0, 0.9, 0])
        sp = spans(notes, info, 4)
        tags = VGroup(*[T(f"{NAMES[c]}의 대표 마디", 20, TC[c], weight=BOLD).move_to([(x0 + x1) / 2, g.get_top()[1] + 0.3, 0])
                        for c, (x0, x1) in zip(seq, sp)])
        self.play(FadeOut(q), FadeOut(qq), nodes[2].animate.scale(0.8), FadeOut(evg), FadeOut(nodes), run_time=0.7)
        self.play(FadeIn(g, lag_ratio=0.1), FadeIn(tags), run_time=1.4)
        stamp = VGroup(T("짜깁기", 44, RED, weight=BOLD), T("새 선율은 생기지 않음", 24, RED)).arrange(DOWN, buff=0.12)
        stamp.move_to([0, -1.6, 0])
        self.play(FadeIn(stamp, scale=1.3), run_time=0.7)
        self.hold()
        # (c) members of one type are variations around the representative (type F)
        self.say("s7c")
        self.play(FadeOut(VGroup(g, tags, stamp)), run_time=0.7)
        ms = [(42, "43마디 · 대표 마디"), (53, "54마디 · 가까운 변주"), (30, "31마디 · 중간 변주"), (29, "30마디 · 먼 변주")]
        med = PHR[42]
        def sounding(t):
            on = 0
            for p, d in med:
                if on <= t < on + d:
                    return p
                on += d
        rows = VGroup()
        for r, (i, lab) in enumerate(ms):
            gg, nn, ll, ii = staff([PHR[i]], 4.6, lsize=20, drop=0.35)
            for (mi, ei, first, p, on), n in zip(ii, nn):
                if p is not None and r > 0 and sounding(on) != p:
                    n.set_fill(RED).set_stroke(RED)
            for k, t in ll:
                if r > 0 and sounding(ii[k][4]) != ii[k][3]:
                    t.set_color(RED)
            col = r % 2; rr = r // 2
            gg.move_to([-3.6 + col * 7.4, 1.25 - rr * 2.65, 0])
            tt = T(lab, 22, TC[5] if r == 0 else INK, weight=BOLD).next_to(gg, UP, buff=0.12).align_to(gg, LEFT).shift(RIGHT * 0.9)
            rows.add(VGroup(gg, tt))
        head = VGroup(T("선율형 F의 마디들", 26, TC[5], weight=BOLD), T("붉은 음 = 대표 마디와 다른 음", 20, RED)).arrange(RIGHT, buff=0.5)
        head.move_to([0, 3.0, 0])
        self.play(FadeIn(head), run_time=0.6)
        for row in rows:
            self.play(FadeIn(row, shift=UP * 0.15), run_time=0.9)
        self.hold()
        # (d) the width of variation differs by type (paper fig. 5)
        self.say("s7d")
        bars = VGroup()
        for k, (c, v) in enumerate([(3, 0.08), (1, 0.10), (7, 0.55)]):
            lab = T(NAMES[c], 28, TC[c], weight=BOLD)
            base = Rectangle(width=4.0, height=0.36, fill_color=LINE, fill_opacity=1, stroke_width=0)
            fill = Rectangle(width=4.0 * v, height=0.36, fill_color=TC[c], fill_opacity=1, stroke_width=0).align_to(base, LEFT)
            val = T(f"{round(v * 100)}%", 24, TC[c], weight=BOLD)
            row = VGroup(lab, VGroup(base, fill), val).arrange(RIGHT, buff=0.25)
            bars.add(row)
        bars.arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        bt = T("대표 마디와 다른 음의 비율", 22, INK, weight=BOLD)
        src = T("(논문 그림 5)", 18, MUTED)
        panel = VGroup(bt, bars, src).arrange(DOWN, buff=0.25)
        bgp = RoundedRectangle(corner_radius=0.2, width=panel.width + 0.6, height=panel.height + 0.5, fill_color=CARD,
                               fill_opacity=0.97, stroke_color=LINE, stroke_width=2)
        P = VGroup(bgp, panel.move_to(bgp)).move_to([0, -0.1, 0]).set_z_index(5)
        self.play(FadeIn(P, scale=0.95), run_time=0.8)
        self.hold(0.6)
        # (e) inflate points into lobes joined by necks: the melodic-type manifold
        self.say("s7e")
        self.play(FadeOut(P), FadeOut(rows), FadeOut(head), run_time=0.7)
        nodes, edges = graph_mobs(gpos)
        evg = VGroup(*edges.values())
        self.play(FadeIn(nodes), FadeIn(evg), run_time=0.8)
        surf = M.surface()
        necks_in = VGroup(surf[1], surf[3])
        # points move to their places and grow into lobes; necks join the lobes whose measures touch
        self.play(*[nodes[k].animate.move_to(M.cen[k]) for k in range(9)], FadeOut(evg), run_time=1.8)
        self.play(*[nodes[k][0].animate.scale(M.r[k] / 0.36).set_fill(SURF, 1).set_stroke(EDGE, 3) for k in range(9)],
                  *[ReplacementTransform(nodes[k][1], surf[6][k]) for k in range(9)], run_time=1.8)
        self.add(surf[0], surf[2]); self.remove(*[nodes[k][0] for k in range(9)])
        self.play(FadeIn(necks_in), run_time=1.2)
        self.play(FadeIn(surf[4]), FadeIn(surf[5]), run_time=0.8)
        dots = VGroup(*[Dot(p, radius=0.03, color=EDGE) for p in M.dots(MAN["original"])]).set_z_index(7)
        self.play(FadeIn(dots, lag_ratio=0.002), run_time=1.0)
        leg = VGroup(T("덩어리 = 한 선율형의 변주들이 모인 자리", 22, INK), T("목 = 선율형이 바뀌는 길", 22, INK),
                     T("검은 점 = 원곡의 실제 음", 22, INK), T("(모양은 개념도)", 18, MUTED)
                     ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([4.6, 1.5, 0])
        name = T("선율형 다양체", 40, ACC, weight=BOLD).move_to([5.4, -0.6, 0])
        self.play(FadeIn(leg), FadeIn(name, scale=1.1), run_time=0.9)
        self.hold()
        # (f) graph = skeleton, manifold = flesh
        self.say("s7f")
        tp = circle_pos((5.4, -1.0), 1.15)
        tn, te = graph_mobs(tp, rad=0.17, size=16)
        thumb = VGroup(*te.values(), tn)
        for e in te.values():
            e.set_stroke(width=1.4)
        arr = Arrow([2.5, -1.0, 0], [3.9, -1.0, 0], color=MUTED, buff=0)
        at = T("덩어리를 점으로 줄이면", 20, MUTED).next_to(arr, DOWN, buff=0.12).shift(LEFT * 0.4)
        self.play(FadeOut(leg), name.animate.move_to([5.4, 2.4, 0]), run_time=0.8)
        self.play(GrowArrow(arr), FadeIn(at), FadeIn(thumb), run_time=1.2)
        sk = VGroup(T("뼈대 = 전이 그래프", 24, INK, weight=BOLD), T("살 = 선율형마다의 변주", 24, INK, weight=BOLD)
                    ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to([5.4, 1.2, 0])
        self.play(FadeIn(sk), run_time=0.8)
        self.hold()
        # (g) the piece is one curve; the return is the same path walked twice
        self.say("s7g")
        self.play(FadeOut(VGroup(arr, at, thumb, sk)), run_time=0.6)
        tr = M.track(MAN["original"]); n = len(tr) - 1
        prog = ValueTracker(0)
        curve = always_redraw(lambda: track_segments(tr, 0, int(prog.get_value()), width=3.2).set_z_index(8))
        self.add(curve)
        ct = T("원곡 = 하나의 곡선", 26, INK, weight=BOLD).move_to([5.4, 1.4, 0])
        self.play(FadeIn(ct), prog.animate.set_value(n), run_time=max(2.0, self.left() - 2.4), rate_func=linear)
        curve.clear_updaters()
        me = MAN["original"]["measures"]; DT = MAN["DT"]
        i0, i1 = int(me[0]["a"] / DT), int(me[3]["b"] / DT)
        hw = track_segments(tr, i0, i1, width=11).set_stroke(RED, opacity=0.55).set_z_index(9)
        hl = T("환입 = 같은 길을 두 번", 26, RED, weight=BOLD).move_to([5.4, 0.6, 0])
        self.play(curve.animate.set_stroke(opacity=0.45), FadeIn(hw), FadeIn(hl), run_time=0.9)
        self.hold()
        # (h) a new piece is a new curve on the manifold
        self.say("s7h")
        g_tr = M.track(GEN); gn = int(GEN["measures"][11]["b"] / MAN["DT"])
        prog2 = ValueTracker(0)
        newc = always_redraw(lambda: track_segments(g_tr, 0, int(prog2.get_value()), width=5.0).set_z_index(10))
        self.add(newc)
        nt = VGroup(T("새 곡 = 새 곡선", 30, ACC, weight=BOLD), T("(생성곡 1~12마디)", 18, MUTED)).arrange(DOWN, buff=0.1).move_to([5.4, -0.5, 0])
        self.play(curve.animate.set_stroke(opacity=0.12), dots.animate.set_opacity(0.3), FadeOut(hw), FadeIn(nt),
                  prog2.animate.set_value(gn), run_time=max(2.0, self.left() - 0.4), rate_func=linear)
        newc.clear_updaters()
        self.end_scene(0.8)


class S08_Walk(Base):
    KEY = "S08_Walk"
    TITLE = "④ 새 선율형 순서 정하기"

    def construct(self):
        pos = circle_pos((-5.15, 0.0), 2.1)
        nodes, edges = graph_mobs(pos, rad=0.31, size=24)
        evg = VGroup(*edges.values()).set_stroke(opacity=0.18)
        idx = list(range(12)) + list(range(24, 28))
        def row(seq):
            cells = strip([seq[i] for i in idx], 0.46, 20, gap=0.06)
            for k in range(12, 16):
                cells[k].shift(RIGHT * 0.42)
            return cells
        rowO, rowN = row(LABELS), row(GEN["walk"])
        VGroup(rowO, rowN).arrange(DOWN, buff=0.5).move_to([2.85, 1.2, 0])
        lo = T("원곡", 22, MUTED, weight=BOLD).next_to(rowO, LEFT, buff=0.2)
        ln = T("새 순서", 22, ACC, weight=BOLD).next_to(rowN, LEFT, buff=0.2)
        nums = VGroup(*[T(str(i + 1), 15, MUTED).next_to(rowO[k], UP, buff=0.08) for k, i in enumerate(idx)])
        dots_ = T("…", 26, MUTED).move_to((rowO[11].get_center() + rowO[12].get_center()) / 2)
        dots2 = dots_.copy().set_y(rowN.get_y())
        self.say("s8a")
        self.play(FadeIn(nodes), FadeIn(evg), FadeIn(rowO), FadeIn(lo), FadeIn(nums), FadeIn(dots_), FadeIn(ln),
                  FadeIn(dots2), run_time=1.0)
        walk = GEN["walk"]
        tok = Dot(pos[walk[0]], radius=0.14, color=RED).set_z_index(6)
        self.play(FadeIn(tok), FadeIn(rowN[0]), run_time=0.5)
        st = 0.55
        for k in range(1, 12):
            a, b = walk[k - 1], walk[k]
            w = PIECE["transition_counts"][a]
            outs = [edges[(a, x)] for x in range(9) if w[x]]
            self.play(*[e.animate.set_stroke(TC[a], opacity=0.9) for e in outs], run_time=st * 0.4)
            self.play(*[e.animate.set_stroke(MUTED, opacity=0.18) for e in outs if e is not edges[(a, b)]],
                      edges[(a, b)].animate.set_stroke(RED, opacity=1), tok.animate.move_to(pos[b]), FadeIn(rowN[k]),
                      run_time=st * 0.6)
        self.play(LaggedStart(*[FadeIn(rowN[k]) for k in range(12, 16)], lag_ratio=0.2), run_time=0.8)
        f1 = DashedVMobject(SurroundingRectangle(rowN[0], color=INK, buff=0.07, stroke_width=3), num_dashes=16)
        h1 = SurroundingRectangle(VGroup(*rowN[0:4]), color=RED, buff=0.1, stroke_width=3.5)
        h2 = SurroundingRectangle(VGroup(*rowN[12:16]), color=RED, buff=0.1, stroke_width=3.5)
        rules = VGroup(T("점선 틀: 처음(과 끝) 마디 = 원곡과 같게", 21, INK),
                       T("붉은 틀: 환입 자리 1~4 = 25~28, 같은 선율형", 21, RED)).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        rules.next_to(rowN, DOWN, buff=0.45).align_to(rowN, LEFT)
        self.wait(max(0.0, self.left() - 2.2))
        self.play(Create(f1), Create(h1), Create(h2), FadeIn(rules), run_time=1.2)
        self.hold()
        self.say("s8b")
        diff = VGroup(*[SurroundingRectangle(rowN[k], color=ACC, buff=0.03, stroke_width=2.5)
                        for k, i in enumerate(idx) if walk[i] != LABELS[i]])
        stat = VGroup(T("뽑은 순서 2,000개", 24, INK, weight=BOLD),
                      T("원곡과 똑같은 순서: 0개", 24, INK),
                      T("원곡과 다른 자리: 평균 80%", 24, ACC, weight=BOLD)).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        stat.move_to([2.85, -1.75, 0])
        self.play(FadeOut(rules), Create(diff), run_time=0.8)
        self.play(FadeIn(stat, shift=UP * 0.15), run_time=0.8)
        self.end_scene(0.8)


class S09_Grid(Base):
    KEY = "S09_Grid"
    TITLE = "⑤ 마디를 칸으로 적기"

    def construct(self):
        ev = PHR[2]
        g, notes, labs, info = staff([ev], 7.4, lsize=28)
        g.move_to([0, 1.75, 0])
        tl = T("3마디 (선율형 C)", 22, TC[2], weight=BOLD).next_to(g, LEFT, buff=0.3)
        cs = 0.78
        cells = cell_row(16, cs).move_to([0, -1.2, 0])
        beats = VGroup(*[T(f"{b + 1}박", 20, MUTED).next_to(VGroup(*cells[4 * b:4 * b + 4]), DOWN, buff=0.14) for b in range(4)])
        braces = VGroup(*[Line(cells[4 * b].get_corner(UL) + UP * 0.12, cells[4 * b + 3].get_corner(UR) + UP * 0.12,
                               stroke_color=FAINT, stroke_width=3) for b in range(4)])
        self.say("s9a")
        self.play(FadeIn(g), FadeIn(tl), run_time=1.0)
        self.play(LaggedStart(*[FadeIn(c, shift=DOWN * 0.1) for c in cells], lag_ratio=0.04), FadeIn(beats),
                  FadeIn(braces), run_time=1.6)
        cnt = T("4박 × 4칸 = 16칸", 26, INK, weight=BOLD).move_to([5.6, -0.2, 0])
        self.play(FadeIn(cnt), run_time=0.6)
        self.hold()
        self.say("s9b")
        toks = tokens(ev)
        txt = VGroup(*[tok_text(t, TC[2], 24).move_to(cells[i]) for i, t in enumerate(toks)])
        on = 0
        steps = []
        for k, (p, d) in enumerate(ev):
            n = int(round(d * 4)); steps.append((k, on, n)); on += n
        per = max(0.6, (self.left() - 2.0) / len(steps))
        for k, a, n in steps:
            note = notes[k]
            hl = SurroundingRectangle(note, color=RED, buff=0.08, stroke_width=3)
            span = SurroundingRectangle(VGroup(*cells[a:a + n]), color=RED, buff=0.05, stroke_width=3)
            self.play(Create(hl), Create(span), FadeIn(txt[a]), *[FadeIn(txt[i]) for i in range(a + 1, a + n)],
                      *[cells[i].animate.set_fill(TC[2], opacity=0.12) for i in range(a, a + n)], run_time=per * 0.7)
            self.play(FadeOut(hl), FadeOut(span), run_time=per * 0.3)
        leg = VGroup(VGroup(T("남", 22, TC[2], weight=BOLD), T("새 음의 시작 (음 이름)", 22, INK)).arrange(RIGHT, buff=0.2),
                     VGroup(T("—", 22, MUTED), T("앞 소리의 이어짐", 22, INK)).arrange(RIGHT, buff=0.2),
                     VGroup(T("쉼", 20, MUTED), T("쉼의 시작", 22, INK)).arrange(RIGHT, buff=0.2)
                     ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to([5.6, 1.7, 0])
        self.play(FadeIn(leg), run_time=0.7)
        self.end_scene(0.8)


def pixel_note():
    bmp = ["........", "...##...", "...#.#..", "...#..#.", "...#....", ".###....", "####....", ".##....."]
    return [[1 if ch == "#" else 0 for ch in r] for r in bmp]


class S10_Diffusion(Base):
    KEY = "S10_Diffusion"
    TITLE = "⑥ 확산 모델(Diffusion): 지우는 과정을 거꾸로 배우기"

    def construct(self):
        ts = [0, 0.25, 0.5, 0.75, 1.0]
        xs = [-5.6, -2.8, 0.0, 2.8, 5.6]
        bmp = pixel_note(); rng = np.random.default_rng(2)
        noise = rng.random((len(ts), 8, 8))
        imgs = VGroup()
        for k, t in enumerate(ts):
            px = VGroup()
            for i in range(8):
                for j in range(8):
                    base = ManimColor(ACC) if bmp[i][j] else ManimColor(CARD)
                    col = interpolate_color(base, interpolate_color(ManimColor(CARD), ManimColor(INK), noise[k, i, j]), t)
                    px.add(Square(0.19, fill_color=col, fill_opacity=1, stroke_width=0).move_to([j * 0.19, -i * 0.19, 0]))
            imgs.add(px.center().move_to([xs[k], 1.55, 0]))
        tlab = VGroup(*[T(lab, 18, MUTED).next_to(imgs[k], UP, buff=0.12) for k, lab in
                        enumerate(["원래 자료", "조금 흐트러짐", "더 흐트러짐", "거의 사라짐", "아무 형체 없음"])])
        fwd = Arrow([-6.6, 0.25, 0], [6.6, 0.25, 0], color=MUTED, buff=0, stroke_width=3)
        fl = T("흐트러뜨리기 (정해 둔 과정)", 20, MUTED).next_to(fwd, DOWN, buff=0.08).shift(RIGHT * 3.6)
        bwd = Arrow([6.6, -0.35, 0], [-6.6, -0.35, 0], color=ACC, buff=0, stroke_width=5)
        bl = T("되돌리기 (모델이 배우는 것)", 22, ACC, weight=BOLD).next_to(bwd, DOWN, buff=0.08).shift(LEFT * 3.4)
        side1 = T("그림", 22, INK, weight=BOLD).next_to(imgs[0], LEFT, buff=0.35)
        self.say("s10a")
        self.play(FadeIn(imgs[0]), FadeIn(tlab[0]), FadeIn(side1), run_time=0.8)
        self.play(GrowArrow(fwd), FadeIn(fl), LaggedStart(*[FadeIn(imgs[k]) for k in range(1, 5)], lag_ratio=0.5),
                  *[FadeIn(tlab[k]) for k in range(1, 5)], run_time=2.6)
        self.wait(max(0.0, self.left() - 3.2))
        self.play(GrowArrow(bwd), FadeIn(bl), run_time=1.4)
        self.hold()
        # masked diffusion on the grid: cells are erased instead of noised
        self.say("s10b")
        toks = tokens(PHR[2])
        order = list(range(16)); random.Random(4).shuffle(order)
        grids = VGroup()
        for k, t in enumerate(ts):
            hidden = set(order[:round(16 * t)])
            cs = 0.36
            g = VGroup()
            for i in range(16):
                r, cc = divmod(i, 4)
                sq = Square(cs * 0.94, fill_color=FAINT if i in hidden else CARD, fill_opacity=0.55 if i in hidden else 1,
                            stroke_color=TC[2], stroke_width=1.5).move_to([cc * cs, -r * cs, 0])
                tt = tok_text("?" if i in hidden else toks[i], TC[2], 13).move_to(sq)
                g.add(VGroup(sq, tt))
            grids.add(g.center().move_to([xs[k], -1.75, 0]))
        side2 = T("음표 칸", 22, INK, weight=BOLD).next_to(grids[0], LEFT, buff=0.35)
        mk = T("마스크형 확산: 잡음 대신 칸을 지운다 (? = 지운 칸)", 22, TC[2], weight=BOLD).move_to([0, -2.85, 0])
        self.play(imgs.animate.set_opacity(0.35), FadeIn(side2), LaggedStart(*[FadeIn(g) for g in grids], lag_ratio=0.4),
                  run_time=2.4)
        self.play(FadeIn(mk), run_time=0.6)
        self.hold()
        # analogy: filling a partly lost old score from its style and context
        self.say("s10c")
        self.play(FadeOut(VGroup(imgs, tlab, fwd, fl, bwd, bl, side1, side2, grids, mk)), run_time=0.8)
        g, notes, labs, info = staff(PHR[0:2], 9.0, lsize=24)
        g.move_to([0, 0.6, 0])
        old = RoundedRectangle(corner_radius=0.2, width=10.2, height=3.2, fill_color="#F1E9D6", fill_opacity=1,
                               stroke_color="#C9B88F", stroke_width=2).move_to(g).set_z_index(-1)
        hide = [3, 4, 7]
        blots = VGroup()
        for k in hide:
            n = notes[k]
            lab = [t for kk, t in labs if kk == k]
            tgt = VGroup(n, *lab)
            b = RoundedRectangle(corner_radius=0.1, width=tgt.width + 0.25, height=g.height * 0.95, fill_color="#D8CDB3",
                                 fill_opacity=1, stroke_width=0).move_to([n.get_center()[0], g.get_center()[1], 0])
            blots.add(VGroup(b, T("?", 30, MUTED, weight=BOLD).move_to(b)))
        ol = T("일부가 지워진 옛 악보 (예시)", 22, "#8A7A55", weight=BOLD).next_to(old, UP, buff=0.15)
        self.play(FadeIn(old), FadeIn(g), FadeIn(blots), FadeIn(ol), run_time=1.2)
        hint = T("양식과 앞뒤 가락에 기대어 채운다", 26, ACC, weight=BOLD).next_to(old, DOWN, buff=0.35)
        self.wait(1.2)
        for k in hide:
            notes[k].set_fill(ACC).set_stroke(ACC)
        for kk, t in labs:
            if kk in hide:
                t.set_color(ACC)
        self.play(FadeOut(blots, lag_ratio=0.4), FadeIn(hint), run_time=1.8)
        self.end_scene(0.8)


def nn_icon(w=3.0, h=1.5, layers=(4, 6, 6, 4), color=ACC):
    """The usual drawing of a neural network: layers of nodes joined by thin lines."""
    xs = np.linspace(-w / 2, w / 2, len(layers))
    cols = [[np.array([x, y, 0]) for y in np.linspace(-h / 2, h / 2, n)] for x, n in zip(xs, layers)]
    lines = VGroup(*[Line(p, q, stroke_color=color, stroke_width=1.2, stroke_opacity=0.35)
                     for c0, c1 in zip(cols, cols[1:]) for p in c0 for q in c1])
    nodes = VGroup(*[Circle(radius=0.085, fill_color=CARD, fill_opacity=1, stroke_color=color, stroke_width=2.5).move_to(p)
                     for col in cols for p in col])
    return VGroup(lines, nodes)


def diff_staff(i, med_i, width, color, lsize=20):
    """Staff of measure i with the notes that differ from the representative measure in red."""
    med = PHR[med_i]
    def sounding(t):
        on = 0
        for p, d in med:
            if on <= t < on + d:
                return p
            on += d
    g, nn, ll, ii = staff([PHR[i]], width, color=color, lsize=lsize, drop=0.35)
    if i != med_i:
        for (mi, ei, first, p, on), n in zip(ii, nn):
            if p is not None and sounding(on) != p:
                n.set_fill(RED).set_stroke(RED)
        for k, t in ll:
            if sounding(ii[k][4]) != ii[k][3]:
                t.set_color(RED)
    return g


class S11_Train(Base):
    KEY = "S11_Train"
    TITLE = "⑦ 국소 확산 모델의 학습"

    def construct(self):
        mi = 8                                  # measure 9, type C
        toks = tokens(PHR[mi])
        cs = 0.6
        inp = cell_row(16, cs).move_to([0.6, 2.25, 0])
        it = VGroup(*[tok_text(t, TC[2], 18).move_to(inp[i]) for i, t in enumerate(toks)])
        il = T("원곡 9마디 (C)", 20, TC[2], weight=BOLD).next_to(inp, LEFT, buff=0.25)
        net = nn_icon(3.0, 1.3).move_to([0.6, 0.35, 0])
        nt = T("신경망", 30, ACC, weight=BOLD).next_to(net, RIGHT, buff=0.4)
        out = cell_row(16, cs).move_to([0.6, -1.5, 0])
        ol = T("맞힌 결과", 20, ACC, weight=BOLD).next_to(out, LEFT, buff=0.25)
        a1 = Arrow(inp.get_bottom(), net.get_top(), color=MUTED, buff=0.15)
        a2 = Arrow(net.get_bottom(), out.get_top(), color=MUTED, buff=0.15)
        order = list(range(16)); random.Random(11).shuffle(order)
        def masked(frac):
            return set(order[:round(16 * frac)])
        hid = masked(0.4)
        self.say("s11a")
        self.play(FadeIn(inp), FadeIn(it), FadeIn(il), run_time=0.9)
        qs = VGroup(*[tok_text("?", MUTED, 18).move_to(inp[i]) for i in hid])
        self.play(*[it[i].animate.set_opacity(0) for i in hid], *[inp[i].animate.set_fill(FAINT, 0.5) for i in hid],
                  FadeIn(qs), run_time=0.8)
        ratio = T("가린 비율 40%", 20, MUTED).next_to(inp, UP, buff=0.15).align_to(inp, RIGHT)
        self.play(FadeIn(ratio), FadeIn(net[0]), LaggedStart(*[FadeIn(n, scale=0.5) for n in net[1]], lag_ratio=0.03),
                  FadeIn(nt), GrowArrow(a1), run_time=1.0)
        guess = VGroup(*[tok_text(toks[i], ACC, 18).move_to(out[i]) for i in hid])
        wrong = sorted(hid)[2]
        guess_w = tok_text("임", RED, 18).move_to(out[wrong])
        self.play(GrowArrow(a2), FadeIn(out), FadeIn(ol), *[FadeIn(g) for g, i in zip(guess, hid) if i != wrong],
                  FadeIn(guess_w), run_time=1.0)
        marks = VGroup(*[T("✓" if i != wrong else "✕", 18, TC[2] if i != wrong else RED, weight=BOLD).next_to(out[i], DOWN, buff=0.1)
                         for i in hid])
        fix = T("틀린 만큼 고쳐 배운다 (예시)", 22, RED, weight=BOLD).next_to(out, DOWN, buff=0.55)
        self.play(FadeIn(marks), FadeIn(fix), run_time=0.7)
        self.wait(max(0.0, self.left() - 2.4))
        for frac in (0.15, 0.7, 1.0):
            h2 = masked(frac)
            self.play(*[inp[i].animate.set_fill(FAINT if i in h2 else CARD, 0.5 if i in h2 else 1) for i in range(16)],
                      *[it[i].animate.set_opacity(0 if i in h2 else 1) for i in range(16)],
                      Transform(qs, VGroup(*[tok_text("?", MUTED, 18).move_to(inp[i]) for i in h2])),
                      Transform(ratio, T(f"가린 비율 {round(frac * 100)}%", 20, MUTED).next_to(inp, UP, buff=0.15).align_to(inp, RIGHT)),
                      run_time=0.6)
        self.hold()

        # clues read off the score: measures 8–10, the middle one being learned
        self.say("s11b")
        self.clear_stage()
        trio = [mi - 1, mi, mi + 1]
        cols3 = [TC[LABELS[k]] for k in trio]
        g, notes, labs, info = staff([PHR[k] for k in trio], 11.0, colors=cols3, lsize=20)
        g.move_to([-1.2, 1.95, 0])
        sp = spans(notes, info, 3)
        mids = [(x0 + x1) / 2 for x0, x1 in sp]
        ytop = g.get_top()[1] + 0.32
        nums = VGroup(*[T(f"{k + 1}마디", 18, MUTED).move_to([mids[j], ytop + 0.45, 0]) for j, k in enumerate(trio)])
        lets = VGroup(*[T(NAMES[LABELS[k]], 30, TC[LABELS[k]], weight=BOLD).move_to([mids[j], ytop, 0]) for j, k in enumerate(trio)])
        veil = Rectangle(width=sp[1][1] - sp[1][0] + 0.1, height=g.height + 0.1, fill_color=FAINT, fill_opacity=0.35,
                         stroke_color=MUTED, stroke_width=2).move_to([mids[1], g.get_center()[1], 0])
        vt = T("배우는 마디 (일부 가림)", 18, MUTED).next_to(veil, DOWN, buff=0.1)
        self.play(FadeIn(g), FadeIn(nums), run_time=1.0)
        self.play(FadeIn(veil), FadeIn(vt), run_time=0.6)

        def clue(num, text):
            t = VGroup(T(num, 26, ACC, weight=BOLD), T(text, 24, INK, weight=MEDIUM)).arrange(RIGHT, buff=0.18)
            box = RoundedRectangle(corner_radius=0.14, width=6.4, height=0.66, fill_color=CARD, fill_opacity=1,
                                   stroke_color=ACC, stroke_width=2)
            t.move_to(box).align_to(box, LEFT).shift(RIGHT * 0.25)
            return VGroup(box, t)
        k1 = clue("①", f"이 마디의 선율형: {NAMES[LABELS[mi]]}")
        k2 = clue("②", f"앞뒤 마디의 선율형: {NAMES[LABELS[mi - 1]]} · {NAMES[LABELS[mi + 1]]}")
        k3 = clue("③", f"대표 마디({MED[LABELS[mi]] + 1}마디)의 오르내림")
        prev_last = [p for p, _ in PHR[mi - 1] if p is not None][-1]
        next_first = [p for p, _ in PHR[mi + 1] if p is not None][0]
        k4 = clue("④", f"맞닿는 음: {HANGUL[prev_last]} · {HANGUL[next_first]}")
        clues = VGroup(k1, k2, k3, k4).arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to([-3.75, -1.3, 0])
        def give(src, box):
            ghost = src.copy().set_z_index(8)
            self.play(ghost.animate.move_to(box.get_center()).scale(0.5).set_opacity(0), FadeIn(box), run_time=1.0)
            self.remove(ghost)
        self.play(FadeIn(lets[1], scale=1.3), run_time=0.7)
        give(lets[1], k1)
        self.wait(max(0.3, self.left() * 0.25))
        self.play(FadeIn(lets[0], scale=1.3), FadeIn(lets[2], scale=1.3), run_time=0.7)
        give(VGroup(lets[0], lets[2]), k2)
        self.hold()
        # ③ the representative's rise and fall, ④ the notes at the joins
        self.say("s11c")
        rep_i = MED[LABELS[mi]]
        rg, rn, rl, ri = staff([PHR[rep_i]], 4.4, color=TC[2], lsize=18, drop=0.35)
        rg.move_to([3.7, -1.0, 0])
        heads = [n.get_center() for n, inf in zip(rn, ri) if inf[3] is not None]
        wave = VMobject().set_points_smoothly([h + UP * 0.32 for h in heads])
        wave = DashedVMobject(wave.set_stroke(ACC, 4), num_dashes=24)
        rt = T(f"C의 대표 마디 ({rep_i + 1}마디)", 20, TC[2], weight=BOLD).next_to(rg, UP, buff=0.45)
        self.play(FadeIn(rg), FadeIn(rt), run_time=0.9)
        self.play(Create(wave), run_time=1.2)
        give(wave, k3)
        pl = [n for n, inf in zip(notes, info) if inf[0] == 0 and inf[3] is not None][-1]
        nf = [n for n, inf in zip(notes, info) if inf[0] == 2 and inf[3] is not None][0]
        rings = VGroup(*[Circle(radius=0.3, stroke_color=RED, stroke_width=4).move_to(n) for n in (pl, nf)])
        self.play(Create(rings), run_time=0.8)
        give(rings, k4)
        self.hold()
        # the clues are packed into short codes and fed to the network; no position in the piece
        self.say("s11d")
        net = nn_icon(3.0, 1.6).move_to([4.9, -1.0, 0])
        nt = T("신경망", 28, ACC, weight=BOLD).next_to(net, UP, buff=0.25)
        self.play(FadeOut(VGroup(rg, rt, wave)), FadeIn(net), FadeIn(nt), run_time=0.9)
        codes = VGroup()
        for k, cl in enumerate(clues):
            row = VGroup(*[Square(0.2, fill_color=ACC, fill_opacity=0.25 + 0.6 * ((k * 3 + j * 5) % 7) / 7, stroke_width=0)
                           for j in range(5)]).arrange(RIGHT, buff=0.04)
            codes.add(row.move_to([1.9, cl.get_y(), 0]))
        self.play(*[ReplacementTransform(cl.copy(), cd) for cl, cd in zip(clues, codes)], run_time=1.4)
        firsts = net[1][:4]
        feeds = VGroup(*[Arrow(cd.get_right(), firsts[3 - k].get_left(), buff=0.1, color=ACC, stroke_width=2.5,
                               tip_length=0.14) for k, cd in enumerate(codes)])
        self.play(LaggedStart(*[GrowArrow(f) for f in feeds], lag_ratio=0.2),
                  LaggedStart(*[n.animate(rate_func=there_and_back).set_fill(ACC, 0.9) for n in net[1]], lag_ratio=0.02),
                  run_time=1.6)
        nope = VGroup(T("✕", 30, RED, weight=BOLD), T("곡 안의 위치 (몇 번째 마디)", 22, RED)).arrange(RIGHT, buff=0.2)
        nope.move_to([4.9, -2.55, 0])
        self.play(FadeIn(nope, shift=UP * 0.15), run_time=0.7)
        self.hold()
        # local: each type learns how its measures vary near the representative
        self.say("s11e")
        self.clear_stage()
        colsD = VGroup(); colsH = VGroup()
        for col, c, rep, vs, x in ((colsD, 3, MED[3], [4, 5], -3.6), (colsH, 7, MED[7], [43, 20], 4.0)):
            head = T(f"선율형 {NAMES[c]}", 30, TC[c], weight=BOLD).move_to([x - 1.0, 3.0, 0])
            col.add(head)
            for r, i in enumerate([rep] + vs):
                st = diff_staff(i, rep, 4.4, TC[c] if r == 0 else INK)
                st.move_to([x, 1.75 - r * 1.7, 0])
                lab = VGroup(T(f"{i + 1}마디", 18, TC[c] if r == 0 else MUTED, weight=BOLD),
                             T("대표" if r == 0 else "변주", 18, TC[c] if r == 0 else MUTED)).arrange(DOWN, buff=0.06)
                col.add(VGroup(st, lab.next_to(st, LEFT, buff=0.2)))
        mid = Line([0, 3.2, 0], [0, -2.9, 0], stroke_color=LINE, stroke_width=2)
        key = T("붉은 음 = 대표 마디와 다른 음", 20, RED).move_to([0, 3.55, 0])
        self.play(FadeIn(mid), FadeIn(key), FadeIn(colsD[0]), FadeIn(colsH[0]), FadeIn(colsD[1]), FadeIn(colsH[1]), run_time=1.0)
        self.play(LaggedStart(*[FadeIn(r, shift=UP * 0.15) for r in list(colsD[2:]) + list(colsH[2:])], lag_ratio=0.35),
                  run_time=self.fill(0.7, 1.5))
        wD = T("변주 폭 8%: 거의 그대로", 22, TC[3], weight=BOLD).next_to(colsD[0], RIGHT, buff=0.3)
        wH = T("변주 폭 55%: 자유롭게", 22, TC[7], weight=BOLD).next_to(colsH[0], RIGHT, buff=0.3)
        self.play(FadeIn(wD), FadeIn(wH), run_time=0.7)
        self.hold()
        self.say("s11f")
        boxD = SurroundingRectangle(colsD[1:], color=TC[3], buff=0.12, corner_radius=0.2, stroke_width=3)
        boxH = SurroundingRectangle(colsH[1:], color=TC[7], buff=0.12, corner_radius=0.2, stroke_width=3)
        self.play(Create(boxD), Create(boxH), run_time=1.0)
        self.wait(max(0.3, self.left() - 3.0))
        name = VGroup(T("국소 확산 모델", 34, ACC, weight=BOLD), T("선율형마다 대표 마디 가까이의 변주를 배운다", 22, INK, weight=MEDIUM)
                      ).arrange(DOWN, buff=0.15)
        nb = RoundedRectangle(corner_radius=0.2, width=name.width + 0.8, height=name.height + 0.45, fill_color=CARD,
                              fill_opacity=0.97, stroke_color=ACC, stroke_width=3)
        badge = VGroup(nb, name.move_to(nb)).move_to([0, 0.05, 0]).set_z_index(6)
        self.play(VGroup(colsD, colsH).animate.set_opacity(0.25), FadeIn(badge, scale=1.15), run_time=0.8)
        self.end_scene(0.8)


class S12_Generate(Base):
    KEY = "S12_Generate"
    TITLE = "⑧ 새 마디 만들기"

    def construct(self):
        k = 2                                   # generated measure 3, type C
        ev = GEN["phrases"][k]; walk = GEN["walk"]
        toks = tokens(ev)
        prev_last = [p for p, _ in GEN["phrases"][k - 1] if p is not None][-1]
        next_first = [p for p, _ in GEN["phrases"][k + 1] if p is not None][0]
        chips = VGroup(chip("선율형: C", TC[2], 22), chip("대표 마디의 윤곽", INK, 22),
                       chip(f"앞뒤: {NAMES[walk[k - 1]]} · {NAMES[walk[k + 1]]}", INK, 22),
                       chip(f"맞닿는 음: {HANGUL[prev_last]} · {HANGUL[next_first]}", INK, 22)).arrange(RIGHT, buff=0.25)
        chips.move_to([0, 2.65, 0])
        cs = 0.78
        cells = cell_row(16, cs).move_to([0, 1.0, 0])
        qs = [tok_text("?", MUTED, 24).move_to(cells[i]) for i in range(16)]
        for c in cells:
            c.set_fill(FAINT, 0.45)
        counter = T("0 / 24 단계", 26, INK, weight=BOLD).next_to(cells, DOWN, buff=0.35)
        self.say("s12a")
        self.play(FadeIn(chips, lag_ratio=0.15), run_time=1.0)
        self.play(FadeIn(cells), *[FadeIn(q) for q in qs], FadeIn(counter), run_time=1.0)
        self.hold()
        self.say("s12b")
        order = list(range(16)); random.Random(5).shuffle(order)
        groups = [order[2 * i:2 * i + 2] for i in range(8)]
        per = max(0.5, (self.left() - 0.8) / 8)
        for g_i, grp in enumerate(groups):
            step = 3 * (g_i + 1)
            self.play(*[Transform(qs[i], tok_text(toks[i], TC[2], 24).move_to(cells[i])) for i in grp],
                      *[cells[i].animate.set_fill(TC[2], 0.12) for i in grp],
                      Transform(counter, T(f"{step} / 24 단계", 26, INK, weight=BOLD).move_to(counter)), run_time=per * 0.75)
            self.wait(per * 0.25)
        sch = T("(드러나는 순서는 도식, 칸의 내용은 실제 생성 마디)", 18, MUTED).next_to(counter, DOWN, buff=0.12)
        self.play(FadeIn(sch), run_time=0.4)
        self.hold()
        self.say("s12c")
        self.play(FadeOut(counter), FadeOut(sch), FadeOut(chips), VGroup(cells, *qs).animate.shift(UP * 1.2), run_time=0.8)
        gN, *_ = staff([ev], 5.8, color=TC[2], lsize=24)
        gM, *_ = staff([PHR[MED[2]]], 5.8, lsize=24)
        gM.move_to([-3.7, -1.15, 0]); gN.move_to([3.7, -1.15, 0])
        lM = T("C의 대표 마디 (원곡 4마디)", 22, INK, weight=BOLD).next_to(gM, UP, buff=0.2)
        lN = T("새로 만든 마디", 22, TC[2], weight=BOLD).next_to(gN, UP, buff=0.2)
        arr = Arrow(cells.get_bottom() + DOWN * 0.05, gN.get_top() + UP * 0.45, color=MUTED, buff=0.05)
        self.play(GrowArrow(arr), FadeIn(gN), FadeIn(lN), run_time=1.0)
        self.play(FadeIn(gM), FadeIn(lM), run_time=0.8)
        tag = T("닮았지만 같지 않은, 그 선율형다운 새 가락", 26, ACC, weight=BOLD).move_to([0, -2.85, 0])
        self.play(FadeIn(tag), run_time=0.6)
        self.end_scene(0.8)


class S13_Loss(Base):
    KEY = "S13_Loss"
    TITLE = "⑨ 구조 손실과 환입"

    def construct(self):
        seq = GEN["walk"][:8]
        given = strip(seq, 0.7, 28).move_to([0.6, 2.3, 0])
        gl = T("주어진 순서", 22, ACC, weight=BOLD).next_to(given, LEFT, buff=0.3)
        made = VGroup(*[VGroup(RoundedRectangle(corner_radius=0.08, width=0.7, height=0.55, fill_color=CARD, fill_opacity=1,
                                                stroke_color=FAINT, stroke_width=2), T("♪", 24, MUTED))
                        for _ in seq])
        for m, gcell in zip(made, given):
            m[1].move_to(m[0]); m.move_to(gcell.get_center() + DOWN * 1.05)
        ml = T("만든 마디", 22, MUTED, weight=BOLD).next_to(made, LEFT, buff=0.3).align_to(gl, RIGHT)
        bad = 2
        read = [7 if i == bad else c for i, c in enumerate(seq)]
        back = strip(read, 0.7, 28).move_to(given.get_center() + DOWN * 2.1)
        bl = T("다시 읽은 선율형", 22, INK, weight=BOLD).next_to(back, LEFT, buff=0.3).align_to(gl, RIGHT)
        marks = VGroup(*[T("✕" if i == bad else "✓", 24, RED if i == bad else TC[2], weight=BOLD).next_to(back[i], DOWN, buff=0.12)
                         for i in range(8)])
        ex = T("(예시)", 18, MUTED).next_to(back, RIGHT, buff=0.3)
        self.say("s13a")
        self.play(FadeIn(given), FadeIn(gl), run_time=0.8)
        self.play(LaggedStart(*[FadeIn(m, shift=DOWN * 0.15) for m in made], lag_ratio=0.12), FadeIn(ml), run_time=1.4)
        self.play(LaggedStart(*[FadeIn(b, shift=DOWN * 0.15) for b in back], lag_ratio=0.12), FadeIn(bl), FadeIn(ex), run_time=1.4)
        self.play(FadeIn(marks), Indicate(back[bad], color=RED), run_time=0.9)
        self.hold()
        self.say("s13b")
        new = T("학습 중에는 원곡에 없는 새 순서로 만들게 한다", 22, MUTED).move_to([0.6, 3.25, 0])
        def arcs(row, seq_, col, extra=None):
            g = VGroup()
            for i in range(len(seq_) - 1):
                cc = RED if extra and i in extra else col
                g.add(ArcBetweenPoints(row[i].get_bottom() + DOWN * 0.05, row[i + 1].get_bottom() + DOWN * 0.05,
                                       angle=PI / 2.4, stroke_color=cc, stroke_width=3 if cc == col else 4.5))
            return g
        bad_edges = {bad - 1, bad}
        ab = arcs(back, read, FAINT, bad_edges)
        pen = VGroup(T("주어진 순서에 없던 연결", 22, RED), T("어긋난 만큼 벌점 = 구조 손실", 30, RED, weight=BOLD)
                     ).arrange(DOWN, buff=0.12).move_to([0.6, -1.35, 0])
        self.play(FadeIn(new), run_time=0.6)
        self.play(FadeOut(marks), Create(ab), run_time=1.2)
        self.play(FadeIn(pen, shift=UP * 0.15), run_time=0.8)
        self.hold()
        # total loss with a weight
        self.say("s13c")
        self.play(FadeOut(VGroup(given, gl, made, ml, back, bl, ex, ab, pen, new)), run_time=0.7)
        b1 = chip("가린 칸을 맞히는 손실", ACC, 28)
        plus = T("+", 40, INK, weight=BOLD)
        lam = T("λ ×", 36, RED, weight=BOLD)
        b2 = chip("구조 손실", RED, 28)
        eq = VGroup(b1, plus, lam, b2).arrange(RIGHT, buff=0.3).move_to([0, 1.6, 0])
        self.play(FadeIn(b1), run_time=0.6)
        self.play(FadeIn(plus), FadeIn(lam), FadeIn(b2), run_time=0.8)
        ax = NumberLine(x_range=[0, 2, 0.5], length=8, color=MUTED, include_numbers=False).move_to([0, -0.6, 0])
        ticks = VGroup(*[T(f"{v:g}", 20, MUTED).next_to(ax.n2p(v), DOWN, buff=0.15) for v in (0, 0.5, 1, 1.5, 2)])
        knob = Dot(ax.n2p(0), radius=0.14, color=RED).set_z_index(3)
        kl = T("λ = 구조 손실의 가중치", 24, RED, weight=BOLD).next_to(ax, UP, buff=0.35)
        ends = VGroup(T("0: 구조 손실 없이", 20, MUTED).next_to(ax.n2p(0), DOWN, buff=0.6),
                      T("클수록 구조를 더 강하게", 20, MUTED).next_to(ax.n2p(2), DOWN, buff=0.6))
        self.play(Create(ax), FadeIn(ticks), FadeIn(knob), FadeIn(kl), FadeIn(ends), run_time=1.0)
        self.play(knob.animate.move_to(ax.n2p(1)), run_time=1.2)
        self.play(knob.animate.move_to(ax.n2p(0.1)), run_time=1.0)
        self.hold()
        # the return: both places drawn as one melody
        self.say("s13d")
        self.play(FadeOut(VGroup(eq, ax, ticks, knob, kl, ends)), run_time=0.7)
        def slot(lab, x):
            b = RoundedRectangle(corner_radius=0.12, width=3.6, height=1.25, fill_color=CARD, fill_opacity=1,
                                 stroke_color=RED, stroke_width=3).move_to([x, 1.75, 0])
            return VGroup(b, T(lab, 24, RED, weight=BOLD).move_to(b))
        s1, s4 = slot("제1장 (1~4마디)", -3.6), slot("제4장 (25~28마디)", 3.6)
        ctx1 = chip("앞뒤 문맥: 곡의 시작 · 5마디", MUTED, 20).next_to(s1, UP, buff=0.2)
        ctx4 = chip("앞뒤 문맥: 24마디 · 29마디", MUTED, 20).next_to(s4, UP, buff=0.2)
        mid = RoundedRectangle(corner_radius=0.2, width=4.6, height=0.9, fill_color=ACC, fill_opacity=1, stroke_width=0).move_to([0, 0.0, 0])
        mt = T("두 문맥을 함께 보고 한 번 뽑기", 24, WHITE, weight=BOLD).move_to(mid)
        a1 = Arrow(s1.get_bottom(), mid.get_left() + UP * 0.1, color=MUTED, buff=0.1)
        a4 = Arrow(s4.get_bottom(), mid.get_right() + UP * 0.1, color=MUTED, buff=0.1)
        self.play(FadeIn(s1), FadeIn(s4), FadeIn(ctx1), FadeIn(ctx4), run_time=1.0)
        self.play(GrowArrow(a1), GrowArrow(a4), FadeIn(mid), FadeIn(mt), run_time=1.0)
        mel, *_ = staff(GEN["phrases"][0:2], 5.4, lsize=20, drop=0.32)
        m1 = mel.copy().move_to([-3.6, -1.85, 0]); m4 = mel.copy().move_to([3.6, -1.85, 0])
        part = T("(새 선율의 처음 두 마디)", 18, MUTED).move_to([0, -2.75, 0])
        d1 = Arrow(mid.get_bottom(), m1.get_top(), color=RED, buff=0.12); d4 = Arrow(mid.get_bottom(), m4.get_top(), color=RED, buff=0.12)
        same = T("=", 44, RED, weight=BOLD).move_to([0, -1.9, 0])
        self.play(GrowArrow(d1), GrowArrow(d4), FadeIn(m1), FadeIn(m4), FadeIn(part), run_time=1.2)
        self.play(FadeIn(same, scale=1.3), run_time=0.5)
        self.end_scene(0.8)


class S13b_Flesh(Base):
    KEY = "S13b_Flesh"
    TITLE = "⑩ 신경망이 다양체에 살을 붙인다"

    def construct(self):
        M = Mani(-2.5, 0.0, 9.6, 5.9)
        surf = M.surface()
        odots = VGroup(*[Dot(p, radius=0.032, color=EDGE) for p in M.dots(MAN["original"])]).set_z_index(7)
        # (a) the original fills only a few points of each lobe
        self.say("s13e")
        self.play(FadeIn(surf), run_time=1.2)
        self.play(FadeIn(odots, lag_ratio=0.003), run_time=1.6)
        k1 = VGroup(Dot(radius=0.07, color=EDGE), T("원곡의 음 (72마디)", 22, INK)).arrange(RIGHT, buff=0.2)
        k1.move_to([4.6, 2.6, 0])
        self.play(FadeIn(k1), run_time=0.6)
        self.hold()
        # (b) the trained network adds new variations: the flesh
        self.say("s13f")
        net = nn_icon(2.4, 1.2).move_to([4.6, 0.4, 0])
        nt = T("학습한 신경망", 24, ACC, weight=BOLD).next_to(net, UP, buff=0.2)
        self.play(FadeIn(net), FadeIn(nt), run_time=0.9)
        newp = M.dots(GEN)
        ndots = VGroup(*[Dot(p, radius=0.034, color=ACC) for p in newp]).set_z_index(8)
        starts = [Dot(net.get_center(), radius=0.034, color=ACC).set_z_index(8) for _ in newp]
        k2 = VGroup(Dot(radius=0.07, color=ACC), T("새 변주의 음 (생성곡 1번)", 22, ACC)).arrange(RIGHT, buff=0.2)
        k2.next_to(k1, DOWN, buff=0.25).align_to(k1, LEFT)
        self.play(LaggedStart(*[st.animate.move_to(d.get_center()) for st, d in zip(starts, ndots)], lag_ratio=0.004),
                  LaggedStart(*[n.animate(rate_func=there_and_back).set_fill(ACC, 0.9) for n in net[1]], lag_ratio=0.03),
                  FadeIn(k2), run_time=self.fill(0.55, 2.5))
        self.remove(*starts); self.add(ndots)
        self.play(surf[4].animate.set_fill(opacity=0.32), odots.animate.set_opacity(0.6), run_time=1.0)
        lab = T("살 = 선율형마다 가능한 변주", 24, INK, weight=BOLD).move_to([4.6, -1.2, 0])
        self.play(FadeIn(lab), run_time=0.6)
        self.hold()
        # (c) generation: order along the skeleton, a new curve over the new flesh
        self.say("s13g")
        walk = GEN["walk"][:12]
        st = strip(walk, cs=0.36, size=16, gap=0.05).move_to([4.6, -2.05, 0])
        sl = T("① 뼈대를 따라 정한 순서", 20, MUTED, weight=BOLD).next_to(st, UP, buff=0.12)
        self.play(FadeOut(lab), FadeOut(net), FadeOut(nt), FadeIn(sl), FadeIn(st), run_time=0.8)
        tr = M.track(GEN); gn = int(GEN["measures"][11]["b"] / MAN["DT"])
        prog = ValueTracker(0)
        cur = always_redraw(lambda: track_segments(tr, 0, int(prog.get_value()), width=5.0).set_z_index(10))
        self.add(cur)
        cl = T("② 살 위로 그린 새 곡선", 22, ACC, weight=BOLD).move_to([4.6, 0.6, 0])
        self.play(FadeIn(cl), odots.animate.set_opacity(0.25), ndots.animate.set_opacity(0.55),
                  prog.animate.set_value(gn), run_time=max(2.5, self.left() - 0.3), rate_func=linear)
        cur.clear_updaters()
        self.end_scene(0.8)


class S14_Play(Base):
    KEY = "S14_Play"
    TITLE = "⑪ 새로 만든 곡 듣기"
    NMEAS = 5

    def construct(self):
        from synth import render
        g = GEN
        M = Mani(-1.6, 1.0, 10.4, 4.6)
        surf = M.surface(tint=0.10)
        tr = M.track(g); me = g["measures"]; DT = MAN["DT"]
        nbeats = me[self.NMEAS - 1]["b"]
        iend = int(nbeats / DT)
        ghost = track_segments(tr, 0, iend, width=2.0).set_stroke(FAINT, opacity=0.8).set_z_index(7)
        wav = HERE / "audio" / "music_gen.wav"
        render(g["phrases"][: self.NMEAS], 90, str(wav))
        walk = g["walk"]
        sg, notes, labs, info = staff(g["phrases"][: self.NMEAS], 14.6, colors=[TC[walk[m]] for m in range(self.NMEAS)], lsize=20)
        sg.move_to([0, -2.45, 0])
        sp = spans(notes, info, self.NMEAS)
        let = VGroup(*[T(NAMES[walk[m]], 26, TC[walk[m]], weight=BOLD).move_to([(x0 + x1) / 2, sg.get_top()[1] + 0.25, 0])
                       for m, (x0, x1) in enumerate(sp)])
        meta = VGroup(T("λ = 0.1 모델 · 생성 1번 곡 1~5마디", 20, MUTED),
                      T("♩ = 90 · 합성음 (피리 녹음 아님)", 20, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to([6.0, 0.4, 0])
        self.say("s14a")
        self.play(FadeIn(surf), run_time=1.0)
        self.play(Create(ghost), FadeIn(sg), FadeIn(let), FadeIn(meta), run_time=1.6)
        self.hold(0.2)
        if self.cap is not None:
            self.play(FadeOut(self.cap), run_time=0.3); self.cap = None
        beat = ValueTracker(0)
        tx = time_to_x(notes, info, nbeats, sp[-1][1] - 0.1)
        yb, yt = sg.get_bottom()[1] - 0.05, sg[0].get_top()[1] + 0.05
        head = always_redraw(lambda: Line([tx(beat.get_value()), yb, 0], [tx(beat.get_value()), yt, 0], color=RED, stroke_width=4))
        curve = always_redraw(lambda: track_segments(tr, 0, max(1, int(beat.get_value() / DT)), width=5.5).set_z_index(8))
        def cur_m():
            b = beat.get_value()
            return min(self.NMEAS - 1, sum(1 for x in me[: self.NMEAS] if x["a"] <= b) - 1)
        def glow_m():
            k = walk[cur_m()]
            return Circle(radius=M.r[k], fill_color=TC[k], fill_opacity=0.30, stroke_width=0).move_to(M.cen[k]).set_z_index(4)
        glow = always_redraw(glow_m)
        def cur_point():
            i = min(iend - 1, int(beat.get_value() / DT)); return tr[i][0]
        dot = always_redraw(lambda: VGroup(Circle(radius=0.12, fill_color=CARD, fill_opacity=1, stroke_color=EDGE, stroke_width=4),
                                           Dot(radius=0.08, color=EDGE)).move_to(cur_point()).set_z_index(10))
        def now_text():
            m = cur_m(); cc = walk[m]
            return VGroup(T(f"{m + 1}마디", 30, MUTED, weight=BOLD), T("선율형", 26, MUTED), T(NAMES[cc], 54, TC[cc], weight=BOLD)
                          ).arrange(RIGHT, buff=0.22).move_to([6.0, 1.6, 0])
        now = always_redraw(now_text)
        self.add(glow, curve, head, dot, now)
        self.add_sound(str(wav))
        self.play(beat.animate.set_value(nbeats), run_time=nbeats * SPB, rate_func=linear)
        x0, x1 = sp[0][0], sp[3][1]
        ret = Line([x0, yb - 0.15, 0], [x1, yb - 0.15, 0], color=RED, stroke_width=4)
        rett = T("1~4마디 = 25~28마디 (환입 자리)", 20, RED, weight=BOLD).next_to(ret, DOWN, buff=0.08)
        self.play(FadeIn(ret), FadeIn(rett), run_time=0.6)
        self.wait(1.4)
        self.end_scene(0)


class S15_Results(Base):
    KEY = "S15_Results"
    TITLE = "결과"

    def construct(self):
        def bar(label, a, b, y, col):
            lab = T(label, 24, INK, weight=BOLD).move_to([-4.0, y + 0.62, 0])
            base = RoundedRectangle(corner_radius=0.1, width=6.6, height=0.44, fill_color="#E9E6F2", fill_opacity=1, stroke_width=0).move_to([-4.0, y, 0])
            fa = RoundedRectangle(corner_radius=0.1, width=6.6 * a, height=0.44, fill_color=MUTED, fill_opacity=0.6, stroke_width=0)
            fa.align_to(base, LEFT).set_y(y)
            fb = RoundedRectangle(corner_radius=0.1, width=6.6 * b, height=0.44, fill_color=col, fill_opacity=1, stroke_width=0)
            fb.align_to(base, LEFT).set_y(y)
            return lab, base, fa, fb
        lab1, b1, f1a, f1b = bar("정해진 선율형을 따르는 마디", 0.812, 0.938, 2.2, ACC)
        t1a = T("구조 손실 없이 81%", 22, MUTED).next_to(b1, DOWN, buff=0.12).align_to(b1, LEFT)
        t1b = T("구조 손실과 함께 94%", 22, ACC, weight=BOLD).next_to(b1, DOWN, buff=0.12).align_to(b1, RIGHT)
        lab2, b2, f2a, f2b = bar("환입 구간이 음높이·길이까지 같은 생성곡", 1.0, 1.0, 0.0, RED)
        t2 = T("100%", 22, RED, weight=BOLD).next_to(b2, DOWN, buff=0.12).align_to(b2, RIGHT)
        src = T("〈양청도드리〉 피리 · 가중치 0과 1 · 학습 3회 × 60곡 (논문 6.1절)", 17, MUTED).move_to([-4.0, -1.2, 0])
        self.say("s15a")
        self.play(FadeIn(lab1), FadeIn(b1), GrowFromEdge(f1a, LEFT), FadeIn(t1a), run_time=1.2)
        self.play(GrowFromEdge(f1b, LEFT), FadeIn(t1b), run_time=1.2)
        self.wait(max(0.0, self.left() - 4.0))
        self.play(FadeIn(lab2), FadeIn(b2), GrowFromEdge(f2b, LEFT), FadeIn(t2), FadeIn(src), run_time=1.2)
        self.hold()
        # the trade-off, from the λ sweep (15 training seeds × 60 pieces per λ)
        self.say("s15b")
        lam = SWEEP["metrics"]["first_pass_membership"]["lambda"]
        mem = SWEEP["metrics"]["first_pass_membership"]["mean"]
        cop = SWEEP["metrics"]["exact_copy_rate"]["mean"]
        def panel(vals, lo, hi, y, col, title, fmt):
            ax = Axes(x_range=[0, len(vals) - 1, 1], y_range=[lo, hi, (hi - lo) / 2], x_length=5.6, y_length=1.7,
                      axis_config={"color": MUTED, "stroke_width": 2, "include_ticks": False}, tips=False).move_to([4.2, y, 0])
            pts = [ax.c2p(i, v) for i, v in enumerate(vals)]
            ln = VMobject(stroke_color=col, stroke_width=5).set_points_as_corners(pts)
            ds = VGroup(*[Dot(p, radius=0.06, color=col) for p in pts])
            tt = T(title, 22, col, weight=BOLD).next_to(ax, UP, buff=0.12).align_to(ax, LEFT)
            v0 = T(fmt(vals[0]), 18, col).next_to(pts[0], LEFT, buff=0.12)
            v1 = T(fmt(vals[-1]), 18, col).next_to(pts[-1], UP, buff=0.1)
            return VGroup(ax, ln, ds, tt, v0, v1), ax
        p1, ax1 = panel(mem, 0.78, 1.0, 1.9, ACC, "선율형을 따르는 마디 ↑", lambda v: f"{v * 100:.0f}%")
        p2, ax2 = panel(cop, 0.06, 0.20, -0.75, RED, "원곡 마디와 똑같은 마디 ↑", lambda v: f"{v * 100:.0f}%")
        xl = VGroup(*[T(f"{v:g}", 17, MUTED).next_to(ax2.c2p(i, 0.06), DOWN, buff=0.12) for i, v in enumerate(lam)])
        xt = T("구조 손실의 가중치 λ", 20, INK, weight=BOLD).next_to(xl, DOWN, buff=0.12)
        band = Rectangle(width=0.4, height=4.3, fill_color=ACC, fill_opacity=0.12, stroke_width=0).move_to([ax2.c2p(2, 0)[0], 0.55, 0])
        bt = T("▮ 앞에서 들은 곡: λ = 0.1", 20, ACC, weight=BOLD).move_to([4.2, 3.4, 0])
        src2 = T("학습 15회 × 60곡 평균 (논문 그림 12)", 16, MUTED).next_to(xt, DOWN, buff=0.08)
        self.play(VGroup(lab1, b1, f1a, f1b, t1a, t1b, lab2, b2, f2a, f2b, t2, src).animate.set_opacity(0.45), run_time=0.6)
        self.play(FadeIn(p1[0]), FadeIn(p2[0]), FadeIn(xl), FadeIn(xt), FadeIn(src2), run_time=0.7)
        self.play(Create(p1[1]), Create(p2[1]), FadeIn(p1[2]), FadeIn(p2[2]), FadeIn(p1[3]), FadeIn(p2[3]),
                  FadeIn(p1[4]), FadeIn(p1[5]), FadeIn(p2[4]), FadeIn(p2[5]), run_time=2.0)
        self.wait(max(0.0, self.left() - 4.5))
        self.play(FadeIn(band), FadeIn(bt), run_time=0.8)
        self.end_scene(1.0)


class S16_Outro(Base):
    KEY = "S16_Outro"

    def construct(self):
        logo = T("DiffDodree", 90, ACC, weight=BOLD).move_to([0, 2.0, 0])
        tag = T("한 곡에서 배워, 구조는 지키고 선율은 새롭게", 40, INK, weight=BOLD).next_to(logo, DOWN, buff=0.45)
        url = T("웹 데모  shlee-tda.github.io/DiffDodree", 30, ACC, weight=BOLD)
        gh = T("코드 · Colab 튜토리얼  github.com/SHlee-TDA/DiffDodree", 26, MUTED)
        links = VGroup(url, gh).arrange(DOWN, buff=0.25).next_to(tag, DOWN, buff=0.7)
        self.say("s16a")
        self.play(FadeIn(logo, shift=DOWN * 0.3), run_time=1.0)
        self.play(FadeIn(tag), run_time=0.9)
        self.play(FadeIn(links, shift=UP * 0.15), run_time=0.9)
        self.hold(2.5)
        self.play(FadeOut(Group(*self.mobjects)), run_time=1.2)
