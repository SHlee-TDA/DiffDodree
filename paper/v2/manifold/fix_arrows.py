"""Replace marker-end arrows (unsupported by MuPDF) with explicit chevron paths; relabel."""
import math, re, sys
from pathlib import Path

src, dst = sys.argv[1], sys.argv[2]
s = Path(src).read_text(encoding="utf-8")

def nums(d):
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", d)]

def chevron(tip, frm, size=6.0, width=1.3):
    ang = math.atan2(tip[1] - frm[1], tip[0] - frm[0])
    pts = []
    for da in (2.6, -2.6):
        pts.append((tip[0] + size * math.cos(ang + da), tip[1] + size * math.sin(ang + da)))
    return (f'<path d="M{pts[0][0]:.2f} {pts[0][1]:.2f}L{tip[0]:.2f} {tip[1]:.2f}L{pts[1][0]:.2f} {pts[1][1]:.2f}" '
            f'fill="none" stroke="#2C2C2A" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"/>')

out = []
def repl(m):
    el = m.group(0)
    if el.startswith("<line"):
        x1, y1, x2, y2 = (float(re.search(fr'{k}="([^"]+)"', el).group(1)) for k in ("x1", "y1", "x2", "y2"))
        tip, frm = (x2, y2), (x1, y1)
    else:
        v = nums(re.search(r'\sd="([^"]+)"', el).group(1))
        tip, frm = (v[-2], v[-1]), (v[-4], v[-3])
    out.append(chevron(tip, frm))
    return el.replace(' marker-end="url(#arrow)"', "")

s = re.sub(r'<(?:line|path)[^>]*marker-end="url\(#arrow\)"[^>]*/>', repl, s)
s = s.replace("</svg>", "".join(out) + "</svg>")
s = s.replace(">phrase graph<", ">프레이즈 그래프<")
Path(dst).write_text(s, encoding="utf-8")
print("arrows replaced:", len(out))
