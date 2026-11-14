import re
import xml.etree.ElementTree as ET
NS = {"svg": "http://www.w3.org/2000/svg"}
ET.register_namespace("", "http://www.w3.org/2000/svg"); ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
SHAPES = {"path", "ellipse", "polygon", "polyline", "rect"}

def clean_verovio_svg(s):
    root = ET.fromstring(s)
    w, h = root.get("viewBox").split()[2:]
    root.set("width", w); root.set("height", h)
    parent = {c: p for p in root.iter() for c in p}
    for el in list(root.iter()):
        tag = el.tag.split("}")[-1]
        cls = el.get("class") or ""
        if tag == "g" and ("cursor" in cls or "placeholder" in cls):
            parent[el].remove(el); continue
        if tag == "svg" and "definition-scale" in cls:
            el.set("width", w); el.set("height", h); el.set("x", "0"); el.set("y", "0")
        if tag == "style":
            el.text = ""
        if tag in SHAPES and el.get("stroke") is None:
            col, a = "#000000", el
            while a is not None:
                c = a.get("color") or (a.get("fill") if a is not el and a.get("fill") not in (None, "none") else None)
                if c and c != "currentColor": col = c; break
                a = parent.get(a)
            el.set("stroke", col)
            if el.get("fill") == "currentColor": el.set("fill", col)
    out = ET.tostring(root, encoding="unicode")
    return out.replace("currentColor", "#000000")
