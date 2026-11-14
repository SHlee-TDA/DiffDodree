"""Inline <style> class rules into style attributes so MuPDF renders the SVG faithfully."""
import re, sys
import xml.etree.ElementTree as ET
from pathlib import Path

NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", NS)

def inline(src, dst, replace_font=None):
    text = Path(src).read_text(encoding="utf-8")
    css = " ".join(re.findall(r"<style>(.*?)</style>", text, re.S))
    rules = {}
    for sel, body in re.findall(r"([^{}]+)\{([^}]*)\}", css):
        for s in sel.split(","):
            rules.setdefault(s.strip(), []).append(body.strip())
    root = ET.fromstring(text)
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        parts = list(rules.get(tag, []))
        for c in (el.get("class") or "").split():
            parts += rules.get("." + c, [])
        if parts:
            style = ";".join(parts + ([el.get("style")] if el.get("style") else []))
            if replace_font:
                style = re.sub(r"font-family:[^;]*", f"font-family:{replace_font}", style)
            el.set("style", style)
    for st in root.iter(f"{{{NS}}}style"):
        st.text = ""
    Path(dst).write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")

if __name__ == "__main__":
    inline(sys.argv[1], sys.argv[2])
