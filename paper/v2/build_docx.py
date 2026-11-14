"""Build the Word manuscript: reference.docx with Korean journal styling, then pandoc."""
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = "manuscript_v3.md"
OUT = HERE / "DiffDodree_논문_초고_v3.docx"
REF = HERE / "reference.docx"
FONT = 'w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="바탕" w:cs="Times New Roman"'


def style_block(xml, sid):
    m = re.search(r'<w:style [^>]*w:styleId="%s".*?</w:style>' % sid, xml, re.S)
    return m


def set_rpr(xml, sid, rpr, ppr=None):
    m = style_block(xml, sid)
    block = m.group(0)
    block = re.sub(r"<w:rPr>.*?</w:rPr>", "", block, flags=re.S)
    if ppr is not None:
        block = re.sub(r"<w:pPr>.*?</w:pPr>", "", block, flags=re.S)
    inner = (ppr or "") + "<w:rPr>" + rpr + "</w:rPr>"
    block = block.replace("</w:style>", inner + "</w:style>")
    return xml[: m.start()] + block + xml[m.end():]


def make_reference():
    default = subprocess.run(["pandoc", "--print-default-data-file", "reference.docx"], capture_output=True, check=True).stdout
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "default.docx"; src.write_bytes(default)
        zin = zipfile.ZipFile(src)
        files = {n: zin.read(n) for n in zin.namelist()}
    s = files["word/styles.xml"].decode("utf-8")
    s = re.sub(r"<w:rPrDefault>.*?</w:rPrDefault>",
               f'<w:rPrDefault><w:rPr><w:rFonts {FONT}/><w:sz w:val="21"/><w:szCs w:val="21"/>'
               '<w:lang w:val="en-US" w:eastAsia="ko-KR" w:bidi="ar-SA"/></w:rPr></w:rPrDefault>', s, flags=re.S)
    s = re.sub(r"<w:pPrDefault>.*?</w:pPrDefault>",
               '<w:pPrDefault><w:pPr><w:spacing w:after="0" w:line="360" w:lineRule="auto"/></w:pPr></w:pPrDefault>', s, flags=re.S)
    body = '<w:pPr><w:spacing w:before="0" w:after="120"/><w:ind w:firstLine="400"/><w:jc w:val="both"/></w:pPr>'
    s = set_rpr(s, "BodyText", f'<w:rFonts {FONT}/>', body)
    s = set_rpr(s, "FirstParagraph", f'<w:rFonts {FONT}/>', body)
    for sid, size, before in [("Heading1", 28, 480), ("Heading2", 24, 360), ("Heading3", 22, 240)]:
        s = set_rpr(s, sid, f'<w:rFonts {FONT}/><w:b/><w:color w:val="000000"/><w:sz w:val="{size}"/><w:szCs w:val="{size}"/>',
                    f'<w:pPr><w:keepNext/><w:keepLines/><w:spacing w:before="{before}" w:after="160"/>'
                    f'<w:outlineLvl w:val="{sid[-1] if sid[-1].isdigit() else 0}"/></w:pPr>')
    s = s.replace('<w:outlineLvl w:val="1"/></w:pPr><w:rPr><w:rFonts ' + FONT + '/><w:b/><w:color w:val="000000"/><w:sz w:val="28"/>',
                  '<w:outlineLvl w:val="0"/></w:pPr><w:rPr><w:rFonts ' + FONT + '/><w:b/><w:color w:val="000000"/><w:sz w:val="28"/>')
    s = s.replace('<w:outlineLvl w:val="2"/></w:pPr><w:rPr><w:rFonts ' + FONT + '/><w:b/><w:color w:val="000000"/><w:sz w:val="24"/>',
                  '<w:outlineLvl w:val="1"/></w:pPr><w:rPr><w:rFonts ' + FONT + '/><w:b/><w:color w:val="000000"/><w:sz w:val="24"/>')
    s = s.replace('<w:outlineLvl w:val="3"/></w:pPr><w:rPr><w:rFonts ' + FONT + '/><w:b/><w:color w:val="000000"/><w:sz w:val="22"/>',
                  '<w:outlineLvl w:val="2"/></w:pPr><w:rPr><w:rFonts ' + FONT + '/><w:b/><w:color w:val="000000"/><w:sz w:val="22"/>')
    s = set_rpr(s, "Title", f'<w:rFonts {FONT}/><w:b/><w:sz w:val="32"/><w:szCs w:val="32"/>',
                '<w:pPr><w:spacing w:before="0" w:after="120" w:line="300" w:lineRule="auto"/><w:jc w:val="center"/></w:pPr>')
    s = set_rpr(s, "Subtitle", f'<w:rFonts {FONT}/><w:color w:val="555555"/><w:sz w:val="21"/>',
                '<w:pPr><w:spacing w:after="360"/><w:jc w:val="center"/></w:pPr>')
    s = set_rpr(s, "FootnoteText", f'<w:rFonts {FONT}/><w:sz w:val="17"/><w:szCs w:val="17"/>',
                '<w:pPr><w:spacing w:after="40" w:line="260" w:lineRule="auto"/><w:jc w:val="both"/></w:pPr>')
    for sid in ("Caption", "ImageCaption", "TableCaption"):
        s = set_rpr(s, sid, f'<w:rFonts {FONT}/><w:i w:val="0"/><w:sz w:val="18"/><w:szCs w:val="18"/>',
                    '<w:pPr><w:spacing w:before="60" w:after="240" w:line="280" w:lineRule="auto"/><w:jc w:val="both"/></w:pPr>')
    s = set_rpr(s, "Compact", f'<w:rFonts {FONT}/><w:sz w:val="18"/><w:szCs w:val="18"/>',
                '<w:pPr><w:spacing w:before="20" w:after="20" w:line="260" w:lineRule="auto"/></w:pPr>')
    tbl = style_block(s, "Table")
    new_tbl = ('<w:style w:type="table" w:default="1" w:styleId="Table"><w:name w:val="Table"/><w:basedOn w:val="TableNormal"/>'
               '<w:qFormat/><w:tblPr><w:jc w:val="center"/><w:tblBorders><w:top w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
               '<w:bottom w:val="single" w:sz="8" w:space="0" w:color="000000"/></w:tblBorders>'
               '<w:tblCellMar><w:top w:w="40" w:type="dxa"/><w:left w:w="100" w:type="dxa"/><w:bottom w:w="40" w:type="dxa"/>'
               '<w:right w:w="100" w:type="dxa"/></w:tblCellMar></w:tblPr>'
               '<w:tblStylePr w:type="firstRow"><w:rPr><w:b/></w:rPr><w:tcPr><w:tcBorders><w:bottom w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
               '</w:tcBorders><w:shd w:val="clear" w:color="auto" w:fill="F2F2F2"/></w:tcPr></w:tblStylePr></w:style>')
    s = s[: tbl.start()] + new_tbl + s[tbl.end():]
    # character style for sentences flagged for co-author review (pandoc: [..]{custom-style="Inquiry"})
    s = s.replace("</w:styles>", '<w:style w:type="character" w:customStyle="1" w:styleId="Inquiry"><w:name w:val="Inquiry"/>'
                  '<w:rPr><w:color w:val="1F4FD1"/><w:shd w:val="clear" w:color="auto" w:fill="E8EEFF"/></w:rPr></w:style></w:styles>')
    files["word/styles.xml"] = s.encode("utf-8")
    d = files["word/document.xml"].decode("utf-8")
    d = re.sub(r"<w:sectPr>.*?</w:sectPr>",
               '<w:sectPr><w:footnotePr><w:numFmt w:val="decimal"/></w:footnotePr><w:pgSz w:w="11906" w:h="16838"/>'
               '<w:pgMar w:top="1701" w:right="1418" w:bottom="1701" w:left="1418" w:header="851" w:footer="992" w:gutter="0"/></w:sectPr>',
               d, flags=re.S)
    files["word/document.xml"] = d.encode("utf-8")
    with zipfile.ZipFile(REF, "w", zipfile.ZIP_DEFLATED) as z:
        for n, b in files.items():
            z.writestr(n, b)


def main():
    make_reference()
    subprocess.run(["pandoc", SRC, "-o", str(OUT), "--reference-doc", str(REF), "--resource-path", str(HERE),
                    "-f", "markdown+tex_math_dollars+pipe_tables+footnotes+raw_attribute+bracketed_spans"], cwd=HERE, check=True)
    subprocess.run(["zip", "-d", str(OUT), "*/"], capture_output=True)
    print("wrote", OUT, OUT.stat().st_size)


if __name__ == "__main__":
    sys.exit(main())
