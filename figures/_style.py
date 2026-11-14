"""Shared settings for the figure scripts: Korean font fallback and repository paths."""

from pathlib import Path

import matplotlib
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = ROOT / "results" / "runs"
OUTPUT = HERE / "output"

# The paper figures were drawn with Apple SD Gothic Neo (macOS). On Colab/Linux install
# fonts-nanum (``apt-get install -y fonts-nanum``); the glyph shapes then differ slightly.
KOREAN_FONTS = ["Apple SD Gothic Neo", "NanumGothic", "Noto Sans CJK KR", "Malgun Gothic"]


def korean_font() -> list[str]:
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = [name for name in KOREAN_FONTS if name in available]
    if not chosen:
        print("warning: no Korean font found; install fonts-nanum for readable labels")
    return chosen + ["DejaVu Sans"]


def read_report(run: Path) -> dict:
    """``report.json`` of a full run directory, or ``report_slim.json`` from ``results/runs``."""

    import json

    for name in ("report.json", "report_slim.json"):
        if (run / name).exists():
            return json.loads((run / name).read_text(encoding="utf-8"))
    raise FileNotFoundError(f"no report.json or report_slim.json in {run}")


matplotlib.use("Agg")
