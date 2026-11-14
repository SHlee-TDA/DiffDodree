"""Rebuild every paper figure from results/runs (see figures/README.md for the mapping).

    python figures/make_all.py

Steps: extract_data.py (run 20261112_2112) -> check_tda_order.py -> make_figures.py
-> make_sweep_figure.py (runs 1653 + 2206) -> render_scores.py (run 1331).
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["extract_data.py", "check_tda_order.py", "make_figures.py", "make_sweep_figure.py", "render_scores.py"]

for step in STEPS:
    print(f"== {step}", flush=True)
    subprocess.run([sys.executable, str(HERE / step)], check=True)
print("done: figures/output/")
