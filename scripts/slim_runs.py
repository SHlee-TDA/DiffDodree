"""Copy the parts of finished runs that the figures and the web page need into ``results/runs``.

A full run directory (``artifacts/chart_diffusion_runs/<RUN_TAG>/``) holds an indented
``report.json`` of 12–200 MB and about one checkpoint per model. This script writes, per run:

* ``meta.json``, ``config_used.yaml`` (if present) and ``evaluation.json`` unchanged;
* ``report_slim.json``: ``report.json`` without the generated pieces (full and copy-test units),
  except for the training seeds listed in ``--keep-pieces`` (all seeds if ``all``). The held-out curves, the selected
  lengths, the training histories, the monitoring and every per-piece metric are kept.

Nothing in ``results/runs/<RUN_TAG>`` is overwritten; delete the folder first to redo it.

    python scripts/slim_runs.py RUN_DIR [RUN_DIR ...] --keep-pieces 260807
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def slim(run: Path, out_root: Path, keep: set[int] | None) -> Path:
    out = out_root / run.name
    if out.exists():
        raise FileExistsError(f"{out} exists; not overwriting")
    out.mkdir(parents=True)
    for name in ("meta.json", "config_used.yaml", "evaluation.json"):
        if (run / name).exists():
            shutil.copy2(run / name, out / name)
    source = run / "report.json"
    report = json.loads(source.read_text(encoding="utf-8"))
    for unit in report.get("full", []):
        if keep is not None and int(unit["seed"]) not in keep:
            unit["generation"].pop("pieces", None)
    for unit in report.get("copytest", []):
        if keep is not None and int(unit["seed"]) not in keep:
            unit.pop("pieces", None)
    report["slim"] = {
        "source": "report.json",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "pieces_kept_for_seeds": "all" if keep is None else sorted(keep),
        "removed": "generated pieces (full and copy-test units) of the other training seeds",
    }
    (out / "report_slim.json").write_text(json.dumps(report, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--keep-pieces", default="", help="comma-separated training seeds, or 'all'")
    parser.add_argument("--out", type=Path, default=ROOT / "results/runs")
    args = parser.parse_args()
    keep = None if args.keep_pieces == "all" else {int(s) for s in args.keep_pieces.split(",") if s}
    for run in args.runs:
        out = slim(run, args.out, keep)
        size = sum(p.stat().st_size for p in out.iterdir()) / 1e6
        print(f"{run.name}: {size:.1f} MB -> {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")


if __name__ == "__main__":
    main()
