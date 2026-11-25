"""Build the frozen measures file of a piece config from its MusicXML sources.

    python scripts/build_measures.py configs/yangcheongdodeuri_haegeum.yaml
    python scripts/build_measures.py configs/yangcheongdodeuri.yaml --check   # piri: compare only

The sources are ``data.source_globs`` in file order (one file per glob). Measures are numbered
across files and assigned to chapters by ``chapters[*].measures``; every measure must fill its
meter. The output has the layout of ``artifacts/measures_v0.json`` (the piri file is reproduced
byte for byte). An existing output is never overwritten: if it differs, the script stops.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gugakgen.config import load_yaml  # noqa: E402
from gugakgen.data.musicxml import combine_scores, parse_musicxml  # noqa: E402


def build(piece_config: Path, root: Path = ROOT) -> tuple[str, Path]:
    piece = load_yaml(piece_config)
    data = piece["data"]
    ranges = [chapter["measures"] for chapter in piece["chapters"]]
    meters = {m: tuple(c["meter"]) for c in piece["chapters"] for m in range(c["measures"][0], c["measures"][1] + 1)}
    segments, offset = [], 0
    for pattern in data["source_globs"]:
        matches = sorted(glob.glob(str(root / pattern)))
        if len(matches) != 1:
            raise FileNotFoundError(f"{pattern!r}: expected one file, found {len(matches)}")
        score = parse_musicxml(
            matches[0], piece_id=piece["piece"]["id"], part_name=piece["piece"]["part_name_in_files"],
            measure_offset=offset, chapter_ranges=ranges, reject_chords=bool(data.get("reject_chords", True)),
        )
        offset += len(score.measures)
        segments.append(score)
    score = combine_scores(segments)
    score.validate_meters()
    if len(score.measures) != len(meters):
        raise ValueError(f"{len(score.measures)} measures; the chapters cover {len(meters)}")
    for measure in score.measures:
        if tuple(measure.meter.to_list()) != meters[measure.number]:
            raise ValueError(f"Measure {measure.number} is {measure.meter.to_list()}, chapters say {meters[measure.number]}")
    records = [
        {
            "global_measure": m.number,
            "meter": m.meter.to_list(),
            "events": [{"pitch": e.pitch, "duration_q": float(e.duration_q)} for e in m.events],
            "source_file": m.source_file,
            "source_measure": m.source_measure,
            "chapter": m.chapter,
        }
        for m in score.measures
    ]
    print(f"{piece['piece']['id']}: {len(records)} measures, chapters {dict(Counter(m.chapter for m in score.measures))}")
    return json.dumps(records, ensure_ascii=False, indent=2), root / data["canonical_measures"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("piece_config", type=Path)
    parser.add_argument("--check", action="store_true", help="only compare with the existing output")
    args = parser.parse_args()
    text, out = build(args.piece_config.resolve())
    if out.exists():
        same = out.read_text(encoding="utf-8") == text
        print(f"{out.relative_to(ROOT)}: {'identical' if same else 'DIFFERENT'}")
        if not same:
            raise SystemExit("The existing file differs; not overwriting. Use a new canonical_measures name.")
        return
    if args.check:
        raise SystemExit(f"{out} does not exist")
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
