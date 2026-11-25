"""Per-instrument configs: the piri measures file is rebuilt from its MusicXML byte for byte, and the
haegeum configs differ from the piri ones only where they must (score, measures file, run folder)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from build_measures import build  # noqa: E402

from gugakgen.config import load_yaml  # noqa: E402


def test_piri_measures_are_rebuilt_byte_for_byte() -> None:
    text, out = build(PROJECT_ROOT / "configs/yangcheongdodeuri.yaml")
    assert out == PROJECT_ROOT / "artifacts/measures_v0.json"
    assert text == out.read_text(encoding="utf-8")


def test_haegeum_configs_match_piri_except_score_and_outputs() -> None:
    piri, haegeum = (load_yaml(PROJECT_ROOT / f"configs/{n}.yaml") for n in ("yangcheongdodeuri", "yangcheongdodeuri_haegeum"))
    assert haegeum["chapters"] == piri["chapters"] and haegeum["pitch"] == piri["pitch"]
    strip = lambda r: {k: v for k, v in r.items() if k != "provenance"}  # noqa: E731
    assert [strip(r) for r in haegeum["structure"]["hwanip_relations"]] == [
        strip(r) for r in piri["structure"]["hwanip_relations"]]
    assert [r["source_measures"] for r in haegeum["structure"]["hwanip_relations"]] == [[1, 4]]
    assert [r["target_measures"] for r in haegeum["structure"]["hwanip_relations"]] == [[25, 28]]
    assert haegeum["data"]["canonical_measures"] != piri["data"]["canonical_measures"]
    assert all("Haegeum" in g for g in haegeum["data"]["source_globs"])

    v1, v1h = (load_yaml(PROJECT_ROOT / f"configs/{n}.yaml") for n in ("chart_diffusion_v1", "chart_diffusion_v1_haegeum"))
    assert v1h["piece_config"] == "configs/yangcheongdodeuri_haegeum.yaml"
    assert v1h["artifacts"]["run_dir"] != v1["artifacts"]["run_dir"]  # never writes into the piri run folder
    same = {k for k in v1 if k not in {"experiment_name", "piece_config", "artifacts"}}
    assert {k: v1h[k] for k in same} == {k: v1[k] for k in same}


@pytest.mark.skipif(not (PROJECT_ROOT / "artifacts/measures_haegeum_v0.json").exists(),
                    reason="the haegeum score is not distributed with the repository")
def test_haegeum_measures_match_their_score_and_fit_the_grid() -> None:
    from gugakgen.chart.grid import Grid
    from gugakgen.chart.typology import load_phrases

    text, out = build(PROJECT_ROOT / "configs/yangcheongdodeuri_haegeum.yaml")
    assert text == out.read_text(encoding="utf-8")
    phrases, chapters = load_phrases(out)
    assert len(phrases) == 72 and chapters.count(7) == 17
    assert phrases[0:4] == phrases[24:28]  # 장 되풀이: 1–4 = 25–28마디
    grid = Grid.from_phrases(phrases, 4)
    assert sorted({len(grid.encode(p)) for p in phrases}) == [16, 24]
