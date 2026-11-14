"""Checks that the public results still match the code and data in this repository.

If you change the generator code or a config on purpose, these tests tell you that the results in
``results/`` were produced by different code; record the new runs instead of editing the hashes.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gugakgen.chart.diffusion import GridDenoiser, Tables  # noqa: E402
from gugakgen.chart.evaluation import as_phrase  # noqa: E402
from gugakgen.workflows import chart_diffusion_experiment as v1  # noqa: E402

RUNS = PROJECT_ROOT / "results/runs"
SWEEP = "20261113_1653_cpu_lambda_sweep"
SWEEP_CONFIGS = {
    "20261113_1653_cpu_lambda_sweep": "configs/chart_diffusion_v1_lambda_sweep.yaml",
    "20261113_2206_cpu_lambda_sweep_seeds12": "configs/chart_diffusion_v1_lambda_sweep_more_seeds.yaml",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def meta(run: str) -> dict:
    return json.loads((RUNS / run / "meta.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("run", sorted(p.name for p in RUNS.iterdir() if p.is_dir()))
def test_runs_used_this_score_and_piece_config(run: str) -> None:
    m = meta(run)
    assert m["measures_sha256"] == sha256(PROJECT_ROOT / "artifacts/measures_v0.json")
    assert m["piece_config_sha256"] == sha256(PROJECT_ROOT / "configs/yangcheongdodeuri.yaml")


@pytest.mark.parametrize("run,config", sorted(SWEEP_CONFIGS.items()))
def test_lambda_sweep_runs_match_current_code_and_configs(run: str, config: str) -> None:
    m = meta(run)
    assert m["source_sha256"] == v1.source_digest(), "generator code differs from the code of the λ sweep runs"
    assert m["config_sha256"] == sha256(PROJECT_ROOT / config)


def test_checkpoint_regenerates_the_papers_piece() -> None:
    setup = v1.load_setup(PROJECT_ROOT / SWEEP_CONFIGS[SWEEP], PROJECT_ROOT)
    report = json.loads((RUNS / SWEEP / "report_slim.json").read_text(encoding="utf-8"))
    unit = next(u for u in report["full"] if u["arm"] == "diffusion" and u["seed"] == 260807)
    path = PROJECT_ROOT / "results/checkpoints" / SWEEP / unit["checkpoint"]
    assert sha256(path) == unit["checkpoint_sha256"]
    saved = torch.load(path, map_location="cpu", weights_only=False)
    model = GridDenoiser(setup["layout"], **setup["config"]["model"])
    model.load_state_dict(saved["state_dict"])
    model.eval()
    tables = Tables(setup["typology"], setup["layout"], torch.device("cpu"))
    piece = unit["generation"]["pieces"][0]
    phrases, walk, _ = v1.generate_piece(model, tables, setup, piece["seed"])
    assert walk == piece["walk"]
    assert [tuple(p) for p in phrases] == [as_phrase(p) for p in piece["phrases"]]


def test_web_models_come_from_the_checkpoints() -> None:
    gallery = json.loads((PROJECT_ROOT / "site/data/gallery.json").read_text(encoding="utf-8"))
    for model in gallery["models"]:
        path = PROJECT_ROOT / "results/checkpoints" / gallery["run"] / model["checkpoint"]
        assert sha256(path) == model["checkpoint_sha256"]
        assert (PROJECT_ROOT / "site" / model["file"]).exists()
