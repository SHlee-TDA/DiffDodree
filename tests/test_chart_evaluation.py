from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ripser")
pytest.importorskip("scipy")

from gugakgen.chart.evaluation import (  # noqa: E402
    _diagram_costs,
    barcode,
    bottleneck,
    copy_test,
    judge,
    model_scores,
    note_graph_distances,
    overlap_area,
    wasserstein1,
)
from gugakgen.workflows import chart_diffusion_experiment as v1  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def setup():
    return v1.load_setup(PROJECT_ROOT / "configs/chart_diffusion_v1.yaml", PROJECT_ROOT)


@pytest.fixture(scope="module")
def ref(setup):
    return v1.build_reference(setup)


def _brute(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    cost = _diagram_costs(a, b)
    best_max, best_sum = np.inf, np.inf
    for perm in itertools.permutations(range(len(cost))):
        values = cost[np.arange(len(cost)), perm]
        best_max, best_sum = min(best_max, values.max()), min(best_sum, values.sum())
    return float(best_max), float(best_sum)


def test_diagram_distances_match_brute_force() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = np.sort(rng.random((rng.integers(0, 3), 2)), axis=1)
        b = np.sort(rng.random((rng.integers(1, 4), 2)), axis=1)
        top, total = _brute(a, b)
        assert bottleneck(a, b) == pytest.approx(top)
        assert wasserstein1(a, b) == pytest.approx(total)
    assert bottleneck(a, a) == 0 and wasserstein1(a, a) == pytest.approx(0)


def test_note_graph_distance_and_cycle() -> None:
    a, b, c, d = (60, 1.0), (62, 1.0), (64, 1.0), (65, 1.0)
    nodes, dist = note_graph_distances([a, b, a, b, c])  # a-b three times, b-c once
    i = {n: k for k, n in enumerate(nodes)}
    assert dist[i[a], i[b]] == pytest.approx(1 / 3)
    assert dist[i[a], i[c]] == pytest.approx(1 / 3 + 1)
    _, square = note_graph_distances([a, b, c, d, a])  # a 4-cycle: one H1 bar
    assert len(barcode(square)) == 1
    _, path = note_graph_distances([a, b, c, d])  # a path: no H1
    assert len(barcode(path)) == 0


def test_overlap_area_bounds() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=500)
    assert overlap_area(x, x) == pytest.approx(1, abs=0.02)
    assert overlap_area(x, x + 20) < 0.01


def test_original_as_generated_piece(ref) -> None:
    piece = list(ref.phrases)
    scores = model_scores([piece, piece], ref, bootstrap=50, seed=0, pair_sample=500)
    p, s = scores["per_piece"], scores["set"]
    assert p["exact_copy_rate"]["mean"] == 1 and s["authenticity"]["value"] == 0
    assert p["ph_bottleneck"]["mean"] == 0 and p["phrase_graph_tv"]["mean"] == pytest.approx(0)
    assert p["beat_fill_error"]["mean"] == 0 and p["offgrid_onset"]["mean"] == 0
    assert p["pitch_outside_scale"]["mean"] == 0
    assert p["chapter_cadence_ok"]["mean"] == 1 and p["final_pitch_ok"]["mean"] == 1
    assert p["interval_jsd"]["mean"] == 0 and p["note_4gram_jsd"]["mean"] == 0
    assert s["coverage"]["value"] == 1 and s["nn_distance_mean"]["value"] == 0


def test_copy_test_detects_copying(setup) -> None:
    from gugakgen.workflows.chart_experiment import heldout_folds

    typology = setup["typology"]
    held = set(heldout_folds(typology, 3, 20261111)[0])
    train = [typology.phrases[t] for t in range(72) if t not in held]
    test = [typology.phrases[t] for t in sorted(held)]
    assert copy_test(train * 3, train, test)["z"] < -2  # pure copies of the training data
    neutral = copy_test(test * 3, train, test)["z"]  # same distribution as the held-out phrases
    assert abs(neutral) < 1e-9


def test_judge_rules() -> None:
    first = {"g": {1: 0.8, 2: 0.7}, "d": {1: 0.6, 2: 0.75}}
    out = judge(first, {"g": {1: 0.5, 2: -3.0}}, main_arm="g", ablation_arm="d")
    assert out["5_graph_loss"]["status"] == "fail" and out["5_graph_loss"]["by_seed"] == {1: True, 2: False}
    assert out["4_novelty"]["copy_test_pass"] is False
    assert out["1_geometry"]["status"].startswith("pending")
    same = judge(first, {}, main_arm="g", ablation_arm="d", graph_active_steps=0)
    assert same["5_graph_loss"]["status"].startswith("invalid")
    assert judge(first, {}, main_arm="g", ablation_arm="d", graph_active_steps=50)["5_graph_loss"]["status"] == "fail"
