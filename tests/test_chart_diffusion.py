from __future__ import annotations

import itertools
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from gugakgen.chart.diffusion import (
    GridDenoiser,
    Layout,
    Tables,
    combine_returns,
    corrupt,
    diffusion_loss,
    graph_loss,
    make_rows,
    reverse_sample,
    soft_types,
    sounding_marginals,
)
from gugakgen.chart.grid import Grid
from gugakgen.chart.typology import build_typology, load_phrases
from gugakgen.chart.walk import TypeWalk
from gugakgen.device import resolve_device
from gugakgen.workflows import chart_diffusion_experiment as v1

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RETURNS = tuple((i, 24 + i) for i in range(4))


@pytest.fixture(scope="module")
def piece():
    phrases, chapters = load_phrases(PROJECT_ROOT / "artifacts/measures_v0.json")
    typology = build_typology(phrases, chapters, k=9)
    grid = Grid.from_phrases(phrases, 4)
    tokens = [grid.encode(p) for p in phrases]
    layout = Layout(grid, 9, (16, 24))
    return typology, grid, tokens, layout


def test_grid_round_trip_and_resolution(piece) -> None:
    typology, grid, tokens, _ = piece
    assert all(grid.decode(t) == list(p) for t, p in zip(tokens, typology.phrases, strict=True))
    assert {len(t) for t in tokens} == {16, 24}
    with pytest.raises(ValueError):  # the transcription has notes at 3/4 of a beat
        Grid.from_phrases(typology.phrases, 3).encode(typology.phrases[2])
    with pytest.raises(ValueError):
        grid.decode([grid.hold, 0])


def test_walk_is_exact_conditional_sampling() -> None:
    counts = np.array([[1, 2, 0], [1, 1, 2], [2, 0, 1]])
    length, start, end, returns = 7, 0, 2, [(1, 4), (2, 5)]
    P = counts / counts.sum(1, keepdims=True)
    exact: dict[tuple, float] = {}
    for middle in itertools.product(range(3), repeat=length - 2):
        y = (start, *middle, end)
        if all(y[s] == y[t] for s, t in returns):
            weight = np.prod([P[a, b] for a, b in zip(y[:-1], y[1:], strict=True)])
            if weight > 0:
                exact[y] = weight
    total = sum(exact.values())
    walk = TypeWalk(counts, length, start, end, returns)
    rng = np.random.default_rng(0)
    drawn = Counter(tuple(walk.sample(rng)) for _ in range(20000))
    tv = 0.5 * sum(abs(drawn[y] / 20000 - w / total) for y, w in exact.items())
    assert set(drawn) <= set(exact) and tv < 0.03


def test_walk_is_exact_with_a_return_at_the_start() -> None:
    counts = np.array([[1, 2, 1], [1, 1, 2], [2, 1, 1]])
    length, start, end, returns = 7, 0, 2, [(0, 3), (1, 4)]
    P = counts / counts.sum(1, keepdims=True)
    exact: dict[tuple, float] = {}
    for middle in itertools.product(range(3), repeat=length - 2):
        y = (start, *middle, end)
        if all(y[s] == y[t] for s, t in returns):
            weight = np.prod([P[a, b] for a, b in zip(y[:-1], y[1:], strict=True)])
            if weight > 0:
                exact[y] = weight
    total = sum(exact.values())
    walk = TypeWalk(counts, length, start, end, returns)
    rng = np.random.default_rng(2)
    drawn = Counter(tuple(walk.sample(rng)) for _ in range(20000))
    tv = 0.5 * sum(abs(drawn[y] / 20000 - w / total) for y, w in exact.items())
    assert set(drawn) <= set(exact) and tv < 0.03


def test_diffusion_loss_by_hand() -> None:
    log_probs = torch.log(torch.tensor([[[0.5, 0.5], [0.25, 0.75], [0.9, 0.1]]]))
    clean = torch.tensor([[0, 1, 1]])
    masked = torch.tensor([[True, True, False]])
    value = diffusion_loss(log_probs, clean, masked, torch.tensor([0.5]), torch.tensor([3]))
    assert float(value) == pytest.approx((np.log(2) + np.log(4 / 3)) / (0.5 * 3))
    batch = {"tokens": torch.zeros(1, 4, dtype=torch.long), "length": torch.tensor([3])}
    _, mask = corrupt(batch, torch.tensor([1.0]), torch.Generator().manual_seed(0), 9)
    assert mask.tolist() == [[True, True, True, False]]  # r = 1 masks every valid cell, never padding


def test_return_predictions_are_averaged_not_multiplied() -> None:
    logits = torch.log_softmax(torch.tensor([[2.0, 0.5, -1.0]]), -1)
    same = combine_returns(logits, logits)
    assert torch.allclose(same - torch.logsumexp(same, -1, keepdim=True), logits)  # agreement does not sharpen
    other = torch.log_softmax(torch.tensor([[-1.0, 0.5, 2.0]]), -1)
    mixed = combine_returns(logits, other)
    assert torch.allclose(mixed, 0.5 * (logits + other))


def test_graph_loss_gradient_reaches_the_note_head(piece) -> None:
    typology, grid, _, layout = piece
    torch.manual_seed(0)
    model = GridDenoiser(layout, d_model=32, n_layers=1, n_heads=2, ffn_dim=64)
    tables = Tables(typology, layout, torch.device("cpu"))
    lengths = [len(t) for t in piece[2]]
    walks = np.array([typology.labels, typology.labels[::-1]])
    batch = {k: torch.as_tensor(v).flatten(0, 1) for k, v in v1.walk_rows(walks, lengths, layout).items()}
    assign = soft_types(model(batch, tables).exp(), batch["meter"], batch["length"], tables, grid, 0.5)
    loss = graph_loss(assign.view(2, -1, layout.num_types), torch.as_tensor(walks), layout.num_types)
    loss.backward()
    assert model.head.weight.grad is not None and float(model.head.weight.grad.abs().sum()) > 0


def test_walks_on_the_piece_keep_the_conditions(piece) -> None:
    typology = piece[0]
    walk = TypeWalk(typology.transition_counts(), 72, typology.labels[0], typology.labels[-1], RETURNS)
    edges = typology.transition_counts() > 0
    rng = np.random.default_rng(1)
    for _ in range(200):
        y = walk.sample(rng)
        assert y[0] == typology.labels[0] and y[-1] == typology.labels[-1] and y[:4] == y[24:28]
        assert all(edges[a, b] for a, b in zip(y[:-1], y[1:], strict=True))


def test_sounding_marginals_match_sampling(piece) -> None:
    grid = piece[1]
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(1, 6, grid.num_outputs), -1)
    probs[0, 0, grid.hold] = 0
    probs = probs / probs.sum(-1, keepdim=True)
    q = sounding_marginals(probs, grid)[0].numpy()
    rng = np.random.default_rng(0)
    draws = np.stack([rng.choice(grid.num_outputs, 40000, p=probs[0, j].numpy()) for j in range(6)], 1)
    for j in range(6):
        states = [grid.sounding(row)[j] for row in draws[:5000]]
        freq = np.bincount(states, minlength=grid.num_states) / len(states)
        assert np.abs(freq - q[j]).max() < 0.03


def test_soft_types_and_graph_loss(piece) -> None:
    typology, grid, tokens, layout = piece
    tables = Tables(typology, layout, torch.device("cpu"))
    rows = make_rows(tokens, typology.labels, layout)
    onehot = torch.nn.functional.one_hot(torch.as_tensor(rows["tokens"]).clamp(max=grid.num_outputs - 1), grid.num_outputs).float()
    assign = soft_types(onehot, torch.as_tensor(rows["meter"]), torch.as_tensor(rows["length"]), tables, grid, 0.05)
    agree = (assign.argmax(-1).numpy() == np.array(typology.labels)).sum()
    assert agree >= 66  # contour typing agrees with the edit-distance types on most phrases
    labels = torch.as_tensor([typology.labels])
    exact = torch.nn.functional.one_hot(labels, 9).float().requires_grad_()
    assert float(graph_loss(exact, labels, 9).detach()) == pytest.approx(0.0, abs=1e-7)
    blurred = (0.5 * exact + 0.5 / 9)
    value = graph_loss(blurred, labels, 9)
    value.backward()
    assert float(value.detach()) > 0 and exact.grad is not None


def test_reverse_sample_respects_ties_givens_and_rhythm(piece) -> None:
    typology, grid, tokens, layout = piece
    torch.manual_seed(0)
    model = GridDenoiser(layout, d_model=32, n_layers=1, n_heads=2, ffn_dim=64).eval()
    tables = Tables(typology, layout, torch.device("cpu"))
    lengths = [len(t) for t in tokens]
    generator = torch.Generator().manual_seed(0)
    out = reverse_sample(model, tables, list(typology.labels), lengths, steps=6, generator=generator,
                         returns=RETURNS, given={10: tokens[10]})
    assert [len(t) for t in out] == lengths and out[10] == tokens[10]
    assert all(out[s] == out[t] for s, t in RETURNS)
    with pytest.raises(ValueError):  # a return pair must have phrases of one length
        reverse_sample(model, tables, list(typology.labels), [lengths[0] + 1, *lengths[1:]], steps=2,
                       generator=generator, returns=RETURNS)
    assert all(t[0] != grid.hold and grid.mask not in t for t in out)
    assert [sum(d for _, d in grid.decode(t)) for t in out] == [sum(d for _, d in p) for p in typology.phrases]


def test_smoke_run_resumes_and_never_overwrites(tmp_path: Path) -> None:
    for name in ("configs/chart_diffusion_v1.yaml", "configs/yangcheongdodeuri.yaml", "artifacts/measures_v0.json"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((PROJECT_ROOT / name).read_bytes())
    torch.set_num_threads(2)
    runtime = resolve_device("cpu")
    config = tmp_path / "configs/chart_diffusion_v1.yaml"
    quiet = {"run_mode": "smoke", "log": lambda _: None}
    report = v1.run_experiment(config, tmp_path, "t", runtime, **quiet)
    arms = {"diffusion", "diffusion_graph_025", "diffusion_graph_050", "diffusion_graph"}
    assert set(report["selected_steps"]) == arms
    assert len(report["full"]) == 4 and all(u["generation"]["per_piece"] for u in report["full"])
    assert v1.curve_rows(report) and v1.monitor_rows(report) and v1.generation_rows(report) and v1.recall_rows(report)
    graph_history = [u for u in report["full"] if u["arm"] == "diffusion_graph"][0]["history"]
    assert any("graph" in h for h in graph_history)  # the graph loss is on from the first steps
    assert report["selected_steps"]["diffusion_graph"]["graph_active_steps"] > 0
    assert report["selected_steps"]["diffusion"]["graph_active_steps"] == 0
    assert all(len(u["monitor"]) >= 2 for u in report["full"])  # monitoring spread over the run
    assert 0 < v1.heldout_baseline(report) < 3
    assert len(report["copytest"]) == 4 and all(len(u["pieces"]) == 2 for u in report["copytest"])
    evaluation = report["evaluation"]
    assert len(evaluation["models"]) == 4 and len(evaluation["copy_tests"]) == 4
    assert set(evaluation["criteria_by_arm"]) == arms - {"diffusion"}
    assert all(c["5_graph_loss"]["status"] in {"pass", "fail"} for c in evaluation["criteria_by_arm"].values())
    assert set(evaluation["selection"]["eligible"]) == arms - {"diffusion"}
    assert v1.selection_rows(evaluation) and len(v1.tradeoff_rows(evaluation)) == 4

    assert "z_novel" in evaluation["copy_tests"][0] and "copy_z_novel_combined" in evaluation
    assert evaluation["reference"]["repeat_rate"] == pytest.approx(33 / 72)

    run_dir = tmp_path / "artifacts/chart_diffusion_runs/t_smoke"
    stamps = {p.name: p.stat().st_mtime_ns for p in run_dir.glob("full_*")}
    again = v1.run_experiment(config, tmp_path, "t", runtime, **quiet)
    assert again["full"] == report["full"]
    assert stamps == {p.name: p.stat().st_mtime_ns for p in run_dir.glob("full_*")}
    # stopped after the checkpoint but before the unit's JSON: resume from the checkpoint, no retraining
    unit_json = run_dir / "full_diffusion_s260807.json"
    unit_json.unlink()
    resumed = v1.run_experiment(config, tmp_path, "t", runtime, **quiet)
    assert stamps["full_diffusion_s260807.pt"] == (run_dir / "full_diffusion_s260807.pt").stat().st_mtime_ns
    first = [u for u in report["full"] if u["arm"] == "diffusion"][0]
    back = [u for u in resumed["full"] if u["arm"] == "diffusion"][0]
    assert back["history"] == first["history"] and back["monitor"] == first["monitor"]
    assert back["generation"]["pieces"] == first["generation"]["pieces"]
    # a checkpoint from other settings is never overwritten
    unit_json.unlink()
    torch.save({"protocol": "other"}, run_dir / "full_diffusion_s260807.pt")
    with pytest.raises(FileExistsError):
        v1.run_experiment(config, tmp_path, "t", runtime, **quiet)
    piece_config = tmp_path / "configs/yangcheongdodeuri.yaml"
    piece_config.write_text(piece_config.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        v1.run_experiment(config, tmp_path, "t", runtime, **quiet)


def test_stale_code_in_memory_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    v1.check_loaded_source()  # unchanged files pass
    monkeypatch.setattr(v1, "LOADED_SOURCE", "loaded-before-a-pull")
    with pytest.raises(RuntimeError, match="Restart"):
        v1.run_experiment(PROJECT_ROOT / "configs/chart_diffusion_v1.yaml", tmp_path, "t", resolve_device("cpu"),
                          run_mode="smoke", log=lambda _: None)


def test_monitor_steps_and_baseline() -> None:
    assert v1.monitor_steps(100, 5) == [20, 40, 60, 80, 100]
    assert v1.monitor_steps(3, 5) == [1, 2, 3]
    setup = v1.load_setup(PROJECT_ROOT / "configs/chart_diffusion_v1.yaml", PROJECT_ROOT)
    value = v1.positional_baseline(setup, list(range(0, 72, 4)))
    assert 0 < value < np.log(setup["grid"].num_outputs)


def test_musicxml_export_round_trip(tmp_path: Path) -> None:
    from gugakgen.data.musicxml import parse_musicxml

    setup = v1.load_setup(PROJECT_ROOT / "configs/chart_diffusion_v1.yaml", PROJECT_ROOT)
    typology = setup["typology"]
    unit = {"arm": "x", "seed": 1, "generation": {"pieces": [
        {"seed": 5, "walk": list(typology.labels), "phrases": [[[p, d] for p, d in ph] for ph in typology.phrases]}
    ]}}
    [path] = v1.export_musicxml(unit, setup, tmp_path)
    score = parse_musicxml(path, piece_id="x", part_name="Piri")
    assert [tuple((e.pitch, float(e.duration_q)) for e in m.events) for m in score.measures] == list(typology.phrases)
    with pytest.raises(FileExistsError):
        v1.export_musicxml(unit, setup, tmp_path)
