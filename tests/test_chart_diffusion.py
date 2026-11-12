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
    assert set(report["selected_steps"]) == {"diffusion", "diffusion_graph"}
    assert len(report["full"]) == 2 and all(u["generation"]["per_piece"] for u in report["full"])
    assert v1.curve_rows(report) and v1.monitor_rows(report) and v1.generation_rows(report) and v1.recall_rows(report)
    graph_history = [u for u in report["full"] if u["arm"] == "diffusion_graph"][0]["history"]
    assert all("graph" not in h for h in graph_history)  # smoke steps end before the graph loss starts

    run_dir = tmp_path / "artifacts/chart_diffusion_runs/t_smoke"
    stamps = {p.name: p.stat().st_mtime_ns for p in run_dir.glob("full_*")}
    again = v1.run_experiment(config, tmp_path, "t", runtime, **quiet)
    assert again["full"] == report["full"]
    assert stamps == {p.name: p.stat().st_mtime_ns for p in run_dir.glob("full_*")}
    (run_dir / "full_diffusion_s260807.json").unlink()
    with pytest.raises(FileExistsError):
        v1.run_experiment(config, tmp_path, "t", runtime, **quiet)
    config.write_text(config.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        v1.run_experiment(config, tmp_path, "t", runtime, **quiet)


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
