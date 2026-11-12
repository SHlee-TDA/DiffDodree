from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from gugakgen.chart.coords import Scale, from_chart, surrogate_tables, to_chart
from gugakgen.chart.edit_distance import align, distance
from gugakgen.chart.metrics import baseline_edit, baseline_recombine, evaluate_piece
from gugakgen.chart.model import (
    ChartDenoiser,
    Layout,
    LossWeights,
    encode_piece,
    hide,
    losses,
    margin_tables,
    pitch_targets,
    random_spans,
)
from gugakgen.chart.sampling import (
    ChainConfig,
    EmpiricalPredictor,
    ModelPredictor,
    run_chain,
    sample_durations,
)
from gugakgen.chart.typology import build_typology, load_phrases
from gugakgen.device import resolve_device
from gugakgen.workflows import chart_experiment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RETURNS = ((0, 24), (1, 25), (2, 26), (3, 27))
# Methods draft 2026-11-11, section 2.4
DRAFT_SEQUENCE = "abcc | ddebcc | abedeefdaghiig | abcc | efffgfgfhhihg | afhbbedgdebfff | eihgafcfcfeeehcch"


@pytest.fixture(scope="module")
def typology():
    phrases, chapters = load_phrases(PROJECT_ROOT / "artifacts/measures_v0.json")
    return build_typology(phrases, chapters, k=9)


@pytest.fixture(scope="module")
def layout(typology):
    return Layout.from_typology(typology)


def test_edit_distance_basics() -> None:
    a = [(63, 1.0), (65, 1.0), (68, 2.0)]
    assert distance(a, a) == 0.0
    assert distance(a, [(63, 1.0), (65, 1.0), (70, 2.0)]) == pytest.approx(2 * 2 / 8)
    # one note split into two of the same pitch costs only the surcharge
    value, steps = align(a, [(63, 1.0), (65, 1.0), (68, 1.0), (68, 1.0)])
    assert value == pytest.approx(0.5 / 8)
    assert [step.kind for step in steps] == ["sub", "sub", "frag"]


def test_typology_reproduces_the_draft(typology) -> None:
    assert typology.sequence_by_chapter() == DRAFT_SEQUENCE
    assert typology.silhouette == pytest.approx(0.424, abs=5e-4)
    assert typology.medoid_phrases[5] + 1 == 43 and typology.names.count("f") == 13
    assert int((typology.transition_counts() > 0).sum()) == 41
    assert typology.contact_summary() == {
        "contact_pairs": 13,
        "transition_pairs": 24,
        "shared_pairs": 10,
        "transitions": 56,
        "transitions_across_contact": 31,
    }
    assert all(typology.is_member(p, label) for p, label in zip(typology.phrases, typology.labels, strict=True))


def test_chart_restores_every_phrase_and_surrogate_matches_distance(typology, layout) -> None:
    scale = layout.scale
    for phrase, label in zip(typology.phrases, typology.labels, strict=True):
        medoid = typology.medoids[label]
        assert tuple(from_chart(to_chart(phrase, medoid, scale), medoid, scale)) == phrase
        cost, const = surrogate_tables(phrase, typology.medoids, scale, layout.max_events)
        tokens = [layout.pitch_token(pitch) for pitch, _ in phrase]
        value = [sum(cost[c, j, v] for j, v in enumerate(tokens)) + const[c] for c in range(typology.k)]
        assert np.allclose(value, typology.medoid_distances(phrase), atol=1e-5)


def test_context_is_an_address_of_most_measures(typology) -> None:
    audit = chart_experiment.address_audit(typology)
    assert audit == {
        "measures": 72,
        "kappa_unique_position": 57,
        "kappa_single_content": 66,
        "kappa_and_boundary_notes_unique_position": 63,
    }


def test_heldout_folds_group_identical_phrases_and_keep_medoids(typology) -> None:
    folds = chart_experiment.heldout_folds(typology, 5, 20261111)
    held = [t for fold in folds for t in fold]
    assert len(held) == len(set(held))
    assert not set(held) & set(typology.medoid_phrases)
    for fold in folds:
        inside = {typology.phrases[t] for t in fold}
        assert all((typology.phrases[t] in inside) == (t in fold) for t in range(72))
    assert {0, 24} <= set(held) or not {0, 24} & set(held)  # measures 1 and 25 are identical


def test_model_outputs_and_losses(typology, layout) -> None:
    torch.manual_seed(0)
    encoded = encode_piece(typology.phrases, typology, layout)
    batch = {key: torch.as_tensor(value) for key, value in encoded.items()}
    generator = torch.Generator().manual_seed(0)
    hidden = random_spans(batch["length"], layout.max_events, generator, 0.15, 1.0)
    valid = torch.arange(layout.max_events)[None] < batch["length"][:, None]
    assert hidden.any(dim=1).all() and not (hidden & ~valid).any()
    # eta tokens and reference steps give back the pitches of the score
    targets = pitch_targets(batch, layout)
    for t, phrase in enumerate(typology.phrases):
        assert targets[t, : len(phrase)].tolist() == [layout.pitch_token(p) for p, _ in phrase]

    model = ChartDenoiser(layout, d_model=32, n_layers=1, n_heads=2, ffn_dim=64, dropout=0.0)
    nobody = torch.zeros(72, dtype=torch.bool)
    log_pitch, log_dur = model(hide(batch, hidden, layout, null_context=~nobody, null_type=nobody))
    assert log_pitch.shape == (72, layout.max_events, len(layout.scale) + 1)
    assert torch.allclose(log_pitch.exp().sum(-1), torch.ones(72, layout.max_events), atol=1e-5)
    cost, const = (torch.as_tensor(x) for x in margin_tables(typology, layout))
    terms = losses(log_pitch, log_dur, batch, hidden, layout, LossWeights(1.0, 1.0), cost, const)
    assert set(terms) == {"pitch", "duration", "rec", "margin", "beats", "total"}
    terms["total"].backward()
    # with certainty on the truth the structural terms vanish: every phrase is inside its type
    certain_pitch = torch.log(torch.nn.functional.one_hot(targets, len(layout.scale) + 1) + 1e-12)
    durations = batch["dur"].clamp(max=len(layout.durations) - 1)
    certain_dur = torch.log(torch.nn.functional.one_hot(durations, len(layout.durations)) + 1e-12)
    exact = losses(
        certain_pitch, certain_dur, batch, hidden, layout,
        LossWeights(1.0, 1.0, margin_size=0.0), cost, const,
    )
    assert float(exact["margin"]) < 1e-3 and float(exact["beats"]) < 1e-9


def test_duration_sampler_keeps_the_total() -> None:
    rng = np.random.default_rng(0)
    quanta = (1, 2, 3, 4, 6, 7, 8, 12)
    log_prob = np.log(np.full((3, len(quanta)), 1 / len(quanta)))
    for _ in range(50):
        chosen = sample_durations(log_prob, quanta, 9, rng)
        assert chosen is not None and sum(quanta[i] for i in chosen) == 9
    assert sample_durations(log_prob, quanta, 2, rng) is None  # three events need at least 3


def test_chain_keeps_the_structure(typology, layout) -> None:
    encoded = encode_piece(typology.phrases, typology, layout)
    predictors = [
        EmpiricalPredictor(encoded, layout),
        ModelPredictor(
            ChartDenoiser(layout, d_model=32, n_layers=1, n_heads=2, ffn_dim=64), torch.device("cpu")
        ),
    ]
    for predictor in predictors:
        chain = ChainConfig(steps=2, rho=0.6, returns=RETURNS)
        phrases, stats = run_chain(typology, layout, predictor, chain, np.random.default_rng(1))
        scores = evaluate_piece(phrases, typology, RETURNS)
        assert scores["type_membership"] == 1.0
        assert scores["returns_exact"] == 1.0 and scores["totals_exact"] == 1.0
        assert stats.proposals == 2 * 68
    again, _ = run_chain(typology, layout, predictors[0], chain, np.random.default_rng(1))
    assert again == run_chain(typology, layout, predictors[0], chain, np.random.default_rng(1))[0]


def test_baselines(typology, layout) -> None:
    rng = np.random.default_rng(0)
    recombined = evaluate_piece(baseline_recombine(typology, RETURNS, rng), typology, RETURNS)
    assert recombined["original_phrase_rate"] == 1.0 and recombined["type_membership"] == 1.0
    edited = evaluate_piece(
        baseline_edit(typology, RETURNS, rng, Scale.from_phrases(typology.phrases), 0.3),
        typology,
        RETURNS,
    )
    assert edited["returns_exact"] == 1.0 and edited["totals_exact"] == 1.0
    original = evaluate_piece(list(typology.phrases), typology, RETURNS)
    assert original["interval_jsd"] == 0.0 and original["unseen_note_transition_rate"] == 0.0


def test_smoke_run_resumes_and_never_overwrites(tmp_path: Path) -> None:
    for name in ("configs/chart_v0.yaml", "configs/yangcheongdodeuri.yaml", "artifacts/measures_v0.json"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((PROJECT_ROOT / name).read_bytes())
    torch.set_num_threads(2)
    runtime = resolve_device("cpu")
    config = tmp_path / "configs/chart_v0.yaml"
    quiet = {"run_mode": "smoke", "log": lambda _: None}
    report = chart_experiment.run_experiment(config, tmp_path, "t", runtime, **quiet)
    assert report["typology"]["sequence"] == DRAFT_SEQUENCE
    assert set(report["selected_steps"]) == {"rec", "rec_drop", "struct"}
    assert len(report["heldout"]) == 6 and len(report["full"]) == 3
    assert chart_experiment.heldout_rows(report) and chart_experiment.curve_rows(report)
    assert chart_experiment.recall_rows(report) and chart_experiment.generation_rows(report)

    run_dir = tmp_path / "artifacts/chart_runs/t_smoke"
    stamps = {path.name: path.stat().st_mtime_ns for path in run_dir.glob("full_*")}
    again = chart_experiment.run_experiment(config, tmp_path, "t", runtime, **quiet)
    assert again["full"] == report["full"]
    assert stamps == {path.name: path.stat().st_mtime_ns for path in run_dir.glob("full_*")}

    # a finished checkpoint without its result is never replaced
    (run_dir / "full_rec_s260807.json").unlink()
    with pytest.raises(FileExistsError):
        chart_experiment.run_experiment(config, tmp_path, "t", runtime, **quiet)
    # a changed configuration cannot continue under the same RUN_TAG
    config.write_text(config.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        chart_experiment.run_experiment(config, tmp_path, "t", runtime, **quiet)
