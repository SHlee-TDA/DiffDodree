from __future__ import annotations

from pathlib import Path

import pytest

from gugakgen.config import load_yaml
from gugakgen.structure.measure_distance import (
    distance_matrix,
    family_labels,
    measure_distance,
    passage_distance,
    score_events,
)
from gugakgen.workflows._common import load_piece_context

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _original_events() -> tuple[list, str]:
    config = load_yaml(PROJECT_ROOT / "configs/train_maskgit.yaml")
    context = load_piece_context(
        config["piece_config"], project_root=PROJECT_ROOT, model_section=config["model"]
    )
    return score_events(context.tokens, context.vocabulary), "".join(context.graph.original_path)


def test_distance_reproduces_frozen_measure_families() -> None:
    events, frozen_path = _original_events()
    labels = family_labels(distance_matrix(events, relative_pitch=True), num_families=9)
    assert "".join(labels) == frozen_path


def test_hwanip_passages_are_identical_and_others_are_not() -> None:
    events, _ = _original_events()
    chapter_one, chapter_four = events[0:4], events[24:28]
    assert passage_distance(chapter_one, chapter_four) == 0.0
    assert passage_distance(chapter_one, events[4:8]) > 0.5


def test_relative_pitch_ignores_transposition_but_absolute_does_not() -> None:
    measure = [(70, 1.0), (72, 0.5), (75, 0.5), (None, 1.0), (72, 1.0)]
    transposed = [(None if pitch is None else pitch + 2, duration) for pitch, duration in measure]
    assert measure_distance(measure, transposed, relative_pitch=True) == pytest.approx(0.0)
    assert measure_distance(measure, transposed, relative_pitch=False) > 0.0


def test_distance_grows_with_larger_changes() -> None:
    measure = [(70, 1.0), (72, 1.0), (72, 1.0), (75, 1.0)]
    one_note = [(70, 1.0), (72, 1.0), (73, 1.0), (75, 1.0)]
    every_note = [(71, 1.0), (73, 1.0), (73, 1.0), (76, 1.0)]
    small = measure_distance(measure, one_note, relative_pitch=False)
    large = measure_distance(measure, every_note, relative_pitch=False)
    assert 0.0 < small < large


def test_passages_must_align() -> None:
    with pytest.raises(ValueError):
        passage_distance([[(70, 1.0)]], [[(70, 1.0)], [(72, 1.0)]])
