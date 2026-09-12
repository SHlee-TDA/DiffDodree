from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from gugakgen.data.musicxml import parse_musicxml, write_musicxml
from gugakgen.data.vocabulary import ScoreVocabulary
from gugakgen.device import resolve_device
from gugakgen.sampling.meter_decoder import ExactMeterDecoder
from gugakgen.structure.transition_graph import TransitionGraph
from gugakgen.workflows.generate import load_generation_bundle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "artifacts"
CHAPTER_RANGES = ((1, 4), (5, 10), (11, 24), (25, 28), (29, 41), (42, 55), (56, 72))


def test_musicxml_parse_and_write_round_trip(tmp_path: Path) -> None:
    source = next((PROJECT_ROOT / "data/02 Cheon-Yangcheongdodeuri_musicxml_1-6").glob("*Piri*.xml*"))
    score = parse_musicxml(
        source,
        piece_id="yangcheongdodeuri_piri",
        part_name="피리",
        chapter_ranges=CHAPTER_RANGES,
    )
    score.validate_meters()
    assert len(score.measures) == 55

    frozen = json.loads((ARTIFACTS / "measures_v0.json").read_text(encoding="utf-8"))
    assert [event.to_dict() for event in score.measures[0].events] == frozen[0]["events"]

    destination = tmp_path / "round_trip.musicxml"
    write_musicxml(score, destination, divisions_per_quarter=4, key_fifths=-4)
    reparsed = parse_musicxml(
        destination,
        piece_id=score.piece_id,
        part_name=score.part,
        chapter_ranges=CHAPTER_RANGES,
    )
    reparsed.validate_meters()
    assert [measure.signature() for measure in reparsed.measures] == [
        measure.signature() for measure in score.measures
    ]


def test_transition_graph_original_path_and_projection() -> None:
    graph = TransitionGraph.from_json(ARTIFACTS / "phrase_graph_v0.json")
    original = np.asarray([graph.node_id(name) for name in graph.original_path], dtype=np.int64)
    assert len(original) == 72
    assert graph.transition_validity(original) == 1.0

    damaged = original.copy()
    damaged[10] = graph.node_id("I")
    assert graph.transition_validity(damaged) < 1.0
    projected, edits = graph.project_path(
        damaged,
        anchors={0: int(original[0]), len(original) - 1: int(original[-1])},
    )
    assert edits >= 1
    assert graph.transition_validity(projected) == 1.0


def test_exact_meter_decoder_for_both_score_meters() -> None:
    vocabulary = ScoreVocabulary.from_json(ARTIFACTS / "vocabulary_v0.json")
    decoder = ExactMeterDecoder(
        vocabulary.duration_values,
        max_events=vocabulary.max_events_per_measure,
        pitch_pad_id=vocabulary.pitch_pad_id,
        pitch_mask_id=vocabulary.pitch_mask_id,
        duration_pad_id=vocabulary.duration_pad_id,
        duration_mask_id=vocabulary.duration_mask_id,
        allowed_pitch_ids=range(vocabulary.pitch_rest_id + 1),
    )
    rng = np.random.default_rng(2026)
    for target_beats in (4.0, 6.0):
        pitch_logits = rng.normal(size=(vocabulary.max_events_per_measure, vocabulary.num_pitch_tokens))
        duration_logits = rng.normal(
            size=(vocabulary.max_events_per_measure, vocabulary.num_duration_tokens)
        )
        for temperature in (0.0, 1.0):
            result = decoder.decode(
                pitch_logits,
                duration_logits,
                target_beats,
                rng=rng,
                temperature=temperature,
            )
            assert decoder.is_valid(
                result.pitch_tokens, result.duration_tokens, target_beats
            )


def test_one_step_sampling_smoke() -> None:
    bundle = load_generation_bundle(
        ARTIFACTS / "m2_denoiser_v0.pt",
        project_root=PROJECT_ROOT,
        runtime=resolve_device("cpu"),
    )
    sampling_config = replace(
        bundle.sampling_config,
        num_steps=1,
        copy_rejection_attempts=1,
    )
    result = bundle.sampler.sample(
        seed=2026,
        config=sampling_config,
        score_anchors=(),
        family_anchors={1: "A", 70: "H", 71: "H", 72: "I"},
    )
    assert result.tokens.pitch.shape == (72, 7)
    assert bundle.context.graph.transition_validity(result.tokens.family) == 1.0
    assert not np.any(result.tokens.pitch == bundle.context.vocabulary.pitch_mask_id)
    assert not np.any(result.tokens.family == bundle.context.vocabulary.family_mask_id)
    assert all(
        bundle.sampler.meter_decoder.is_valid(
            result.tokens.pitch[index],
            result.tokens.duration[index],
            float(result.tokens.target_beats[index]),
        )
        for index in range(result.tokens.num_measures)
    )
