"""Internal assembly of piece-specific artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from gugakgen.config import load_yaml, project_path, require_mapping
from gugakgen.data.musicxml import load_measure_artifact, score_to_tokens
from gugakgen.data.records import ScoreRecord, TokenizedScore
from gugakgen.data.vocabulary import ScoreVocabulary
from gugakgen.models.denoiser import DenoiserConfig
from gugakgen.structure.hwanip_relations import HwanipRelations
from gugakgen.structure.transition_graph import TransitionGraph


@dataclass(frozen=True)
class PieceContext:
    config: Mapping[str, Any]
    config_path: Path
    score: ScoreRecord
    vocabulary: ScoreVocabulary
    graph: TransitionGraph
    tokens: TokenizedScore
    hwanip_relations: HwanipRelations
    allowed_families: NDArray[np.bool_]
    meter_id_to_value: Mapping[int, tuple[int, int]]


def load_piece_context(
    piece_config_path: str | Path,
    *,
    project_root: str | Path,
    model_section: Mapping[str, Any],
) -> PieceContext:
    root = Path(project_root).resolve()
    path = project_path(root, piece_config_path)
    config = load_yaml(path)
    piece = require_mapping(config.get("piece"), "piece")
    data = require_mapping(config.get("data"), "data")
    structure = require_mapping(config.get("structure"), "structure")

    vocabulary = ScoreVocabulary.from_json(project_path(root, data["vocabulary"]))
    graph = TransitionGraph.from_json(project_path(root, data["transition_graph"]))
    score = load_measure_artifact(
        project_path(root, data["canonical_measures"]),
        piece_id=str(piece["id"]),
        part=str(piece["part"]),
        original_path=graph.original_path,
    )
    score.validate_meters()
    position_buckets = int(model_section.get("num_position_buckets", 4))
    position_encoding = str(model_section.get("position_encoding", "relative_chapter_quartile"))
    tokens = score_to_tokens(
        score,
        vocabulary,
        original_path=graph.original_path,
        num_position_buckets=position_buckets,
        position_encoding=position_encoding,
    )
    relations = HwanipRelations.from_config(
        structure.get("hwanip_relations"), num_measures=tokens.num_measures
    )

    allowed = np.ones((tokens.num_measures, vocabulary.num_families), dtype=bool)
    for terminal_name in structure.get("terminal_families", []):
        terminal_id = vocabulary.encode_family(str(terminal_name))
        observed_positions = {
            index for index, value in enumerate(tokens.family) if int(value) == terminal_id
        }
        allowed[:, terminal_id] = False
        for index in observed_positions:
            allowed[index, terminal_id] = True
    if bool((~allowed.any(axis=1)).any()):
        raise ValueError("Piece configuration leaves a measure with no allowed family")

    meter_values = [
        (measure.meter.numerator, measure.meter.denominator) for measure in score.measures
    ]
    meter_id_to_value: dict[int, tuple[int, int]] = {}
    for token_id, value in zip(tokens.meter, meter_values):
        existing = meter_id_to_value.setdefault(int(token_id), value)
        if existing != value:
            raise ValueError("One meter token id maps to multiple MusicXML meters")
    return PieceContext(
        config,
        path,
        score,
        vocabulary,
        graph,
        tokens,
        relations,
        allowed,
        meter_id_to_value,
    )


def make_model_config(
    model_section: Mapping[str, Any], context: PieceContext
) -> DenoiserConfig:
    tokens = context.tokens
    vocabulary = context.vocabulary
    return DenoiserConfig(
        num_pitch_tokens=vocabulary.num_pitch_tokens,
        num_duration_tokens=vocabulary.num_duration_tokens,
        num_family_tokens=vocabulary.num_family_tokens,
        num_families=vocabulary.num_families,
        num_chapters=int(tokens.chapter.max()) + 1,
        num_meters=int(tokens.meter.max()) + 1,
        num_position_buckets=int(model_section.get("num_position_buckets", 4)),
        max_events_per_measure=vocabulary.max_events_per_measure,
        d_model=int(model_section.get("d_model", 128)),
        n_layers=int(model_section.get("n_layers", 4)),
        n_heads=int(model_section.get("n_heads", 4)),
        ffn_dim=int(model_section.get("ffn_dim", 256)),
        dropout=float(model_section.get("dropout", 0.1)),
        embedding_init_std=float(model_section.get("embedding_init_std", 0.02)),
        hwanip_message_scale=float(model_section.get("hwanip_message_scale", 0.0)),
    )
