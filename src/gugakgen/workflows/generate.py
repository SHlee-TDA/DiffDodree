"""Load a portable Checkpoint and generate an exact-meter symbolic score."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from gugakgen.config import find_project_root, load_yaml, project_path, require_mapping
from gugakgen.data.musicxml import tokens_to_score, write_musicxml
from gugakgen.data.records import ScoreRecord, TokenizedScore
from gugakgen.device import RuntimeDevice, resolve_device
from gugakgen.models.checkpoint import LoadedCheckpoint, load_checkpoint, load_model_state
from gugakgen.models.denoiser import DenoiserConfig, GugakDenoiser
from gugakgen.sampling.maskgit import MaskGITConfig, MaskGITSampler, SamplingResult
from gugakgen.workflows._common import PieceContext, load_piece_context, make_model_config


@dataclass(frozen=True)
class GenerationBundle:
    model: GugakDenoiser
    sampler: MaskGITSampler
    context: PieceContext
    checkpoint: LoadedCheckpoint
    runtime: RuntimeDevice
    sampling_config: MaskGITConfig
    score_anchors: tuple[int, ...]
    family_anchors: Mapping[int, str | int]


@dataclass(frozen=True)
class GenerationResult:
    score: ScoreRecord
    sampling: SamplingResult
    checkpoint_sha256: str
    output_path: Path | None = None


def _legacy_model_config(
    checkpoint: LoadedCheckpoint,
    context: PieceContext,
    model_section: Mapping[str, Any],
) -> DenoiserConfig:
    legacy = checkpoint.payload.get("cfg")
    if not isinstance(legacy, Mapping):
        return make_model_config(model_section, context)
    values = dict(model_section)
    aliases = {
        "d_model": "d_model",
        "n_layers": "n_layers",
        "n_heads": "n_heads",
        "ffn_dim": "ffn_dim",
        "dropout": "dropout",
        "emb_init_std": "embedding_init_std",
    }
    for old_name, new_name in aliases.items():
        if old_name in legacy:
            values[new_name] = legacy[old_name]
    values["num_position_buckets"] = int(
        checkpoint.payload.get("pos_buckets", values.get("num_position_buckets", 4))
    )
    # Legacy Checkpoints were not trained with explicit source→target 환입 messages.
    values["hwanip_message_scale"] = 0.0
    return make_model_config(values, context)


def _sampling_config(
    sampling: Mapping[str, Any], loss: Mapping[str, Any]
) -> MaskGITConfig:
    return MaskGITConfig(
        num_steps=int(sampling.get("num_steps", 12)),
        schedule=str(sampling.get("schedule", "surface_first")),
        temperature_pitch_duration=float(sampling.get("temperature_pitch_duration", 1.4)),
        temperature_family=float(sampling.get("temperature_family", 1.0)),
        graph_guidance_scale=float(sampling.get("graph_guidance_scale", 0.5)),
        hwanip_guidance_scale=float(sampling.get("hwanip_guidance_scale", 0.0)),
        corruption=float(sampling.get("corruption", 1.0)),
        forbid_unobserved_transitions=bool(
            sampling.get("forbid_unobserved_transitions", True)
        ),
        project_transition_path=bool(sampling.get("project_transition_path", True)),
        transition_log_floor=float(loss.get("transition_log_floor", -6.0)),
        copy_rejection_attempts=int(sampling.get("copy_rejection_attempts", 4)),
    )


def _with_checkpoint_positions(
    tokens: TokenizedScore, checkpoint: LoadedCheckpoint
) -> TokenizedScore:
    metadata = checkpoint.payload.get("metadata", {})
    positions = checkpoint.payload.get("pos_array")
    if positions is None and isinstance(metadata, Mapping):
        positions = metadata.get("position_array")
    if positions is None:
        return tokens
    position_array = np.asarray(positions, dtype=np.int64)
    if position_array.shape != (tokens.num_measures,):
        raise ValueError("Checkpoint position_array has an incompatible shape")
    return replace(tokens, position=position_array)


def load_generation_bundle(
    checkpoint_path: str | Path,
    *,
    config_path: str | Path = "configs/train_maskgit.yaml",
    project_root: str | Path | None = None,
    runtime: RuntimeDevice | None = None,
) -> GenerationBundle:
    root = find_project_root(project_root)
    config = load_yaml(project_path(root, config_path))
    runtime_section = require_mapping(config.get("runtime"), "runtime")
    model_section = require_mapping(config.get("model"), "model")
    sampling_section = require_mapping(config.get("sampling"), "sampling")
    loss_section = require_mapping(config.get("loss"), "loss")
    runtime = runtime or resolve_device(str(runtime_section.get("device", "auto")))
    checkpoint = load_checkpoint(project_path(root, checkpoint_path))
    context = load_piece_context(
        config["piece_config"], project_root=root, model_section=model_section
    )
    context = replace(context, tokens=_with_checkpoint_positions(context.tokens, checkpoint))

    raw_model_config = checkpoint.payload.get("model_config")
    model_config = (
        DenoiserConfig.from_dict(raw_model_config)
        if isinstance(raw_model_config, Mapping)
        else _legacy_model_config(checkpoint, context, model_section)
    )
    expected = make_model_config(
        {**model_section, "num_position_buckets": model_config.num_position_buckets}, context
    )
    compatibility_fields = (
        "num_pitch_tokens",
        "num_duration_tokens",
        "num_family_tokens",
        "num_families",
        "num_chapters",
        "num_meters",
        "max_events_per_measure",
    )
    for name in compatibility_fields:
        if getattr(model_config, name) != getattr(expected, name):
            raise ValueError(
                f"Checkpoint model_config.{name}={getattr(model_config, name)} does not match "
                f"the current dataset value {getattr(expected, name)}"
            )

    model = GugakDenoiser(
        model_config,
        torch.as_tensor(context.graph.transition_matrix, dtype=torch.float32),
    )
    load_model_state(model, checkpoint, strict=True)
    model.to(runtime.device).eval()
    piece_data = require_mapping(context.config.get("data"), "data")
    sampler = MaskGITSampler(
        model=model,
        vocabulary=context.vocabulary,
        transition_graph=context.graph,
        template=context.tokens,
        runtime=runtime,
        duration_quantum_q=float(piece_data.get("duration_quantum_q", 0.25)),
        hwanip_relations=context.hwanip_relations,
        allowed_families=context.allowed_families,
    )
    family_anchors = {
        int(number): value for number, value in sampling_section.get("family_anchors", {}).items()
    }
    return GenerationBundle(
        model,
        sampler,
        context,
        checkpoint,
        runtime,
        _sampling_config(sampling_section, loss_section),
        tuple(int(value) for value in sampling_section.get("score_anchors", [])),
        family_anchors,
    )


def generate_from_config(
    checkpoint_path: str | Path,
    *,
    seed: int,
    config_path: str | Path = "configs/train_maskgit.yaml",
    project_root: str | Path | None = None,
    runtime: RuntimeDevice | None = None,
    sampling_overrides: Mapping[str, Any] | None = None,
    score_anchors: Sequence[int] | None = None,
    family_anchors: Mapping[int, str | int] | None = None,
    output_path: str | Path | None = None,
) -> GenerationResult:
    root = find_project_root(project_root)
    bundle = load_generation_bundle(
        checkpoint_path, config_path=config_path, project_root=root, runtime=runtime
    )
    sampling_config = bundle.sampling_config
    if sampling_overrides:
        unknown = set(sampling_overrides) - set(MaskGITConfig.__dataclass_fields__)
        if unknown:
            raise KeyError(f"Unknown sampling override fields: {sorted(unknown)}")
        sampling_config = replace(sampling_config, **dict(sampling_overrides))
    sampled = bundle.sampler.sample(
        seed=seed,
        config=sampling_config,
        score_anchors=bundle.score_anchors if score_anchors is None else score_anchors,
        family_anchors=bundle.family_anchors if family_anchors is None else family_anchors,
    )
    score = tokens_to_score(
        sampled.tokens,
        bundle.context.vocabulary,
        piece_id=bundle.context.score.piece_id,
        part=bundle.context.score.part,
        meter_id_to_value=bundle.context.meter_id_to_value,
    )
    piece = require_mapping(bundle.context.config.get("piece"), "piece")
    score = replace(
        score,
        metadata={
            "title": piece.get("title_ko", piece.get("title_en", piece["id"])),
            "generated_seed": seed,
            "checkpoint_sha256": bundle.checkpoint.sha256,
        },
    )

    destination: Path | None = None
    if output_path is not None:
        destination = project_path(root, output_path)
        export = require_mapping(bundle.context.config.get("export"), "export")
        write_musicxml(
            score,
            destination,
            divisions_per_quarter=int(export.get("divisions_per_quarter", 4)),
            key_fifths=int(export.get("key_fifths", -4)),
            part_id=str(export.get("part_id", "P1")),
            part_name=str(export.get("part_name", score.part)),
            prefer_flats=int(export.get("key_fifths", -4)) <= 0,
        )
    return GenerationResult(score, sampled, bundle.checkpoint.sha256, destination)
