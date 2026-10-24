from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from gugakgen.config import load_yaml, require_mapping
from gugakgen.device import resolve_device
from gugakgen.models.denoiser import GugakDenoiser
from gugakgen.sampling.maskgit import MaskGITConfig, MaskGITSampler
from gugakgen.workflows._common import PieceContext, load_piece_context, make_model_config
from gugakgen.workflows.conddrop_experiment import (
    evaluate_checkpoint,
    summary_rows,
    write_runtime_configs,
)
from gugakgen.workflows.generate import load_generation_bundle
from gugakgen.workflows.train import _make_batch, memorization_recall

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONDDROP_CONFIG = PROJECT_ROOT / "configs/train_maskgit_conddrop.yaml"


def _context_and_model() -> tuple[PieceContext, GugakDenoiser, dict]:
    config = load_yaml(CONDDROP_CONFIG)
    model_section = require_mapping(config.get("model"), "model")
    context = load_piece_context(
        config["piece_config"], project_root=PROJECT_ROOT, model_section=model_section
    )
    torch.manual_seed(2026)
    model = GugakDenoiser(
        make_model_config(model_section, context),
        torch.as_tensor(context.graph.transition_matrix, dtype=torch.float32),
    )
    return context, model, require_mapping(config.get("training"), "training")


def test_null_tokens_add_one_embedding_row_each() -> None:
    context, model, _ = _context_and_model()
    config = model.config
    assert config.null_conditions
    assert model.emb_ch.num_embeddings == config.num_chapters + 1
    assert model.emb_pos.num_embeddings == config.num_position_buckets + 1
    assert config.null_chapter_id == int(context.tokens.chapter.max()) + 1


def test_condition_dropout_replaces_chapter_and_position_together() -> None:
    context, model, training = _context_and_model()
    null_ids = (model.config.null_chapter_id, model.config.null_position_id)
    device = torch.device("cpu")

    kept = _make_batch(
        np.random.default_rng(0),
        context=context,
        batch_size=4,
        training={**training, "condition_dropout": 0.0},
        device=device,
        null_ids=null_ids,
    )
    assert (kept["chapter"].numpy() == context.tokens.chapter).all()
    assert (kept["position"].numpy() == context.tokens.position).all()

    dropped = _make_batch(
        np.random.default_rng(0),
        context=context,
        batch_size=64,
        training=training,
        device=device,
        null_ids=null_ids,
    )
    chapter_null = dropped["chapter"].numpy() == null_ids[0]
    position_null = dropped["position"].numpy() == null_ids[1]
    assert np.array_equal(chapter_null, position_null)
    assert 0.4 < chapter_null.mean() < 0.6

    with pytest.raises(ValueError, match="null_conditions"):
        _make_batch(
            np.random.default_rng(0),
            context=context,
            batch_size=1,
            training=training,
            device=device,
        )


def test_memorization_recall_reports_keyed_and_null() -> None:
    context, model, _ = _context_and_model()
    pairs = torch.as_tensor(context.hwanip_relations.pairs, dtype=torch.long).reshape(-1, 2)
    null_ids = (model.config.null_chapter_id, model.config.null_position_id)
    result = memorization_recall(model, context, pairs, device=torch.device("cpu"), null_ids=null_ids)
    assert set(result) == {"recall_keyed", "recall_null"}
    assert all(0.0 <= value <= 1.0 for value in result.values())
    assert set(memorization_recall(model, context, pairs, device=torch.device("cpu"))) == {
        "recall_keyed"
    }


def test_sampler_feeds_null_conditions_only_to_chosen_measures() -> None:
    context, model, _ = _context_and_model()
    sampler = MaskGITSampler(
        model=model,
        vocabulary=context.vocabulary,
        transition_graph=context.graph,
        template=context.tokens,
        runtime=resolve_device("cpu"),
        hwanip_relations=context.hwanip_relations,
        allowed_families=context.allowed_families,
    )
    seen: list[tuple[np.ndarray, np.ndarray]] = []
    forward = model.forward

    def recording_forward(*args: torch.Tensor) -> object:
        seen.append((args[5].cpu().numpy().copy(), args[7].cpu().numpy().copy()))
        return forward(*args)

    model.forward = recording_forward  # type: ignore[method-assign]
    sampler.sample(
        seed=2026,
        config=MaskGITConfig(num_steps=1, copy_rejection_attempts=1),
        family_anchors={1: "A", 70: "H", 71: "H", 72: "I"},
        null_condition_measures=(1, 2, 3, 4),
    )

    expected_chapter = context.tokens.chapter.copy()
    expected_position = context.tokens.position.copy()
    expected_chapter[:4] = model.config.null_chapter_id
    expected_position[:4] = model.config.null_position_id
    assert seen
    for chapter, position in seen:
        assert np.array_equal(chapter, expected_chapter)
        assert np.array_equal(position, expected_position)


def test_legacy_checkpoint_rejects_null_conditions() -> None:
    bundle = load_generation_bundle(
        PROJECT_ROOT / "artifacts/m2_denoiser_v0.pt",
        project_root=PROJECT_ROOT,
        runtime=resolve_device("cpu"),
    )
    assert not bundle.model.config.null_conditions
    with pytest.raises(ValueError, match="null chapter/position"):
        bundle.sampler.sample(
            seed=0,
            config=replace(bundle.sampling_config, num_steps=1, copy_rejection_attempts=1),
            null_condition_measures=(1,),
        )


def test_runtime_configs_use_tagged_names_and_never_overwrite(tmp_path: Path) -> None:
    shutil.copytree(PROJECT_ROOT / "configs", tmp_path / "configs")
    written = write_runtime_configs(tmp_path, run_tag="t1", steps=10)
    assert set(written) == {"baseline", "conddrop"}
    baseline = load_yaml(written["baseline"])
    conddrop = load_yaml(written["conddrop"])
    assert baseline["artifacts"]["checkpoint_name"] == "yangcheongdodeuri_maskgit_t1.pt"
    assert conddrop["artifacts"]["checkpoint_name"] == "yangcheongdodeuri_maskgit_conddrop_t1.pt"
    assert baseline["training"]["steps"] == conddrop["training"]["steps"] == 10
    assert conddrop["training"]["condition_dropout"] > 0
    assert baseline["training"].get("condition_dropout", 0.0) == 0.0

    with pytest.raises(FileExistsError):
        write_runtime_configs(tmp_path, run_tag="t1")

    checkpoint_dir = tmp_path / "artifacts/checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "yangcheongdodeuri_maskgit_conddrop_t2_smoke.pt").write_bytes(b"")
    with pytest.raises(FileExistsError, match="already exist"):
        write_runtime_configs(tmp_path, run_tag="t2")
    assert not (tmp_path / "artifacts/runtime_configs/yangcheongdodeuri_maskgit_t2.yaml").exists()


def test_evaluate_checkpoint_on_legacy_model() -> None:
    report = evaluate_checkpoint(
        PROJECT_ROOT / "artifacts/m2_denoiser_v0.pt",
        "configs/train_maskgit.yaml",
        project_root=PROJECT_ROOT,
        runtime=resolve_device("cpu"),
        seeds=(260807,),
    )
    assert set(report["recall"]) == {"recall_keyed"}
    assert set(report["settings"]) == {"keyed"}
    assert set(report["two_stage"]) == {"stage_one_ch1_novel_mean", "true", "none", "negative"}
    sampling, two_stage = summary_rows({"legacy": report})
    assert len(sampling) == 1 and len(two_stage) == 3
