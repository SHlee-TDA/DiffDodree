"""Baseline vs condition-dropout comparison for the 제1장/제4장 memorization study.

The two training configurations differ only in whether chapter/position conditions are
randomly hidden during training. Evaluation reports how strongly each checkpoint recalls
original measures from their address, and whether a newly written 제1장 returns in 제4장,
both as exact repetition and as DTW distance (``gugakgen.structure.measure_distance``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from gugakgen.config import load_yaml, project_path, require_mapping
from gugakgen.data.records import TokenizedScore
from gugakgen.device import RuntimeDevice
from gugakgen.structure.hwanip_relations import HwanipRelation, HwanipRelations
from gugakgen.structure.measure_distance import Event, passage_distance, score_events
from gugakgen.workflows.generate import load_generation_bundle
from gugakgen.workflows.train import memorization_recall

BASE_CONFIGS: Mapping[str, str] = {
    "baseline": "configs/train_maskgit.yaml",
    "conddrop": "configs/train_maskgit_conddrop.yaml",
}
CHAPTER_ONE = (1, 2, 3, 4)
CHAPTER_FOUR = (25, 26, 27, 28)


def write_runtime_configs(
    project_root: str | Path,
    *,
    run_tag: str,
    steps: int | None = None,
    output_dir: str | Path = "artifacts/runtime_configs",
) -> dict[str, Path]:
    """Write one runtime config per model whose checkpoint names carry ``run_tag``.

    Refuses to proceed if a config or any checkpoint/history file with those names
    already exists, so a rerun never overwrites earlier results.
    """

    root = Path(project_root)
    if not run_tag or any(character in run_tag for character in "/\\ "):
        raise ValueError("run_tag must be a non-empty name without spaces or slashes")
    planned: dict[str, tuple[Path, dict[str, Any]]] = {}
    for name, base in BASE_CONFIGS.items():
        config = load_yaml(project_path(root, base))
        experiment = f"{config['experiment_name']}_{run_tag}"
        config["experiment_name"] = experiment
        artifacts = require_mapping(config.get("artifacts"), "artifacts")
        artifacts["checkpoint_name"] = f"{experiment}.pt"
        artifacts["history_name"] = f"{experiment}_history.json"
        config["artifacts"] = artifacts
        if steps is not None:
            config["training"]["steps"] = int(steps)
        checkpoint_dir = project_path(root, artifacts["output_dir"])
        existing = [
            path
            for path in (
                checkpoint_dir / f"{experiment}.pt",
                checkpoint_dir / f"{experiment}_smoke.pt",
                checkpoint_dir / f"{experiment}_history.json",
                checkpoint_dir / f"{experiment}_history_smoke.json",
            )
            if path.exists()
        ]
        if existing:
            raise FileExistsError(f"Results for run_tag {run_tag!r} already exist: {existing}")
        planned[name] = (project_path(root, output_dir) / f"{experiment}.yaml", config)

    for path, _ in planned.values():
        if path.exists():
            raise FileExistsError(f"Runtime config already exists: {path}")
    written: dict[str, Path] = {}
    for name, (path, config) in planned.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        written[name] = path
    return written


def _signatures(tokens: TokenizedScore) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    return [
        (tuple(int(x) for x in tokens.pitch[index]), tuple(int(x) for x in tokens.duration[index]))
        for index in range(tokens.num_measures)
    ]


def _jensen_shannon(left: np.ndarray, right: np.ndarray) -> float:
    left = left / left.sum()
    right = right / right.sum()
    mixture = 0.5 * (left + right)

    def divergence(p: np.ndarray, q: np.ndarray) -> float:
        support = p > 0
        return float(np.sum(p[support] * np.log(p[support] / q[support])))

    return 0.5 * divergence(left, mixture) + 0.5 * divergence(right, mixture)


def _comparison_windows(tokens: TokenizedScore) -> tuple[int, ...]:
    """Start measures of 4-measure passages in 제4장's meter that overlap neither 제1장 nor 제4장."""

    length = len(CHAPTER_FOUR)
    meter = int(tokens.meter[CHAPTER_FOUR[0] - 1])
    excluded = set(CHAPTER_ONE) | set(CHAPTER_FOUR)
    starts = []
    for start in range(1, tokens.num_measures - length + 2):
        span = range(start, start + length)
        if excluded & set(span):
            continue
        if all(int(tokens.meter[number - 1]) == meter for number in span):
            starts.append(start)
    return tuple(starts)


def _return_distances(
    events: Sequence[Sequence[Event]], windows: Sequence[int]
) -> dict[str, float]:
    """DTW distance from 제4장 to 제1장, compared with other passages of the same piece."""

    def passage(start: int) -> list[Sequence[Event]]:
        return [events[number - 1] for number in range(start, start + len(CHAPTER_FOUR))]

    fourth = passage(CHAPTER_FOUR[0])
    to_first = passage_distance(fourth, passage(CHAPTER_ONE[0]), relative_pitch=False)
    to_others = np.asarray(
        [passage_distance(fourth, passage(start), relative_pitch=False) for start in windows]
    )
    return {
        "dtw_abs": to_first,
        "dtw_rel": passage_distance(fourth, passage(CHAPTER_ONE[0]), relative_pitch=True),
        "dtw_abs_others": float(to_others.mean()),
        "windows_farther": float((to_others > to_first).mean()),
        "ch1_nearest": float(to_first < to_others.min()),
    }


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    config_path: str | Path,
    *,
    project_root: str | Path,
    runtime: RuntimeDevice,
    seeds: Sequence[int],
) -> dict[str, Any]:
    """Measure memorization, 제1장 novelty and 제1장→제4장 return for one checkpoint."""

    bundle = load_generation_bundle(
        checkpoint_path, config_path=config_path, project_root=project_root, runtime=runtime
    )
    sampler, context = bundle.sampler, bundle.context
    vocabulary = context.vocabulary
    model_config = bundle.model.config
    null_ids = (
        (model_config.null_chapter_id, model_config.null_position_id)
        if model_config.null_conditions
        else None
    )
    pairs = torch.as_tensor(context.hwanip_relations.pairs, dtype=torch.long).reshape(-1, 2)
    report: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": bundle.checkpoint.sha256,
        "recall": memorization_recall(
            bundle.model, context, pairs.to(runtime.device), device=runtime.device, null_ids=null_ids
        ),
        "settings": {},
        "two_stage": {},
    }

    original = _signatures(context.tokens)
    original_set = set(original)
    first = [number - 1 for number in CHAPTER_ONE]
    fourth = [number - 1 for number in CHAPTER_FOUR]
    playable = vocabulary.pitch_rest_id + 1
    original_pitch = np.bincount(
        context.tokens.pitch[context.tokens.pitch < playable], minlength=playable
    ).astype(float)

    settings: dict[str, tuple[int, ...]] = {"keyed": ()}
    if null_ids is not None:
        settings["null_ch1"] = CHAPTER_ONE
        settings["null_ch1_ch4"] = CHAPTER_ONE + CHAPTER_FOUR
        settings["null_all"] = tuple(range(1, context.tokens.num_measures + 1))
    for setting, null_measures in settings.items():
        rows = []
        for seed in seeds:
            tokens = sampler.sample(
                seed=int(seed),
                config=bundle.sampling_config,
                family_anchors=bundle.family_anchors,
                null_condition_measures=null_measures,
                record_trace=False,
            ).tokens
            generated = _signatures(tokens)
            pitch = np.bincount(tokens.pitch[tokens.pitch < playable], minlength=playable)
            rows.append(
                {
                    "ch1_original": sum(generated[i] == original[i] for i in first),
                    "ch1_novel": sum(generated[i] not in original_set for i in first),
                    "ch1_ch4_equal": sum(generated[a] == generated[b] for a, b in zip(first, fourth, strict=True)),
                    "novel_returned": sum(
                        generated[a] == generated[b] and generated[a] not in original_set
                        for a, b in zip(first, fourth, strict=True)
                    ),
                    "same_position_copy": float(
                        np.mean([g == o for g, o in zip(generated, original, strict=True)])
                    ),
                    "any_original": float(np.mean([g in original_set for g in generated])),
                    "transition_validity": float(context.graph.transition_validity(tokens.family.tolist())),
                    "pitch_jsd": _jensen_shannon(pitch + 1e-9, original_pitch + 1e-9),
                }
            )
        report["settings"][setting] = {
            "null_condition_measures": list(null_measures),
            "samples": len(rows),
            "ch1_all_original": sum(row["ch1_original"] == 4 for row in rows),
            "ch1_novel_mean": float(np.mean([row["ch1_novel"] for row in rows])),
            "ch1_ch4_all_equal": sum(row["ch1_ch4_equal"] == 4 for row in rows),
            "ch1_ch4_equal_mean": float(np.mean([row["ch1_ch4_equal"] for row in rows])),
            "novel_returned_total": sum(row["novel_returned"] for row in rows),
            "same_position_copy_rate": float(np.mean([row["same_position_copy"] for row in rows])),
            "any_original_rate": float(np.mean([row["any_original"] for row in rows])),
            "transition_validity": float(np.mean([row["transition_validity"] for row in rows])),
            "pitch_jsd": float(np.mean([row["pitch_jsd"] for row in rows])),
        }

    # Tutorial 05 protocol: write a new 제1장 with 제2–7장 fixed, then regenerate only 제4장.
    relations: dict[str, HwanipRelations | None] = {
        "true": context.hwanip_relations,
        "none": None,
        "negative": HwanipRelations(
            relations=(
                HwanipRelation(
                    relation_id="negative_chapter_2_to_chapter_4",
                    source_measures=(5, 8),
                    target_measures=(25, 28),
                    relation_type="return_of",
                ),
            ),
            num_measures=context.tokens.num_measures,
        ),
    }
    # Hide the addresses of both 제1장 and 제4장 in stage two: with 제1장's chapter/position
    # visible, the 환입 message carries that address and 제4장 recalls the original 제1장.
    stage_one_null = CHAPTER_ONE if null_ids is not None else ()
    stage_two_null = CHAPTER_ONE + CHAPTER_FOUR if null_ids is not None else ()
    stage_two_fixed = tuple(
        number for number in range(1, context.tokens.num_measures + 1) if number not in CHAPTER_FOUR
    )
    windows = _comparison_windows(context.tokens)
    outcomes: dict[str, list[tuple[bool, bool, bool]]] = {name: [] for name in relations}
    distances: dict[str, list[dict[str, float]]] = {name: [] for name in relations}
    stage_one_novel = []
    for seed in seeds:
        stage_one = sampler.sample(
            seed=int(seed),
            config=bundle.sampling_config,
            score_anchors=tuple(range(CHAPTER_ONE[-1] + 1, context.tokens.num_measures + 1)),
            family_anchors=bundle.family_anchors,
            null_condition_measures=stage_one_null,
            record_trace=False,
        ).tokens
        new_first = _signatures(stage_one)
        stage_one_novel.append(sum(new_first[i] not in original_set for i in first))
        for name, relation in relations.items():
            saved = sampler.hwanip_relations
            sampler.hwanip_relations = relation
            try:
                stage_two = sampler.sample(
                    seed=int(seed) + 100,
                    config=bundle.sampling_config,
                    score_anchors=stage_two_fixed,
                    family_anchors=bundle.family_anchors,
                    initial=stage_one,
                    null_condition_measures=stage_two_null,
                    record_trace=False,
                ).tokens
            finally:
                sampler.hwanip_relations = saved
            generated = _signatures(stage_two)
            for a, b in zip(first, fourth, strict=True):
                outcomes[name].append(
                    (
                        new_first[a] not in original_set,
                        generated[b] == new_first[a],
                        generated[b] == original[b],
                    )
                )
            distances[name].append(
                _return_distances(score_events(stage_two, vocabulary), windows)
            )

    report["two_stage"]["stage_one_ch1_novel_mean"] = float(np.mean(stage_one_novel))
    report["two_stage"]["comparison_window_starts"] = list(windows)
    baseline = [row["dtw_abs"] for row in distances["none"]]
    for name, values in outcomes.items():
        array = np.asarray(values, dtype=bool)
        novel, copied = array[array[:, 0]], array[~array[:, 0]]
        rows = distances[name]
        absolute = [row["dtw_abs"] for row in rows]
        report["two_stage"][name] = {
            "novel_source_measures": int(len(novel)),
            "return_rate_novel": float(novel[:, 1].mean()) if len(novel) else None,
            "copied_source_measures": int(len(copied)),
            "return_rate_copied": float(copied[:, 1].mean()) if len(copied) else None,
            "ch4_original_rate": float(array[:, 2].mean()),
            "dtw_abs_ch4_ch1": float(np.mean(absolute)),
            "dtw_rel_ch4_ch1": float(np.mean([row["dtw_rel"] for row in rows])),
            "dtw_abs_ch4_other_windows": float(np.mean([row["dtw_abs_others"] for row in rows])),
            "windows_farther_than_ch1": float(np.mean([row["windows_farther"] for row in rows])),
            "ch1_nearest_seeds": int(sum(row["ch1_nearest"] for row in rows)),
            "closer_than_none_seeds": (
                None
                if name == "none"
                else int(sum(d < b for d, b in zip(absolute, baseline, strict=True)))
            ),
            "seeds": len(rows),
            "dtw_abs_per_seed": absolute,
        }
    return report


def summary_rows(reports: Mapping[str, Mapping[str, Any]]) -> tuple[list[dict], list[dict]]:
    """Flatten reports into rows for a sampling table and a two-stage table."""

    sampling, two_stage = [], []
    for model, report in reports.items():
        for setting, values in report["settings"].items():
            sampling.append(
                {
                    "model": model,
                    "setting": setting,
                    **{key: value for key, value in values.items() if key != "null_condition_measures"},
                }
            )
        for relation, values in report["two_stage"].items():
            if isinstance(values, Mapping):
                two_stage.append(
                    {
                        "model": model,
                        "relation": relation,
                        **{key: value for key, value in values.items() if key != "dtw_abs_per_seed"},
                    }
                )
    return sampling, two_stage
