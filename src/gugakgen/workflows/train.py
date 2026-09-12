"""Piece-specific MaskGIT training workflow used by scripts and notebooks."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor

from gugakgen.config import find_project_root, load_yaml, project_path, require_mapping
from gugakgen.device import RuntimeDevice, resolve_device
from gugakgen.models.checkpoint import (
    load_model_state,
    move_optimizer_state,
    save_checkpoint,
    sha256_file,
)
from gugakgen.models.denoiser import DenoiserOutput, GugakDenoiser
from gugakgen.reproducibility import set_global_seed
from gugakgen.workflows._common import PieceContext, load_piece_context, make_model_config


@dataclass(frozen=True)
class TrainingResult:
    checkpoint_path: Path
    checkpoint_sha256: str
    history_path: Path
    history: tuple[Mapping[str, float | int], ...]
    runtime: RuntimeDevice
    steps: int


def _sample_anchor(rng: np.random.Generator, measure_count: int, probability: float) -> NDArray[np.bool_]:
    anchor = np.zeros(measure_count, dtype=bool)
    if rng.random() >= probability:
        return anchor
    if rng.random() < 0.5:
        anchor[0] = True
        anchor[-1] = True
    else:
        length = int(rng.integers(1, min(5, measure_count + 1)))
        start = int(rng.integers(0, measure_count - length + 1))
        anchor[start : start + length] = True
    return anchor


def _corrupt(
    rng: np.random.Generator,
    *,
    rate: float,
    mode: str,
    anchor: NDArray[np.bool_],
    event_counts: NDArray[np.int64],
    slots: int,
    hwanip_pairs: Sequence[tuple[int, int]],
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    measures = len(event_counts)
    pitch_mask = np.zeros((measures, slots), dtype=bool)
    family_mask = np.zeros(measures, dtype=bool)
    real_events = np.arange(slots)[None, :] < event_counts[:, None]

    if mode == "hwanip_target":
        if not hwanip_pairs:
            raise ValueError("hwanip_target Masking requires at least one 환입 relation")
        targets = np.asarray(sorted({target for _, target in hwanip_pairs}), dtype=np.int64)
        pitch_mask[targets, :] = True
        family_mask[targets] = True
    elif mode == "event":
        pitch_mask = (rng.random((measures, slots)) < rate) & real_events
    elif mode == "structure":
        family_mask = rng.random(measures) < rate
    else:
        if mode == "span":
            length = int(rng.integers(2, min(9, measures + 1)))
            start = int(rng.integers(0, measures - length + 1))
            selected = np.arange(start, start + length)
        else:
            count = max(1, int(round(rate * measures)))
            selected = rng.choice(measures, size=min(count, measures), replace=False)
        pitch_mask[selected, :] = True
        if mode == "joint":
            family_mask[selected] = True

    pitch_mask[anchor, :] = False
    family_mask[anchor] = False
    return pitch_mask, family_mask


def _make_batch(
    rng: np.random.Generator,
    *,
    context: PieceContext,
    batch_size: int,
    training: Mapping[str, Any],
    device: torch.device,
    forced_modes: Sequence[str] | None = None,
) -> dict[str, Tensor]:
    tokens = context.tokens
    modes = tuple(str(value) for value in training["masking_modes"])
    probabilities = np.asarray(training["masking_probabilities"], dtype=np.float64)
    if len(modes) != len(probabilities) or not np.isclose(probabilities.sum(), 1.0):
        raise ValueError("masking_modes and masking_probabilities must align and sum to one")

    values: dict[str, list[NDArray[Any]]] = {
        "pitch": [],
        "duration": [],
        "family": [],
        "pitch_mask": [],
        "family_mask": [],
        "anchor": [],
        "corruption": [],
    }
    if forced_modes and any(mode not in modes for mode in forced_modes):
        raise ValueError("forced_modes contains a mode absent from masking_modes")
    for batch_index in range(batch_size):
        rate = float(
            rng.uniform(float(training["corruption_min"]), float(training["corruption_max"]))
        )
        anchor = _sample_anchor(
            rng, tokens.num_measures, float(training["anchor_probability"])
        )
        mode = (
            forced_modes[batch_index % len(forced_modes)]
            if forced_modes
            else str(rng.choice(modes, p=probabilities))
        )
        pitch_mask, family_mask = _corrupt(
            rng,
            rate=rate,
            mode=mode,
            anchor=anchor,
            event_counts=tokens.event_counts,
            slots=tokens.max_events,
            hwanip_pairs=context.hwanip_relations.pairs,
        )
        values["pitch"].append(
            np.where(pitch_mask, context.vocabulary.pitch_mask_id, tokens.pitch)
        )
        values["duration"].append(
            np.where(pitch_mask, context.vocabulary.duration_mask_id, tokens.duration)
        )
        values["family"].append(
            np.where(family_mask, context.vocabulary.family_mask_id, tokens.family)
        )
        values["pitch_mask"].append(pitch_mask)
        values["family_mask"].append(family_mask)
        values["anchor"].append(anchor)
        values["corruption"].append(
            np.asarray([pitch_mask.mean(), family_mask.mean()], dtype=np.float32)
        )

    result: dict[str, Tensor] = {}
    for name, arrays in values.items():
        stacked = np.stack(arrays)
        if name in {"pitch", "duration", "family"}:
            result[name] = torch.as_tensor(stacked, dtype=torch.long, device=device)
        elif name == "corruption":
            result[name] = torch.as_tensor(stacked, dtype=torch.float32, device=device)
        else:
            result[name] = torch.as_tensor(stacked, dtype=torch.bool, device=device)
    return result


def _masked_cross_entropy(logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    if bool(mask.any()):
        return F.cross_entropy(logits[mask], target[mask])
    return logits.sum() * 0.0


def _js_divergence(left_logits: Tensor, right_logits: Tensor) -> Tensor:
    """Jensen-Shannon divergence along the final vocabulary dimension."""

    left_log = F.log_softmax(left_logits, dim=-1)
    right_log = F.log_softmax(right_logits, dim=-1)
    left = left_log.exp()
    right = right_log.exp()
    mixture = 0.5 * (left + right)
    mixture_log = torch.log(mixture.clamp(min=1e-12))
    return 0.5 * (
        (left * (left_log - mixture_log)).sum(dim=-1)
        + (right * (right_log - mixture_log)).sum(dim=-1)
    )


def _compute_losses(
    output: DenoiserOutput,
    batch: Mapping[str, Tensor],
    *,
    context: PieceContext,
    loss_config: Mapping[str, Any],
    device: torch.device,
) -> tuple[Tensor, dict[str, float]]:
    pitch_logits, duration_logits, family_logits = output
    batch_size = pitch_logits.shape[0]
    pitch_target = torch.as_tensor(context.tokens.pitch, dtype=torch.long, device=device)
    duration_target = torch.as_tensor(context.tokens.duration, dtype=torch.long, device=device)
    family_target = torch.as_tensor(context.tokens.family, dtype=torch.long, device=device)
    pitch_target = pitch_target.unsqueeze(0).expand(batch_size, -1, -1)
    duration_target = duration_target.unsqueeze(0).expand(batch_size, -1, -1)
    family_target = family_target.unsqueeze(0).expand(batch_size, -1)
    pitch_mask = batch["pitch_mask"]
    family_mask = batch["family_mask"]

    pitch_loss = _masked_cross_entropy(pitch_logits, pitch_target, pitch_mask)
    duration_loss = _masked_cross_entropy(duration_logits, duration_target, pitch_mask)
    family_loss = _masked_cross_entropy(family_logits, family_target, family_mask)

    transition = torch.as_tensor(
        context.graph.transition_matrix, dtype=torch.float32, device=device
    )
    log_transition = torch.log(transition + 1e-12).clamp(
        min=float(loss_config["transition_log_floor"])
    )
    family_probability = F.softmax(family_logits, dim=-1)
    target_one_hot = F.one_hot(family_target, context.vocabulary.num_families).float()
    mixed_family = torch.where(family_mask.unsqueeze(-1), family_probability, target_one_hot)
    affected_edges = (family_mask[:, :-1] | family_mask[:, 1:]).float()
    graph_nll = -torch.einsum(
        "btu,btv,uv->bt", mixed_family[:, :-1], mixed_family[:, 1:], log_transition
    )
    graph_loss = (graph_nll * affected_edges).sum() / affected_edges.sum().clamp(min=1.0)

    duration_values = torch.zeros(
        context.vocabulary.num_duration_tokens, dtype=torch.float32, device=device
    )
    duration_values[: context.vocabulary.duration_pad_id] = torch.tensor(
        [float(value) for value in context.vocabulary.duration_values], device=device
    )
    meter_logits = duration_logits.clone()
    meter_logits[..., context.vocabulary.duration_mask_id] = -torch.inf
    expected_duration = (F.softmax(meter_logits, dim=-1) * duration_values).sum(dim=-1)
    target_duration = duration_values[duration_target]
    mixed_duration = torch.where(pitch_mask, expected_duration, target_duration)
    target_beats = torch.as_tensor(
        context.tokens.target_beats, dtype=torch.float32, device=device
    )
    meter_loss = (mixed_duration.sum(dim=-1) - target_beats.unsqueeze(0)).abs().mean()

    pairs = context.hwanip_relations.pairs
    zero = pitch_logits.sum() * 0.0
    hwanip_family_loss = zero
    hwanip_surface_loss = zero
    if pairs:
        source = torch.as_tensor([pair[0] for pair in pairs], dtype=torch.long, device=device)
        target = torch.as_tensor([pair[1] for pair in pairs], dtype=torch.long, device=device)
        family_affected = family_mask[:, source] | family_mask[:, target]
        family_js = _js_divergence(family_logits[:, source], family_logits[:, target])
        if bool(family_affected.any()):
            hwanip_family_loss = (
                family_js * family_affected.float()
            ).sum() / family_affected.sum().clamp(min=1)

        surface_affected = pitch_mask[:, source].any(dim=-1) | pitch_mask[:, target].any(dim=-1)
        pitch_js = _js_divergence(pitch_logits[:, source], pitch_logits[:, target]).mean(dim=-1)
        duration_js = _js_divergence(
            duration_logits[:, source], duration_logits[:, target]
        ).mean(dim=-1)
        surface_js = 0.5 * (pitch_js + duration_js)
        if bool(surface_affected.any()):
            hwanip_surface_loss = (
                surface_js * surface_affected.float()
            ).sum() / surface_affected.sum().clamp(min=1)

    total = (
        float(loss_config["pitch"]) * pitch_loss
        + float(loss_config["duration"]) * duration_loss
        + float(loss_config["family"]) * family_loss
        + float(loss_config["graph"]) * graph_loss
        + float(loss_config["meter"]) * meter_loss
        + float(loss_config.get("hwanip_family", 0.0)) * hwanip_family_loss
        + float(loss_config.get("hwanip_surface", 0.0)) * hwanip_surface_loss
    )
    components = {
        "total": float(total.detach().cpu()),
        "pitch": float(pitch_loss.detach().cpu()),
        "duration": float(duration_loss.detach().cpu()),
        "family": float(family_loss.detach().cpu()),
        "graph": float(graph_loss.detach().cpu()),
        "meter": float(meter_loss.detach().cpu()),
        "hwanip_family": float(hwanip_family_loss.detach().cpu()),
        "hwanip_surface": float(hwanip_surface_loss.detach().cpu()),
    }
    return total, components


def _parameter_groups(model: GugakDenoiser, weight_decay: float) -> list[dict[str, Any]]:
    decay: list[Tensor] = []
    no_decay: list[Tensor] = []
    for name, parameter in model.named_parameters():
        target = no_decay if parameter.ndim <= 1 or "emb" in name or "role_mask" in name else decay
        target.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _cosine_schedule(step: int, *, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _grad_scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train_from_config(
    config_path: str | Path = "configs/train_maskgit.yaml",
    *,
    project_root: str | Path | None = None,
    run_mode: str = "smoke",
    runtime: RuntimeDevice | None = None,
    resume_checkpoint: str | Path | None = None,
) -> TrainingResult:
    """Train the piece-specific denoiser and publish a portable Checkpoint."""

    if run_mode not in {"smoke", "full"}:
        raise ValueError("run_mode must be 'smoke' or 'full'")
    root = find_project_root(project_root)
    resolved_config_path = project_path(root, config_path)
    config = load_yaml(resolved_config_path)
    runtime_config = require_mapping(config.get("runtime"), "runtime")
    model_section = require_mapping(config.get("model"), "model")
    training = require_mapping(config.get("training"), "training")
    loss_config = require_mapping(config.get("loss"), "loss")
    artifacts = require_mapping(config.get("artifacts"), "artifacts")
    runtime = runtime or resolve_device(str(runtime_config.get("device", "auto")))
    if (
        run_mode == "full"
        and bool(runtime_config.get("require_cuda_for_full", True))
        and runtime.backend != "cuda"
    ):
        raise RuntimeError(
            "Full training requires CUDA by configuration. Use Colab/Linux CUDA or set "
            "runtime.require_cuda_for_full=false explicitly."
        )

    seed = int(config["seed"])
    set_global_seed(seed)
    context = load_piece_context(
        config["piece_config"], project_root=root, model_section=model_section
    )
    model_config = make_model_config(model_section, context)
    model = GugakDenoiser(
        model_config,
        torch.as_tensor(context.graph.transition_matrix, dtype=torch.float32),
    ).to(runtime.device)
    optimizer = torch.optim.AdamW(
        _parameter_groups(model, float(training["weight_decay"])),
        lr=float(training["learning_rate"]),
    )
    total_steps = int(training["smoke_steps"] if run_mode == "smoke" else training["steps"])
    batch_size = int(
        training["smoke_batch_size"] if run_mode == "smoke" else training["batch_size"]
    )
    warmup = min(int(training["warmup_steps"]), max(0, total_steps - 1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _cosine_schedule(step, warmup=warmup, total=total_steps)
    )
    start_step = 0
    if resume_checkpoint is not None:
        loaded = load_model_state(model, project_path(root, resume_checkpoint), strict=True)
        if "optimizer_state_dict" in loaded.payload:
            optimizer.load_state_dict(loaded.payload["optimizer_state_dict"])
            move_optimizer_state(optimizer, runtime.device)
        if "scheduler_state_dict" in loaded.payload:
            scheduler.load_state_dict(loaded.payload["scheduler_state_dict"])
        start_step = loaded.step
        if start_step >= total_steps:
            raise ValueError("Resume Checkpoint step is not smaller than the requested total steps")

    amp_setting = runtime_config.get("amp", "auto")
    use_amp = runtime.amp_enabled if amp_setting == "auto" else bool(amp_setting)
    if use_amp and runtime.backend != "cuda":
        raise ValueError("AMP is supported only for CUDA in this project")
    scaler = _grad_scaler(use_amp)
    rng = np.random.default_rng(seed)
    evaluation_rng = np.random.default_rng(seed + 1)
    history: list[Mapping[str, float | int]] = []
    evaluation_interval = max(1, int(training["evaluation_interval"]))
    checkpoint_interval = max(1, int(training["checkpoint_interval"]))
    gradient_clip = float(training["gradient_clip_norm"])
    configured_checkpoint_name = Path(str(artifacts["checkpoint_name"]))
    configured_history_name = Path(str(artifacts["history_name"]))
    checkpoint_name = (
        configured_checkpoint_name.name
        if run_mode == "full"
        else f"{configured_checkpoint_name.stem}_smoke{configured_checkpoint_name.suffix}"
    )
    history_name = (
        configured_history_name.name
        if run_mode == "full"
        else f"{configured_history_name.stem}_smoke{configured_history_name.suffix}"
    )

    scratch_base = Path(
        os.environ.get(str(runtime_config.get("scratch_env", "GUGAKGEN_SCRATCH")), tempfile.gettempdir())
    ).expanduser()
    scratch_dir = scratch_base / "gugakgen" / str(config["experiment_name"])
    scratch_dir.mkdir(parents=True, exist_ok=True)
    scratch_checkpoint = scratch_dir / checkpoint_name
    started = time.monotonic()

    def checkpoint_metadata(step: int) -> dict[str, Any]:
        return {
            "experiment_name": config["experiment_name"],
            "seed": seed,
            "run_mode": run_mode,
            "runtime": runtime.as_dict(),
            "piece_config": str(Path(config["piece_config"])),
            "train_config": str(resolved_config_path.relative_to(root)),
            "position_array": context.tokens.position.tolist(),
            "step": step,
        }

    model.train()
    hwanip_pairs = torch.as_tensor(
        context.hwanip_relations.pairs, dtype=torch.long, device=runtime.device
    ).reshape(-1, 2)
    for step in range(start_step + 1, total_steps + 1):
        batch = _make_batch(
            rng,
            context=context,
            batch_size=batch_size,
            training=training,
            device=runtime.device,
        )
        optimizer.zero_grad(set_to_none=True)
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=runtime.dtype) if use_amp else nullcontext()
        )
        with autocast_context:
            output = model(
                batch["pitch"],
                batch["duration"],
                batch["family"],
                batch["corruption"],
                batch["anchor"],
                torch.as_tensor(context.tokens.chapter, dtype=torch.long, device=runtime.device),
                torch.as_tensor(context.tokens.meter, dtype=torch.long, device=runtime.device),
                torch.as_tensor(context.tokens.position, dtype=torch.long, device=runtime.device),
                hwanip_pairs,
            )
            loss, parts = _compute_losses(
                output,
                batch,
                context=context,
                loss_config=loss_config,
                device=runtime.device,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip))
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        should_evaluate = step == 1 or step == total_steps or step % evaluation_interval == 0
        if should_evaluate:
            model.eval()
            with torch.inference_mode():
                eval_batch = _make_batch(
                    evaluation_rng,
                    context=context,
                    batch_size=max(batch_size, len(training["masking_modes"])),
                    training=training,
                    device=runtime.device,
                    forced_modes=tuple(str(mode) for mode in training["masking_modes"]),
                )
                eval_output = model(
                    eval_batch["pitch"],
                    eval_batch["duration"],
                    eval_batch["family"],
                    eval_batch["corruption"],
                    eval_batch["anchor"],
                    torch.as_tensor(context.tokens.chapter, dtype=torch.long, device=runtime.device),
                    torch.as_tensor(context.tokens.meter, dtype=torch.long, device=runtime.device),
                    torch.as_tensor(context.tokens.position, dtype=torch.long, device=runtime.device),
                    hwanip_pairs,
                )
                _, eval_parts = _compute_losses(
                    eval_output,
                    eval_batch,
                    context=context,
                    loss_config=loss_config,
                    device=runtime.device,
                )
            record: dict[str, float | int] = {
                "step": step,
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "gradient_norm": gradient_norm,
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            record.update({f"train_{key}": value for key, value in parts.items()})
            record.update({f"eval_{key}": value for key, value in eval_parts.items()})
            history.append(record)
            model.train()

        if step % checkpoint_interval == 0 or step == total_steps:
            save_checkpoint(
                scratch_checkpoint,
                model,
                step=step,
                model_config=model_config.to_dict(),
                vocabulary=context.vocabulary.to_dict(),
                optimizer=optimizer,
                scheduler=scheduler,
                metadata=checkpoint_metadata(step),
            )

    output_dir = project_path(root, artifacts["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / checkpoint_name
    temporary_destination = output_dir / f".{destination.name}.copying"
    shutil.copy2(scratch_checkpoint, temporary_destination)
    if sha256_file(scratch_checkpoint) != sha256_file(temporary_destination):
        temporary_destination.unlink(missing_ok=True)
        raise OSError("Checkpoint checksum changed while copying from scratch storage")
    os.replace(temporary_destination, destination)

    history_path = output_dir / history_name
    _write_json_atomic(
        history_path,
        {
            "schema_version": 1,
            "experiment_name": config["experiment_name"],
            "seed": seed,
            "run_mode": run_mode,
            "runtime": runtime.as_dict(),
            "steps": total_steps,
            "history": history,
            "checkpoint": str(destination.relative_to(root)),
            "checkpoint_sha256": sha256_file(destination),
        },
    )
    return TrainingResult(
        destination,
        sha256_file(destination),
        history_path,
        tuple(history),
        runtime,
        total_steps,
    )
