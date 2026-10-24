"""Iterative MaskGIT sampler with local Graph and exact-meter constraints."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from numpy.typing import NDArray

from gugakgen.data.records import TokenizedScore
from gugakgen.data.vocabulary import ScoreVocabulary
from gugakgen.device import RuntimeDevice
from gugakgen.models.denoiser import GugakDenoiser
from gugakgen.sampling.meter_decoder import ExactMeterDecoder, MeterDecodeResult
from gugakgen.structure.hwanip_relations import HwanipRelations
from gugakgen.structure.transition_graph import TransitionGraph


SCHEDULE_WINDOWS: Mapping[str, Mapping[str, tuple[float, float]]] = {
    "structure_first": {"family": (0.00, 0.50), "surface": (0.20, 1.00)},
    "joint": {"family": (0.00, 1.00), "surface": (0.00, 1.00)},
    "surface_first": {"family": (0.20, 1.00), "surface": (0.00, 0.50)},
}


@dataclass(frozen=True, slots=True)
class MaskGITConfig:
    num_steps: int = 12
    schedule: str = "surface_first"
    temperature_pitch_duration: float = 1.4
    temperature_family: float = 1.0
    graph_guidance_scale: float = 0.5
    hwanip_guidance_scale: float = 0.0
    corruption: float = 1.0
    forbid_unobserved_transitions: bool = True
    project_transition_path: bool = True
    transition_log_floor: float = -6.0
    copy_rejection_attempts: int = 4

    def __post_init__(self) -> None:
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if self.schedule not in SCHEDULE_WINDOWS:
            raise ValueError(f"Unknown schedule {self.schedule!r}")
        if self.temperature_pitch_duration < 0 or self.temperature_family < 0:
            raise ValueError("Sampling temperatures must be non-negative")
        if not 0.0 <= self.corruption <= 1.0:
            raise ValueError("corruption must be between zero and one")
        if self.copy_rejection_attempts <= 0:
            raise ValueError("copy_rejection_attempts must be positive")


@dataclass(frozen=True)
class SamplingResult:
    tokens: TokenizedScore
    raw_family: NDArray[np.int64]
    structure_edits: int
    trace: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)

    @property
    def raw_transition_changed(self) -> bool:
        return self.structure_edits > 0


def _remaining_fraction(step: int, num_steps: int, window: tuple[float, float]) -> float:
    low, high = window
    progress = (step + 1) / num_steps
    normalized = np.clip((progress - low) / max(high - low, 1e-9), 0.0, 1.0)
    return float(math.cos(math.pi * normalized / 2.0))


def _softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    maximum = np.max(logits, axis=-1, keepdims=True)
    safe_maximum = np.where(np.isfinite(maximum), maximum, 0.0)
    exponent = np.exp(np.where(np.isfinite(logits), logits - safe_maximum, -np.inf))
    totals = exponent.sum(axis=-1, keepdims=True)
    if bool((totals <= 0).any()):
        raise ValueError("At least one Logits row has no admissible value")
    return exponent / totals


def _sample_categorical(
    logits: NDArray[np.float64], temperature: float, rng: np.random.Generator
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    probabilities = _softmax(logits if temperature == 0 else logits / temperature)
    if temperature == 0:
        values = np.argmax(logits, axis=-1).astype(np.int64)
    else:
        flattened = probabilities.reshape(-1, probabilities.shape[-1])
        draws = [rng.choice(row.size, p=row) for row in flattened]
        values = np.asarray(draws, dtype=np.int64).reshape(logits.shape[:-1])
    confidence = np.take_along_axis(probabilities, values[..., None], axis=-1)[..., 0]
    return values, confidence


class MaskGITSampler:
    """Generate one score at a time from a piece-specific denoiser."""

    def __init__(
        self,
        *,
        model: GugakDenoiser,
        vocabulary: ScoreVocabulary,
        transition_graph: TransitionGraph,
        template: TokenizedScore,
        runtime: RuntimeDevice,
        duration_quantum_q: float = 0.25,
        hwanip_relations: HwanipRelations | None = None,
        allowed_families: NDArray[np.bool_] | None = None,
    ) -> None:
        if transition_graph.num_nodes != vocabulary.num_families:
            raise ValueError("Transition Graph and family vocabulary sizes differ")
        if model.config.num_families != vocabulary.num_families:
            raise ValueError("Model and family vocabulary sizes differ")
        if model.config.max_events_per_measure != template.max_events:
            raise ValueError("Model and tokenized template use different slot counts")
        if template.num_measures != len(transition_graph.original_path):
            raise ValueError("Template and original transition path lengths differ")
        if hwanip_relations is not None and hwanip_relations.num_measures != template.num_measures:
            raise ValueError("환입 relations and template lengths differ")

        self.model = model.to(runtime.device)
        self.model.eval()
        self.vocabulary = vocabulary
        self.graph = transition_graph
        self.template = template
        self.runtime = runtime
        self.hwanip_relations = hwanip_relations
        if allowed_families is None:
            allowed_families = np.ones(
                (template.num_measures, vocabulary.num_families), dtype=bool
            )
        self.allowed_families = np.asarray(allowed_families, dtype=bool)
        if self.allowed_families.shape != (template.num_measures, vocabulary.num_families):
            raise ValueError("allowed_families has an invalid shape")
        if bool((~self.allowed_families.any(axis=1)).any()):
            raise ValueError("Each measure must allow at least one family")

        self.meter_decoder = ExactMeterDecoder(
            vocabulary.duration_values,
            max_events=vocabulary.max_events_per_measure,
            pitch_pad_id=vocabulary.pitch_pad_id,
            pitch_mask_id=vocabulary.pitch_mask_id,
            duration_pad_id=vocabulary.duration_pad_id,
            duration_mask_id=vocabulary.duration_mask_id,
            allowed_pitch_ids=range(vocabulary.pitch_rest_id + 1),
            duration_quantum_q=duration_quantum_q,
        )
        self._original_measure_tokens = {
            self._measure_signature(template.pitch[index], template.duration[index])
            for index in range(template.num_measures)
        }

    @staticmethod
    def _measure_signature(
        pitch: Sequence[int], duration: Sequence[int]
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return tuple(int(value) for value in pitch), tuple(int(value) for value in duration)

    def _anchor_arrays(
        self,
        score_anchors: Sequence[int],
        family_anchors: Mapping[int, str | int],
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.int64], NDArray[np.bool_]]:
        measures = self.template.num_measures
        score_mask = np.zeros(measures, dtype=bool)
        family_mask = np.zeros(measures, dtype=bool)
        fixed_family = self.template.family.copy()
        for number in score_anchors:
            if not 1 <= int(number) <= measures:
                raise ValueError(f"Score anchor measure is out of range: {number}")
            score_mask[int(number) - 1] = True
            family_mask[int(number) - 1] = True
        for number, family in family_anchors.items():
            if not 1 <= int(number) <= measures:
                raise ValueError(f"Family anchor measure is out of range: {number}")
            position = int(number) - 1
            family_mask[position] = True
            fixed_family[position] = (
                self.vocabulary.encode_family(family) if isinstance(family, str) else int(family)
            )
        if ((fixed_family[family_mask] < 0) | (fixed_family[family_mask] >= self.vocabulary.num_families)).any():
            raise ValueError("A family anchor contains an invalid token")
        # Training marks a measure as anchored only when its surface and family are both
        # visible, so a family-only anchor must not set the model's anchor flag.
        model_anchor = score_mask.copy()
        return score_mask, family_mask, fixed_family, model_anchor

    def _condition_arrays(
        self, null_condition_measures: Sequence[int]
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        chapter = self.template.chapter.copy()
        position = self.template.position.copy()
        if not null_condition_measures:
            return chapter, position
        config = self.model.config
        for number in null_condition_measures:
            if not 1 <= int(number) <= self.template.num_measures:
                raise ValueError(f"Null-condition measure is out of range: {number}")
            chapter[int(number) - 1] = config.null_chapter_id
            position[int(number) - 1] = config.null_position_id
        return chapter, position

    @torch.inference_mode()
    def _predict(
        self,
        pitch: NDArray[np.int64],
        duration: NDArray[np.int64],
        family: NDArray[np.int64],
        model_anchor: NDArray[np.bool_],
        conditions: tuple[NDArray[np.int64], NDArray[np.int64]] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        device = self.runtime.device
        chapter, position = conditions or (self.template.chapter, self.template.position)
        corruption = torch.tensor(
            [[float(np.mean(pitch == self.vocabulary.pitch_mask_id)), float(np.mean(family == self.vocabulary.family_mask_id))]],
            dtype=torch.float32,
            device=device,
        )
        output = self.model(
            torch.as_tensor(pitch[None], dtype=torch.long, device=device),
            torch.as_tensor(duration[None], dtype=torch.long, device=device),
            torch.as_tensor(family[None], dtype=torch.long, device=device),
            corruption,
            torch.as_tensor(model_anchor, dtype=torch.bool, device=device),
            torch.as_tensor(chapter, dtype=torch.long, device=device),
            torch.as_tensor(self.template.meter, dtype=torch.long, device=device),
            torch.as_tensor(position, dtype=torch.long, device=device),
            torch.as_tensor(
                self.hwanip_relations.pairs if self.hwanip_relations is not None else (),
                dtype=torch.long,
                device=device,
            ).reshape(-1, 2),
        )
        pitch_logits = output.pitch_logits[0].float().cpu().numpy().astype(np.float64)
        duration_logits = output.duration_logits[0].float().cpu().numpy().astype(np.float64)
        family_logits = output.family_logits[0].float().cpu().numpy().astype(np.float64)
        pitch_logits[..., self.vocabulary.pitch_mask_id] = -np.inf
        duration_logits[..., self.vocabulary.duration_mask_id] = -np.inf
        return pitch_logits, duration_logits, family_logits

    def _decode_with_copy_rejection(
        self,
        pitch_logits: NDArray[np.float64],
        duration_logits: NDArray[np.float64],
        target_beats: float,
        config: MaskGITConfig,
        rng: np.random.Generator,
    ) -> MeterDecodeResult:
        result: MeterDecodeResult | None = None
        for _ in range(config.copy_rejection_attempts):
            result = self.meter_decoder.decode(
                pitch_logits,
                duration_logits,
                target_beats,
                rng=rng,
                temperature=config.temperature_pitch_duration,
            )
            if self._measure_signature(result.pitch_tokens, result.duration_tokens) not in self._original_measure_tokens:
                return result
        if result is None:
            raise RuntimeError("Exact-meter decoder produced no proposal")
        return result

    def _realize_changed_families(
        self,
        pitch: NDArray[np.int64],
        duration: NDArray[np.int64],
        family: NDArray[np.int64],
        changed: NDArray[np.int64],
        score_anchor: NDArray[np.bool_],
        model_anchor: NDArray[np.bool_],
        conditions: tuple[NDArray[np.int64], NDArray[np.int64]],
        config: MaskGITConfig,
        rng: np.random.Generator,
    ) -> None:
        editable = np.asarray([index for index in changed if not score_anchor[index]], dtype=np.int64)
        if len(editable) == 0:
            return
        pitch[editable] = self.vocabulary.pitch_mask_id
        duration[editable] = self.vocabulary.duration_mask_id
        pitch_logits, duration_logits, _ = self._predict(
            pitch, duration, family, model_anchor, conditions
        )
        for index in editable:
            proposal = self._decode_with_copy_rejection(
                pitch_logits[index],
                duration_logits[index],
                float(self.template.target_beats[index]),
                config,
                rng,
            )
            pitch[index] = proposal.pitch_tokens
            duration[index] = proposal.duration_tokens

    @torch.inference_mode()
    def sample(
        self,
        *,
        seed: int,
        config: MaskGITConfig | None = None,
        score_anchors: Sequence[int] = (),
        family_anchors: Mapping[int, str | int] | None = None,
        initial: TokenizedScore | None = None,
        record_trace: bool = True,
        null_condition_measures: Sequence[int] = (),
    ) -> SamplingResult:
        config = config or MaskGITConfig()
        rng = np.random.default_rng(seed)
        initial_tokens = initial or self.template
        if initial_tokens.pitch.shape != self.template.pitch.shape:
            raise ValueError("initial score shape differs from the model template")
        pitch = initial_tokens.pitch.copy()
        duration = initial_tokens.duration.copy()
        family = initial_tokens.family.copy()
        score_anchor, family_anchor, fixed_family, model_anchor = self._anchor_arrays(
            score_anchors, family_anchors or {}
        )
        family[family_anchor] = fixed_family[family_anchor]
        conditions = self._condition_arrays(null_condition_measures)

        free_surface = np.flatnonzero(~score_anchor)
        free_family = np.flatnonzero(~family_anchor)
        surface_count = int(round(config.corruption * len(free_surface)))
        family_count = int(round(config.corruption * len(free_family)))
        selected_surface = (
            rng.choice(free_surface, size=surface_count, replace=False)
            if surface_count
            else np.asarray([], dtype=np.int64)
        )
        selected_family = (
            rng.choice(free_family, size=family_count, replace=False)
            if family_count
            else np.asarray([], dtype=np.int64)
        )
        surface_mask = np.zeros(self.template.num_measures, dtype=bool)
        family_mask = np.zeros(self.template.num_measures, dtype=bool)
        surface_mask[selected_surface] = True
        family_mask[selected_family] = True
        pitch[surface_mask] = self.vocabulary.pitch_mask_id
        duration[surface_mask] = self.vocabulary.duration_mask_id
        family[family_mask] = self.vocabulary.family_mask_id
        initial_surface_count = int(surface_mask.sum())
        initial_family_count = int(family_mask.sum())
        windows = SCHEDULE_WINDOWS[config.schedule]
        trace: list[Mapping[str, Any]] = []

        for step in range(config.num_steps):
            if not surface_mask.any() and not family_mask.any():
                break
            pitch_logits, duration_logits, family_logits = self._predict(
                pitch, duration, family, model_anchor, conditions
            )

            if family_mask.any():
                guided = family_logits.copy()
                relation_bias = (
                    self.hwanip_relations.family_guidance_bias(
                        family,
                        num_families=self.vocabulary.num_families,
                        mask_id=self.vocabulary.family_mask_id,
                        scale=config.hwanip_guidance_scale,
                    )
                    if self.hwanip_relations is not None and config.hwanip_guidance_scale
                    else np.zeros_like(guided)
                )
                for index in np.flatnonzero(family_mask):
                    guided[index] = self.graph.guide_logits(
                        guided[index] + relation_bias[index],
                        int(index),
                        family,
                        mask_id=self.vocabulary.family_mask_id,
                        scale=config.graph_guidance_scale,
                        floor=config.transition_log_floor,
                        forbid_unobserved=config.forbid_unobserved_transitions,
                        allowed=self.allowed_families[index],
                    )
                proposed, confidence = _sample_categorical(
                    guided, config.temperature_family, rng
                )
                remaining = int(
                    round(
                        _remaining_fraction(step, config.num_steps, windows["family"])
                        * initial_family_count
                    )
                )
                commit_count = max(0, int(family_mask.sum()) - remaining)
                if step == config.num_steps - 1:
                    commit_count = int(family_mask.sum())
                candidates = np.flatnonzero(family_mask)
                order = candidates[np.argsort(-confidence[candidates])][:commit_count]
                family[order] = proposed[order]
                family_mask[order] = False

            if surface_mask.any():
                proposals: dict[int, MeterDecodeResult] = {}
                for index in np.flatnonzero(surface_mask):
                    proposals[int(index)] = self._decode_with_copy_rejection(
                        pitch_logits[index],
                        duration_logits[index],
                        float(self.template.target_beats[index]),
                        config,
                        rng,
                    )
                remaining = int(
                    round(
                        _remaining_fraction(step, config.num_steps, windows["surface"])
                        * initial_surface_count
                    )
                )
                commit_count = max(0, int(surface_mask.sum()) - remaining)
                if step == config.num_steps - 1:
                    commit_count = int(surface_mask.sum())
                order = sorted(proposals, key=lambda index: -proposals[index].confidence)[:commit_count]
                for index in order:
                    pitch[index] = proposals[index].pitch_tokens
                    duration[index] = proposals[index].duration_tokens
                    surface_mask[index] = False

            if record_trace:
                trace.append(
                    {
                        "step": step + 1,
                        "surface_masks_remaining": int(surface_mask.sum()),
                        "family_masks_remaining": int(family_mask.sum()),
                        "family_path": "".join(
                            "·" if value == self.vocabulary.family_mask_id else self.vocabulary.decode_family(int(value))
                            for value in family
                        ),
                    }
                )

        if surface_mask.any() or family_mask.any():
            raise RuntimeError("MaskGIT finished with unresolved MASK tokens")
        raw_family = family.copy()
        structure_edits = 0
        if config.project_transition_path:
            anchors_zero_based = {
                int(index): int(fixed_family[index]) for index in np.flatnonzero(family_anchor)
            }
            family, structure_edits = self.graph.project_path(
                family, anchors=anchors_zero_based, allowed=self.allowed_families
            )
            changed = np.flatnonzero(family != raw_family)
            self._realize_changed_families(
                pitch,
                duration,
                family,
                changed,
                score_anchor,
                model_anchor,
                conditions,
                config,
                rng,
            )

        for index in range(self.template.num_measures):
            if not self.meter_decoder.is_valid(
                pitch[index], duration[index], float(self.template.target_beats[index])
            ):
                raise RuntimeError(f"Generated measure {index + 1} violates exact meter")

        tokens = TokenizedScore(
            pitch=pitch,
            duration=duration,
            family=family,
            chapter=self.template.chapter.copy(),
            meter=self.template.meter.copy(),
            position=self.template.position.copy(),
            event_counts=np.sum(duration != self.vocabulary.duration_pad_id, axis=1),
            target_beats=self.template.target_beats.copy(),
        )
        return SamplingResult(tokens, raw_family, structure_edits, tuple(trace))
