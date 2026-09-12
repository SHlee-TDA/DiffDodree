"""Exact dynamic-programming decoder for measure-level meter constraints."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

import numpy as np
import torch
from numpy.typing import NDArray

from gugakgen.data.records import as_fraction

NEGATIVE_INFINITY = -np.inf


def _as_numpy(value: NDArray[np.floating] | torch.Tensor) -> NDArray[np.float64]:
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _logsumexp(values: NDArray[np.float64]) -> float:
    if values.size == 0:
        return NEGATIVE_INFINITY
    maximum = float(np.max(values))
    if not np.isfinite(maximum):
        return NEGATIVE_INFINITY
    return maximum + float(np.log(np.exp(values - maximum).sum()))


def _log_softmax(values: NDArray[np.float64]) -> NDArray[np.float64]:
    maximum = np.max(values, axis=-1, keepdims=True)
    shifted = values - maximum
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def _sample_log_weights(values: NDArray[np.float64], rng: np.random.Generator) -> int:
    normalizer = _logsumexp(values)
    if not np.isfinite(normalizer):
        raise ValueError("Cannot sample from an empty or non-finite distribution")
    probabilities = np.exp(values - normalizer)
    probabilities /= probabilities.sum()
    return int(rng.choice(len(probabilities), p=probabilities))


@dataclass(frozen=True, slots=True)
class MeterDecodeResult:
    pitch_tokens: NDArray[np.int64]
    duration_tokens: NDArray[np.int64]
    event_count: int
    log_weight: float
    confidence: float


class ExactMeterDecoder:
    """Sample or maximize the joint token distribution under exact meter.

    Valid measures contain one or more event slots followed by a contiguous PAD
    suffix, and active durations must sum exactly to the target number of beats.
    """

    def __init__(
        self,
        durations_q: Sequence[int | float | str | Fraction],
        *,
        max_events: int,
        pitch_pad_id: int,
        pitch_mask_id: int,
        duration_pad_id: int,
        duration_mask_id: int,
        allowed_pitch_ids: Sequence[int],
        duration_quantum_q: int | float | str | Fraction = Fraction(1, 4),
    ) -> None:
        if max_events <= 0:
            raise ValueError("max_events must be positive")
        quantum = as_fraction(duration_quantum_q)
        durations = tuple(as_fraction(value) for value in durations_q)
        weights: list[int] = []
        for duration in durations:
            units = duration / quantum
            if units.denominator != 1:
                raise ValueError(f"Duration {duration} is not aligned to quantum {quantum}")
            weights.append(int(units))
        allowed = np.asarray(tuple(int(value) for value in allowed_pitch_ids), dtype=np.int64)
        if len(allowed) == 0 or len(set(allowed.tolist())) != len(allowed):
            raise ValueError("allowed_pitch_ids must contain unique values")

        self.durations_q = durations
        self.duration_weights = np.asarray(weights, dtype=np.int64)
        self.quantum = quantum
        self.max_events = int(max_events)
        self.pitch_pad_id = int(pitch_pad_id)
        self.pitch_mask_id = int(pitch_mask_id)
        self.duration_pad_id = int(duration_pad_id)
        self.duration_mask_id = int(duration_mask_id)
        self.allowed_pitch_ids = allowed

    def target_units(self, target_beats: int | float | str | Fraction) -> int:
        units = as_fraction(target_beats) / self.quantum
        if units.denominator != 1:
            raise ValueError(f"Target meter {target_beats} is not aligned to quantum {self.quantum}")
        return int(units)

    def decode(
        self,
        pitch_logits: NDArray[np.floating] | torch.Tensor,
        duration_logits: NDArray[np.floating] | torch.Tensor,
        target_beats: int | float | str | Fraction,
        *,
        rng: np.random.Generator,
        temperature: float = 1.0,
    ) -> MeterDecodeResult:
        pitch = _as_numpy(pitch_logits)
        duration = _as_numpy(duration_logits)
        if pitch.ndim != 2 or duration.ndim != 2 or pitch.shape[0] != self.max_events:
            raise ValueError("Logits must have shapes (max_events, pitch/duration vocabulary)")
        if duration.shape != (self.max_events, self.duration_mask_id + 1):
            raise ValueError(
                f"duration_logits must have shape {(self.max_events, self.duration_mask_id + 1)}"
            )
        if pitch.shape[1] <= max(self.pitch_pad_id, self.pitch_mask_id, int(self.allowed_pitch_ids.max())):
            raise ValueError("pitch_logits vocabulary is smaller than configured token ids")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")

        # MASK is never a decoded value.
        pitch = pitch.copy()
        duration = duration.copy()
        pitch[:, self.pitch_mask_id] = NEGATIVE_INFINITY
        duration[:, self.duration_mask_id] = NEGATIVE_INFINITY
        target = self.target_units(target_beats)
        if temperature == 0:
            return self._decode_map(pitch, duration, target)
        return self._decode_sample(pitch / temperature, duration / temperature, target, rng)

    def _weights(
        self, pitch_logits: NDArray[np.float64], duration_logits: NDArray[np.float64]
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        pitch_log_probability = _log_softmax(pitch_logits)
        duration_log_probability = _log_softmax(duration_logits)
        event_weight = np.asarray(
            [_logsumexp(row[self.allowed_pitch_ids]) for row in pitch_log_probability],
            dtype=np.float64,
        )
        pad_weight = (
            pitch_log_probability[:, self.pitch_pad_id]
            + duration_log_probability[:, self.duration_pad_id]
        )
        return pitch_log_probability, duration_log_probability, event_weight, pad_weight

    def _forward_sum(
        self,
        duration_log_probability: NDArray[np.float64],
        event_weight: NDArray[np.float64],
        target: int,
    ) -> NDArray[np.float64]:
        table = np.full((self.max_events + 1, target + 1), NEGATIVE_INFINITY)
        table[0, 0] = 0.0
        for slot in range(self.max_events):
            for token, weight in enumerate(self.duration_weights):
                if weight > target:
                    continue
                source = table[slot, : target + 1 - weight]
                candidate = source + event_weight[slot] + duration_log_probability[slot, token]
                table[slot + 1, weight:] = np.logaddexp(table[slot + 1, weight:], candidate)
        return table

    def _decode_sample(
        self,
        pitch_logits: NDArray[np.float64],
        duration_logits: NDArray[np.float64],
        target: int,
        rng: np.random.Generator,
    ) -> MeterDecodeResult:
        pitch_lp, duration_lp, event_weight, pad_weight = self._weights(
            pitch_logits, duration_logits
        )
        forward = self._forward_sum(duration_lp, event_weight, target)
        suffix_pad = np.concatenate((np.cumsum(pad_weight[::-1])[::-1], [0.0]))
        event_count_weights = np.asarray(
            [forward[count, target] + suffix_pad[count] for count in range(self.max_events + 1)]
        )
        event_count_weights[0] = NEGATIVE_INFINITY
        if not np.isfinite(event_count_weights).any():
            raise ValueError("No exact-meter measure can be formed with this duration vocabulary")
        event_count = _sample_log_weights(event_count_weights, rng)

        duration_tokens = np.full(self.max_events, self.duration_pad_id, dtype=np.int64)
        remaining = target
        for slot in range(event_count - 1, -1, -1):
            candidates = np.full(len(self.duration_weights), NEGATIVE_INFINITY)
            for token, weight in enumerate(self.duration_weights):
                previous = remaining - int(weight)
                if previous >= 0:
                    candidates[token] = (
                        forward[slot, previous]
                        + event_weight[slot]
                        + duration_lp[slot, token]
                    )
            token = _sample_log_weights(candidates, rng)
            duration_tokens[slot] = token
            remaining -= int(self.duration_weights[token])
        if remaining != 0:
            raise RuntimeError("Exact-meter duration backtracking failed")

        pitch_tokens = np.full(self.max_events, self.pitch_pad_id, dtype=np.int64)
        pitch_confidences: list[float] = []
        for slot in range(event_count):
            active = pitch_lp[slot, self.allowed_pitch_ids]
            local_index = _sample_log_weights(active, rng)
            pitch_tokens[slot] = int(self.allowed_pitch_ids[local_index])
            pitch_confidences.append(float(np.exp(active[local_index] - _logsumexp(active))))

        normalizer = _logsumexp(event_count_weights)
        selected_weight = float(event_count_weights[event_count])
        confidence_parts = [float(np.exp(selected_weight - normalizer)), *pitch_confidences]
        confidence = float(np.exp(np.mean(np.log(np.clip(confidence_parts, 1e-12, 1.0)))))
        return MeterDecodeResult(
            pitch_tokens,
            duration_tokens,
            event_count,
            selected_weight,
            confidence,
        )

    def _decode_map(
        self,
        pitch_logits: NDArray[np.float64],
        duration_logits: NDArray[np.float64],
        target: int,
    ) -> MeterDecodeResult:
        pitch_lp, duration_lp, _, pad_weight = self._weights(pitch_logits, duration_logits)
        best_pitch_local = np.argmax(pitch_lp[:, self.allowed_pitch_ids], axis=1)
        best_pitch = self.allowed_pitch_ids[best_pitch_local]
        event_weight = pitch_lp[np.arange(self.max_events), best_pitch]

        table = np.full((self.max_events + 1, target + 1), NEGATIVE_INFINITY)
        back = np.full((self.max_events + 1, target + 1), -1, dtype=np.int64)
        table[0, 0] = 0.0
        for slot in range(self.max_events):
            for current in range(target + 1):
                if not np.isfinite(table[slot, current]):
                    continue
                for token, weight in enumerate(self.duration_weights):
                    destination = current + int(weight)
                    if destination > target:
                        continue
                    candidate = table[slot, current] + event_weight[slot] + duration_lp[slot, token]
                    if candidate > table[slot + 1, destination]:
                        table[slot + 1, destination] = candidate
                        back[slot + 1, destination] = token

        suffix_pad = np.concatenate((np.cumsum(pad_weight[::-1])[::-1], [0.0]))
        totals = np.asarray(
            [table[count, target] + suffix_pad[count] for count in range(self.max_events + 1)]
        )
        totals[0] = NEGATIVE_INFINITY
        event_count = int(np.argmax(totals))
        if not np.isfinite(totals[event_count]):
            raise ValueError("No exact-meter measure can be formed with this duration vocabulary")

        duration_tokens = np.full(self.max_events, self.duration_pad_id, dtype=np.int64)
        remaining = target
        for count in range(event_count, 0, -1):
            token = int(back[count, remaining])
            if token < 0:
                raise RuntimeError("Exact-meter MAP backtracking failed")
            duration_tokens[count - 1] = token
            remaining -= int(self.duration_weights[token])
        pitch_tokens = np.full(self.max_events, self.pitch_pad_id, dtype=np.int64)
        pitch_tokens[:event_count] = best_pitch[:event_count]
        return MeterDecodeResult(
            pitch_tokens,
            duration_tokens,
            event_count,
            float(totals[event_count]),
            1.0,
        )

    def is_valid(
        self,
        pitch_tokens: Sequence[int],
        duration_tokens: Sequence[int],
        target_beats: int | float | str | Fraction,
    ) -> bool:
        if len(pitch_tokens) != self.max_events or len(duration_tokens) != self.max_events:
            return False
        pad_started = False
        units = 0
        count = 0
        for pitch, duration in zip(pitch_tokens, duration_tokens):
            if pitch == self.pitch_pad_id and duration == self.duration_pad_id:
                pad_started = True
                continue
            if pad_started or pitch not in self.allowed_pitch_ids or not 0 <= duration < len(self.duration_weights):
                return False
            units += int(self.duration_weights[duration])
            count += 1
        return count > 0 and units == self.target_units(target_beats)
