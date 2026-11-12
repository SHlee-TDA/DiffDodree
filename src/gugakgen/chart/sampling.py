"""Corrupt-restore chain in chart coordinates (Methods draft, 2.6 "생성 절차").

Setting (ga): the type sequence Y, the return set R and the phrase totals B are
fixed. Every phrase starts from an observed phrase of its type. At each step a
span of its events is hidden and re-drawn from a predictor; the hidden span
keeps its total duration (hard L_B), a candidate is accepted only if it is
still nearest to its own medoid (hard L_M, eq. 9) and return targets are
copies of their sources (hard L_R).

A predictor maps a batch of masked phrases to log-probabilities over pitch
tokens and durations. ``ModelPredictor`` wraps the trained network;
``EmpiricalPredictor`` is the learning-free reference that draws deviations and
durations from per-type counts without looking at the visible events.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from gugakgen.chart.coords import QUANTUM, cached_distance
from gugakgen.chart.edit_distance import Event
from gugakgen.chart.model import ChartDenoiser, Layout, encode_phrase, hide
from gugakgen.chart.typology import Typology

Phrase = tuple[Event, ...]
Predictor = Callable[[dict[str, np.ndarray], np.ndarray], tuple[np.ndarray, np.ndarray]]
"""(encoded phrases, hidden mask) -> (log p(pitch token), log p(duration)) per event."""


def is_member(phrase: Sequence[Event], type_index: int, typology: Typology) -> bool:
    distances = [cached_distance(phrase, medoid) for medoid in typology.medoids]
    return distances[type_index] <= min(distances) + 1e-9


class ModelPredictor:
    def __init__(
        self,
        model: ChartDenoiser,
        device: torch.device,
        *,
        null_context: bool = False,
        null_type: bool = False,
    ) -> None:
        self.model, self.device = model.eval(), device
        self.null_context, self.null_type = null_context, null_type

    @torch.no_grad()
    def __call__(
        self, encoded: dict[str, np.ndarray], hidden: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        batch = {key: torch.as_tensor(value, device=self.device) for key, value in encoded.items()}
        mask = torch.as_tensor(hidden, device=self.device)
        rows = torch.ones(mask.shape[0], dtype=torch.bool, device=self.device)
        inputs = hide(
            batch,
            mask,
            self.model.layout,
            null_context=rows if self.null_context else None,
            null_type=rows if self.null_type else None,
        )
        log_pitch, log_dur = self.model(inputs)
        return log_pitch.float().cpu().numpy(), log_dur.float().cpu().numpy()


class EmpiricalPredictor:
    """Per-type counts of eta and duration with add-``alpha`` smoothing; ignores context."""

    def __init__(
        self,
        encoded: Mapping[str, np.ndarray],
        layout: Layout,
        *,
        alpha: float = 0.5,
        by_type: bool = True,
        rows: Sequence[int] | None = None,
    ) -> None:
        self.layout, self.by_type = layout, by_type
        groups = layout.num_types if by_type else 1
        self.eta = np.full((groups, layout.num_eta + 1), alpha)
        self.dur = np.full((groups, len(layout.durations)), alpha)
        for row in range(len(encoded["length"])) if rows is None else rows:
            group = int(encoded["type"][row]) if by_type else 0
            for j in range(int(encoded["length"][row])):
                self.eta[group, encoded["eta"][row, j]] += 1
                self.dur[group, encoded["dur"][row, j]] += 1

    def __call__(
        self, encoded: dict[str, np.ndarray], hidden: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        layout = self.layout
        rows, size = hidden.shape
        log_pitch = np.full((rows, size, layout.num_steps + 1), -np.inf)
        log_dur = np.zeros((rows, size, len(layout.durations)))
        for row in range(rows):
            group = int(encoded["type"][row]) if self.by_type else 0
            log_dur[row] = np.log(self.dur[group] / self.dur[group].sum())
            for j in range(size):
                index = np.arange(layout.num_steps) - int(encoded["ref"][row, j]) + layout.eta_offset
                weights = np.append(self.eta[group, index], self.eta[group, layout.eta_rest])
                log_pitch[row, j] = np.log(weights / weights.sum())
        return log_pitch, log_dur


def sample_durations(
    log_prob: np.ndarray, quanta: Sequence[int], total: int, rng: np.random.Generator
) -> list[int] | None:
    """Draw one duration index per event from independent categoricals given their sum.

    Exact conditional sampling by a backward table over partial sums; returns
    ``None`` if no combination reaches ``total``.
    """

    count = log_prob.shape[0]
    table = np.full((count + 1, total + 1), -np.inf)
    table[count, 0] = 0.0
    for j in range(count - 1, -1, -1):
        for rest in range(total + 1):
            terms = [
                log_prob[j, v] + table[j + 1, rest - q]
                for v, q in enumerate(quanta)
                if q <= rest and np.isfinite(table[j + 1, rest - q])
            ]
            if terms:
                table[j, rest] = np.logaddexp.reduce(terms)
    if not np.isfinite(table[0, total]):
        return None
    chosen: list[int] = []
    rest = total
    for j in range(count):
        scores = np.array(
            [
                log_prob[j, v] + table[j + 1, rest - q] if q <= rest else -np.inf
                for v, q in enumerate(quanta)
            ]
        )
        probabilities = np.exp(scores - np.logaddexp.reduce(scores))
        choice = int(rng.choice(len(quanta), p=probabilities / probabilities.sum()))
        chosen.append(choice)
        rest -= quanta[choice]
    return chosen


@dataclass
class ChainStats:
    proposals: int = 0
    accepted: int = 0
    rejected_membership: int = 0
    unchanged: int = 0
    first_step_proposals: int = 0
    first_step_members: int = 0

    def as_dict(self) -> dict[str, float]:
        changed = max(self.proposals - self.unchanged, 1)
        return {
            "proposals": self.proposals,
            "unchanged_share": self.unchanged / max(self.proposals, 1),
            "membership_rejection_rate": self.rejected_membership / changed,
            "first_step_membership_rate": self.first_step_members
            / max(self.first_step_proposals, 1),
        }


@dataclass(frozen=True)
class ChainConfig:
    steps: int = 8
    rho: float = 0.5  # share of a phrase's events hidden at each step
    temperature: float = 1.0
    initial: str = "member"  # "member": random observed phrase of the type; "medoid"
    enforce_membership: bool = True
    returns: tuple[tuple[int, int], ...] = field(default_factory=tuple)  # 0-based (source, target)


def _tempered(log_prob: np.ndarray, temperature: float) -> np.ndarray:
    scaled = log_prob / temperature
    return scaled - np.logaddexp.reduce(scaled, axis=-1, keepdims=True)


def run_chain(
    typology: Typology,
    layout: Layout,
    predictor: Predictor,
    config: ChainConfig,
    rng: np.random.Generator,
) -> tuple[list[Phrase], ChainStats]:
    """Generate one piece in setting (ga). Returns the phrases and proposal statistics."""

    labels = typology.labels
    count = len(labels)
    target_of = {target: source for source, target in config.returns}
    free = [t for t in range(count) if t not in target_of]
    members: dict[int, list[Phrase]] = {}
    for phrase, label in zip(typology.phrases, labels, strict=True):
        members.setdefault(label, []).append(phrase)

    state: list[Phrase] = list(typology.phrases)
    for t in free:
        if config.initial == "medoid":
            state[t] = typology.medoids[labels[t]]
        else:  # an observed phrase of the same type and the same total duration
            total = sum(d for _, d in typology.phrases[t])
            pool = [p for p in members[labels[t]] if abs(sum(d for _, d in p) - total) < 1e-9]
            state[t] = pool[int(rng.integers(len(pool)))]
    for target, source in target_of.items():
        state[target] = state[source]

    stats = ChainStats()
    scale = layout.scale
    for step in range(config.steps):
        rows = [
            encode_phrase(
                state[t],
                labels[t],
                None if t == 0 else (labels[t - 1], state[t - 1]),
                None if t == count - 1 else (labels[t + 1], state[t + 1]),
                typology,
                layout,
            )
            for t in free
        ]
        encoded = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
        hidden = np.zeros((len(free), layout.max_events), dtype=bool)
        for row, t in enumerate(free):
            length = len(state[t])
            span = min(max(1, round(config.rho * length)), length)
            start = int(rng.integers(length - span + 1))
            hidden[row, start : start + span] = True
        log_pitch, log_dur = predictor(encoded, hidden)
        log_pitch = _tempered(log_pitch, config.temperature)
        log_dur = _tempered(log_dur, config.temperature)

        proposal = list(state)
        for row, t in enumerate(free):
            events = list(state[t])
            where = np.flatnonzero(hidden[row])
            total = sum(round(events[j][1] / QUANTUM) for j in where)
            durations = sample_durations(log_dur[row, where], layout.durations, total, rng)
            for position, j in enumerate(where):
                probabilities = np.exp(log_pitch[row, j])
                token = int(rng.choice(len(probabilities), p=probabilities / probabilities.sum()))
                pitch = None if token == layout.pitch_rest else scale.pitches[token]
                quanta = (
                    round(events[j][1] / QUANTUM)
                    if durations is None
                    else layout.durations[durations[position]]
                )
                events[j] = (pitch, quanta * QUANTUM)
            candidate = tuple(events)
            stats.proposals += 1
            if candidate == state[t]:
                stats.unchanged += 1
                continue
            member = is_member(candidate, labels[t], typology)
            if step == 0:
                stats.first_step_proposals += 1
                stats.first_step_members += int(member)
            if member or not config.enforce_membership:
                proposal[t] = candidate
                stats.accepted += 1
            else:
                stats.rejected_membership += 1
        state = proposal
        for target, source in target_of.items():
            state[target] = state[source]
    return state, stats
