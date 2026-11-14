"""Local coordinates of a phrase around the medoid of its type (Methods draft, 2.5).

A phrase is aligned to the medoid with the edit alignment of 2.3. Each event
then carries a slot (the medoid note it sits on), a deviation ``eta`` from that
note and its own duration. ``eta`` is counted in steps of the piece's scale
(the sorted set of pitches in the score), so one step is the neighbouring
scale tone. ``(slot, eta, duration)`` restores the phrase exactly.

Working choices where the draft leaves the coordinate open: an event with no
medoid counterpart keeps the slot of the event before it and is flagged
``inserted``; a rest has ``eta = None``; a medoid rest is replaced as reference
by the nearest pitched medoid note.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from gugakgen.chart.edit_distance import (
    INDEL_WEIGHT,
    Event,
    Step,
    align,
    pitch_cost,
)

QUANTUM = 0.25


@dataclass(frozen=True, slots=True)
class ChartEvent:
    slot: int
    inserted: bool
    eta: int | None  # scale steps from the reference medoid note; None for a rest
    quanta: int  # duration in units of QUANTUM


@dataclass(frozen=True)
class Scale:
    """The pitches of the piece in ascending order; a step is an index into it."""

    pitches: tuple[int, ...]

    @classmethod
    def from_phrases(cls, phrases: Sequence[Sequence[Event]]) -> Scale:
        return cls(tuple(sorted({p for phrase in phrases for p, _ in phrase if p is not None})))

    def step(self, pitch: int) -> int:
        return self.pitches.index(pitch)

    def __len__(self) -> int:
        return len(self.pitches)


def to_quanta(duration: float) -> int:
    quanta = round(duration / QUANTUM)
    if quanta <= 0 or abs(quanta * QUANTUM - duration) > 1e-6:
        raise ValueError(f"Duration {duration} is not a positive multiple of {QUANTUM}")
    return quanta


def reference_steps(medoid: Sequence[Event], scale: Scale) -> list[int]:
    """Scale step of each medoid note; a rest takes the nearest pitched note's step."""

    pitched = [i for i, (pitch, _) in enumerate(medoid) if pitch is not None]
    if not pitched:
        raise ValueError("A medoid needs at least one pitched note")
    return [
        scale.step(medoid[min(pitched, key=lambda i: (abs(i - j), i))][0])  # type: ignore[arg-type]
        for j in range(len(medoid))
    ]


@lru_cache(maxsize=200_000)
def _cached_align(phrase: tuple[Event, ...], medoid: tuple[Event, ...]) -> tuple[float, tuple[Step, ...]]:
    value, steps = align(phrase, medoid)
    return value, tuple(steps)


def cached_distance(phrase: Sequence[Event], medoid: Sequence[Event]) -> float:
    return _cached_align(tuple(phrase), tuple(medoid))[0]


def to_chart(phrase: Sequence[Event], medoid: Sequence[Event], scale: Scale) -> list[ChartEvent]:
    _, steps = _cached_align(tuple(phrase), tuple(medoid))
    refs = reference_steps(medoid, scale)
    slots: list[tuple[int, bool]] = []
    last = 0
    for step in steps:
        if step.kind == "ins":  # medoid note with no phrase event
            continue
        inserted = step.kind == "del"
        if not inserted:
            last = step.j0
        slots.extend((last, inserted) for _ in range(step.i1 - step.i0))
    return [
        ChartEvent(
            slot=slot,
            inserted=inserted,
            eta=None if pitch is None else scale.step(pitch) - refs[slot],
            quanta=to_quanta(duration),
        )
        for (pitch, duration), (slot, inserted) in zip(phrase, slots, strict=True)
    ]


def from_chart(chart: Sequence[ChartEvent], medoid: Sequence[Event], scale: Scale) -> list[Event]:
    refs = reference_steps(medoid, scale)
    return [
        (
            None if event.eta is None else scale.pitches[refs[event.slot] + event.eta],
            event.quanta * QUANTUM,
        )
        for event in chart
    ]


def surrogate_tables(
    phrase: Sequence[Event],
    medoids: Sequence[Sequence[Event]],
    scale: Scale,
    max_events: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Distance to every medoid as a function of the phrase's pitches, alignment held fixed.

    Returns ``cost`` of shape ``(K, max_events, len(scale) + 1)`` and ``const`` of
    shape ``(K,)`` such that, for pitch tokens ``v_j`` (scale step, or
    ``len(scale)`` for a rest) at the phrase's own durations,
    ``sum_j cost[c, j, v_j] + const[c]`` is the eq. 2 cost of the phrase's
    optimal alignment to medoid ``c``. At the phrase's own pitches this equals
    the distance; elsewhere it is an upper bound, because the alignment is not
    re-optimised. Used for the margin loss L_M.
    """

    tokens: list[int | None] = [*scale.pitches, None]
    cost = np.zeros((len(medoids), max_events, len(tokens)), dtype=np.float32)
    const = np.zeros(len(medoids), dtype=np.float32)
    for c, medoid in enumerate(medoids):
        _, steps = _cached_align(tuple(phrase), tuple(medoid))
        norm = sum(d for _, d in phrase) + sum(d for _, d in medoid)
        for step in steps:
            if step.kind == "sub":
                (_, dx), (py, dy) = phrase[step.i0], medoid[step.j0]
                const[c] += abs(dx - dy) / norm
                for v, pitch in enumerate(tokens):
                    cost[c, step.i0, v] += pitch_cost(pitch, py) * (dx + dy) / 2.0 / norm
            elif step.kind == "frag":  # one phrase event against several medoid notes
                dx = phrase[step.i0][1]
                notes = medoid[step.j0 : step.j1]
                const[c] += (abs(dx - sum(d for _, d in notes)) + 0.5) / norm
                for v, pitch in enumerate(tokens):
                    cost[c, step.i0, v] += sum(pitch_cost(pitch, py) * dy for py, dy in notes) / norm
            elif step.kind == "cons":  # several phrase events against one medoid note
                py, dy = medoid[step.j0]
                events = range(step.i0, step.i1)
                const[c] += (abs(dy - sum(phrase[i][1] for i in events)) + 0.5) / norm
                for i in events:
                    for v, pitch in enumerate(tokens):
                        cost[c, i, v] += pitch_cost(pitch, py) * phrase[i][1] / norm
            elif step.kind == "del":
                const[c] += INDEL_WEIGHT * phrase[step.i0][1] / norm
            else:  # ins
                const[c] += INDEL_WEIGHT * medoid[step.j0][1] / norm
    return cost, const
