"""Duration-weighted edit distance between phrases (Methods draft 2026-11-11, eq. 2).

Substitution, insertion, deletion, and fragmentation/consolidation (one note
against two or three) in the manner of Mongeau & Sankoff (1990). Pitches are
compared as absolute MIDI numbers. The constants are the draft's working values
and were not fitted to any labels.

The draft states the fragmentation surcharge (0.5) but not the rest of that
cost. Here a fragmentation of ``x`` into ``y_1..y_k`` costs
``sum_i |p_x - p_yi| * d_yi + |d_x - sum_i d_yi| + 0.5``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Event = tuple[int | None, float]
"""(MIDI pitch or None for a rest, duration in quarter notes)."""

REST_PITCH_COST = 6.0
INDEL_WEIGHT = 3.0  # kappa in eq. 2
FRAGMENT_SURCHARGE = 0.5
MAX_FRAGMENTS = 3


@dataclass(frozen=True, slots=True)
class Step:
    """One alignment step: ``left[i0:i1]`` against ``right[j0:j1]``.

    ``kind`` is ``sub`` (1:1), ``del`` (left only), ``ins`` (right only),
    ``frag`` (one left note against several right notes) or ``cons`` (several
    left notes against one right note).
    """

    kind: str
    i0: int
    i1: int
    j0: int
    j1: int
    cost: float


def pitch_cost(left: int | None, right: int | None) -> float:
    if left is None and right is None:
        return 0.0
    if left is None or right is None:
        return REST_PITCH_COST
    return float(abs(left - right))


def _substitution(x: Event, y: Event) -> float:
    return pitch_cost(x[0], y[0]) * (x[1] + y[1]) / 2.0 + abs(x[1] - y[1])


def _fragmentation(one: Event, many: Sequence[Event]) -> float:
    total = sum(duration for _, duration in many)
    return (
        sum(pitch_cost(one[0], pitch) * duration for pitch, duration in many)
        + abs(one[1] - total)
        + FRAGMENT_SURCHARGE
    )


def align(left: Sequence[Event], right: Sequence[Event]) -> tuple[float, list[Step]]:
    """Return the normalised distance and one optimal alignment."""

    if not left or not right:
        raise ValueError("Phrases must contain at least one event")
    rows, columns = len(left), len(right)
    table = np.full((rows + 1, columns + 1), np.inf)
    back: dict[tuple[int, int], Step] = {}
    table[0, 0] = 0.0

    def relax(i: int, j: int, step: Step) -> None:
        value = table[step.i0, step.j0] + step.cost
        if value < table[i, j] - 1e-12:
            table[i, j] = value
            back[(i, j)] = step

    for i in range(rows + 1):
        for j in range(columns + 1):
            if i > 0 and j > 0:
                relax(i, j, Step("sub", i - 1, i, j - 1, j, _substitution(left[i - 1], right[j - 1])))
            if i > 0:
                relax(i, j, Step("del", i - 1, i, j, j, INDEL_WEIGHT * left[i - 1][1]))
            if j > 0:
                relax(i, j, Step("ins", i, i, j - 1, j, INDEL_WEIGHT * right[j - 1][1]))
            for k in range(2, MAX_FRAGMENTS + 1):
                if i > 0 and j >= k:
                    cost = _fragmentation(left[i - 1], right[j - k : j])
                    relax(i, j, Step("frag", i - 1, i, j - k, j, cost))
                if j > 0 and i >= k:
                    cost = _fragmentation(right[j - 1], left[i - k : i])
                    relax(i, j, Step("cons", i - k, i, j - 1, j, cost))

    steps: list[Step] = []
    i, j = rows, columns
    while (i, j) != (0, 0):
        step = back[(i, j)]
        steps.append(step)
        i, j = step.i0, step.j0
    steps.reverse()
    norm = sum(d for _, d in left) + sum(d for _, d in right)
    return float(table[rows, columns] / norm), steps


def distance(left: Sequence[Event], right: Sequence[Event]) -> float:
    return align(left, right)[0]


def distance_matrix(phrases: Sequence[Sequence[Event]]) -> NDArray[np.float64]:
    count = len(phrases)
    matrix = np.zeros((count, count), dtype=np.float64)
    for i in range(count):
        for j in range(i + 1, count):
            matrix[i, j] = matrix[j, i] = distance(phrases[i], phrases[j])
    return matrix
