"""DTW distance between measures, as used to build the measure-family Graph.

Follows the phrase-graph method note (§3–4): each measure is a sequence of
(pitch, duration) events compared by Dynamic Time Warping. The original
definition uses pitch relative to the first note of each measure; for comparing
passages that should return at the same pitch, ``relative_pitch=False`` uses
absolute MIDI pitch with the same costs. The coefficients are the exploratory
values of that note, not musicologically fixed ones.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from gugakgen.data.records import TokenizedScore
from gugakgen.data.vocabulary import ScoreVocabulary

Event = tuple[int | None, float]
"""(MIDI pitch or None for a rest, duration in quarter notes)."""

DURATION_WEIGHT = 0.7
PITCH_DIVISOR = 4.0
PITCH_CAP = 3.0
REST_MISMATCH_COST = 2.5
LAST_PITCH_WEIGHT = 0.08
EVENT_COUNT_WEIGHT = 0.08
_EPSILON = 1e-6


def measure_events(
    tokens: TokenizedScore, vocabulary: ScoreVocabulary, index: int
) -> list[Event]:
    """Decode one tokenized measure (zero-based ``index``) into events."""

    events: list[Event] = []
    for pitch_id, duration_id in zip(tokens.pitch[index], tokens.duration[index], strict=True):
        if int(duration_id) == vocabulary.duration_pad_id:
            continue
        pitch = (
            None
            if int(pitch_id) == vocabulary.pitch_rest_id
            else int(vocabulary.pitch_values[int(pitch_id)])
        )
        events.append((pitch, float(vocabulary.duration_values[int(duration_id)])))
    return events


def score_events(tokens: TokenizedScore, vocabulary: ScoreVocabulary) -> list[list[Event]]:
    return [measure_events(tokens, vocabulary, index) for index in range(tokens.num_measures)]


def _relative(events: Sequence[Event]) -> list[Event]:
    first = next((pitch for pitch, _ in events if pitch is not None), 0)
    return [(None if pitch is None else pitch - first, duration) for pitch, duration in events]


def _pitch_cost(left: int | None, right: int | None) -> float:
    if left is None and right is None:
        return 0.0
    if left is None or right is None:
        return REST_MISMATCH_COST
    return min(abs(left - right) / PITCH_DIVISOR, PITCH_CAP)


def _last_pitch(events: Sequence[Event]) -> int:
    return next((pitch for pitch, _ in reversed(events) if pitch is not None), 0)


def measure_distance(
    left: Sequence[Event], right: Sequence[Event], *, relative_pitch: bool = True
) -> float:
    """DTW distance between two measures; zero for identical measures."""

    if not left or not right:
        raise ValueError("Measures must contain at least one event")
    if relative_pitch:
        left, right = _relative(left), _relative(right)
    rows, columns = len(left), len(right)
    cost = np.full((rows + 1, columns + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, rows + 1):
        pitch_i, duration_i = left[i - 1]
        for j in range(1, columns + 1):
            pitch_j, duration_j = right[j - 1]
            local = _pitch_cost(pitch_i, pitch_j) + DURATION_WEIGHT * abs(
                math.log2((duration_i + _EPSILON) / (duration_j + _EPSILON))
            )
            cost[i, j] = local + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(
        cost[rows, columns] / max(rows, columns)
        + LAST_PITCH_WEIGHT * abs(_last_pitch(left) - _last_pitch(right))
        + EVENT_COUNT_WEIGHT * abs(rows - columns)
    )


def passage_distance(
    left: Sequence[Sequence[Event]],
    right: Sequence[Sequence[Event]],
    *,
    relative_pitch: bool = False,
) -> float:
    """Mean distance between measure-aligned passages (``left[k]`` vs ``right[k]``)."""

    if len(left) != len(right) or not left:
        raise ValueError("Passages must be non-empty and have the same number of measures")
    return float(
        np.mean(
            [
                measure_distance(a, b, relative_pitch=relative_pitch)
                for a, b in zip(left, right, strict=True)
            ]
        )
    )


def distance_matrix(
    measures: Sequence[Sequence[Event]], *, relative_pitch: bool = True
) -> NDArray[np.float64]:
    count = len(measures)
    matrix = np.zeros((count, count), dtype=np.float64)
    for i in range(count):
        for j in range(i + 1, count):
            value = measure_distance(measures[i], measures[j], relative_pitch=relative_pitch)
            matrix[i, j] = matrix[j, i] = value
    return matrix


def family_labels(
    matrix: NDArray[np.float64], *, num_families: int = 9, max_iterations: int = 30
) -> list[str]:
    """k-medoids families named A, B, ... in order of first appearance (method note §4)."""

    medoids = [0]
    while len(medoids) < num_families:
        medoids.append(int(np.argmax(matrix[:, medoids].min(axis=1))))
    for _ in range(max_iterations):
        assignment = np.argmin(matrix[:, medoids], axis=1)
        updated = []
        for cluster in range(num_families):
            members = np.flatnonzero(assignment == cluster)
            within = matrix[np.ix_(members, members)].sum(axis=1)
            updated.append(int(members[np.argmin(within)]))
        if updated == medoids:
            break
        medoids = updated
    assignment = np.argmin(matrix[:, medoids], axis=1)
    names: dict[int, str] = {}
    for cluster in assignment:
        names.setdefault(int(cluster), chr(ord("A") + len(names)))
    return [names[int(cluster)] for cluster in assignment]
