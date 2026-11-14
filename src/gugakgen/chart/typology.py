"""Phrase types from weighted k-medoids, and the structures built on them.

Methods draft 2026-11-11, sections 2.3-2.5: distinct phrases weighted by how
often they occur are clustered with PAM; the medoid of each type is its
representative phrase. From the typed sequence come the type sequence Y, the
transition counts W (graph G) and the contact relation between types.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from gugakgen.chart.edit_distance import Event, distance, distance_matrix


def load_phrases(path: Path) -> tuple[list[list[Event]], list[int]]:
    """Read ``measures_v0.json``; returns the events and chapter of each measure."""

    records = json.loads(Path(path).read_text(encoding="utf-8"))
    phrases = [
        [
            (None if event["pitch"] is None else int(event["pitch"]), float(event["duration_q"]))
            for event in record["events"]
        ]
        for record in records
    ]
    return phrases, [int(record["chapter"]) for record in records]


def weighted_pam(
    matrix: NDArray[np.float64], weights: NDArray[np.float64], k: int, max_swaps: int = 200
) -> list[int]:
    """PAM (BUILD then SWAP) minimising the weighted sum of distances to medoids."""

    count = matrix.shape[0]

    def total(medoids: Sequence[int]) -> float:
        return float((weights * matrix[:, list(medoids)].min(axis=1)).sum())

    medoids = [int(np.argmin((weights[:, None] * matrix).sum(axis=0)))]
    while len(medoids) < k:
        nearest = matrix[:, medoids].min(axis=1)
        gain = (weights[:, None] * np.maximum(nearest[:, None] - matrix, 0.0)).sum(axis=0)
        gain[medoids] = -np.inf
        medoids.append(int(np.argmax(gain)))
    best = total(medoids)
    for _ in range(max_swaps):
        improvement, swap = 1e-12, None
        for slot in range(k):
            for candidate in range(count):
                if candidate in medoids:
                    continue
                trial = medoids[:slot] + [candidate] + medoids[slot + 1 :]
                gain = best - total(trial)
                if gain > improvement:
                    improvement, swap = gain, trial
        if swap is None:
            break
        medoids, best = swap, total(swap)
    return medoids


def weighted_silhouette(
    matrix: NDArray[np.float64], weights: NDArray[np.float64], assignment: NDArray[np.int64]
) -> float:
    """Mean silhouette over all occurrences (duplicates count as distance-0 neighbours)."""

    clusters = sorted(set(int(c) for c in assignment))
    scores = np.zeros(len(weights))
    for i in range(len(weights)):
        means: dict[int, float | None] = {}
        for cluster in clusters:
            members = assignment == cluster
            mass = weights[members].sum() - (1.0 if cluster == assignment[i] else 0.0)
            means[cluster] = (
                float((weights[members] * matrix[i, members]).sum() / mass) if mass > 0 else None
            )
        own = means[int(assignment[i])]
        if own is None:  # singleton cluster
            continue
        other = min(v for c, v in means.items() if c != assignment[i] and v is not None)
        scores[i] = (other - own) / max(other, own) if max(other, own) > 0 else 0.0
    return float((weights * scores).sum() / weights.sum())


@dataclass(frozen=True)
class Typology:
    """Types of one piece. ``labels[t]`` is the type index of phrase ``t``."""

    phrases: tuple[tuple[Event, ...], ...]
    chapters: tuple[int, ...]
    labels: tuple[int, ...]
    medoid_phrases: tuple[int, ...]  # phrase index (0-based) of each type's medoid
    silhouette: float

    @property
    def k(self) -> int:
        return len(self.medoid_phrases)

    @property
    def medoids(self) -> list[tuple[Event, ...]]:
        return [self.phrases[index] for index in self.medoid_phrases]

    @property
    def names(self) -> str:
        return "".join(chr(ord("a") + label) for label in self.labels)

    def sequence_by_chapter(self) -> str:
        parts: list[str] = []
        for chapter in sorted(set(self.chapters)):
            parts.append(
                "".join(n for n, c in zip(self.names, self.chapters, strict=True) if c == chapter)
            )
        return " | ".join(parts)

    def transition_counts(self) -> NDArray[np.int64]:
        counts = np.zeros((self.k, self.k), dtype=np.int64)
        for a, b in zip(self.labels[:-1], self.labels[1:], strict=True):
            counts[a, b] += 1
        return counts

    def medoid_distances(self, phrase: Sequence[Event]) -> list[float]:
        return [distance(phrase, medoid) for medoid in self.medoids]

    def nearest_types(self, phrase: Sequence[Event]) -> list[int]:
        """Type indices ordered by distance from ``phrase`` to each medoid."""

        return [int(i) for i in np.argsort(self.medoid_distances(phrase), kind="stable")]

    def is_member(self, phrase: Sequence[Event], type_index: int, tolerance: float = 1e-9) -> bool:
        """Eq. 9: the phrase is no farther from its own medoid than from any other."""

        distances = self.medoid_distances(phrase)
        return distances[type_index] <= min(distances) + tolerance

    def contact_pairs(self) -> set[tuple[int, int]]:
        """Eq. 6': types that are the two nearest medoids of some phrase.

        When several medoids tie for second nearest, each of them is in contact
        with the nearest one; this reproduces the draft's counts (13 pairs).
        """

        pairs: set[tuple[int, int]] = set()
        for phrase in self.phrases:
            distances = np.array(self.medoid_distances(phrase))
            order = np.argsort(distances, kind="stable")
            first = int(order[0])
            for other in order[1:]:
                if distances[other] > distances[order[1]] + 1e-9:
                    break
                pairs.add((min(first, int(other)), max(first, int(other))))
        return pairs

    def contact_summary(self) -> dict[str, int]:
        contacts = self.contact_pairs()
        moves = [
            (min(a, b), max(a, b))
            for a, b in zip(self.labels[:-1], self.labels[1:], strict=True)
            if a != b
        ]
        return {
            "contact_pairs": len(contacts),
            "transition_pairs": len(set(moves)),
            "shared_pairs": len(contacts & set(moves)),
            "transitions": len(moves),
            "transitions_across_contact": sum(move in contacts for move in moves),
        }


def build_typology(
    phrases: Sequence[Sequence[Event]], chapters: Sequence[int], k: int = 9
) -> Typology:
    """Cluster distinct phrases (weighted by multiplicity); name types by first appearance."""

    frozen = [tuple(phrase) for phrase in phrases]
    unique = list(dict.fromkeys(frozen))
    weights = np.array([frozen.count(phrase) for phrase in unique], dtype=np.float64)
    matrix = distance_matrix(unique)
    medoids = weighted_pam(matrix, weights, k)
    assignment = np.argmin(matrix[:, medoids], axis=1)
    cluster_of = {phrase: int(assignment[i]) for i, phrase in enumerate(unique)}
    order: dict[int, int] = {}
    for phrase in frozen:
        order.setdefault(cluster_of[phrase], len(order))
    return Typology(
        phrases=tuple(frozen),
        chapters=tuple(int(c) for c in chapters),
        labels=tuple(order[cluster_of[phrase]] for phrase in frozen),
        medoid_phrases=tuple(
            frozen.index(unique[medoids[cluster]])
            for cluster in sorted(order, key=lambda cluster: order[cluster])
        ),
        silhouette=weighted_silhouette(matrix, weights, assignment),
    )
