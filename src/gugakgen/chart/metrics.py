"""Evaluation of generated pieces on the four axes of the Methods draft (2.7), and rule baselines.

Values that hold by construction (type membership after rejection, exact
returns after copying, phrase totals) are reported so that a broken constraint
is visible, but they are not evidence about the model.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

from gugakgen.chart.coords import QUANTUM, Scale, cached_distance
from gugakgen.chart.edit_distance import Event
from gugakgen.chart.sampling import Phrase, is_member
from gugakgen.chart.typology import Typology


def _notes(phrases: Sequence[Sequence[Event]]) -> list[Event]:
    return [event for phrase in phrases for event in phrase]


def _ngrams(items: Sequence, n: int) -> list[tuple]:
    return [tuple(items[i : i + n]) for i in range(len(items) - n + 1)]


def _jsd(left: Counter, right: Counter) -> float:
    keys = sorted(set(left) | set(right), key=repr)
    p = np.array([left[k] for k in keys], dtype=float)
    q = np.array([right[k] for k in keys], dtype=float)
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2

    def kl(a: np.ndarray) -> float:
        keep = a > 0
        return float((a[keep] * np.log2(a[keep] / m[keep])).sum())

    return (kl(p) + kl(q)) / 2


def _intervals(notes: Sequence[Event]) -> Counter:
    pitches = [p for p, _ in notes if p is not None]
    return Counter(b - a for a, b in zip(pitches[:-1], pitches[1:], strict=True))


def _pitch_mass(notes: Sequence[Event]) -> Counter:
    mass: Counter = Counter()
    for pitch, duration in notes:
        mass[pitch] += duration
    return mass


def _boundaries(phrases: Sequence[Sequence[Event]]) -> list[tuple]:
    return [(a[-1], b[0]) for a, b in zip(phrases[:-1], phrases[1:], strict=True)]


def _longest_shared_run(generated: Sequence[Event], original: Sequence[Event]) -> int:
    best = 0
    previous = [0] * (len(original) + 1)
    for g in generated:
        current = [0] * (len(original) + 1)
        for j, o in enumerate(original, start=1):
            if g == o:
                current[j] = previous[j - 1] + 1
                best = max(best, current[j])
        previous = current
    return best


def evaluate_piece(
    generated: Sequence[Phrase],
    typology: Typology,
    returns: Sequence[tuple[int, int]],
) -> dict[str, float]:
    original = typology.phrases
    labels = typology.labels
    originals = set(original)
    notes, original_notes = _notes(generated), _notes(original)
    free = [t for t in range(len(labels)) if t not in {target for _, target in returns}]
    own = [cached_distance(generated[t], typology.medoids[labels[t]]) for t in free]
    seen_bigrams = set(_ngrams(original_notes, 2))
    seen_4grams = set(_ngrams(original_notes, 4))
    seen_boundaries = set(_boundaries(original))
    bigrams, fourgrams = _ngrams(notes, 2), _ngrams(notes, 4)
    boundaries = _boundaries(generated)
    return {
        # structure (constructive in setting (ga) with the hard constraints on)
        "type_membership": float(np.mean([is_member(generated[t], labels[t], typology) for t in free])),
        "returns_exact": float(all(generated[s] == generated[t] for s, t in returns)),
        "totals_exact": float(
            all(
                abs(sum(d for _, d in g) - sum(d for _, d in o)) < QUANTUM / 2
                for g, o in zip(generated, original, strict=True)
            )
        ),
        # novelty
        "original_phrase_rate": float(np.mean([generated[t] in originals for t in free])),
        "same_position_copy_rate": float(np.mean([generated[t] == original[t] for t in free])),
        "unseen_note_transition_rate": float(np.mean([b not in seen_bigrams for b in bigrams])),
        "note_4gram_novelty": float(np.mean([g not in seen_4grams for g in fourgrams])),
        "longest_shared_run_events": float(_longest_shared_run(notes, original_notes)),
        # within-type plausibility
        "own_medoid_distance": float(np.mean(own)),
        # style
        "interval_jsd": _jsd(_intervals(notes), _intervals(original_notes)),
        "pitch_jsd": _jsd(_pitch_mass(notes), _pitch_mass(original_notes)),
        "duration_jsd": _jsd(Counter(d for _, d in notes), Counter(d for _, d in original_notes)),
        "unseen_boundary_rate": float(np.mean([b not in seen_boundaries for b in boundaries])),
    }


def original_own_medoid_distance(typology: Typology) -> float:
    return float(
        np.mean(
            [
                cached_distance(phrase, typology.medoids[label])
                for phrase, label in zip(typology.phrases, typology.labels, strict=True)
            ]
        )
    )


def summarise(rows: Sequence[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Mean and a 95% bootstrap interval over pieces (pieces from one model are not independent trainings)."""

    rng = np.random.default_rng(0)
    out: dict[str, dict[str, float]] = {}
    for key in rows[0]:
        values = np.array([row[key] for row in rows], dtype=float)
        resampled = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
        out[key] = {
            "mean": float(values.mean()),
            "low": float(np.quantile(resampled, 0.025)),
            "high": float(np.quantile(resampled, 0.975)),
        }
    return out


# ----------------------------------------------------------------------------------------------
# Rule baselines (no learning)


def _copy_returns(state: list[Phrase], returns: Sequence[tuple[int, int]]) -> list[Phrase]:
    for source, target in returns:
        state[target] = state[source]
    return state


def baseline_recombine(
    typology: Typology, returns: Sequence[tuple[int, int]], rng: np.random.Generator
) -> list[Phrase]:
    """T0: every phrase is a random observed phrase of its type with the same total; returns copied."""

    state: list[Phrase] = []
    for t, label in enumerate(typology.labels):
        total = sum(d for _, d in typology.phrases[t])
        pool = [
            phrase
            for phrase, other in zip(typology.phrases, typology.labels, strict=True)
            if other == label and abs(sum(d for _, d in phrase) - total) < 1e-9
        ]
        state.append(pool[int(rng.integers(len(pool)))])
    return _copy_returns(state, returns)


def baseline_edit(
    typology: Typology,
    returns: Sequence[tuple[int, int]],
    rng: np.random.Generator,
    scale: Scale,
    rate: float = 0.3,
) -> list[Phrase]:
    """T2: T0, then each note moves to a neighbouring scale tone with probability ``rate``
    and each adjacent pair of unequal durations is swapped with probability ``rate / 2``."""

    state = baseline_recombine(typology, returns, rng)
    for t, phrase in enumerate(state):
        events = list(phrase)
        for j, (pitch, duration) in enumerate(events):
            if pitch is not None and rng.random() < rate:
                step = scale.step(pitch) + (1 if rng.random() < 0.5 else -1)
                events[j] = (scale.pitches[min(max(step, 0), len(scale) - 1)], duration)
        for j in range(len(events) - 1):
            if events[j][1] != events[j + 1][1] and rng.random() < rate / 2:
                events[j], events[j + 1] = (events[j][0], events[j + 1][1]), (
                    events[j + 1][0],
                    events[j][1],
                )
        state[t] = tuple(events)
    return _copy_returns(state, returns)
