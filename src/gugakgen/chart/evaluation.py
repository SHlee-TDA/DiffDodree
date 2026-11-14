"""Evaluation of generated pieces, fixed with the user on 2026-11-12.

Four axes, all computed from saved pieces (no model needed), so any generator
(our arms now; rule baselines and the TDA composition method later) can be put
through the same functions:

1. geometry   - phrase level: surface types by edit distance, graph G;
                note level: the note graph of Tran, Lee & Jung (2022, eq. 1) and
                its 1-dimensional persistence barcode, compared by bottleneck and
                1-Wasserstein distance;
2. statistics - Jensen-Shannon divergences of note-level distributions, and the
                overlapping area of intra-set / inter-set distance distributions
                (Yang & Lerch);
3. music      - rule checks (scale, beat fill, grid, chapter cadences);
4. novelty    - k-NN metrics in the edit-distance space of measures: density and
                coverage (Naeem et al., ICML 2020), authenticity in the
                nearest-neighbour form (after Alaa et al., ICML 2022), exact copies,
                longest shared run, and the data-copying test C_T (Meehan et al.,
                AISTATS 2020) with a single cell.

Checks that hold by construction for the grid model (scale, beat fill, grid)
are still reported; they matter for generators that are not built on the grid.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from gugakgen.chart.coords import cached_distance
from gugakgen.chart.edit_distance import Event, distance_matrix
from gugakgen.chart.metrics import _intervals, _jsd, _longest_shared_run, _ngrams, _notes

EVAL_PROTOCOL = "chart_eval_v2_2026-11-12"  # v2: criteria per λ arm and the λ selection rule
Phrase = tuple[Event, ...]


def as_phrase(raw: Iterable[Sequence[Any]]) -> Phrase:
    """JSON ``[[pitch, duration], ...]`` back to the event tuple used everywhere else."""

    return tuple((None if p is None else int(p), float(d)) for p, d in raw)


# ----------------------------------------------------------------------------------------------
# Note graph and persistence (geometry of the TDA composition work)


def note_graph_distances(notes: Sequence[Event]) -> tuple[list[Event], NDArray[np.float64]]:
    """Nodes are distinct (pitch, duration) events; time-adjacent events share an edge whose
    weight w counts the adjacencies. Distance = shortest path with edge length 1/w (eq. 1 of
    Tran, Lee & Jung 2022, computed with Floyd-Warshall). Rests are nodes like notes."""

    nodes = sorted(set(notes), key=lambda e: (-1 if e[0] is None else e[0], e[1]))
    index = {node: i for i, node in enumerate(nodes)}
    weight = np.zeros((len(nodes), len(nodes)))
    for a, b in zip(notes[:-1], notes[1:], strict=True):
        if a != b:
            weight[index[a], index[b]] += 1
            weight[index[b], index[a]] += 1
    dist = np.full_like(weight, np.inf)
    dist[weight > 0] = 1.0 / weight[weight > 0]
    np.fill_diagonal(dist, 0.0)
    for k in range(len(nodes)):
        dist = np.minimum(dist, dist[:, k : k + 1] + dist[k : k + 1, :])
    return nodes, dist


def barcode(dist: NDArray[np.float64], dim: int = 1) -> NDArray[np.float64]:
    """Vietoris-Rips persistence diagram (finite bars only) of a distance matrix."""

    from ripser import ripser  # optional dependency, installed by requirements/colab.txt

    finite = dist.copy()
    if not np.isfinite(finite).all():  # disconnected graph: put components far apart
        finite[~np.isfinite(finite)] = 2 * finite[np.isfinite(finite)].max() + 1
    diagram = ripser(finite, distance_matrix=True, maxdim=dim)["dgms"][dim]
    return diagram[np.isfinite(diagram[:, 1])] if len(diagram) else np.zeros((0, 2))


def _diagram_costs(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Square cost matrix of the matching problem with diagonal copies (L-infinity ground)."""

    m, n = len(a), len(b)
    cost = np.zeros((m + n, m + n))
    if m and n:
        cost[:m, :n] = np.maximum(np.abs(a[:, None, 0] - b[None, :, 0]), np.abs(a[:, None, 1] - b[None, :, 1]))
    cost[:m, n:] = ((a[:, 1] - a[:, 0]) / 2)[:, None] if m else 0
    cost[m:, :n] = ((b[:, 1] - b[:, 0]) / 2)[None, :] if n else 0
    return cost


def wasserstein1(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    from scipy.optimize import linear_sum_assignment

    cost = _diagram_costs(a, b)
    if cost.size == 0:
        return 0.0
    rows, cols = linear_sum_assignment(cost)
    return float(cost[rows, cols].sum())


def bottleneck(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching

    cost = _diagram_costs(a, b)
    if cost.size == 0:
        return 0.0
    size = len(cost)
    for threshold in np.unique(cost):
        allowed = csr_matrix((cost <= threshold + 1e-12).astype(np.int8))
        if (maximum_bipartite_matching(allowed, perm_type="column") >= 0).sum() == size:
            return float(threshold)
    return float(cost.max())  # unreachable: the full matrix always has a perfect matching


# ----------------------------------------------------------------------------------------------
# Small statistics helpers


def _counter(items: Iterable[Any]) -> Counter:
    return Counter(items)


def overlap_area(left: Sequence[float], right: Sequence[float], points: int = 512) -> float:
    """Overlapping area of two densities (Gaussian KDE; histogram when a KDE is singular)."""

    left_a, right_a = np.asarray(left, float), np.asarray(right, float)
    lo, hi = min(left_a.min(), right_a.min()), max(left_a.max(), right_a.max())
    if hi - lo < 1e-12:
        return 1.0
    grid = np.linspace(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo), points)
    try:
        from scipy.stats import gaussian_kde

        p, q = gaussian_kde(left_a)(grid), gaussian_kde(right_a)(grid)
        integrate = getattr(np, "trapezoid", None) or np.trapz
        return float(integrate(np.minimum(p, q), grid))
    except (np.linalg.LinAlgError, ValueError):
        edges = np.linspace(lo, hi, 41)
        p, _ = np.histogram(left_a, edges, density=True)
        q, _ = np.histogram(right_a, edges, density=True)
        return float((np.minimum(p, q) * np.diff(edges)).sum())


def bootstrap_ci(values: Sequence[float], draws: int, seed: int) -> tuple[float, float]:
    data = np.asarray(values, float)
    if len(data) < 2:
        return (float(data.mean()), float(data.mean())) if len(data) else (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, len(data), (draws, len(data)))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _final_pitch(phrase: Sequence[Event]) -> int | None:
    pitches = [p for p, _ in phrase if p is not None]
    return pitches[-1] if pitches else None


# ----------------------------------------------------------------------------------------------
# Reference: everything computed once from the original


class Reference:
    """The original piece and the fixed objects every generated piece is compared with."""

    def __init__(
        self,
        phrases: Sequence[Sequence[Event]],
        chapters: Sequence[int],
        medoids: Sequence[Sequence[Event]],
        edges: NDArray[np.bool_],
        returns: Sequence[tuple[int, int]],
        *,
        quantum: float,
        knn_k: int,
    ) -> None:
        self.phrases = [tuple(p) for p in phrases]
        self.chapters = list(chapters)
        self.medoids = [tuple(m) for m in medoids]
        self.edges = edges
        self.returns = list(returns)
        self.quantum = quantum
        self.knn_k = knn_k
        targets = {t for _, t in returns}
        self.free = [t for t in range(len(self.phrases)) if t not in targets]
        self.notes = _notes(self.phrases)
        self.pitch_set = {p for p, _ in self.notes if p is not None}
        ends = [t for t in range(len(chapters)) if t == len(chapters) - 1 or chapters[t + 1] != chapters[t]]
        self.chapter_ends = ends
        self.cadence_pitches = {_final_pitch(self.phrases[t]) for t in ends}
        self.final_pitch = _final_pitch(self.phrases[-1])
        self.surface = self.surface_types(self.phrases)
        self.graph = self._counts(self.surface)
        counts = Counter(self.phrases)
        # share of measures whose notes recur elsewhere in the piece: what a fresh draw from the
        # piece's own phrase distribution would match exactly (Good-Turing seen mass, 0.458 here)
        self.repeat_rate = float(np.mean([counts[p] > 1 for p in self.phrases]))
        self.repeat_rate_free = float(np.mean([counts[self.phrases[t]] > 1 for t in self.free]))
        self.distinct = sorted(set(self.phrases), key=repr)
        self.distinct_index = {p: i for i, p in enumerate(self.distinct)}
        matrix = distance_matrix(self.distinct)
        ordered = np.sort(matrix + np.diag(np.full(len(matrix), np.inf)), axis=1)
        self.nn1 = ordered[:, 0]
        self.radius = ordered[:, knn_k - 1]
        intra = [cached_distance(self.phrases[i], self.phrases[j])
                 for i in range(len(self.phrases)) for j in range(len(self.phrases)) if i < j]
        self.intra = np.asarray(intra)
        _, note_dist = note_graph_distances(self.notes)
        self.barcode = barcode(note_dist)

    def surface_types(self, phrases: Sequence[Sequence[Event]]) -> list[int]:
        return [int(np.argmin([cached_distance(tuple(p), m) for m in self.medoids])) for p in phrases]

    def _counts(self, labels: Sequence[int]) -> NDArray[np.float64]:
        counts = np.zeros(self.edges.shape)
        for a, b in zip(labels[:-1], labels[1:], strict=True):
            counts[a, b] += 1
        return counts

    def to_distinct(self, phrase: Phrase) -> NDArray[np.float64]:
        return np.array([cached_distance(phrase, x) for x in self.distinct])


def _tv(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    return float(0.5 * np.abs(a / a.sum() - b / b.sum()).sum())


# ----------------------------------------------------------------------------------------------
# Per piece


def piece_scores(phrases: Sequence[Phrase], ref: Reference) -> dict[str, float]:
    """Scores of one 72-measure piece that do not need the other pieces."""

    notes = _notes(phrases)
    surface = ref.surface_types(phrases)
    _, note_dist = note_graph_distances(notes)
    bars = barcode(note_dist)
    pitches = [p for p, _ in notes if p is not None]
    onsets_off = 0
    for phrase in phrases:
        position = 0.0
        for _, duration in phrase:
            onsets_off += abs(position / ref.quantum - round(position / ref.quantum)) > 1e-6
            position += duration
    durations = [sum(d for _, d in p) for p in phrases]
    target = [sum(d for _, d in p) for p in ref.phrases]
    ngram = {n: (_counter(_ngrams(notes, n)), _counter(_ngrams(ref.notes, n))) for n in (3, 4)}
    return {
        # 1. geometry
        "phrase_graph_tv": _tv(ref._counts(surface), ref.graph),
        "surface_edge_validity": float(np.mean([bool(ref.edges[a, b]) for a, b in zip(surface[:-1], surface[1:], strict=True)])),
        "ph_bottleneck": bottleneck(bars, ref.barcode),
        "ph_wasserstein": wasserstein1(bars, ref.barcode),
        "ph_cycles": float(len(bars)),
        # 2. statistics
        "interval_jsd": _jsd(_intervals(notes), _intervals(ref.notes)),
        "duration_jsd": _jsd(_counter(d for _, d in notes), _counter(d for _, d in ref.notes)),
        "pitch_jsd": _jsd(_counter(p for p, _ in notes), _counter(p for p, _ in ref.notes)),
        "note_3gram_jsd": _jsd(*ngram[3]),
        "note_4gram_jsd": _jsd(*ngram[4]),
        # 3. music (the first three hold by construction on the grid)
        "pitch_outside_scale": float(np.mean([p not in ref.pitch_set for p in pitches])) if pitches else 0.0,
        "beat_fill_error": float(np.mean([abs(a - b) > 1e-6 for a, b in zip(durations, target, strict=True)])),
        "offgrid_onset": float(onsets_off / max(len(notes), 1)),
        "chapter_cadence_ok": float(np.mean([_final_pitch(phrases[t]) in ref.cadence_pitches for t in ref.chapter_ends])),
        "final_pitch_ok": float(_final_pitch(phrases[-1]) == ref.final_pitch),
        # 4. novelty (per piece)
        "exact_copy_rate": float(np.mean([phrases[t] in ref.distinct_index for t in ref.free])),
        "longest_shared_run": float(_longest_shared_run(notes, ref.notes)),
        "longest_shared_run_share": float(_longest_shared_run(notes, ref.notes) / max(len(notes), 1)),
    }


# ----------------------------------------------------------------------------------------------
# Per model (set of pieces)


def novelty_scores(
    pieces: Sequence[Sequence[Phrase]], ref: Reference, *, pair_sample: int, seed: int
) -> tuple[dict[str, float], dict[Phrase, NDArray[np.float64]]]:
    """k-NN metrics over the free measures of all pieces (with multiplicity)."""

    generated = [piece[t] for piece in pieces for t in ref.free]
    cache = {g: ref.to_distinct(g) for g in set(generated)}
    dists = np.stack([cache[g] for g in generated])  # (M, N)
    inside = dists <= ref.radius[None, :] + 1e-12
    nearest = dists.argmin(axis=1)
    nn = dists[np.arange(len(dists)), nearest]
    rng = np.random.default_rng(seed)
    pairs = rng.integers(0, len(generated), (pair_sample, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    intra_gen = np.array([cached_distance(generated[i], generated[j]) for i, j in pairs])
    weights = Counter(generated)
    distinct = list(weights)
    inter = np.array([[cached_distance(g, x) for x in ref.phrases] for g in distinct])
    inter_w = np.repeat(inter, [weights[g] for g in distinct], axis=0).ravel()
    return {
        "density": float(inside.sum() / (ref.knn_k * len(generated))),
        "coverage": float(inside.any(axis=0).mean()),
        "authenticity": float(np.mean(nn >= ref.nn1[nearest] - 1e-12)),
        "nn_distance_mean": float(nn.mean()),
        "distinct_measure_rate": float(len(weights) / len(generated)),
        "intra_generated_distance": float(intra_gen.mean()),
        "oa_intra_original_vs_inter": overlap_area(ref.intra, inter_w),
        "oa_intra_original_vs_intra_generated": overlap_area(ref.intra, intra_gen),
    }, cache


def model_scores(
    pieces: Sequence[Sequence[Phrase]], ref: Reference, *, bootstrap: int, seed: int, pair_sample: int
) -> dict[str, Any]:
    rows = [piece_scores(piece, ref) for piece in pieces]
    per_piece = {
        key: {"mean": float(np.mean([r[key] for r in rows])),
              "ci95": bootstrap_ci([r[key] for r in rows], bootstrap, seed)}
        for key in rows[0]
    }
    pooled_surface = sum(ref._counts(ref.surface_types(piece)) for piece in pieces)
    novelty, cache = novelty_scores(pieces, ref, pair_sample=pair_sample, seed=seed)
    # piece-level bootstrap of the set metrics that are cheap given the cached distances
    rng = np.random.default_rng(seed)
    # coverage has no interval: resampling pieces with replacement leaves fewer distinct pieces, so
    # its bootstrap values sit at or below the point estimate (code review of 45618a9)
    boot: dict[str, list[float]] = {"density": [], "authenticity": []}
    blocks = [np.stack([cache[piece[t]] for t in ref.free]) for piece in pieces]
    for _ in range(bootstrap):
        d = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        inside = d <= ref.radius[None, :] + 1e-12
        nearest = d.argmin(axis=1)
        boot["density"].append(inside.sum() / (ref.knn_k * len(d)))
        boot["authenticity"].append(np.mean(d[np.arange(len(d)), nearest] >= ref.nn1[nearest] - 1e-12))
    set_level = {key: {"value": value, "ci95": (float(np.quantile(boot[key], 0.025)), float(np.quantile(boot[key], 0.975)))
                       if key in boot else None} for key, value in novelty.items()}
    set_level["pooled_phrase_graph_tv"] = {"value": _tv(pooled_surface, ref.graph), "ci95": None}
    return {"pieces": len(pieces), "per_piece": per_piece, "set": set_level, "rows": rows}


# ----------------------------------------------------------------------------------------------
# Data-copying test C_T (Meehan et al. 2020), one cell


def mann_whitney_z(b: NDArray[np.float64], a: NDArray[np.float64]) -> float:
    """Normal approximation of the Mann-Whitney U of ``b`` vs ``a`` with the tie correction."""

    m, n = len(b), len(a)
    if m == 0 or n == 0:
        return float("nan")
    u = float((b[:, None] > a[None, :]).sum() + 0.5 * (b[:, None] == a[None, :]).sum())
    total = m + n
    _, ties = np.unique(np.concatenate([a, b]), return_counts=True)
    correction = float((ties**3 - ties).sum()) / (total * (total - 1)) if total > 1 else 0.0
    variance = m * n / 12 * ((total + 1) - correction)
    return (u - m * n / 2) / math.sqrt(variance) if variance > 0 else 0.0


def copy_test(
    generated: Sequence[Phrase], train: Sequence[Phrase], test: Sequence[Phrase]
) -> dict[str, float]:
    """Mann-Whitney Z of nearest-training distances: generated vs held-out. Z << 0 = copying.

    Two versions (code review of 45618a9). ``z`` uses every generated measure. The held-out
    measures here never repeat a training measure (folds are built from distinct non-medoid
    contents), while the piece repeats 46% of its measures, so ``z`` is pushed negative even for
    a generator that only repeats as the piece does. ``z_novel`` therefore drops the generated
    (and held-out) measures identical to a training measure and compares the rest; the dropped
    share is reported as ``exact_train_share``, to be read against the piece's own repeat rate.
    """

    train_set = sorted(set(train), key=repr)
    known = set(train_set)

    def nearest(phrase: Phrase) -> float:
        return min(cached_distance(phrase, x) for x in train_set)

    cache = {g: nearest(g) for g in set(generated)}
    b = np.array([cache[g] for g in generated])  # generated
    a = np.array([nearest(p) for p in test])  # held-out
    new_b = np.array([cache[g] for g in generated if g not in known])
    new_a = np.array([nearest(p) for p in test if p not in known])
    return {"z": mann_whitney_z(b, a), "z_novel": mann_whitney_z(new_b, new_a),
            "generated": len(b), "heldout": len(a), "generated_novel": len(new_b),
            "exact_train_share": float(1 - len(new_b) / len(b)) if len(b) else float("nan"),
            "generated_nn_mean": float(b.mean()), "heldout_nn_mean": float(a.mean()),
            "generated_novel_nn_mean": float(new_b.mean()) if len(new_b) else float("nan")}


# ----------------------------------------------------------------------------------------------
# Criteria fixed on 2026-11-12 (only those that need no other generator are judged now)


CRITERIA = {
    "1_geometry": "λ=1 < TLJ-A in phrase-graph TV and PH bottleneck (needs TLJ-A)",
    "2_statistics": "λ=1 < TLJ-A and R2 in interval/duration/4-gram JSD, OA larger (needs TLJ-A, R2)",
    "3_music": "λ=1 ≤ TLJ-A in out-of-scale, beat-fill and cadence violations (needs TLJ-A)",
    "4_novelty": "authenticity ≥ T2-0.3, |Z_CT| < 2 on every training seed on measures not identical to a training measure "
                 "(exact share reported against the piece's repeat rate), density > R0, R1, R2, TLJ-A (only |Z| judged now)",
    "5_graph_loss": "this arm (λ > 0) > λ=0 in first-pass membership for every training seed",
}


SELECTION_RULE = (
    "among arms with λ > 0 that pass criterion 4 (|z_novel| < 2) and criterion 5 on every training seed, "
    "the one with the lowest held-out ELBO at its selected training length"
)


def select_arm(criteria_by_arm: Mapping[str, Mapping[str, Any]], heldout_elbo: Mapping[str, float]) -> dict[str, Any]:
    """λ selection fixed before the λ sweep was run (2026-11-12)."""

    eligible = {arm: bool(c["5_graph_loss"]["status"] == "pass" and c["4_novelty"]["copy_test_pass"])
                for arm, c in criteria_by_arm.items()}
    passing = [arm for arm, ok in eligible.items() if ok]
    chosen = min(passing, key=lambda arm: heldout_elbo[arm]) if passing else None
    return {"rule": SELECTION_RULE, "eligible": eligible,
            "heldout_elbo": {arm: float(heldout_elbo[arm]) for arm in criteria_by_arm}, "selected": chosen}


def judge(
    first_pass: Mapping[str, Mapping[int, float]], copy_z: Mapping[str, Mapping[int, float]],
    *, main_arm: str, ablation_arm: str, graph_active_steps: int | None = None,
) -> dict[str, Any]:
    seeds = sorted(set(first_pass.get(main_arm, {})) & set(first_pass.get(ablation_arm, {})))
    c5 = {s: first_pass[main_arm][s] > first_pass[ablation_arm][s] for s in seeds}
    z = copy_z.get(main_arm, {})
    c4 = {s: bool(abs(v) < 2) for s, v in z.items()}  # NaN (nothing new generated) fails
    out: dict[str, Any] = {key: {"rule": rule, "status": "pending: needs comparison generators"} for key, rule in CRITERIA.items()}
    out["5_graph_loss"].update(status=("pass" if seeds and all(c5.values()) else "fail"), by_seed=c5)
    if graph_active_steps is not None:
        out["5_graph_loss"]["graph_active_steps"] = graph_active_steps
        if graph_active_steps <= 0:  # the main arm never felt the graph loss: both arms are one model
            out["5_graph_loss"]["status"] = "invalid: graph loss inactive at the selected length"
    out["4_novelty"].update(status="partial: only |Z_CT| < 2 judged", copy_test_by_seed=c4,
                            copy_test_pass=bool(c4) and all(c4.values()))
    return out
