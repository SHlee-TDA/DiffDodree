"""Typed transition Graph for expert-segmented measure-family sequences."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class TransitionEdge:
    source: str
    target: str
    count: int
    probability: float

    def __post_init__(self) -> None:
        if self.count < 0 or not 0.0 <= self.probability <= 1.0:
            raise ValueError("Transition edge count/probability is invalid")


@dataclass(frozen=True)
class TransitionGraph:
    node_order: tuple[str, ...]
    transition_matrix: NDArray[np.float64]
    edges: tuple[TransitionEdge, ...]
    original_path: tuple[str, ...] = ()
    chapter_paths: tuple[tuple[str, ...], ...] = ()
    metadata: Mapping[str, Any] | None = None
    _node_ids: Mapping[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.node_order or len(set(self.node_order)) != len(self.node_order):
            raise ValueError("node_order must contain unique node names")
        matrix = np.asarray(self.transition_matrix, dtype=np.float64)
        size = len(self.node_order)
        if matrix.shape != (size, size):
            raise ValueError(f"transition_matrix must have shape {(size, size)}, got {matrix.shape}")
        if not np.isfinite(matrix).all() or (matrix < 0).any():
            raise ValueError("transition_matrix must contain finite non-negative values")
        row_sums = matrix.sum(axis=1)
        active = row_sums > 0
        if active.any() and not np.allclose(row_sums[active], 1.0, atol=1e-4):
            raise ValueError("Every non-terminal transition row must sum to one")
        known = set(self.node_order)
        if any(edge.source not in known or edge.target not in known for edge in self.edges):
            raise ValueError("An edge references an unknown node")
        if any(label not in known for label in self.original_path):
            raise ValueError("original_path contains an unknown family")
        object.__setattr__(self, "transition_matrix", matrix)
        object.__setattr__(self, "_node_ids", {name: i for i, name in enumerate(self.node_order)})

    @property
    def num_nodes(self) -> int:
        return len(self.node_order)

    def node_id(self, name: str) -> int:
        try:
            return self._node_ids[name]
        except KeyError as error:
            raise KeyError(f"Unknown transition node {name!r}") from error

    def node_name(self, node_id: int) -> str:
        if node_id < 0 or node_id >= self.num_nodes:
            raise IndexError(f"Node id {node_id} is out of range")
        return self.node_order[node_id]

    def probability(self, source: str | int, target: str | int) -> float:
        source_id = self.node_id(source) if isinstance(source, str) else int(source)
        target_id = self.node_id(target) if isinstance(target, str) else int(target)
        return float(self.transition_matrix[source_id, target_id])

    def observed(self, source: int, target: int) -> bool:
        return bool(self.transition_matrix[int(source), int(target)] > 0)

    def log_matrix(self, floor: float = -6.0) -> NDArray[np.float64]:
        if floor > 0:
            raise ValueError("floor must be a non-positive log-probability")
        with np.errstate(divide="ignore"):
            values = np.log(self.transition_matrix)
        return np.maximum(values, floor)

    def transition_validity(self, path: Sequence[int | str]) -> float:
        if len(path) < 2:
            return 1.0
        ids = [self.node_id(item) if isinstance(item, str) else int(item) for item in path]
        return float(np.mean([self.observed(left, right) for left, right in zip(ids, ids[1:])]))

    def path_log_probability(self, path: Sequence[int | str], floor: float = -6.0) -> float:
        if len(path) < 2:
            return 0.0
        ids = [self.node_id(item) if isinstance(item, str) else int(item) for item in path]
        log_matrix = self.log_matrix(floor)
        return float(sum(log_matrix[left, right] for left, right in zip(ids, ids[1:])))

    def guide_logits(
        self,
        logits: NDArray[np.floating[Any]],
        position: int,
        current_path: Sequence[int],
        *,
        mask_id: int,
        scale: float,
        floor: float = -6.0,
        forbid_unobserved: bool = False,
        allowed: NDArray[np.bool_] | None = None,
    ) -> NDArray[np.float64]:
        """Apply local left/right transition evidence to one family-logit row."""

        row = np.asarray(logits, dtype=np.float64).copy()
        if row.shape != (self.num_nodes,):
            raise ValueError(f"Expected logits shape {(self.num_nodes,)}, got {row.shape}")
        if len(current_path) == 0 or not 0 <= position < len(current_path):
            raise IndexError("position is outside current_path")
        log_matrix = self.log_matrix(floor)
        left = None if position == 0 else int(current_path[position - 1])
        right = None if position + 1 == len(current_path) else int(current_path[position + 1])
        left = None if left == mask_id else left
        right = None if right == mask_id else right
        bonus = np.zeros(self.num_nodes, dtype=np.float64)
        if left is not None:
            bonus += log_matrix[left]
        if right is not None:
            bonus += log_matrix[:, right]
        row += float(scale) * bonus

        permitted = np.ones(self.num_nodes, dtype=bool) if allowed is None else np.asarray(allowed, bool).copy()
        if permitted.shape != (self.num_nodes,):
            raise ValueError("allowed must have one value per transition node")
        if forbid_unobserved:
            if left is not None:
                candidate = permitted & (self.transition_matrix[left] > 0)
                if candidate.any():
                    permitted = candidate
            if right is not None:
                candidate = permitted & (self.transition_matrix[:, right] > 0)
                if candidate.any():
                    permitted = candidate
        if not permitted.any():
            raise ValueError(f"No family is allowed at position {position}")
        return np.where(permitted, row, -np.inf)

    def project_path(
        self,
        proposed: Sequence[int],
        *,
        anchors: Mapping[int, int] | None = None,
        allowed: NDArray[np.bool_] | None = None,
        change_cost: float = 1.0,
        probability_weight: float = 0.0,
        log_floor: float = -6.0,
    ) -> tuple[NDArray[np.int64], int]:
        """Find a valid observed-edge path with minimum label-change cost.

        Anchor keys are zero-based positions. ``allowed`` has shape ``(T, K)``.
        A small positive ``probability_weight`` breaks equal-edit ties in favour
        of transition paths with higher empirical probability.
        """

        values = np.asarray(proposed, dtype=np.int64)
        if values.ndim != 1 or len(values) == 0:
            raise ValueError("proposed must be a non-empty one-dimensional path")
        if ((values < 0) | (values >= self.num_nodes)).any():
            raise ValueError("proposed contains an invalid node id")
        length = len(values)
        permitted = (
            np.ones((length, self.num_nodes), dtype=bool)
            if allowed is None
            else np.asarray(allowed, dtype=bool).copy()
        )
        if permitted.shape != (length, self.num_nodes):
            raise ValueError(f"allowed must have shape {(length, self.num_nodes)}")
        for position, node in (anchors or {}).items():
            if not 0 <= position < length or not 0 <= node < self.num_nodes:
                raise ValueError(f"Invalid anchor {position}: {node}")
            permitted[position] = False
            permitted[position, node] = True

        infinity = np.inf
        cost = np.full((length, self.num_nodes), infinity)
        back = np.full((length, self.num_nodes), -1, dtype=np.int64)
        cost[0, permitted[0]] = change_cost * (np.arange(self.num_nodes)[permitted[0]] != values[0])
        log_matrix = self.log_matrix(log_floor)

        for position in range(1, length):
            for target in np.flatnonzero(permitted[position]):
                sources = np.flatnonzero((self.transition_matrix[:, target] > 0) & np.isfinite(cost[position - 1]))
                if len(sources) == 0:
                    continue
                candidate_costs = cost[position - 1, sources]
                if probability_weight:
                    candidate_costs = candidate_costs - probability_weight * log_matrix[sources, target]
                best_index = int(np.argmin(candidate_costs))
                source = int(sources[best_index])
                cost[position, target] = candidate_costs[best_index] + change_cost * (
                    target != values[position]
                )
                back[position, target] = source

        if not np.isfinite(cost[-1]).any():
            raise ValueError("No valid transition path satisfies the supplied constraints")
        node = int(np.argmin(cost[-1]))
        path = np.empty(length, dtype=np.int64)
        path[-1] = node
        for position in range(length - 1, 0, -1):
            node = int(back[position, node])
            if node < 0:
                raise RuntimeError("Transition path backtracking failed")
            path[position - 1] = node
        return path, int(np.count_nonzero(path != values))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TransitionGraph":
        nodes = tuple(str(item) for item in value["node_order"])
        edges = tuple(
            TransitionEdge(
                source=str(edge["source"]),
                target=str(edge["target"]),
                count=int(edge.get("count", 0)),
                probability=float(edge["probability"]),
            )
            for edge in value.get("edges", [])
        )
        chapter_paths_value = value.get("chapter_paths", [])
        if isinstance(chapter_paths_value, Mapping):
            chapter_paths_iter: Iterable[Sequence[str]] = chapter_paths_value.values()
        else:
            chapter_paths_iter = chapter_paths_value
        known_keys = {
            "node_order",
            "transition_matrix",
            "edges",
            "original_path",
            "chapter_paths",
        }
        return cls(
            node_order=nodes,
            transition_matrix=np.asarray(value["transition_matrix"], dtype=np.float64),
            edges=edges,
            original_path=tuple(str(item) for item in value.get("original_path", [])),
            chapter_paths=tuple(tuple(str(item) for item in path) for path in chapter_paths_iter),
            metadata={key: item for key, item in value.items() if key not in known_keys},
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "TransitionGraph":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise TypeError("Transition Graph JSON must contain an object")
        return cls.from_dict(value)

    @classmethod
    def from_path(cls, path: Sequence[str]) -> "TransitionGraph":
        if not path:
            raise ValueError("path cannot be empty")
        nodes = tuple(dict.fromkeys(path))
        ids = {name: index for index, name in enumerate(nodes)}
        counts = np.zeros((len(nodes), len(nodes)), dtype=np.int64)
        for source, target in zip(path, path[1:]):
            counts[ids[source], ids[target]] += 1
        totals = counts.sum(axis=1, keepdims=True)
        matrix = np.divide(counts, totals, out=np.zeros_like(counts, dtype=float), where=totals > 0)
        edges = tuple(
            TransitionEdge(nodes[i], nodes[j], int(counts[i, j]), float(matrix[i, j]))
            for i, j in zip(*np.nonzero(counts))
        )
        return cls(nodes, matrix, edges, tuple(path))
