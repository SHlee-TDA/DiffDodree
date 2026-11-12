"""New type sequences Y' as first-order Markov walks on the phrase graph G.

The transition probabilities are the row-normalised transition counts of the
original (the maximum-likelihood first-order chain). A walk has the original
length, starts with the original first type and ends with the original last
type. The return relation R ties a block of positions to an earlier block
(Methods eq. 5); both blocks must carry the same types, which is a
long-range condition a Markov chain cannot impose step by step. Walks are
therefore drawn exactly from the chain conditioned on all of this: the block's
types are drawn first with their exact weight, then the gaps are filled by
bridge sampling between fixed end points.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


class TypeWalk:
    def __init__(
        self,
        counts: NDArray[np.int64],
        length: int,
        start: int,
        end: int,
        returns: Sequence[tuple[int, int]] = (),
    ) -> None:
        counts = np.asarray(counts, dtype=np.float64)
        rows = counts.sum(axis=1, keepdims=True)
        if (rows == 0).any():
            raise ValueError("Every type needs an outgoing transition")
        self.P = counts / rows
        self.length, self.start, self.end = length, start, end
        self.powers = [np.eye(len(counts))]
        for _ in range(length):
            self.powers.append(self.powers[-1] @ self.P)
        self.block = self._block(returns)
        self.blocks, self.block_weights = self._enumerate_blocks()

    @staticmethod
    def _block(returns: Sequence[tuple[int, int]]) -> tuple[int, int, int] | None:
        """(source start, target start, block length) of one contiguous return block."""

        if not returns:
            return None
        pairs = sorted(returns)
        source, target = pairs[0]
        size = len(pairs)
        if pairs != [(source + i, target + i) for i in range(size)] or source + size > target:
            raise ValueError("Returns must form one contiguous, non-overlapping block")
        return source, target, size

    def _enumerate_blocks(self) -> tuple[list[tuple[int, ...]], NDArray[np.float64]]:
        if self.block is None:
            return [()], np.ones(1)
        source, target, size = self.block
        P, power = self.P, self.powers
        sequences: list[tuple[int, ...]] = [(a,) for a in range(len(P))]
        for _ in range(size - 1):
            sequences = [s + (b,) for s in sequences for b in np.flatnonzero(P[s[-1]] > 0)]
        weights = []
        for s in sequences:
            inner = np.prod([P[a, b] for a, b in zip(s[:-1], s[1:], strict=True)]) if size > 1 else 1.0
            lead = power[source][self.start, s[0]]
            gap = power[target - (source + size - 1)][s[-1], s[0]]
            tail = power[self.length - 1 - (target + size - 1)][s[-1], self.end]
            weights.append(lead * inner * gap * inner * tail)
        weights = np.array(weights)
        if weights.sum() == 0:
            raise ValueError("No walk satisfies the start, end and return conditions")
        keep = weights > 0
        return [s for s, k in zip(sequences, keep, strict=True) if k], weights[keep] / weights[keep].sum()

    def _bridge(self, rng: np.random.Generator, left: int, right: int, a: int, b: int) -> list[int]:
        """Types at positions left+1 .. right-1 given type ``a`` at ``left`` and ``b`` at ``right``."""

        out, current = [], a
        for position in range(left + 1, right):
            weights = self.P[current] * self.powers[right - position][:, b]
            current = int(rng.choice(len(weights), p=weights / weights.sum()))
            out.append(current)
        return out

    def sample(self, rng: np.random.Generator) -> list[int]:
        walk = [-1] * self.length
        walk[0], walk[-1] = self.start, self.end
        fixed = {0: self.start, self.length - 1: self.end}
        if self.block is not None:
            source, target, size = self.block
            block = self.blocks[int(rng.choice(len(self.blocks), p=self.block_weights))]
            for i, value in enumerate(block):
                if (source + i in fixed and fixed[source + i] != value) or (
                    target + i in fixed and fixed[target + i] != value
                ):
                    raise RuntimeError("Inconsistent block")  # excluded by the weights
                fixed[source + i] = fixed[target + i] = value
        anchors = sorted(fixed)
        for position in anchors:
            walk[position] = fixed[position]
        for left, right in zip(anchors[:-1], anchors[1:], strict=True):
            walk[left + 1 : right] = self._bridge(rng, left, right, fixed[left], fixed[right])
        return walk
