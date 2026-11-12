"""Phrases as tokens on a fixed grid of cells per beat (generator v1).

A phrase of ``B`` beats becomes ``R * B`` cells. Each cell holds one token:
the onset of one of the piece's pitches, the onset of a rest, or ``hold``
(the previous note or rest continues). Every token string whose first cell is
not ``hold`` is a valid rhythm, and its total length is fixed by the number of
cells, so the phrase total ``B_t`` holds by construction.

``R`` (cells per beat) is a setting. The 양청도드리 MusicXML needs 4 (onsets at
0, 1/2 and 3/4 of a quarter). Whether the 정간보 unit of this piece is 3 소박
per beat is open; with ``R = 3`` the current transcription cannot be encoded
and ``encode`` raises.

The sounding state of a cell is the pitch (or rest) that is heard there:
states ``0..P-1`` are the scale's pitches and ``P`` is a rest.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gugakgen.chart.edit_distance import REST_PITCH_COST, Event


@dataclass(frozen=True)
class Grid:
    """Token alphabet for one piece: ``P`` pitches, then ON_REST, HOLD, MASK."""

    pitches: tuple[int, ...]  # ascending MIDI numbers of the piece's scale
    cells_per_beat: int

    @property
    def num_pitches(self) -> int:
        return len(self.pitches)

    @property
    def on_rest(self) -> int:
        return self.num_pitches

    @property
    def hold(self) -> int:
        return self.num_pitches + 1

    @property
    def mask(self) -> int:
        return self.num_pitches + 2

    @property
    def num_outputs(self) -> int:  # what the denoiser predicts (never MASK)
        return self.num_pitches + 2

    @property
    def rest_state(self) -> int:
        return self.num_pitches

    @property
    def num_states(self) -> int:
        return self.num_pitches + 1

    @classmethod
    def from_phrases(cls, phrases: Sequence[Sequence[Event]], cells_per_beat: int) -> Grid:
        pitches = sorted({p for phrase in phrases for p, _ in phrase if p is not None})
        return cls(tuple(pitches), cells_per_beat)

    def cells(self, beats: float) -> int:
        count = round(beats * self.cells_per_beat)
        if abs(count - beats * self.cells_per_beat) > 1e-6:
            raise ValueError(f"{beats} beats is not a whole number of cells at R={self.cells_per_beat}")
        return count

    def encode(self, phrase: Sequence[Event]) -> list[int]:
        tokens: list[int] = []
        for pitch, duration in phrase:
            count = self.cells(duration)
            if count <= 0:
                raise ValueError("Every event needs at least one cell")
            tokens.append(self.on_rest if pitch is None else self.pitches.index(pitch))
            tokens.extend([self.hold] * (count - 1))
        return tokens

    def decode(self, tokens: Sequence[int]) -> list[Event]:
        events: list[list] = []
        for token in tokens:
            token = int(token)
            if token == self.hold:
                if not events:
                    raise ValueError("A phrase cannot start with hold")
                events[-1][1] += 1
            elif token == self.on_rest:
                events.append([None, 1])
            elif 0 <= token < self.num_pitches:
                events.append([self.pitches[token], 1])
            else:
                raise ValueError(f"Token {token} is not a decoded token")
        return [(pitch, count / self.cells_per_beat) for pitch, count in events]

    def sounding(self, tokens: Sequence[int]) -> list[int | None]:
        """Sounding state per cell; ``None`` where it depends on a masked cell."""

        states: list[int | None] = []
        current: int | None = None
        for token in tokens:
            token = int(token)
            if token == self.mask:
                current = None
            elif token == self.on_rest:
                current = self.rest_state
            elif token != self.hold:
                current = token
            states.append(current)
        return states

    def contour(self, phrase: Sequence[Event]) -> list[int]:
        """Pitch index per cell; rests take the nearest pitched cell (Methods eq. 6)."""

        states = self.sounding(self.encode(phrase))
        pitched = [j for j, s in enumerate(states) if s is not None and s != self.rest_state]
        if not pitched:
            raise ValueError("A reference phrase needs a pitched note")
        return [
            s if s is not None and s != self.rest_state else states[min(pitched, key=lambda i: (abs(i - j), i))]
            for j, s in enumerate(states)
        ]

    def phase_index(self, length: int, cells: int) -> NDArray[np.int64]:
        """Index into a reference of ``length`` cells for each of ``cells`` cells (phase u = j / cells)."""

        return np.floor(np.arange(cells) * length / cells).astype(np.int64)

    def reference(self, medoid: Sequence[Event], cells: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Medoid pitch index and medoid-onset flag at each cell, stretched to ``cells`` cells."""

        tokens = self.encode(medoid)
        contour = np.array(self.contour(medoid))
        onset = np.array([int(t != self.hold) for t in tokens])
        index = self.phase_index(len(tokens), cells)
        return contour[index], onset[index]

    def contour_costs(self, medoids: Sequence[Sequence[Event]], cells: int) -> NDArray[np.float32]:
        """``cost[c, j, s]``: semitone distance of state ``s`` at cell ``j`` from medoid ``c``."""

        pitches = np.array(self.pitches, dtype=np.float32)
        cost = np.zeros((len(medoids), cells, self.num_states), dtype=np.float32)
        for c, medoid in enumerate(medoids):
            reference, _ = self.reference(medoid, cells)
            cost[c, :, : self.num_pitches] = np.abs(pitches[None, :] - pitches[reference][:, None])
            cost[c, :, self.rest_state] = REST_PITCH_COST
        return cost
