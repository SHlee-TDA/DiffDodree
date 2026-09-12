"""Validated records used at the symbolic-score boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Iterable, Mapping

import numpy as np
from numpy.typing import NDArray


def as_fraction(value: int | float | str | Fraction) -> Fraction:
    """Convert serialized duration values without binary floating-point drift."""

    if isinstance(value, bool):
        raise TypeError("A boolean is not a valid musical duration")
    if isinstance(value, Fraction):
        result = value
    else:
        result = Fraction(str(value)).limit_denominator(384)
    if result <= 0:
        raise ValueError(f"Duration must be positive, got {value!r}")
    return result


@dataclass(frozen=True, slots=True)
class Meter:
    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if self.numerator <= 0 or self.denominator <= 0:
            raise ValueError("Meter values must be positive")

    @property
    def quarter_length(self) -> Fraction:
        return Fraction(self.numerator * 4, self.denominator)

    def to_list(self) -> list[int]:
        return [self.numerator, self.denominator]

    @classmethod
    def from_value(cls, value: "Meter | Iterable[int]") -> "Meter":
        if isinstance(value, cls):
            return value
        items = tuple(int(item) for item in value)
        if len(items) != 2:
            raise ValueError(f"Meter requires two values, got {items!r}")
        return cls(*items)


@dataclass(frozen=True, slots=True)
class EventRecord:
    """One monophonic symbolic event; ``pitch=None`` denotes a rest."""

    pitch: int | None
    duration_q: Fraction

    def __post_init__(self) -> None:
        if self.pitch is not None and (isinstance(self.pitch, bool) or not isinstance(self.pitch, int)):
            raise TypeError("pitch must be an integer MIDI value or None")
        object.__setattr__(self, "duration_q", as_fraction(self.duration_q))

    def to_dict(self) -> dict[str, int | float | None]:
        return {"pitch": self.pitch, "duration_q": float(self.duration_q)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EventRecord":
        pitch = value.get("pitch")
        return cls(None if pitch is None else int(pitch), as_fraction(value["duration_q"]))


@dataclass(frozen=True, slots=True)
class MeasureRecord:
    number: int
    chapter: int
    meter: Meter
    events: tuple[EventRecord, ...]
    family: str | None = None
    source_file: str | None = None
    source_measure: int | None = None
    annotations: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.number <= 0 or self.chapter <= 0:
            raise ValueError("Measure number and chapter must be positive")
        object.__setattr__(self, "meter", Meter.from_value(self.meter))
        object.__setattr__(self, "events", tuple(self.events))
        if not self.events:
            raise ValueError(f"Measure {self.number} contains no duration-bearing events")

    @property
    def duration_q(self) -> Fraction:
        return sum((event.duration_q for event in self.events), start=Fraction(0))

    @property
    def meter_error_q(self) -> Fraction:
        return self.duration_q - self.meter.quarter_length

    def validate_meter(self) -> None:
        if self.meter_error_q != 0:
            raise ValueError(
                f"Measure {self.number} totals {float(self.duration_q)} quarter notes; "
                f"expected {float(self.meter.quarter_length)}"
            )

    def signature(self) -> tuple[tuple[int | None, Fraction], ...]:
        return tuple((event.pitch, event.duration_q) for event in self.events)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "global_measure": self.number,
            "chapter": self.chapter,
            "meter": self.meter.to_list(),
            "events": [event.to_dict() for event in self.events],
        }
        if self.family is not None:
            result["family"] = self.family
        if self.source_file is not None:
            result["source_file"] = self.source_file
        if self.source_measure is not None:
            result["source_measure"] = self.source_measure
        if self.annotations:
            result["annotations"] = dict(self.annotations)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MeasureRecord":
        return cls(
            number=int(value.get("global_measure", value.get("number"))),
            chapter=int(value.get("chapter", 1)),
            meter=Meter.from_value(value["meter"]),
            events=tuple(EventRecord.from_dict(event) for event in value["events"]),
            family=value.get("family"),
            source_file=value.get("source_file"),
            source_measure=(
                None if value.get("source_measure") is None else int(value["source_measure"])
            ),
            annotations=dict(value.get("annotations", {})),
        )


@dataclass(frozen=True, slots=True)
class ScoreRecord:
    piece_id: str
    part: str
    measures: tuple[MeasureRecord, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.piece_id or not self.part:
            raise ValueError("piece_id and part are required")
        object.__setattr__(self, "measures", tuple(self.measures))
        if not self.measures:
            raise ValueError("A score must contain at least one measure")
        numbers = [measure.number for measure in self.measures]
        if len(numbers) != len(set(numbers)):
            raise ValueError("Measure numbers must be unique")

    def validate_meters(self) -> None:
        for measure in self.measures:
            measure.validate_meter()

    def to_measure_dicts(self) -> list[dict[str, Any]]:
        return [measure.to_dict() for measure in self.measures]


@dataclass(frozen=True, slots=True)
class TokenizedScore:
    """Dense model tensors represented as NumPy arrays.

    Shapes are pitch/duration ``(T, S)`` and all measure-level arrays ``(T,)``.
    """

    pitch: NDArray[np.int64]
    duration: NDArray[np.int64]
    family: NDArray[np.int64]
    chapter: NDArray[np.int64]
    meter: NDArray[np.int64]
    position: NDArray[np.int64]
    event_counts: NDArray[np.int64]
    target_beats: NDArray[np.float32]

    def __post_init__(self) -> None:
        pitch = np.asarray(self.pitch, dtype=np.int64)
        duration = np.asarray(self.duration, dtype=np.int64)
        if pitch.ndim != 2 or pitch.shape != duration.shape:
            raise ValueError("pitch and duration must have identical (T, S) shapes")
        measure_count = pitch.shape[0]
        for name in ("family", "chapter", "meter", "position", "event_counts", "target_beats"):
            value = np.asarray(getattr(self, name))
            if value.shape != (measure_count,):
                raise ValueError(f"{name} must have shape ({measure_count},), got {value.shape}")

        object.__setattr__(self, "pitch", pitch)
        object.__setattr__(self, "duration", duration)
        object.__setattr__(self, "family", np.asarray(self.family, dtype=np.int64))
        object.__setattr__(self, "chapter", np.asarray(self.chapter, dtype=np.int64))
        object.__setattr__(self, "meter", np.asarray(self.meter, dtype=np.int64))
        object.__setattr__(self, "position", np.asarray(self.position, dtype=np.int64))
        object.__setattr__(self, "event_counts", np.asarray(self.event_counts, dtype=np.int64))
        object.__setattr__(self, "target_beats", np.asarray(self.target_beats, dtype=np.float32))

    @property
    def num_measures(self) -> int:
        return int(self.pitch.shape[0])

    @property
    def max_events(self) -> int:
        return int(self.pitch.shape[1])

    def copy(self) -> "TokenizedScore":
        return TokenizedScore(
            **{
                name: np.asarray(getattr(self, name)).copy()
                for name in (
                    "pitch",
                    "duration",
                    "family",
                    "chapter",
                    "meter",
                    "position",
                    "event_counts",
                    "target_beats",
                )
            }
        )
