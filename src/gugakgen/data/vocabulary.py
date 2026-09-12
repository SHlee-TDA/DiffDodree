"""Stable token mappings for pitch, duration, and structural families."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Generic, Iterable, Mapping, Sequence, TypeVar

import numpy as np

from gugakgen.data.records import EventRecord, MeasureRecord, ScoreRecord, TokenizedScore, as_fraction

T = TypeVar("T")

REST = "REST"
PAD = "PAD"
MASK = "MASK"


@dataclass(frozen=True)
class TokenVocabulary(Generic[T]):
    values: tuple[T, ...]
    _ids: Mapping[T, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if len(self.values) != len(set(self.values)):
            raise ValueError("Vocabulary values must be unique")
        object.__setattr__(self, "_ids", {value: index for index, value in enumerate(self.values)})

    def __len__(self) -> int:
        return len(self.values)

    def encode(self, value: T) -> int:
        try:
            return self._ids[value]
        except KeyError as error:
            raise KeyError(f"Value {value!r} is not present in the vocabulary") from error

    def decode(self, token_id: int) -> T:
        if token_id < 0 or token_id >= len(self.values):
            raise IndexError(f"Token id {token_id} is outside [0, {len(self.values)})")
        return self.values[token_id]

    def contains(self, value: T) -> bool:
        return value in self._ids


@dataclass(frozen=True)
class ScoreVocabulary:
    """Vocabulary whose ordering is compatible with the frozen v0 artifacts."""

    pitch_values: tuple[int, ...]
    duration_values: tuple[Fraction, ...]
    family_values: tuple[str, ...]
    max_events_per_measure: int
    _pitch_ids: Mapping[int, int] = field(init=False, repr=False, compare=False)
    _duration_ids: Mapping[Fraction, int] = field(init=False, repr=False, compare=False)
    _family_ids: Mapping[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.pitch_values or not self.duration_values or not self.family_values:
            raise ValueError("Pitch, duration, and family vocabularies cannot be empty")
        if self.max_events_per_measure <= 0:
            raise ValueError("max_events_per_measure must be positive")
        if len(set(self.pitch_values)) != len(self.pitch_values):
            raise ValueError("Pitch values must be unique")
        if len(set(self.duration_values)) != len(self.duration_values):
            raise ValueError("Duration values must be unique")
        if len(set(self.family_values)) != len(self.family_values):
            raise ValueError("Family values must be unique")

        object.__setattr__(self, "_pitch_ids", {v: i for i, v in enumerate(self.pitch_values)})
        object.__setattr__(self, "_duration_ids", {v: i for i, v in enumerate(self.duration_values)})
        object.__setattr__(self, "_family_ids", {v: i for i, v in enumerate(self.family_values)})

    @property
    def pitch_rest_id(self) -> int:
        return len(self.pitch_values)

    @property
    def pitch_pad_id(self) -> int:
        return len(self.pitch_values) + 1

    @property
    def pitch_mask_id(self) -> int:
        return len(self.pitch_values) + 2

    @property
    def duration_pad_id(self) -> int:
        return len(self.duration_values)

    @property
    def duration_mask_id(self) -> int:
        return len(self.duration_values) + 1

    @property
    def family_mask_id(self) -> int:
        return len(self.family_values)

    @property
    def num_pitch_tokens(self) -> int:
        return len(self.pitch_values) + 3

    @property
    def num_duration_tokens(self) -> int:
        return len(self.duration_values) + 2

    @property
    def num_family_tokens(self) -> int:
        return len(self.family_values) + 1

    @property
    def num_families(self) -> int:
        return len(self.family_values)

    def encode_pitch(self, pitch: int | None) -> int:
        if pitch is None:
            return self.pitch_rest_id
        try:
            return self._pitch_ids[pitch]
        except KeyError as error:
            raise KeyError(f"Pitch {pitch} is outside the expert-validated vocabulary") from error

    def decode_pitch(self, token_id: int) -> int | None:
        if token_id == self.pitch_rest_id:
            return None
        if token_id in (self.pitch_pad_id, self.pitch_mask_id):
            raise ValueError(f"Token {token_id} is not a musical pitch event")
        if token_id < 0 or token_id >= len(self.pitch_values):
            raise IndexError(f"Pitch token {token_id} is out of range")
        return self.pitch_values[token_id]

    def encode_duration(self, duration_q: int | float | str | Fraction) -> int:
        duration = as_fraction(duration_q)
        try:
            return self._duration_ids[duration]
        except KeyError as error:
            raise KeyError(f"Duration {float(duration)} is outside the vocabulary") from error

    def decode_duration(self, token_id: int) -> Fraction:
        if token_id in (self.duration_pad_id, self.duration_mask_id):
            raise ValueError(f"Token {token_id} is not a musical duration")
        if token_id < 0 or token_id >= len(self.duration_values):
            raise IndexError(f"Duration token {token_id} is out of range")
        return self.duration_values[token_id]

    def encode_family(self, family: str) -> int:
        try:
            return self._family_ids[family]
        except KeyError as error:
            raise KeyError(f"Unknown family {family!r}") from error

    def decode_family(self, token_id: int) -> str:
        if token_id == self.family_mask_id:
            return MASK
        if token_id < 0 or token_id >= len(self.family_values):
            raise IndexError(f"Family token {token_id} is out of range")
        return self.family_values[token_id]

    def encode_score(
        self,
        score: ScoreRecord,
        *,
        original_path: Sequence[str] | None = None,
        num_position_buckets: int = 4,
        position_encoding: str = "relative_chapter_quartile",
    ) -> TokenizedScore:
        measures = score.measures
        measure_count = len(measures)
        slots = self.max_events_per_measure
        pitch = np.full((measure_count, slots), self.pitch_pad_id, dtype=np.int64)
        duration = np.full((measure_count, slots), self.duration_pad_id, dtype=np.int64)
        family = np.empty(measure_count, dtype=np.int64)
        chapter = np.empty(measure_count, dtype=np.int64)
        meter_values: list[tuple[int, int]] = []
        event_counts = np.empty(measure_count, dtype=np.int64)
        target_beats = np.empty(measure_count, dtype=np.float32)

        if original_path is not None and len(original_path) != measure_count:
            raise ValueError("original_path length must equal the number of measures")

        for index, measure in enumerate(measures):
            if len(measure.events) > slots:
                raise ValueError(
                    f"Measure {measure.number} has {len(measure.events)} events; maximum is {slots}"
                )
            for slot, event in enumerate(measure.events):
                pitch[index, slot] = self.encode_pitch(event.pitch)
                duration[index, slot] = self.encode_duration(event.duration_q)
            label = original_path[index] if original_path is not None else measure.family
            if label is None:
                raise ValueError(f"Measure {measure.number} has no family label")
            family[index] = self.encode_family(label)
            chapter[index] = measure.chapter - 1
            meter_values.append((measure.meter.numerator, measure.meter.denominator))
            event_counts[index] = len(measure.events)
            target_beats[index] = float(measure.meter.quarter_length)

        unique_meters = list(dict.fromkeys(meter_values))
        meter_id = {value: index for index, value in enumerate(unique_meters)}
        meter = np.asarray([meter_id[value] for value in meter_values], dtype=np.int64)
        position = self._positions(measures, num_position_buckets, position_encoding)
        return TokenizedScore(
            pitch=pitch,
            duration=duration,
            family=family,
            chapter=chapter,
            meter=meter,
            position=position,
            event_counts=event_counts,
            target_beats=target_beats,
        )

    def decode_measure(
        self,
        pitch_tokens: Sequence[int],
        duration_tokens: Sequence[int],
        *,
        number: int,
        chapter: int,
        meter: Iterable[int],
        family_token: int | None = None,
    ) -> MeasureRecord:
        if len(pitch_tokens) != len(duration_tokens):
            raise ValueError("Pitch and duration rows must have equal lengths")
        events: list[EventRecord] = []
        pad_started = False
        for slot, (pitch_token, duration_token) in enumerate(zip(pitch_tokens, duration_tokens)):
            pitch_id, duration_id = int(pitch_token), int(duration_token)
            is_pad = pitch_id == self.pitch_pad_id and duration_id == self.duration_pad_id
            if is_pad:
                pad_started = True
                continue
            if pad_started:
                raise ValueError(f"Non-PAD token after PAD suffix at slot {slot}")
            if pitch_id in (self.pitch_pad_id, self.pitch_mask_id):
                raise ValueError(f"Invalid pitch token pair at slot {slot}: {pitch_id}, {duration_id}")
            if duration_id in (self.duration_pad_id, self.duration_mask_id):
                raise ValueError(f"Invalid duration token pair at slot {slot}: {pitch_id}, {duration_id}")
            events.append(EventRecord(self.decode_pitch(pitch_id), self.decode_duration(duration_id)))
        if not events:
            raise ValueError("Decoded measure contains no events")
        family = None if family_token is None else self.decode_family(int(family_token))
        return MeasureRecord(number, chapter, meter, tuple(events), family=family)

    @staticmethod
    def _positions(
        measures: Sequence[MeasureRecord], num_buckets: int, encoding: str
    ) -> np.ndarray:
        if num_buckets <= 0:
            raise ValueError("num_position_buckets must be positive")
        result = np.zeros(len(measures), dtype=np.int64)
        chapter_indices: dict[int, list[int]] = {}
        for index, measure in enumerate(measures):
            chapter_indices.setdefault(measure.chapter, []).append(index)
        for indices in chapter_indices.values():
            length = len(indices)
            for local_index, global_index in enumerate(indices):
                if encoding == "relative_chapter_quartile":
                    result[global_index] = min(local_index * num_buckets // length, num_buckets - 1)
                elif encoding == "absolute_window":
                    result[global_index] = min(local_index // num_buckets, num_buckets - 1)
                else:
                    raise ValueError(f"Unknown position encoding: {encoding}")
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "pitch": [*self.pitch_values, REST, PAD, MASK],
            "duration": [*[float(value) for value in self.duration_values], PAD, MASK],
            "family": [*self.family_values, MASK],
            "max_events_per_measure": self.max_events_per_measure,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ScoreVocabulary":
        pitch_raw = list(value["pitch"])
        duration_raw = list(value["duration"])
        family_raw = list(value["family"])
        expected_pitch_suffix = [REST, PAD, MASK]
        expected_duration_suffix = [PAD, MASK]
        if pitch_raw[-3:] != expected_pitch_suffix:
            raise ValueError(f"Pitch vocabulary must end with {expected_pitch_suffix}")
        if duration_raw[-2:] != expected_duration_suffix:
            raise ValueError(f"Duration vocabulary must end with {expected_duration_suffix}")
        if family_raw[-1:] != [MASK]:
            raise ValueError("Family vocabulary must end with MASK")
        return cls(
            pitch_values=tuple(int(item) for item in pitch_raw[:-3]),
            duration_values=tuple(as_fraction(item) for item in duration_raw[:-2]),
            family_values=tuple(str(item) for item in family_raw[:-1]),
            max_events_per_measure=int(value["max_events_per_measure"]),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ScoreVocabulary":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise TypeError("Vocabulary JSON must contain an object")
        return cls.from_dict(value)
