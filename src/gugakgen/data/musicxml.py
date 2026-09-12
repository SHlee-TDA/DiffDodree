"""Monophonic MusicXML parsing, export, and token conversion.

The parser intentionally preserves the expert-transcribed MIDI pitches and
quarter-note durations. It does not infer or synthesize 시김새, 농현, or timing.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from gugakgen.data.records import EventRecord, MeasureRecord, Meter, ScoreRecord, TokenizedScore
from gugakgen.data.vocabulary import ScoreVocabulary

_STEP_TO_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_SHARP_SPELLING = {
    0: ("C", 0),
    1: ("C", 1),
    2: ("D", 0),
    3: ("D", 1),
    4: ("E", 0),
    5: ("F", 0),
    6: ("F", 1),
    7: ("G", 0),
    8: ("G", 1),
    9: ("A", 0),
    10: ("A", 1),
    11: ("B", 0),
}
_FLAT_SPELLING = {
    0: ("C", 0),
    1: ("D", -1),
    2: ("D", 0),
    3: ("E", -1),
    4: ("E", 0),
    5: ("F", 0),
    6: ("G", -1),
    7: ("G", 0),
    8: ("A", -1),
    9: ("A", 0),
    10: ("B", -1),
    11: ("B", 0),
}


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in element if _local_name(child.tag) == name]


def _child(element: ET.Element, name: str) -> ET.Element | None:
    return next((child for child in element if _local_name(child.tag) == name), None)


def _descendants(element: ET.Element, name: str) -> list[ET.Element]:
    return [item for item in element.iter() if _local_name(item.tag) == name]


def _required_text(element: ET.Element, name: str, *, context: str) -> str:
    child = _child(element, name)
    if child is None or child.text is None:
        raise ValueError(f"Missing <{name}> in {context}")
    return child.text.strip()


def _measure_chapter(number: int, chapter_ranges: Sequence[Sequence[int]] | None) -> int:
    if chapter_ranges is None:
        return 1
    for chapter, limits in enumerate(chapter_ranges, start=1):
        if len(limits) != 2:
            raise ValueError(f"Invalid chapter range: {limits!r}")
        if int(limits[0]) <= number <= int(limits[1]):
            return chapter
    raise ValueError(f"Measure {number} is outside the configured chapter ranges")


def _select_part(root: ET.Element, part_id: str | None) -> ET.Element:
    parts = _descendants(root, "part")
    if not parts:
        raise ValueError("MusicXML contains no <part>")
    if part_id is None:
        if len(parts) > 1:
            raise ValueError("MusicXML has multiple parts; specify part_id explicitly")
        return parts[0]
    for part in parts:
        if part.get("id") == part_id:
            return part
    raise ValueError(f"MusicXML contains no part with id={part_id!r}")


def _parse_pitch(note: ET.Element, *, context: str) -> int:
    pitch = _child(note, "pitch")
    if pitch is None:
        raise ValueError(f"Pitched note has no <pitch> in {context}")
    step = _required_text(pitch, "step", context=context)
    octave = int(_required_text(pitch, "octave", context=context))
    alter_element = _child(pitch, "alter")
    alter = Fraction(alter_element.text.strip()) if alter_element is not None and alter_element.text else 0
    if alter.denominator != 1:
        raise ValueError(
            f"Microtonal alter={float(alter)} in {context} cannot be represented by MIDI integers"
        )
    try:
        semitone = _STEP_TO_SEMITONE[step]
    except KeyError as error:
        raise ValueError(f"Unknown pitch step {step!r} in {context}") from error
    return 12 * (octave + 1) + semitone + int(alter)


def parse_musicxml(
    path: str | Path,
    *,
    piece_id: str,
    part_name: str,
    part_id: str | None = None,
    measure_offset: int = 0,
    default_meter: Meter | Iterable[int] | None = None,
    chapter_ranges: Sequence[Sequence[int]] | None = None,
    reject_chords: bool = True,
) -> ScoreRecord:
    """Parse one monophonic MusicXML file into validated records.

    Grace notes without a duration are ignored. Tied notes remain separate
    symbolic events, matching the original M0 pipeline. Polyphonic ``<chord>``
    notes are rejected by default because the Phase 1 representation is
    monophonic.
    """

    source = Path(path)
    root = ET.parse(source).getroot()
    part = _select_part(root, part_id)
    current_divisions: int | None = None
    current_meter = None if default_meter is None else Meter.from_value(default_meter)
    records: list[MeasureRecord] = []

    for ordinal, measure_element in enumerate(_children(part, "measure"), start=1):
        source_number_text = measure_element.get("number", str(ordinal))
        try:
            source_number = int(source_number_text)
        except ValueError:
            source_number = ordinal
        global_number = measure_offset + ordinal
        context = f"{source.name}, measure {source_number_text}"

        attributes = _child(measure_element, "attributes")
        if attributes is not None:
            divisions_element = _child(attributes, "divisions")
            if divisions_element is not None and divisions_element.text:
                current_divisions = int(divisions_element.text)
                if current_divisions <= 0:
                    raise ValueError(f"divisions must be positive in {context}")
            time_element = _child(attributes, "time")
            if time_element is not None:
                current_meter = Meter(
                    int(_required_text(time_element, "beats", context=context)),
                    int(_required_text(time_element, "beat-type", context=context)),
                )

        if current_divisions is None:
            raise ValueError(f"No MusicXML divisions value is active in {context}")
        if current_meter is None:
            raise ValueError(f"No meter is active and no default_meter was supplied in {context}")

        if _children(measure_element, "backup") or _children(measure_element, "forward"):
            raise ValueError(f"Multiple voices are not supported in Phase 1 ({context})")

        events: list[EventRecord] = []
        lyric_count = 0
        breath_count = 0
        for note in _children(measure_element, "note"):
            if _child(note, "chord") is not None:
                if reject_chords:
                    raise ValueError(f"Chord encountered in monophonic parser ({context})")
                continue
            duration_element = _child(note, "duration")
            if duration_element is None or duration_element.text is None:
                # Grace/cue notes without duration are outside the Phase 1 token stream.
                continue
            duration = Fraction(int(duration_element.text), current_divisions)
            is_rest = _child(note, "rest") is not None
            pitch = None if is_rest else _parse_pitch(note, context=context)
            events.append(EventRecord(pitch=pitch, duration_q=duration))
            lyric_count += len(_children(note, "lyric"))
            breath_count += len(_descendants(note, "breath-mark"))

        if not events:
            raise ValueError(f"No duration-bearing notes found in {context}")
        records.append(
            MeasureRecord(
                number=global_number,
                chapter=_measure_chapter(global_number, chapter_ranges),
                meter=current_meter,
                events=tuple(events),
                source_file=source.name,
                source_measure=source_number,
                annotations={"lyric_count": lyric_count, "breath_mark_count": breath_count},
            )
        )

    if not records:
        raise ValueError(f"MusicXML contains no measures: {source}")
    return ScoreRecord(
        piece_id=piece_id,
        part=part_name,
        measures=tuple(records),
        metadata={"source_files": [str(source)]},
    )


def combine_scores(
    scores: Sequence[ScoreRecord], *, piece_id: str | None = None, part: str | None = None
) -> ScoreRecord:
    """Concatenate already-numbered score segments after consistency checks."""

    if not scores:
        raise ValueError("At least one score segment is required")
    resolved_piece = piece_id or scores[0].piece_id
    resolved_part = part or scores[0].part
    for score in scores:
        if score.piece_id != resolved_piece or score.part != resolved_part:
            raise ValueError("All score segments must have the same piece_id and part")
    measures = tuple(measure for score in scores for measure in score.measures)
    sources = [source for score in scores for source in score.metadata.get("source_files", [])]
    return ScoreRecord(resolved_piece, resolved_part, measures, {"source_files": sources})


def load_measure_artifact(
    path: str | Path,
    *,
    piece_id: str,
    part: str,
    original_path: Sequence[str] | None = None,
) -> ScoreRecord:
    """Load the frozen ``measures_v0.json`` representation."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise TypeError("Measure artifact must contain a JSON list")
    if original_path is not None and len(original_path) != len(raw):
        raise ValueError("original_path length does not match measure artifact")
    measures: list[MeasureRecord] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise TypeError(f"Measure artifact item {index} is not an object")
        value = dict(item)
        if original_path is not None:
            value["family"] = original_path[index]
        measures.append(MeasureRecord.from_dict(value))
    return ScoreRecord(piece_id, part, tuple(measures), {"artifact": str(path)})


def score_to_tokens(
    score: ScoreRecord,
    vocabulary: ScoreVocabulary,
    *,
    original_path: Sequence[str] | None = None,
    num_position_buckets: int = 4,
    position_encoding: str = "relative_chapter_quartile",
) -> TokenizedScore:
    return vocabulary.encode_score(
        score,
        original_path=original_path,
        num_position_buckets=num_position_buckets,
        position_encoding=position_encoding,
    )


def tokens_to_score(
    tokens: TokenizedScore,
    vocabulary: ScoreVocabulary,
    *,
    piece_id: str,
    part: str,
    meter_id_to_value: Mapping[int, Sequence[int]],
) -> ScoreRecord:
    measures: list[MeasureRecord] = []
    for index in range(tokens.num_measures):
        meter_id = int(tokens.meter[index])
        if meter_id not in meter_id_to_value:
            raise KeyError(f"No meter value configured for meter id {meter_id}")
        measures.append(
            vocabulary.decode_measure(
                tokens.pitch[index],
                tokens.duration[index],
                number=index + 1,
                chapter=int(tokens.chapter[index]) + 1,
                meter=meter_id_to_value[meter_id],
                family_token=int(tokens.family[index]),
            )
        )
    return ScoreRecord(piece_id, part, tuple(measures))


def _lcm(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result = math.lcm(result, value)
    return result


def _sub(parent: ET.Element, tag: str, text: str | int | None = None, **attributes: str) -> ET.Element:
    child = ET.SubElement(parent, tag, attributes)
    if text is not None:
        child.text = str(text)
    return child


def write_musicxml(
    score: ScoreRecord,
    path: str | Path,
    *,
    divisions_per_quarter: int | None = None,
    key_fifths: int = -4,
    part_id: str = "P1",
    part_name: str | None = None,
    prefer_flats: bool = True,
) -> Path:
    """Write a monophonic ``score-partwise`` MusicXML document."""

    denominators = [event.duration_q.denominator for m in score.measures for event in m.events]
    minimum_divisions = _lcm(denominators)
    divisions = divisions_per_quarter or minimum_divisions
    if divisions <= 0 or any((event.duration_q * divisions).denominator != 1 for m in score.measures for event in m.events):
        raise ValueError(
            f"divisions_per_quarter={divisions} cannot represent every score duration exactly"
        )

    root = ET.Element("score-partwise", version="4.0")
    work = _sub(root, "work")
    _sub(work, "work-title", score.metadata.get("title", score.piece_id))
    part_list = _sub(root, "part-list")
    score_part = _sub(part_list, "score-part", id=part_id)
    _sub(score_part, "part-name", part_name or score.part)
    part = _sub(root, "part", id=part_id)
    last_meter: Meter | None = None

    spelling = _FLAT_SPELLING if prefer_flats else _SHARP_SPELLING
    for measure in score.measures:
        measure_element = _sub(part, "measure", number=str(measure.number))
        if last_meter != measure.meter or measure.number == score.measures[0].number:
            attributes = _sub(measure_element, "attributes")
            _sub(attributes, "divisions", divisions)
            key = _sub(attributes, "key")
            _sub(key, "fifths", key_fifths)
            time = _sub(attributes, "time")
            _sub(time, "beats", measure.meter.numerator)
            _sub(time, "beat-type", measure.meter.denominator)
            clef = _sub(attributes, "clef")
            _sub(clef, "sign", "G")
            _sub(clef, "line", 2)
            last_meter = measure.meter

        for event in measure.events:
            note = _sub(measure_element, "note")
            if event.pitch is None:
                _sub(note, "rest")
            else:
                pitch = _sub(note, "pitch")
                step, alter = spelling[event.pitch % 12]
                _sub(pitch, "step", step)
                if alter:
                    _sub(pitch, "alter", alter)
                _sub(pitch, "octave", event.pitch // 12 - 1)
            _sub(note, "duration", int(event.duration_q * divisions))
            _sub(note, "voice", 1)
            notation = _duration_notation(event.duration_q)
            if notation is not None:
                note_type, dots = notation
                _sub(note, "type", note_type)
                for _ in range(dots):
                    _sub(note, "dot")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    return destination


def _duration_notation(duration_q: Fraction) -> tuple[str, int] | None:
    """Return an exact MusicXML note type and dot count when representable."""

    bases = (
        (Fraction(4), "whole"),
        (Fraction(2), "half"),
        (Fraction(1), "quarter"),
        (Fraction(1, 2), "eighth"),
        (Fraction(1, 4), "16th"),
        (Fraction(1, 8), "32nd"),
    )
    for base, name in bases:
        value = base
        addition = base
        for dots in range(4):
            if duration_q == value:
                return name, dots
            addition /= 2
            value += addition
    # Numeric <duration> remains exact even when a display type is unavailable.
    return None
