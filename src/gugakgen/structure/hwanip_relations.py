"""Non-local 환입 relations used as explicit, auditable constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class HwanipRelation:
    """A one-to-one return relation between two inclusive measure spans."""

    relation_id: str
    source_measures: tuple[int, int]
    target_measures: tuple[int, int]
    relation_type: str = "return_of"
    family_weight: float = 1.0
    surface_weight: float = 1.0
    provenance: str | None = None
    expert_validation: str | None = None

    def __post_init__(self) -> None:
        source_start, source_end = self.source_measures
        target_start, target_end = self.target_measures
        if source_start <= 0 or target_start <= 0 or source_end < source_start or target_end < target_start:
            raise ValueError("환입 measure spans must be positive inclusive ranges")
        if source_end - source_start != target_end - target_start:
            raise ValueError("One-to-one 환입 spans must have equal lengths")
        if self.relation_type != "return_of":
            raise ValueError(f"Unsupported 환입 relation type: {self.relation_type}")
        if self.family_weight < 0 or self.surface_weight < 0:
            raise ValueError("환입 relation weights must be non-negative")

    @property
    def pairs(self) -> tuple[tuple[int, int], ...]:
        """Return zero-based source/target position pairs."""

        source_start, source_end = self.source_measures
        target_start, _ = self.target_measures
        return tuple(
            (source - 1, target_start - 1 + (source - source_start))
            for source in range(source_start, source_end + 1)
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HwanipRelation":
        if value.get("alignment", "one_to_one") != "one_to_one":
            raise ValueError("Only one_to_one 환입 alignment is supported in Phase 1")
        source_values = tuple(int(v) for v in value["source_measures"])
        target_values = tuple(int(v) for v in value["target_measures"])
        if len(source_values) != 2 or len(target_values) != 2:
            raise ValueError("환입 source_measures and target_measures must each contain two values")
        return cls(
            relation_id=str(value.get("id", "hwanip")),
            source_measures=(source_values[0], source_values[1]),
            target_measures=(target_values[0], target_values[1]),
            relation_type=str(value.get("type", "return_of")),
            family_weight=float(value.get("family_weight", 1.0)),
            surface_weight=float(value.get("surface_weight", 1.0)),
            provenance=value.get("provenance"),
            expert_validation=value.get("expert_validation"),
        )


@dataclass(frozen=True, slots=True)
class HwanipRelations:
    relations: tuple[HwanipRelation, ...]
    num_measures: int

    def __post_init__(self) -> None:
        if self.num_measures <= 0:
            raise ValueError("num_measures must be positive")
        ids = [relation.relation_id for relation in self.relations]
        if len(ids) != len(set(ids)):
            raise ValueError("환입 relation ids must be unique")
        for relation in self.relations:
            for source, target in relation.pairs:
                if source >= self.num_measures or target >= self.num_measures:
                    raise ValueError(f"Relation {relation.relation_id} exceeds score length")

    @property
    def pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple(pair for relation in self.relations for pair in relation.pairs)

    def family_guidance_bias(
        self,
        current_path: Sequence[int],
        *,
        num_families: int,
        mask_id: int,
        scale: float = 1.0,
        bidirectional: bool = True,
    ) -> NDArray[np.float64]:
        """Create additive Logits bias from currently known paired families."""

        values = np.asarray(current_path, dtype=np.int64)
        if values.shape != (self.num_measures,):
            raise ValueError(f"current_path must have shape {(self.num_measures,)}")
        bias = np.zeros((self.num_measures, num_families), dtype=np.float64)
        for relation in self.relations:
            strength = scale * relation.family_weight
            for source, target in relation.pairs:
                source_value, target_value = int(values[source]), int(values[target])
                if source_value != mask_id:
                    bias[target, source_value] += strength
                if bidirectional and target_value != mask_id:
                    bias[source, target_value] += strength
        return bias

    def family_satisfaction(self, path: Sequence[int]) -> float:
        values = np.asarray(path, dtype=np.int64)
        if values.shape != (self.num_measures,):
            raise ValueError(f"path must have shape {(self.num_measures,)}")
        pairs = self.pairs
        return 1.0 if not pairs else float(np.mean([values[source] == values[target] for source, target in pairs]))

    def surface_satisfaction(self, signatures: Sequence[object]) -> float:
        if len(signatures) != self.num_measures:
            raise ValueError("signature count must equal num_measures")
        pairs = self.pairs
        return 1.0 if not pairs else float(
            np.mean([signatures[source] == signatures[target] for source, target in pairs])
        )

    def counterfactual_return_rates(
        self,
        signatures: Sequence[object],
        reference_signatures: set[object],
    ) -> dict[str, float | int | None]:
        """Compute r_new and r_copy over declared source/target pairs."""

        if len(signatures) != self.num_measures:
            raise ValueError("signature count must equal num_measures")
        is_new: list[bool] = []
        returned: list[bool] = []
        for source, target in self.pairs:
            is_new.append(signatures[source] not in reference_signatures)
            returned.append(signatures[source] == signatures[target])
        new_array = np.asarray(is_new, dtype=bool)
        return_array = np.asarray(returned, dtype=bool)

        def rate(mask: NDArray[np.bool_]) -> float | None:
            return None if not mask.any() else float(return_array[mask].mean())

        return {
            "n_new": int(new_array.sum()),
            "n_copy": int((~new_array).sum()),
            "r_new": rate(new_array),
            "r_copy": rate(~new_array),
        }

    @classmethod
    def from_config(
        cls, values: Sequence[Mapping[str, Any]] | None, *, num_measures: int
    ) -> "HwanipRelations":
        return cls(tuple(HwanipRelation.from_dict(value) for value in (values or [])), num_measures)
