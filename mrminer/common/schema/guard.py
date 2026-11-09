"""Guard-domain predicates and conservative overlap semantics."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import Field, model_validator
from pydantic.dataclasses import dataclass

from mrminer.common.schema.base import SCHEMA_CONFIG, SchemaRecord

NonEmptyStr = Annotated[str, Field(min_length=1)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]


class GuardMode(str, Enum):
    SET = "set"
    ANY = "any"
    UNKNOWN = "unknown"


class OverlapResult(str, Enum):
    OVERLAP = "overlap"
    DISJOINT = "disjoint"
    UNKNOWN = "unknown"


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class GuardDimension(SchemaRecord):
    RECORD_KIND = "guard_dimension"

    mode: GuardMode
    values: tuple[NonEmptyStr, ...] = ()
    evidence_refs: tuple[NonEmptyStr, ...] = ()

    @model_validator(mode="after")
    def _validate_mode_values(self) -> GuardDimension:
        if self.mode is GuardMode.SET and not self.values:
            raise ValueError("set guard dimensions require at least one value")
        if self.mode is not GuardMode.SET and self.values:
            raise ValueError("only set guard dimensions may define values")
        if self.values != tuple(sorted(set(self.values))):
            raise ValueError("guard values must be sorted and unique")
        if self.evidence_refs != tuple(sorted(set(self.evidence_refs))):
            raise ValueError("evidence_refs must be sorted and unique")
        return self

    def overlap(self, other: GuardDimension) -> OverlapResult:
        if GuardMode.UNKNOWN in {self.mode, other.mode}:
            return OverlapResult.UNKNOWN
        if GuardMode.ANY in {self.mode, other.mode}:
            return OverlapResult.OVERLAP
        if set(self.values).isdisjoint(other.values):
            return OverlapResult.DISJOINT
        return OverlapResult.OVERLAP

    def accepts(self, value: str) -> bool:
        if self.mode is GuardMode.UNKNOWN:
            return False
        if self.mode is GuardMode.ANY:
            return True
        return value in self.values


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class ShapeConstraints(SchemaRecord):
    RECORD_KIND = "shape_constraints"

    rank: tuple[NonNegativeInt, NonNegativeInt] | None = None
    dim_ranges: tuple[tuple[NonNegativeInt | None, NonNegativeInt | None], ...] = ()
    equalities: tuple[tuple[NonNegativeInt, NonNegativeInt], ...] = ()

    @model_validator(mode="after")
    def _validate_ranges(self) -> ShapeConstraints:
        if self.rank is not None and self.rank[0] > self.rank[1]:
            raise ValueError("rank lower bound cannot exceed upper bound")
        for lower, upper in self.dim_ranges:
            if lower is not None and upper is not None and lower > upper:
                raise ValueError("dimension lower bound cannot exceed upper bound")
        if self.equalities != tuple(sorted(set(self.equalities))):
            raise ValueError("shape equalities must be sorted and unique")
        return self

    def accepts(self, dims: tuple[int, ...]) -> bool:
        if self.rank is not None and not self.rank[0] <= len(dims) <= self.rank[1]:
            return False
        if self.dim_ranges and len(self.dim_ranges) != len(dims):
            return False
        for dim, (lower, upper) in zip(dims, self.dim_ranges, strict=False):
            if lower is not None and dim < lower:
                return False
            if upper is not None and dim > upper:
                return False
        for left, right in self.equalities:
            if left >= len(dims) or right >= len(dims) or dims[left] != dims[right]:
                return False
        return True


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class GuardDomain(SchemaRecord):
    RECORD_KIND = "guard_domain"

    dtype: GuardDimension
    device: GuardDimension
    overload: GuardDimension
    layout: GuardDimension
    shape_mode: GuardDimension
    phase: GuardDimension
    backend: GuardDimension
    shape_constraints: ShapeConstraints | None = None

    def overlap(self, other: GuardDomain) -> OverlapResult:
        results = tuple(
            left.overlap(right)
            for left, right in (
                (self.dtype, other.dtype),
                (self.device, other.device),
                (self.overload, other.overload),
                (self.layout, other.layout),
                (self.shape_mode, other.shape_mode),
                (self.phase, other.phase),
                (self.backend, other.backend),
            )
        )
        if OverlapResult.DISJOINT in results:
            return OverlapResult.DISJOINT
        if OverlapResult.UNKNOWN in results:
            return OverlapResult.UNKNOWN
        if (
            self.shape_constraints is not None
            and other.shape_constraints is not None
            and self.shape_constraints != other.shape_constraints
        ):
            return OverlapResult.UNKNOWN
        return OverlapResult.OVERLAP

    def accepts(
        self,
        *,
        dtype: str,
        device: str,
        overload: str,
        layout: str,
        shape_mode: str,
        dims: tuple[int, ...],
        phase: str,
        backend: str,
    ) -> bool:
        dimensions_match = all(
            dimension.accepts(value)
            for dimension, value in (
                (self.dtype, dtype),
                (self.device, device),
                (self.overload, overload),
                (self.layout, layout),
                (self.shape_mode, shape_mode),
                (self.phase, phase),
                (self.backend, backend),
            )
        )
        return dimensions_match and (
            self.shape_constraints is None or self.shape_constraints.accepts(dims)
        )
