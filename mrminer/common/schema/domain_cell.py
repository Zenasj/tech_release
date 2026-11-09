"""Concrete, fully observed points in a GuardDomain."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator
from pydantic.dataclasses import dataclass

from mrminer.common.canonical import content_id
from mrminer.common.schema.base import SCHEMA_CONFIG, SchemaRecord, dump_value

ContentId = Annotated[
    str,
    StringConstraints(pattern=r"^cell:sha256:[0-9a-f]{64}$"),
]
ProgramId = Annotated[
    str,
    StringConstraints(pattern=r"^prog:sha256:[0-9a-f]{64}$"),
]
InputId = Annotated[
    str,
    StringConstraints(pattern=r"^inp:sha256:[0-9a-f]{64}$"),
]
ArtifactId = Annotated[str, StringConstraints(pattern=r"^art:[^\s/]+$")]
NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]
PositiveInt = Annotated[int, Field(gt=0)]


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class ShapeAssignment(SchemaRecord):
    RECORD_KIND = "shape_assignment"

    rank: int = Field(ge=0)
    dims: tuple[PositiveInt, ...]

    @model_validator(mode="after")
    def _validate_rank(self) -> ShapeAssignment:
        if self.rank != len(self.dims):
            raise ValueError("shape rank must equal the number of dimensions")
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class DomainValues(SchemaRecord):
    RECORD_KIND = "domain_values"

    dtype: NonEmptyStr
    device: NonEmptyStr
    overload: NonEmptyStr
    layout: NonEmptyStr
    shape_mode: NonEmptyStr
    shape_assignment: ShapeAssignment
    phase: NonEmptyStr
    backend: NonEmptyStr

    @model_validator(mode="after")
    def _reject_unknown_placeholders(self) -> DomainValues:
        dimensions = (
            self.dtype,
            self.device,
            self.overload,
            self.layout,
            self.shape_mode,
            self.phase,
            self.backend,
        )
        if any(value.strip().lower() == "unknown" for value in dimensions):
            raise ValueError(
                "domain cells require observed values; 'unknown' is reserved"
            )
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class ObservedFrom(SchemaRecord):
    RECORD_KIND = "observed_from"

    program_id: ProgramId
    input_id: InputId
    artifact_id: ArtifactId


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class DomainCell(SchemaRecord):
    RECORD_KIND = "domain_cell"

    schema_version: Literal["domain_cell_v1"] = "domain_cell_v1"
    cell_id: ContentId
    values: DomainValues
    observed_from: ObservedFrom

    @classmethod
    def create(
        cls,
        *,
        values: DomainValues,
        observed_from: ObservedFrom,
    ) -> DomainCell:
        payload = {
            "values": dump_value(values),
            "observed_from": dump_value(observed_from),
        }
        return cls(
            cell_id=content_id("cell", payload),
            values=values,
            observed_from=observed_from,
        )

    def to_hash_payload(self) -> dict[str, object]:
        return {
            "values": dump_value(self.values),
            "observed_from": dump_value(self.observed_from),
        }

    @model_validator(mode="after")
    def _validate_content_id(self) -> DomainCell:
        expected = content_id("cell", self.to_hash_payload())
        if self.cell_id != expected:
            raise ValueError(f"cell_id does not match semantic payload: expected {expected}")
        return self
