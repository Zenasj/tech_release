"""Stable component-boundary records that are not all persisted independently."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import Field, StringConstraints, model_validator
from pydantic.dataclasses import dataclass

from mrminer.common.canonical import canonical_bytes
from mrminer.common.schema.base import SCHEMA_CONFIG, SchemaRecord
from mrminer.common.schema.oracle import OracleId, ValiditySpec
from mrminer.common.schema.run import RelationCheckSpec, RunId, RunSpec

NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]
ArtifactId = Annotated[str, StringConstraints(pattern=r"^art:[^\s/]+$")]
ObservationId = Annotated[
    str, StringConstraints(pattern=r"^obs:sha256:[0-9a-f]{64}$")
]
EntityId = Annotated[str, StringConstraints(pattern=r"^[a-z]+:[^\s/]+$")]
CellId = Annotated[
    str, StringConstraints(pattern=r"^cell:sha256:[0-9a-f]{64}$")
]
Sha256Ref = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]


class ValidationCellStatus(str, Enum):
    VALIDATED = "validated"
    UNSUPPORTED = "unsupported"
    INVALID = "invalid"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class ActivationDecision(SchemaRecord):
    RECORD_KIND = "activation_decision"

    action: Literal["activate", "reject", "suspend"]
    active_support_cells: tuple[CellId, ...] = ()
    reason: NonEmptyStr | None = None

    @model_validator(mode="after")
    def _validate_decision(self) -> ActivationDecision:
        if self.action == "activate":
            if not self.active_support_cells or self.reason is not None:
                raise ValueError("activation requires cells and no rejection reason")
        elif self.active_support_cells or self.reason is None:
            raise ValueError("reject/suspend decisions require only a reason")
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class ValidationReport(SchemaRecord):
    RECORD_KIND = "validation_report"

    oracle_id: OracleId
    generation: int = Field(ge=1)
    cells: dict[CellId, ValidationCellStatus]
    per_check: dict[NonEmptyStr, bool]
    activation_decision: ActivationDecision


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class CapabilityEntry(SchemaRecord):
    RECORD_KIND = "capability_entry"

    capability_id: NonEmptyStr
    schema_version: NonEmptyStr
    supported_artifact_range: ValiditySpec
    probe_semantics_version: NonEmptyStr


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class CapabilitySet(SchemaRecord):
    RECORD_KIND = "capability_set"

    entries: tuple[CapabilityEntry, ...]

    @model_validator(mode="after")
    def _validate_entries(self) -> CapabilitySet:
        ids = tuple(entry.capability_id for entry in self.entries)
        if ids != tuple(sorted(set(ids))):
            raise ValueError("capability entries must be sorted and unique by ID")
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class ProvisionedEnvironment(SchemaRecord):
    RECORD_KIND = "provisioned_environment"

    artifact_id: ArtifactId
    env_paths: dict[NonEmptyStr, NonEmptyStr]
    functional_env_hash: Sha256Ref
    performance_env_hash: Sha256Ref


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class EntityTrace(SchemaRecord):
    RECORD_KIND = "entity_trace"

    obs_ref: ObservationId
    entity_id: EntityId
    trigger_count: int = Field(ge=0)
    probe_version: NonEmptyStr


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class DedupReport(SchemaRecord):
    RECORD_KIND = "dedup_report"

    requested_runs: int = Field(ge=0)
    emitted_runs: int = Field(ge=0)
    reused_observations: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_counts(self) -> DedupReport:
        if self.emitted_runs > self.requested_runs:
            raise ValueError("emitted runs cannot exceed requested runs")
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class ExecutionPlan(SchemaRecord):
    RECORD_KIND = "execution_plan"

    run_specs: tuple[RunSpec, ...]
    checks: tuple[RelationCheckSpec, ...]
    dedup_report: DedupReport


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class RetryPolicy(SchemaRecord):
    RECORD_KIND = "retry_policy"

    max_attempts: int = Field(alias="max", ge=0)
    on: tuple[Literal["environment_error"], ...] = ("environment_error",)

    @model_validator(mode="after")
    def _validate_on(self) -> RetryPolicy:
        if self.on != ("environment_error",):
            raise ValueError("retry is supported only for environment_error")
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class RunDispatchPolicy(SchemaRecord):
    RECORD_KIND = "run_dispatch_policy"

    run_id: RunId
    retry: RetryPolicy


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class ArbitrationRecord(SchemaRecord):
    RECORD_KIND = "arbitration_record"

    oracle_id: OracleId
    suspects: tuple[NonEmptyStr, ...]
    materials: dict[str, Any]
    resolution: NonEmptyStr | None

    @model_validator(mode="after")
    def _validate_materials(self) -> ArbitrationRecord:
        canonical_bytes(self.materials)
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class IndexReport(SchemaRecord):
    RECORD_KIND = "index_report"

    store: NonEmptyStr
    records: int = Field(ge=0)
    rebuilt_at: NonEmptyStr
    checksum_ok: bool
