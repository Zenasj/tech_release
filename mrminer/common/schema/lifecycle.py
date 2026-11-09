"""Append-only lifecycle commands, events, and derived state."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import StringConstraints, model_validator
from pydantic.dataclasses import dataclass

from mrminer.common.canonical import canonical_bytes, content_id
from mrminer.common.schema.base import SCHEMA_CONFIG, SchemaRecord, dump_value
from mrminer.common.schema.evidence import EvidenceSpec

NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]
EventId = Annotated[
    str, StringConstraints(pattern=r"^evt:sha256:[0-9a-f]{64}$")
]
OracleId = Annotated[
    str, StringConstraints(pattern=r"^orc:sha256:[0-9a-f]{64}$")
]
CellId = Annotated[
    str, StringConstraints(pattern=r"^cell:sha256:[0-9a-f]{64}$")
]
ArtifactId = Annotated[str, StringConstraints(pattern=r"^art:[^\s/]+$")]


class LifecycleEventType(str, Enum):
    CREATED = "created"
    VALIDATED = "validated"
    ACTIVATED = "activated"
    VALIDATION_FAILED = "validation_failed"
    SUSPENDED_PROBE_DRIFT = "suspended_probe_drift"
    SUSPENDED_ENVIRONMENT = "suspended_environment"
    SUSPENDED_PENDING_ARBITRATION = "suspended_pending_arbitration"
    SUSPENDED_EVIDENCE_WEAK = "suspended_evidence_weak"
    SUPERSEDED = "superseded"
    REBASELINED = "rebaselined"
    RETIRED_ENTITY_REMOVED = "retired_entity_removed"
    RETIRED_EXPLICIT_CHANGE = "retired_explicit_change"
    RESUMED = "resumed"


class OracleState(str, Enum):
    DRAFT = "draft"
    VALIDATED = "validated"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    RETIRED = "retired"


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class LifecycleEvent(SchemaRecord):
    RECORD_KIND = "lifecycle_event"

    event_version: Literal["v1"] = "v1"
    event_id: EventId
    command_id: NonEmptyStr
    oracle_id: OracleId
    event: LifecycleEventType
    predecessor_event_id: EventId | None
    payload: dict[str, Any]
    evidence: tuple[EvidenceSpec, ...]
    successor_spec: OracleId | None
    actor: NonEmptyStr
    at: NonEmptyStr

    @classmethod
    def create(cls, **fields: Any) -> LifecycleEvent:
        payload = cls._id_payload(**fields)
        return cls(event_id=content_id("evt", payload), **fields)

    @staticmethod
    def _id_payload(**fields: Any) -> dict[str, Any]:
        return {
            "command_id": fields["command_id"],
            "oracle_id": fields["oracle_id"],
            "event": (
                fields["event"].value
                if isinstance(fields["event"], LifecycleEventType)
                else fields["event"]
            ),
            "predecessor_event_id": fields["predecessor_event_id"],
            "payload": fields["payload"],
            "evidence": [dump_value(item) for item in fields["evidence"]],
            "successor_spec": fields["successor_spec"],
            "actor": fields["actor"],
        }

    def to_hash_payload(self) -> dict[str, Any]:
        return self._id_payload(
            command_id=self.command_id,
            oracle_id=self.oracle_id,
            event=self.event,
            predecessor_event_id=self.predecessor_event_id,
            payload=dict(self.payload),
            evidence=self.evidence,
            successor_spec=self.successor_spec,
            actor=self.actor,
        )

    @model_validator(mode="after")
    def _validate_event(self) -> LifecycleEvent:
        canonical_bytes(self.payload)
        if self.event is LifecycleEventType.CREATED:
            if self.predecessor_event_id is not None:
                raise ValueError("created event cannot have a predecessor")
        elif self.predecessor_event_id is None:
            raise ValueError("non-created lifecycle events require a predecessor")
        if self.event in {
            LifecycleEventType.SUPERSEDED,
            LifecycleEventType.REBASELINED,
        }:
            if self.successor_spec is None:
                raise ValueError("supersede/rebaseline events require successor_spec")
        elif self.successor_spec is not None:
            raise ValueError("only supersede/rebaseline events may define successor_spec")
        expected = content_id("evt", self.to_hash_payload())
        if self.event_id != expected:
            raise ValueError("event_id does not match command payload")
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class OracleStateProjection(SchemaRecord):
    RECORD_KIND = "oracle_state_projection"

    projection_version: Literal["v1"] = "v1"
    oracle_id: OracleId
    state: OracleState
    reason: NonEmptyStr | None
    head_event_id: EventId
    active_support_cells: tuple[CellId, ...]
    last_pass_artifact: ArtifactId | None
    successor_specs: tuple[OracleId, ...]

    @model_validator(mode="after")
    def _validate_projection(self) -> OracleStateProjection:
        if self.state is OracleState.SUSPENDED and self.reason is None:
            raise ValueError("suspended projections require a reason")
        if self.state is not OracleState.SUSPENDED and self.reason is not None:
            raise ValueError("only suspended projections may define a reason")
        if self.active_support_cells != tuple(
            sorted(set(self.active_support_cells))
        ):
            raise ValueError("active support cells must be sorted and unique")
        if self.successor_specs != tuple(sorted(set(self.successor_specs))):
            raise ValueError("successor specs must be sorted and unique")
        return self
