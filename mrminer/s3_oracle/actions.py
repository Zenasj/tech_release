"""Content-addressed S3 oracle actions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from mrminer.common.canonical import content_id
from mrminer.common.schema import OracleSpec


class OracleActionKind(str, Enum):
    CREATE = "create"
    ADD_EVIDENCE = "add_evidence"
    REFINE_GUARD = "refine_guard"
    SPLIT = "split"
    MERGE = "merge"
    SUPERSEDE = "supersede"
    REBASELINE = "rebaseline"
    SUSPEND = "suspend"
    NO_OP = "no_op"


class OracleActionStatus(str, Enum):
    READY = "ready"
    PENDING_ARBITRATION = "pending_arbitration"
    NO_OP = "no_op"


@dataclass(frozen=True, kw_only=True)
class OracleAction:
    action_id: str
    kind: OracleActionKind
    status: OracleActionStatus
    entity_id: str
    aspect: str
    source_claim_ids: tuple[str, ...]
    predecessor_ids: tuple[str, ...]
    new_specs: tuple[OracleSpec, ...]
    reason: str

    @classmethod
    def create(
        cls,
        *,
        kind: OracleActionKind,
        status: OracleActionStatus,
        entity_id: str,
        aspect: str,
        source_claim_ids: tuple[str, ...],
        predecessor_ids: tuple[str, ...] = (),
        new_specs: tuple[OracleSpec, ...] = (),
        reason: str,
    ) -> OracleAction:
        normalized_claims = tuple(sorted(set(source_claim_ids)))
        normalized_predecessors = tuple(sorted(set(predecessor_ids)))
        payload = {
            "kind": kind.value,
            "status": status.value,
            "entity_id": entity_id,
            "aspect": aspect,
            "source_claim_ids": list(normalized_claims),
            "predecessor_ids": list(normalized_predecessors),
            "new_spec_ids": [spec.oracle_id for spec in new_specs],
            "reason": reason,
        }
        return cls(
            action_id=content_id("act", payload),
            kind=kind,
            status=status,
            entity_id=entity_id,
            aspect=aspect,
            source_claim_ids=normalized_claims,
            predecessor_ids=normalized_predecessors,
            new_specs=new_specs,
            reason=reason,
        )

    def __post_init__(self) -> None:
        if not self.entity_id or not self.aspect or not self.reason:
            raise ValueError("action entity, aspect, and reason must be non-empty")
        if self.source_claim_ids != tuple(sorted(set(self.source_claim_ids))):
            raise ValueError("source claim IDs must be sorted and unique")
        if self.predecessor_ids != tuple(sorted(set(self.predecessor_ids))):
            raise ValueError("predecessor IDs must be sorted and unique")
        if self.status is OracleActionStatus.READY and not self.new_specs:
            raise ValueError("ready actions require at least one new spec")
        if self.status is not OracleActionStatus.READY and self.new_specs:
            raise ValueError("non-ready actions cannot carry new specs")
        if self.kind is OracleActionKind.CREATE and self.predecessor_ids:
            raise ValueError("create cannot have predecessors")
        if (
            self.kind
            in {
                OracleActionKind.ADD_EVIDENCE,
                OracleActionKind.REFINE_GUARD,
                OracleActionKind.SUPERSEDE,
                OracleActionKind.REBASELINE,
            }
            and self.status is OracleActionStatus.READY
            and (len(self.predecessor_ids) != 1 or len(self.new_specs) != 1)
        ):
            raise ValueError("single-successor action has invalid cardinality")
        if (
            self.kind is OracleActionKind.MERGE
            and self.status is OracleActionStatus.READY
            and (len(self.predecessor_ids) < 2 or len(self.new_specs) != 1)
        ):
            raise ValueError("merge requires multiple predecessors and one successor")
        if (
            self.kind is OracleActionKind.SPLIT
            and self.status is OracleActionStatus.READY
            and (len(self.predecessor_ids) != 1 or len(self.new_specs) < 2)
        ):
            raise ValueError("split requires one predecessor and multiple successors")

    @property
    def persistable(self) -> bool:
        return self.status is OracleActionStatus.READY

    def to_record(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "kind": self.kind.value,
            "status": self.status.value,
            "entity_id": self.entity_id,
            "aspect": self.aspect,
            "source_claim_ids": list(self.source_claim_ids),
            "predecessor_ids": list(self.predecessor_ids),
            "new_specs": [spec.to_record() for spec in self.new_specs],
            "reason": self.reason,
        }
