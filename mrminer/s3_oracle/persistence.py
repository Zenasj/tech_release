"""Persistence of ready S3 actions into graph and lifecycle stores."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mrminer.common.canonical import canonical_json, fs_safe
from mrminer.common.lifecycle_store import LifecycleEventStore
from mrminer.common.schema import (
    EvidenceSpec,
    LifecycleEvent,
    LifecycleEventType,
)
from mrminer.s2_graph import (
    ClaimCandidate,
    EntityGraph,
    GraphEdge,
    GraphEdgeKind,
    GraphNodeKind,
    NodeRef,
)
from mrminer.s3_oracle.actions import (
    OracleAction,
    OracleActionKind,
    OracleActionStatus,
)


class ActionPersistenceError(RuntimeError):
    pass


@dataclass(frozen=True, kw_only=True)
class ActionApplyReport:
    action_ids: tuple[str, ...]
    oracle_ids: tuple[str, ...]
    edges_appended: int
    events_appended: int
    index_records: int

    def to_record(self) -> dict[str, Any]:
        return {
            "action_ids": list(self.action_ids),
            "oracle_ids": list(self.oracle_ids),
            "edges_appended": self.edges_appended,
            "events_appended": self.events_appended,
            "index_records": self.index_records,
        }


def _unique_evidence(
    claims: tuple[ClaimCandidate, ...],
) -> tuple[EvidenceSpec, ...]:
    by_record = {
        canonical_json(evidence.to_record()): evidence
        for claim in claims
        for evidence in claim.evidence
    }
    return tuple(by_record[key] for key in sorted(by_record))


class OracleActionApplier:
    def __init__(
        self,
        graph: EntityGraph,
        *,
        lifecycle_root: Path,
    ):
        self.graph = graph
        self.lifecycle_root = lifecycle_root

    def apply(
        self,
        actions: list[OracleAction] | tuple[OracleAction, ...],
        *,
        claims: list[ClaimCandidate] | tuple[ClaimCandidate, ...],
        at: str,
        actor: str = "s3.inducer",
    ) -> ActionApplyReport:
        if not actions:
            raise ActionPersistenceError("no actions supplied")
        if len({action.entity_id for action in actions}) != 1:
            raise ActionPersistenceError(
                "one apply batch must target exactly one entity"
            )
        claim_by_id = {claim.claim_id: claim for claim in claims}
        self.graph.rebuild_index()
        stored_claim_ids = {
            claim.claim_id for claim in self.graph.claims(actions[0].entity_id)
        }
        for action in actions:
            self._preflight(
                action,
                claim_by_id=claim_by_id,
                stored_claim_ids=stored_claim_ids,
            )

        edges_appended = 0
        events_appended = 0
        oracle_ids: list[str] = []
        for action in actions:
            action_claims = tuple(
                claim_by_id[claim_id] for claim_id in action.source_claim_ids
            )
            evidence = _unique_evidence(action_claims)
            for spec in action.new_specs:
                oracle_ids.append(spec.oracle_id)
                self.graph.add_oracle(spec)
                for binding in spec.binding:
                    edges_appended += int(
                        self.graph.add_edge(
                            GraphEdge.create(
                                kind=GraphEdgeKind.BOUND_TO,
                                source=NodeRef(
                                    kind=GraphNodeKind.ORACLE,
                                    key=spec.oracle_id,
                                ),
                                target=NodeRef(
                                    kind=GraphNodeKind.ENTITY,
                                    key=binding.entity_id,
                                ),
                                evidence=evidence,
                                confidence=min(
                                    claim.confidence for claim in action_claims
                                ),
                                context={"role": binding.role},
                                at=at,
                            )
                        )
                    )
                for claim_id in spec.claim.claim_refs:
                    claim = claim_by_id[claim_id]
                    edges_appended += int(
                        self.graph.add_edge(
                            GraphEdge.create(
                                kind=GraphEdgeKind.DERIVED_FROM_CLAIM,
                                source=NodeRef(
                                    kind=GraphNodeKind.ORACLE,
                                    key=spec.oracle_id,
                                ),
                                target=NodeRef(
                                    kind=GraphNodeKind.CLAIM,
                                    key=claim_id,
                                ),
                                evidence=claim.evidence,
                                confidence=claim.confidence,
                                context={},
                                at=at,
                            )
                        )
                    )
                events_appended += self._append_created(
                    action,
                    oracle_id=spec.oracle_id,
                    evidence=evidence,
                    at=at,
                    actor=actor,
                )

            successor_id = action.new_specs[0].oracle_id
            for predecessor_id in action.predecessor_ids:
                events_appended += self._append_replacement(
                    action,
                    predecessor_id=predecessor_id,
                    successor_id=successor_id,
                    evidence=evidence,
                    at=at,
                    actor=actor,
                )
                if action.kind is OracleActionKind.SUPERSEDE:
                    edges_appended += int(
                        self.graph.add_edge(
                            GraphEdge.create(
                                kind=GraphEdgeKind.SUPERSEDES,
                                source=NodeRef(
                                    kind=GraphNodeKind.ORACLE,
                                    key=successor_id,
                                ),
                                target=NodeRef(
                                    kind=GraphNodeKind.ORACLE,
                                    key=predecessor_id,
                                ),
                                evidence=evidence,
                                confidence=min(
                                    claim.confidence for claim in action_claims
                                ),
                                context={"action_id": action.action_id},
                                at=at,
                            )
                        )
                    )
        index = self.graph.rebuild_index()
        if not index.checksum_ok:
            raise ActionPersistenceError(
                f"graph rebuild failed: {self.graph.last_rebuild_errors}"
            )
        return ActionApplyReport(
            action_ids=tuple(action.action_id for action in actions),
            oracle_ids=tuple(oracle_ids),
            edges_appended=edges_appended,
            events_appended=events_appended,
            index_records=index.records,
        )

    def _preflight(
        self,
        action: OracleAction,
        *,
        claim_by_id: dict[str, ClaimCandidate],
        stored_claim_ids: set[str],
    ) -> None:
        if action.status is not OracleActionStatus.READY:
            raise ActionPersistenceError(
                f"action {action.action_id} is not ready"
            )
        if action.kind is OracleActionKind.SPLIT:
            raise ActionPersistenceError(
                "split persistence is blocked by single successor_spec lifecycle schema"
            )
        missing = set(action.source_claim_ids) - set(claim_by_id)
        if missing:
            raise ActionPersistenceError(
                f"action claims were not supplied: {sorted(missing)}"
            )
        unstored = set(action.source_claim_ids) - stored_claim_ids
        if unstored:
            raise ActionPersistenceError(
                f"action claims are absent from EntityGraph: {sorted(unstored)}"
            )
        for spec in action.new_specs:
            unresolved = set(spec.claim.claim_refs) - set(action.source_claim_ids)
            if unresolved:
                raise ActionPersistenceError(
                    f"spec contains claims outside action: {sorted(unresolved)}"
                )
            if set(spec.lineage.predecessors) != set(action.predecessor_ids):
                raise ActionPersistenceError(
                    "spec lineage predecessors disagree with action"
                )
        existing = {
            oracle.oracle_id
            for oracle in self.graph.oracles(action.entity_id, state=None)
        }
        missing_predecessors = set(action.predecessor_ids) - existing
        if missing_predecessors:
            raise ActionPersistenceError(
                f"predecessor oracles are absent: {sorted(missing_predecessors)}"
            )
        for predecessor_id in action.predecessor_ids:
            store = self._lifecycle_store(predecessor_id)
            if store.projection(predecessor_id) is None:
                raise ActionPersistenceError(
                    f"predecessor {predecessor_id} has no lifecycle stream"
                )

    def _append_created(
        self,
        action: OracleAction,
        *,
        oracle_id: str,
        evidence: tuple[EvidenceSpec, ...],
        at: str,
        actor: str,
    ) -> int:
        store = self._lifecycle_store(oracle_id)
        event = LifecycleEvent.create(
            command_id=f"{action.action_id}:create:{oracle_id}",
            oracle_id=oracle_id,
            event=LifecycleEventType.CREATED,
            predecessor_event_id=None,
            payload={"action": action.kind.value, "action_id": action.action_id},
            evidence=evidence,
            successor_spec=None,
            actor=actor,
            at=at,
        )
        appended = store.append(event)
        projection = store.projection(oracle_id)
        assert projection is not None
        self.graph.add_projection(projection)
        return int(appended)

    def _append_replacement(
        self,
        action: OracleAction,
        *,
        predecessor_id: str,
        successor_id: str,
        evidence: tuple[EvidenceSpec, ...],
        at: str,
        actor: str,
    ) -> int:
        store = self._lifecycle_store(predecessor_id)
        command_id = (
            f"{action.action_id}:replace:{predecessor_id}:{successor_id}"
        )
        for existing in store.events():
            if existing.command_id == command_id:
                if existing.successor_spec != successor_id:
                    raise ActionPersistenceError(
                        "existing replacement command points to another successor"
                    )
                projection = store.projection(predecessor_id)
                assert projection is not None
                self.graph.add_projection(projection)
                return 0
        projection = store.projection(predecessor_id)
        assert projection is not None
        event_type = (
            LifecycleEventType.REBASELINED
            if action.kind is OracleActionKind.REBASELINE
            else LifecycleEventType.SUPERSEDED
        )
        event = LifecycleEvent.create(
            command_id=command_id,
            oracle_id=predecessor_id,
            event=event_type,
            predecessor_event_id=projection.head_event_id,
            payload={"action": action.kind.value, "action_id": action.action_id},
            evidence=evidence,
            successor_spec=successor_id,
            actor=actor,
            at=at,
        )
        appended = store.append(event)
        next_projection = store.projection(predecessor_id)
        assert next_projection is not None
        self.graph.add_projection(next_projection)
        return int(appended)

    def _lifecycle_store(self, oracle_id: str) -> LifecycleEventStore:
        return LifecycleEventStore(
            self.lifecycle_root / f"{fs_safe(oracle_id)}.jsonl"
        )
