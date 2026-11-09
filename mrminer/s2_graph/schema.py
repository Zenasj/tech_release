"""Persisted ClaimCandidate and EntityGraph records."""

from __future__ import annotations

import datetime as dt
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import Field, StringConstraints, model_validator
from pydantic.dataclasses import dataclass

from mrminer.common.canonical import canonical_bytes, content_id, sha256_digest
from mrminer.common.schema.base import SCHEMA_CONFIG, SchemaRecord, dump_value
from mrminer.common.schema.evidence import EvidenceSpec
from mrminer.common.schema.guard import GuardDomain
from mrminer.common.schema.oracle import OracleAspect, OracleProvenance

NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]
EntityId = Annotated[str, StringConstraints(pattern=r"^[a-z]+:[^\s/]+$")]
OracleId = Annotated[
    str, StringConstraints(pattern=r"^orc:sha256:[0-9a-f]{64}$")
]
ClaimId = Annotated[
    str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")
]
ArtifactId = Annotated[str, StringConstraints(pattern=r"^art:[^\s/]+$")]
EdgeId = Annotated[
    str, StringConstraints(pattern=r"^edge:sha256:[0-9a-f]{64}$")
]
AnnotationId = Annotated[
    str, StringConstraints(pattern=r"^orcaud:sha256:[0-9a-f]{64}$")
]


class GraphNodeKind(str, Enum):
    ENTITY = "entity"
    PR = "pr"
    ISSUE = "issue"
    CLAIM = "claim"
    ORACLE = "oracle"
    ARTIFACT = "artifact"


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class NodeRef(SchemaRecord):
    RECORD_KIND = "graph_node_ref"

    kind: GraphNodeKind
    key: NonEmptyStr

    @model_validator(mode="after")
    def _validate_key(self) -> NodeRef:
        valid = {
            GraphNodeKind.ENTITY: (
                self.key.startswith(
                    ("op:", "api:", "pat:", "cfg:", "pass:", "feat:", "sys:")
                )
            ),
            GraphNodeKind.PR: (
                self.key.startswith("pr:")
                and self.key.removeprefix("pr:").isdigit()
                and int(self.key.removeprefix("pr:")) > 0
            ),
            GraphNodeKind.ISSUE: (
                self.key.startswith("issue:")
                and self.key.removeprefix("issue:").isdigit()
                and int(self.key.removeprefix("issue:")) > 0
            ),
            GraphNodeKind.CLAIM: (
                self.key.startswith("sha256:") and len(self.key) == 71
            ),
            GraphNodeKind.ORACLE: (
                self.key.startswith("orc:sha256:") and len(self.key) == 75
            ),
            GraphNodeKind.ARTIFACT: self.key.startswith("art:"),
        }
        if not valid[self.kind]:
            raise ValueError(f"invalid {self.kind.value} NodeRef key")
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class PatternFallback(SchemaRecord):
    RECORD_KIND = "claim_pattern_fallback"

    member_ops: tuple[EntityId, ...]
    reason: Literal["pattern_unsupported"]

    @model_validator(mode="after")
    def _validate_members(self) -> PatternFallback:
        if not self.member_ops or self.member_ops != tuple(
            sorted(set(self.member_ops))
        ):
            raise ValueError("fallback member_ops must be non-empty, sorted, and unique")
        if any(not item.startswith("op:") for item in self.member_ops):
            raise ValueError("fallback members must be operator entities")
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class ClaimPredicate(SchemaRecord):
    RECORD_KIND = "claim_predicate"

    subject: EntityId
    property: NonEmptyStr
    operator: Literal["eq", "ne", "ge", "gt", "le", "lt", "contains"]
    expected: Any
    probe_phase: NonEmptyStr
    preferred_probe: NonEmptyStr

    @model_validator(mode="after")
    def _validate_expected(self) -> ClaimPredicate:
        canonical_bytes(self.expected)
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class ClaimedStatement(SchemaRecord):
    RECORD_KIND = "claimed_statement"

    kind: NonEmptyStr
    text_normalized: NonEmptyStr
    predicate: ClaimPredicate


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class ClaimCandidate(SchemaRecord):
    RECORD_KIND = "claim_candidate"

    schema_version: Literal["claim_candidate_v1"] = "claim_candidate_v1"
    claim_id: ClaimId
    source_pr: int = Field(gt=0)
    effective_artifact: ArtifactId
    target_entities: tuple[EntityId, ...]
    pattern_fallback: PatternFallback | None
    aspect: OracleAspect
    claimed: ClaimedStatement
    domain: GuardDomain
    evidence: tuple[EvidenceSpec, ...]
    confidence: float = Field(ge=0, le=1)
    extractor_version: NonEmptyStr

    @classmethod
    def create(cls, **fields: Any) -> ClaimCandidate:
        payload = cls._payload(**fields)
        return cls(claim_id=sha256_digest(payload), **fields)

    @staticmethod
    def _payload(**fields: Any) -> dict[str, Any]:
        return {
            "source_pr": fields["source_pr"],
            "effective_artifact": fields["effective_artifact"],
            "target_entities": list(fields["target_entities"]),
            "pattern_fallback": (
                dump_value(fields["pattern_fallback"])
                if fields["pattern_fallback"] is not None
                else None
            ),
            "aspect": (
                fields["aspect"].value
                if isinstance(fields["aspect"], OracleAspect)
                else fields["aspect"]
            ),
            "claimed": dump_value(fields["claimed"]),
            "domain": dump_value(fields["domain"]),
            "evidence": [dump_value(item) for item in fields["evidence"]],
            "confidence": fields["confidence"],
            "extractor_version": fields["extractor_version"],
        }

    def to_hash_payload(self) -> dict[str, Any]:
        return self._payload(
            source_pr=self.source_pr,
            effective_artifact=self.effective_artifact,
            target_entities=self.target_entities,
            pattern_fallback=self.pattern_fallback,
            aspect=self.aspect,
            claimed=self.claimed,
            domain=self.domain,
            evidence=self.evidence,
            confidence=self.confidence,
            extractor_version=self.extractor_version,
        )

    @model_validator(mode="after")
    def _validate_claim(self) -> ClaimCandidate:
        if not self.target_entities or self.target_entities != tuple(
            sorted(set(self.target_entities))
        ):
            raise ValueError(
                "claim target_entities must be non-empty, sorted, and unique"
            )
        if self.claimed.predicate.subject not in self.target_entities:
            raise ValueError("claim predicate subject must be a target entity")
        if not self.evidence:
            raise ValueError("claims require evidence")
        expected = sha256_digest(self.to_hash_payload())
        if self.claim_id != expected:
            raise ValueError("claim_id does not match claim payload")
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class IssueNode(SchemaRecord):
    RECORD_KIND = "issue_node"

    issue_version: Literal["v1"] = "v1"
    issue_number: int = Field(gt=0)
    title: str
    html_url: str | None
    created_at: NonEmptyStr


class GraphEdgeKind(str, Enum):
    IMPLEMENTATION_TOUCHES = "implementation_touches"
    HAS_CLAIM = "has_claim"
    CLAIM_TARGETS = "claim_targets"
    STATICALLY_MENTIONS = "statically_mentions"
    SYMPTOM_MENTIONS = "symptom_mentions"
    FIXES = "fixes"
    REVERTS = "reverts"
    RELANDS = "relands"
    COMPOSED_OF = "composed_of"
    DECOMPOSES_TO = "decomposes_to"
    BOUND_TO = "bound_to"
    DERIVED_FROM_CLAIM = "derived_from_claim"
    SUPERSEDES = "supersedes"
    CONTAINS = "contains"


_EDGE_ENDPOINTS = {
    GraphEdgeKind.IMPLEMENTATION_TOUCHES: (
        GraphNodeKind.PR,
        GraphNodeKind.ENTITY,
    ),
    GraphEdgeKind.HAS_CLAIM: (GraphNodeKind.PR, GraphNodeKind.CLAIM),
    GraphEdgeKind.CLAIM_TARGETS: (
        GraphNodeKind.CLAIM,
        GraphNodeKind.ENTITY,
    ),
    GraphEdgeKind.STATICALLY_MENTIONS: (
        GraphNodeKind.PR,
        GraphNodeKind.ENTITY,
    ),
    GraphEdgeKind.SYMPTOM_MENTIONS: (
        GraphNodeKind.ISSUE,
        GraphNodeKind.ENTITY,
    ),
    GraphEdgeKind.FIXES: (GraphNodeKind.PR, GraphNodeKind.ISSUE),
    GraphEdgeKind.REVERTS: (GraphNodeKind.PR, GraphNodeKind.PR),
    GraphEdgeKind.RELANDS: (GraphNodeKind.PR, GraphNodeKind.PR),
    GraphEdgeKind.COMPOSED_OF: (
        GraphNodeKind.ENTITY,
        GraphNodeKind.ENTITY,
    ),
    GraphEdgeKind.DECOMPOSES_TO: (
        GraphNodeKind.ENTITY,
        GraphNodeKind.ENTITY,
    ),
    GraphEdgeKind.BOUND_TO: (GraphNodeKind.ORACLE, GraphNodeKind.ENTITY),
    GraphEdgeKind.DERIVED_FROM_CLAIM: (
        GraphNodeKind.ORACLE,
        GraphNodeKind.CLAIM,
    ),
    GraphEdgeKind.SUPERSEDES: (GraphNodeKind.ORACLE, GraphNodeKind.ORACLE),
    GraphEdgeKind.CONTAINS: (GraphNodeKind.ARTIFACT, GraphNodeKind.PR),
}


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class GraphEdge(SchemaRecord):
    RECORD_KIND = "graph_edge"

    edge_version: Literal["v1"] = "v1"
    edge_id: EdgeId
    kind: GraphEdgeKind
    source: NodeRef
    target: NodeRef
    evidence: tuple[EvidenceSpec, ...]
    confidence: float = Field(ge=0, le=1)
    context: dict[str, Any]
    at: NonEmptyStr

    @classmethod
    def create(cls, **fields: Any) -> GraphEdge:
        payload = cls._payload(**fields)
        return cls(edge_id=content_id("edge", payload), **fields)

    @staticmethod
    def _payload(**fields: Any) -> dict[str, Any]:
        return {
            "kind": (
                fields["kind"].value
                if isinstance(fields["kind"], GraphEdgeKind)
                else fields["kind"]
            ),
            "source": dump_value(fields["source"]),
            "target": dump_value(fields["target"]),
            "evidence": [dump_value(item) for item in fields["evidence"]],
            "confidence": fields["confidence"],
            "context": fields["context"],
            "at": fields["at"],
        }

    def to_hash_payload(self) -> dict[str, Any]:
        return self._payload(
            kind=self.kind,
            source=self.source,
            target=self.target,
            evidence=self.evidence,
            confidence=self.confidence,
            context=dict(self.context),
            at=self.at,
        )

    @model_validator(mode="after")
    def _validate_edge(self) -> GraphEdge:
        if (self.source.kind, self.target.kind) != _EDGE_ENDPOINTS[self.kind]:
            raise ValueError(f"invalid endpoints for {self.kind.value}")
        if self.kind is GraphEdgeKind.COMPOSED_OF and not (
            self.source.key.startswith("pat:") and self.target.key.startswith("op:")
        ):
            raise ValueError("composed_of requires pattern to operator entities")
        if self.kind is GraphEdgeKind.DECOMPOSES_TO and not (
            self.source.key.startswith("op:") and self.target.key.startswith("op:")
        ):
            raise ValueError("decomposes_to requires operator entities")
        if not self.evidence:
            raise ValueError("graph edges require evidence")
        if self.kind is GraphEdgeKind.STATICALLY_MENTIONS and (
            not isinstance(self.context.get("test_index"), int)
            or not isinstance(self.context.get("test_node_id"), str)
        ):
            raise ValueError(
                "statically_mentions requires test_index and test_node_id context"
            )
        canonical_bytes(self.context)
        try:
            dt.datetime.fromisoformat(self.at.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("edge timestamp must be ISO-8601") from error
        expected = content_id("edge", self.to_hash_payload())
        if self.edge_id != expected:
            raise ValueError("edge_id does not match edge payload")
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class OracleProvenanceAnnotation(SchemaRecord):
    RECORD_KIND = "oracle_provenance_annotation"

    annotation_version: Literal["v1"] = "v1"
    annotation_id: AnnotationId
    oracle_id: OracleId
    provenance: OracleProvenance

    @classmethod
    def create(
        cls,
        *,
        oracle_id: str,
        provenance: OracleProvenance,
    ) -> OracleProvenanceAnnotation:
        payload = {
            "oracle_id": oracle_id,
            "provenance": dump_value(provenance),
        }
        return cls(
            annotation_id=content_id("orcaud", payload),
            oracle_id=oracle_id,
            provenance=provenance,
        )

    def to_hash_payload(self) -> dict[str, Any]:
        return {
            "oracle_id": self.oracle_id,
            "provenance": dump_value(self.provenance),
        }

    @model_validator(mode="after")
    def _validate_id(self) -> OracleProvenanceAnnotation:
        expected = content_id("orcaud", self.to_hash_payload())
        if self.annotation_id != expected:
            raise ValueError("annotation_id does not match annotation payload")
        return self
