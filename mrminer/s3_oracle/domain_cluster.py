"""Conservative guard-domain relations and exact-only claim clustering."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from mrminer.common.canonical import canonical_json, sha256_digest
from mrminer.common.schema import GuardDimension, GuardDomain, GuardMode
from mrminer.s2_graph import ClaimCandidate

_DIMENSIONS = (
    "dtype",
    "device",
    "overload",
    "layout",
    "shape_mode",
    "phase",
    "backend",
)


class DomainRelation(str, Enum):
    EXACT = "exact"
    CONTAINS = "contains"
    CONTAINED_BY = "contained_by"
    PARTIAL = "partial"
    DISJOINT = "disjoint"
    UNKNOWN = "unknown"


@dataclass(frozen=True, kw_only=True)
class ClaimCluster:
    cluster_id: str
    entity_id: str
    aspect: str
    domain: GuardDomain
    claims: tuple[ClaimCandidate, ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "entity_id": self.entity_id,
            "aspect": self.aspect,
            "domain": self.domain.to_record(),
            "claim_ids": [claim.claim_id for claim in self.claims],
        }


@dataclass(frozen=True, kw_only=True)
class ClusterRelation:
    left_cluster_id: str
    right_cluster_id: str
    relation: DomainRelation


@dataclass(frozen=True, kw_only=True)
class ClusterResult:
    clusters: tuple[ClaimCluster, ...]
    relations: tuple[ClusterRelation, ...]


def domain_semantic_payload(domain: GuardDomain) -> dict[str, Any]:
    return {
        name: {
            "mode": getattr(domain, name).mode.value,
            "values": list(getattr(domain, name).values),
        }
        for name in _DIMENSIONS
    } | {
        "shape_constraints": (
            domain.shape_constraints.to_record()
            if domain.shape_constraints is not None
            else None
        )
    }


def domain_semantic_key(domain: GuardDomain) -> str:
    return canonical_json(domain_semantic_payload(domain))


def _dimension_relation(
    left: GuardDimension,
    right: GuardDimension,
) -> DomainRelation:
    if left.mode is right.mode and left.values == right.values:
        return DomainRelation.EXACT
    if GuardMode.UNKNOWN in {left.mode, right.mode}:
        return DomainRelation.UNKNOWN
    if left.mode is GuardMode.ANY:
        return DomainRelation.CONTAINS
    if right.mode is GuardMode.ANY:
        return DomainRelation.CONTAINED_BY
    left_values = set(left.values)
    right_values = set(right.values)
    if left_values.isdisjoint(right_values):
        return DomainRelation.DISJOINT
    if left_values > right_values:
        return DomainRelation.CONTAINS
    if left_values < right_values:
        return DomainRelation.CONTAINED_BY
    return DomainRelation.PARTIAL


def _shape_relation(left: GuardDomain, right: GuardDomain) -> DomainRelation:
    if left.shape_constraints == right.shape_constraints:
        return DomainRelation.EXACT
    if left.shape_constraints is None:
        return DomainRelation.CONTAINS
    if right.shape_constraints is None:
        return DomainRelation.CONTAINED_BY
    return DomainRelation.UNKNOWN


def domain_relation(left: GuardDomain, right: GuardDomain) -> DomainRelation:
    relations = [
        _dimension_relation(getattr(left, name), getattr(right, name))
        for name in _DIMENSIONS
    ]
    relations.append(_shape_relation(left, right))
    if DomainRelation.DISJOINT in relations:
        return DomainRelation.DISJOINT
    if DomainRelation.UNKNOWN in relations:
        return DomainRelation.UNKNOWN
    directional = {
        item for item in relations if item is not DomainRelation.EXACT
    }
    if not directional:
        return DomainRelation.EXACT
    if directional == {DomainRelation.CONTAINS}:
        return DomainRelation.CONTAINS
    if directional == {DomainRelation.CONTAINED_BY}:
        return DomainRelation.CONTAINED_BY
    return DomainRelation.PARTIAL


def cluster_claims(
    claims: list[ClaimCandidate] | tuple[ClaimCandidate, ...],
    *,
    entity_id: str,
) -> ClusterResult:
    relevant = [claim for claim in claims if entity_id in claim.target_entities]
    grouped: dict[tuple[str, str], list[ClaimCandidate]] = {}
    for claim in relevant:
        key = (claim.aspect.value, domain_semantic_key(claim.domain))
        grouped.setdefault(key, []).append(claim)
    clusters: list[ClaimCluster] = []
    for (aspect, domain_key), members in sorted(grouped.items()):
        ordered = tuple(
            sorted(members, key=lambda item: (item.source_pr, item.claim_id))
        )
        cluster_id = sha256_digest(
            {
                "entity_id": entity_id,
                "aspect": aspect,
                "domain": domain_key,
            }
        )
        clusters.append(
            ClaimCluster(
                cluster_id=cluster_id,
                entity_id=entity_id,
                aspect=aspect,
                domain=ordered[0].domain,
                claims=ordered,
            )
        )
    relations = tuple(
        ClusterRelation(
            left_cluster_id=left.cluster_id,
            right_cluster_id=right.cluster_id,
            relation=domain_relation(left.domain, right.domain),
        )
        for index, left in enumerate(clusters)
        for right in clusters[index + 1 :]
        if left.aspect == right.aspect
    )
    return ClusterResult(clusters=tuple(clusters), relations=relations)
