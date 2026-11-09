"""Entity and restricted pattern schemas."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import Field, StringConstraints, model_validator
from pydantic.dataclasses import dataclass

from mrminer.common.canonical import sha256_digest
from mrminer.common.schema.base import SCHEMA_CONFIG, SchemaRecord, dump_value

NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]
EntityId = Annotated[str, StringConstraints(pattern=r"^[a-z]+:[^\s/]+$")]


class EntityKind(str, Enum):
    OPERATOR = "operator"
    API = "api"
    PATTERN = "pattern"
    CONFIG = "config"
    PASS = "pass"
    FEATURE = "feature"
    SUBSYSTEM = "subsystem"


class EntitySource(str, Enum):
    REGISTRY = "registry"
    LINKER = "minted_by_linker"
    MANUAL = "manual"


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class PatternNode(SchemaRecord):
    RECORD_KIND = "pattern_node"

    idx: int = Field(ge=0)
    kind: Literal["wildcard", "op"]
    wc: NonEmptyStr | None = None
    type_constraint: Literal["tensor", "scalar"] | None = None
    op: EntityId | None = None
    overload: NonEmptyStr | None = None

    @model_validator(mode="after")
    def _validate_kind_fields(self) -> PatternNode:
        if self.kind == "wildcard":
            if self.wc is None or self.type_constraint is None:
                raise ValueError("wildcard nodes require wc and type_constraint")
            if self.op is not None or self.overload is not None:
                raise ValueError("wildcard nodes cannot define op or overload")
        else:
            if self.op is None or not self.op.startswith("op:"):
                raise ValueError("op nodes require an operator entity ID")
            if self.overload is None:
                raise ValueError("op nodes require overload")
            if self.wc is not None or self.type_constraint is not None:
                raise ValueError("op nodes cannot define wildcard fields")
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class PatternEdge(SchemaRecord):
    RECORD_KIND = "pattern_edge"

    from_idx: int = Field(alias="from", ge=0)
    to_idx: int = Field(alias="to", ge=0)
    arg_pos: int = Field(ge=0)


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class PatternConstraint(SchemaRecord):
    RECORD_KIND = "pattern_constraint"

    dimension: Literal["dtype", "device"]
    operator: Literal["eq", "in"]
    values: tuple[NonEmptyStr, ...]

    @model_validator(mode="after")
    def _validate_values(self) -> PatternConstraint:
        if not self.values or self.values != tuple(sorted(set(self.values))):
            raise ValueError("constraint values must be non-empty, sorted, and unique")
        if self.operator == "eq" and len(self.values) != 1:
            raise ValueError("eq constraints require exactly one value")
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class PatternBoundary(SchemaRecord):
    RECORD_KIND = "pattern_boundary"

    inputs: tuple[int, ...]
    output: int = Field(ge=0)


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class PatternSpec(SchemaRecord):
    RECORD_KIND = "pattern_spec"

    pattern_spec_version: Literal["restricted_v1"] = "restricted_v1"
    nodes: tuple[PatternNode, ...]
    edges: tuple[PatternEdge, ...]
    constraints: tuple[PatternConstraint, ...] = ()
    boundary: PatternBoundary
    canonical_node_order: tuple[int, ...]
    canonical_hash: Annotated[
        str,
        StringConstraints(pattern=r"^restricted_v1:sha256:[0-9a-f]{64}$"),
    ]

    @classmethod
    def create(
        cls,
        *,
        nodes: tuple[PatternNode, ...],
        edges: tuple[PatternEdge, ...],
        boundary: PatternBoundary,
        canonical_node_order: tuple[int, ...],
        constraints: tuple[PatternConstraint, ...] = (),
    ) -> PatternSpec:
        payload = cls._hash_payload(
            nodes, edges, constraints, boundary, canonical_node_order
        )
        return cls(
            nodes=nodes,
            edges=edges,
            constraints=constraints,
            boundary=boundary,
            canonical_node_order=canonical_node_order,
            canonical_hash=f"restricted_v1:{sha256_digest(payload)}",
        )

    @staticmethod
    def _hash_payload(
        nodes: tuple[PatternNode, ...],
        edges: tuple[PatternEdge, ...],
        constraints: tuple[PatternConstraint, ...],
        boundary: PatternBoundary,
        canonical_node_order: tuple[int, ...],
    ) -> dict[str, Any]:
        return {
            "nodes": [dump_value(node) for node in nodes],
            "edges": [dump_value(edge) for edge in edges],
            "constraints": [dump_value(item) for item in constraints],
            "boundary": dump_value(boundary),
            "canonical_node_order": list(canonical_node_order),
        }

    def to_hash_payload(self) -> dict[str, Any]:
        return self._hash_payload(
            self.nodes,
            self.edges,
            self.constraints,
            self.boundary,
            self.canonical_node_order,
        )

    @model_validator(mode="after")
    def _validate_graph(self) -> PatternSpec:
        if not 1 <= len(self.nodes) <= 5:
            raise ValueError("restricted patterns require one to five nodes")
        node_by_idx = {node.idx: node for node in self.nodes}
        if len(node_by_idx) != len(self.nodes):
            raise ValueError("pattern node indices must be unique")
        node_indices = set(node_by_idx)
        if set(self.canonical_node_order) != node_indices or len(
            self.canonical_node_order
        ) != len(self.nodes):
            raise ValueError("canonical_node_order must be a permutation of node indices")
        edge_keys: set[tuple[int, int, int]] = set()
        for edge in self.edges:
            if edge.from_idx not in node_indices or edge.to_idx not in node_indices:
                raise ValueError("pattern edges must reference existing nodes")
            key = (edge.from_idx, edge.to_idx, edge.arg_pos)
            if key in edge_keys:
                raise ValueError("pattern edges must be unique")
            edge_keys.add(key)
        if not self.boundary.inputs:
            raise ValueError("pattern boundary requires at least one input")
        if any(
            idx not in node_by_idx or node_by_idx[idx].kind != "wildcard"
            for idx in self.boundary.inputs
        ):
            raise ValueError("pattern boundary inputs must reference wildcard nodes")
        output = node_by_idx.get(self.boundary.output)
        if output is None or output.kind != "op":
            raise ValueError("pattern boundary output must reference an op node")
        expected = f"restricted_v1:{sha256_digest(self.to_hash_payload())}"
        if self.canonical_hash != expected:
            raise ValueError("canonical_hash does not match pattern payload")
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class OperatorIntrinsic(SchemaRecord):
    RECORD_KIND = "operator_intrinsic"

    overloads: tuple[NonEmptyStr, ...]
    has_lowering: bool
    has_decomp: bool
    has_meta: bool
    decomp_targets: tuple[EntityId, ...] = ()


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class ApiIntrinsic(SchemaRecord):
    """Application-layer public Python API operator (graph_model_v1 §3).

    ``resolved_aten`` records the aten operators this API is known to use (via
    lowering/tracing); it may be empty when resolution is unavailable, in which
    case the graph keeps an aten-operator node as fallback rather than an API
    node (see graph_model_v1 §5).
    """

    RECORD_KIND = "api_intrinsic"

    qualname: NonEmptyStr
    api_kind: Literal["function", "module", "method"]
    resolved_aten: tuple[EntityId, ...] = ()


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class PatternIntrinsic(SchemaRecord):
    RECORD_KIND = "pattern_intrinsic"

    pattern_spec: PatternSpec


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class ConfigIntrinsic(SchemaRecord):
    RECORD_KIND = "config_intrinsic"

    config_path: NonEmptyStr
    value_space: NonEmptyStr


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class PassIntrinsic(SchemaRecord):
    RECORD_KIND = "pass_intrinsic"

    pass_path: NonEmptyStr


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class FeatureIntrinsic(SchemaRecord):
    RECORD_KIND = "feature_intrinsic"

    scope: Literal["frontend", "backend"]


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class SubsystemIntrinsic(SchemaRecord):
    RECORD_KIND = "subsystem_intrinsic"

    label: NonEmptyStr


Intrinsic = (
    OperatorIntrinsic
    | ApiIntrinsic
    | PatternIntrinsic
    | ConfigIntrinsic
    | PassIntrinsic
    | FeatureIntrinsic
    | SubsystemIntrinsic
)


def api_operator_id(qualname: str) -> str:
    """Content-independent ID for a public Python API operator, e.g.
    ``api:torch.nn.functional.linear``."""

    if not qualname or any(ch.isspace() or ch == "/" for ch in qualname):
        raise ValueError(f"invalid API qualname: {qualname!r}")
    return f"api:{qualname}"


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class EntitySpec(SchemaRecord):
    RECORD_KIND = "entity"

    schema_version: Literal["entity_spec_v1"] = "entity_spec_v1"
    entity_kind: EntityKind
    entity_id: EntityId
    display_name: NonEmptyStr
    intrinsic: Intrinsic
    first_seen: NonEmptyStr
    source: EntitySource

    @model_validator(mode="after")
    def _validate_kind(self) -> EntitySpec:
        expected = {
            EntityKind.OPERATOR: ("op:", OperatorIntrinsic),
            EntityKind.API: ("api:", ApiIntrinsic),
            EntityKind.PATTERN: ("pat:", PatternIntrinsic),
            EntityKind.CONFIG: ("cfg:", ConfigIntrinsic),
            EntityKind.PASS: ("pass:", PassIntrinsic),
            EntityKind.FEATURE: ("feat:", FeatureIntrinsic),
            EntityKind.SUBSYSTEM: ("sys:", SubsystemIntrinsic),
        }
        prefix, intrinsic_type = expected[self.entity_kind]
        if not self.entity_id.startswith(prefix):
            raise ValueError(f"{self.entity_kind.value} entity IDs must start with {prefix}")
        if not isinstance(self.intrinsic, intrinsic_type):
            raise ValueError("entity intrinsic does not match entity_kind")
        if isinstance(self.intrinsic, PatternIntrinsic):
            expected_id = f"pat:{self.intrinsic.pattern_spec.canonical_hash}"
            if self.entity_id != expected_id:
                raise ValueError("pattern entity_id must derive from canonical_hash")
        return self
