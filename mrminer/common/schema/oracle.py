"""Immutable oracle definitions."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import Field, StringConstraints, model_validator
from pydantic.dataclasses import dataclass

from mrminer.common.canonical import canonical_json, content_id
from mrminer.common.schema.base import SCHEMA_CONFIG, SchemaRecord, dump_value
from mrminer.common.schema.evidence import EvidenceStrength
from mrminer.common.schema.guard import GuardDomain
from mrminer.common.schema.program import ProvenanceOrigin
from mrminer.common.schema.variation import JsonScalar, VariationTemplateId

NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]
EntityId = Annotated[str, StringConstraints(pattern=r"^[a-z]+:[^\s/]+$")]
ProgramId = Annotated[
    str, StringConstraints(pattern=r"^prog:sha256:[0-9a-f]{64}$")
]
ArtifactId = Annotated[str, StringConstraints(pattern=r"^art:[^\s/]+$")]
OracleId = Annotated[
    str, StringConstraints(pattern=r"^orc:sha256:[0-9a-f]{64}$")
]
ClaimId = Annotated[str, StringConstraints(min_length=1)]


class SpecClass(str, Enum):
    ORACLE = "oracle"
    HYPOTHESIS = "hypothesis"


class OracleAspect(str, Enum):
    SEMANTIC = "semantic"
    EFFECT = "effect"
    DECISION = "decision"
    PERFORMANCE = "performance"


class TemplateId(str, Enum):
    MR_C1 = "MR-C1"
    MR_C2 = "MR-C2"
    MR_C3 = "MR-C3"
    MR_P1 = "MR-P1"
    MR_P2 = "MR-P2"
    MR_P3 = "MR-P3"


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class OracleBinding(SchemaRecord):
    RECORD_KIND = "oracle_binding"

    entity_id: EntityId
    role: Literal["primary", "member"]


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class StaticGuard(SchemaRecord):
    RECORD_KIND = "oracle_static_guard"

    domain: GuardDomain


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class TriggerSpec(SchemaRecord):
    RECORD_KIND = "oracle_trigger"

    probe: NonEmptyStr
    predicate: NonEmptyStr
    entity: EntityId
    min_count: int = Field(ge=1)


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class DynamicGuard(SchemaRecord):
    RECORD_KIND = "oracle_dynamic_guard"

    trigger: TriggerSpec | None = None


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class EnvironmentGuard(SchemaRecord):
    RECORD_KIND = "oracle_environment_guard"

    required_capabilities: tuple[NonEmptyStr, ...]

    @model_validator(mode="after")
    def _validate_capabilities(self) -> EnvironmentGuard:
        if self.required_capabilities != tuple(
            sorted(set(self.required_capabilities))
        ):
            raise ValueError("required_capabilities must be sorted and unique")
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class OracleGuard(SchemaRecord):
    RECORD_KIND = "oracle_guard"

    static: StaticGuard
    dynamic: DynamicGuard
    environment: EnvironmentGuard


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class OracleClaim(SchemaRecord):
    RECORD_KIND = "oracle_claim"

    claim_refs: tuple[ClaimId, ...]
    summary: NonEmptyStr

    @model_validator(mode="after")
    def _validate_refs(self) -> OracleClaim:
        if not self.claim_refs or self.claim_refs != tuple(
            sorted(set(self.claim_refs))
        ):
            raise ValueError("claim_refs must be non-empty, sorted, and unique")
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class EvidencePolicy(SchemaRecord):
    RECORD_KIND = "evidence_policy"

    min_strength: EvidenceStrength


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class SourceSelector(SchemaRecord):
    RECORD_KIND = "source_selector"

    origin: tuple[ProvenanceOrigin, ...]
    program_ids: tuple[ProgramId, ...]

    @model_validator(mode="after")
    def _validate_selector(self) -> SourceSelector:
        if not self.origin or self.origin != tuple(sorted(set(self.origin))):
            raise ValueError("selector origins must be non-empty, sorted, and unique")
        if not self.program_ids or self.program_ids != tuple(
            sorted(set(self.program_ids))
        ):
            raise ValueError("selector program_ids must be non-empty, sorted, and unique")
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class OracleSource(SchemaRecord):
    RECORD_KIND = "oracle_source"

    selector: SourceSelector


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class OracleVariation(SchemaRecord):
    RECORD_KIND = "oracle_variation"

    variation_template_id: VariationTemplateId
    param_grid: dict[NonEmptyStr, tuple[JsonScalar, ...]]

    @model_validator(mode="after")
    def _validate_grid(self) -> OracleVariation:
        for name, values in self.param_grid.items():
            if not name.startswith("$"):
                raise ValueError("param_grid names must start with $")
            encoded = tuple(canonical_json(value) for value in values)
            if not encoded or encoded != tuple(sorted(set(encoded))):
                raise ValueError("param_grid values must be canonical-sorted and unique")
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class RequiredObservation(SchemaRecord):
    RECORD_KIND = "required_observation"

    role: NonEmptyStr
    run_kind: Literal["single", "paired_semantic", "performance"]
    probes: tuple[NonEmptyStr, ...]

    @model_validator(mode="after")
    def _validate_probes(self) -> RequiredObservation:
        if self.probes != tuple(sorted(set(self.probes))):
            raise ValueError("observation probes must be sorted and unique")
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class RelationSpec(SchemaRecord):
    RECORD_KIND = "oracle_relation"

    checker_id: NonEmptyStr
    checker_version: NonEmptyStr
    params: dict[str, Any]
    required_observations: tuple[RequiredObservation, ...]

    @model_validator(mode="after")
    def _validate_relation(self) -> RelationSpec:
        canonical_json(self.params)
        roles = tuple(item.role for item in self.required_observations)
        if not roles or len(roles) != len(set(roles)):
            raise ValueError("required observation roles must be non-empty and unique")
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class ValiditySpec(SchemaRecord):
    RECORD_KIND = "oracle_validity"

    from_artifact: ArtifactId
    to_artifact: ArtifactId | None = None

    @model_validator(mode="after")
    def _validate_range(self) -> ValiditySpec:
        if self.to_artifact == self.from_artifact:
            raise ValueError("validity bounds must differ")
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class LineageSpec(SchemaRecord):
    RECORD_KIND = "oracle_lineage"

    generation: int = Field(ge=1)
    predecessors: tuple[OracleId, ...]

    @model_validator(mode="after")
    def _validate_generation(self) -> LineageSpec:
        if self.predecessors != tuple(sorted(set(self.predecessors))):
            raise ValueError("lineage predecessors must be sorted and unique")
        if self.generation == 1 and self.predecessors:
            raise ValueError("first-generation oracles cannot have predecessors")
        if self.generation > 1 and not self.predecessors:
            raise ValueError("later oracle generations require predecessors")
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class LlmAudit(SchemaRecord):
    RECORD_KIND = "llm_audit"

    model: NonEmptyStr
    prompt_sha: NonEmptyStr


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class OracleProvenance(SchemaRecord):
    RECORD_KIND = "oracle_provenance"

    pipeline_version: NonEmptyStr
    llm: LlmAudit | None
    created_at: NonEmptyStr


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class OracleSpec(SchemaRecord):
    RECORD_KIND = "oracle"

    schema_version: Literal["oracle_spec_v1"] = "oracle_spec_v1"
    oracle_id: OracleId
    spec_class: SpecClass
    aspect: OracleAspect
    template_id: TemplateId
    priority: int | None
    binding: tuple[OracleBinding, ...]
    guard: OracleGuard
    claim: OracleClaim
    evidence_policy: EvidencePolicy
    source: OracleSource
    variation: OracleVariation | None
    relation: RelationSpec
    missing_probe_outcome: Literal["unsupported"] = "unsupported"
    validity: ValiditySpec
    lineage: LineageSpec
    provenance: OracleProvenance

    @classmethod
    def create(cls, **fields: Any) -> OracleSpec:
        payload = cls._payload(**fields)
        return cls(oracle_id=content_id("orc", payload), **fields)

    @staticmethod
    def _payload(**fields: Any) -> dict[str, Any]:
        return {
            name: (
                dump_value(fields[name])
                if not isinstance(fields[name], (str, int, float, bool, type(None)))
                else fields[name]
            )
            for name in (
                "spec_class",
                "aspect",
                "template_id",
                "priority",
                "binding",
                "guard",
                "claim",
                "evidence_policy",
                "source",
                "variation",
                "relation",
                "missing_probe_outcome",
                "validity",
                "lineage",
            )
        }

    def to_hash_payload(self) -> dict[str, Any]:
        return self._payload(
            spec_class=self.spec_class,
            aspect=self.aspect,
            template_id=self.template_id,
            priority=self.priority,
            binding=self.binding,
            guard=self.guard,
            claim=self.claim,
            evidence_policy=self.evidence_policy,
            source=self.source,
            variation=self.variation,
            relation=self.relation,
            missing_probe_outcome=self.missing_probe_outcome,
            validity=self.validity,
            lineage=self.lineage,
        )

    @model_validator(mode="after")
    def _validate_oracle(self) -> OracleSpec:
        primary = [item for item in self.binding if item.role == "primary"]
        if len(primary) != 1:
            raise ValueError("oracle binding requires exactly one primary entity")
        if len({item.entity_id for item in self.binding}) != len(self.binding):
            raise ValueError("oracle binding entities must be unique")
        if self.aspect in {OracleAspect.EFFECT, OracleAspect.DECISION} and (
            self.evidence_policy.min_strength is not EvidenceStrength.EXPLICIT_CLAIM
        ):
            raise ValueError("effect and decision oracles require explicit_claim")
        expected = content_id("orc", self.to_hash_payload())
        if self.oracle_id != expected:
            raise ValueError("oracle_id does not match semantic payload")
        return self
