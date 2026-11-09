"""Immutable observations produced by RunSpec attempts."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import Field, StringConstraints, model_validator
from pydantic.dataclasses import dataclass

from mrminer.common.canonical import canonical_bytes, content_id
from mrminer.common.schema.base import SCHEMA_CONFIG, SchemaRecord, dump_value
from mrminer.common.schema.run import RunId, RunKind

NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]
ObservationId = Annotated[
    str, StringConstraints(pattern=r"^obs:sha256:[0-9a-f]{64}$")
]
ProgramId = Annotated[
    str, StringConstraints(pattern=r"^prog:sha256:[0-9a-f]{64}$")
]
InputId = Annotated[str, StringConstraints(pattern=r"^inp:sha256:[0-9a-f]{64}$")]
ArtifactId = Annotated[str, StringConstraints(pattern=r"^art:[^\s/]+$")]
Sha256Ref = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]


class ObservationStatus(str, Enum):
    OK = "ok"
    EAGER_FAIL = "eager_fail"
    COMPILE_FAIL = "compile_fail"
    CRASH = "crash"
    TIMEOUT = "timeout"


class SemanticStability(str, Enum):
    STABLE = "stable"
    REFERENCE_UNSTABLE = "reference_unstable"


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class ObservationIdentity(SchemaRecord):
    RECORD_KIND = "observation_identity"

    program_id: ProgramId
    input_id: InputId
    artifact_id: ArtifactId
    compile_config_hash: Sha256Ref
    runtime_config_hash: Sha256Ref
    device_class: NonEmptyStr
    phase: NonEmptyStr
    run_kind: RunKind
    probe_bundle_version: NonEmptyStr
    probes: tuple[NonEmptyStr, ...]
    probe_semantics_versions: dict[NonEmptyStr, NonEmptyStr]
    functional_env_hash: Sha256Ref
    performance_env_hash: Sha256Ref

    @model_validator(mode="after")
    def _validate_probes(self) -> ObservationIdentity:
        if not self.probes or self.probes != tuple(sorted(set(self.probes))):
            raise ValueError("observation probes must be non-empty, sorted, and unique")
        if set(self.probe_semantics_versions) != set(self.probes):
            raise ValueError("probe semantics versions must cover exactly the probes")
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class ObservationProvenance(SchemaRecord):
    RECORD_KIND = "observation_provenance"

    env_fingerprint: dict[str, Any]
    stdout_ref: NonEmptyStr | None
    stderr_ref: NonEmptyStr | None
    exit_code: int | None
    timeout_phase: NonEmptyStr | None
    native_signal: NonEmptyStr | None

    @model_validator(mode="after")
    def _validate_environment(self) -> ObservationProvenance:
        canonical_bytes(self.env_fingerprint)
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class ObservationError(SchemaRecord):
    RECORD_KIND = "observation_error"

    stage: Literal["eager", "compile", "runtime", "probe", "environment", "harness"]
    exception_type: NonEmptyStr
    stack_ref: NonEmptyStr


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class TensorSummary(SchemaRecord):
    RECORD_KIND = "tensor_summary"

    shape: tuple[int, ...]
    dtype: NonEmptyStr
    stride: tuple[int, ...]
    value_digest: Sha256Ref
    nan_count: int = Field(ge=0)
    inf_count: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_tensor(self) -> TensorSummary:
        if len(self.shape) != len(self.stride):
            raise ValueError("tensor shape and stride ranks must match")
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class SemanticSide(SchemaRecord):
    RECORD_KIND = "semantic_side"

    output_tree_spec: NonEmptyStr
    tensor_summaries: tuple[TensorSummary, ...]
    mutation_summary: dict[str, Any]
    alias_summary: dict[str, Any]
    gradient_summary: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _validate_summaries(self) -> SemanticSide:
        canonical_bytes(
            {
                "mutation": self.mutation_summary,
                "alias": self.alias_summary,
                "gradient": self.gradient_summary,
            }
        )
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class PairwiseCompare(SchemaRecord):
    RECORD_KIND = "pairwise_compare"

    allclose: bool
    max_abs_err: float = Field(ge=0)
    max_rel_err: float = Field(ge=0)
    rtol_used: float = Field(ge=0)
    atol_used: float = Field(ge=0)
    mismatch_ratio: float = Field(ge=0, le=1)
    structure_equal: bool
    mutation_equal: bool
    alias_equal: bool


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class SemanticObservation(SchemaRecord):
    RECORD_KIND = "semantic_observation"

    reference: SemanticSide
    candidate: SemanticSide
    pairwise_compare: PairwiseCompare
    stability: SemanticStability
    exception: ObservationError | None = None


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class DecisionObservation(SchemaRecord):
    RECORD_KIND = "decision_observation"

    pre_optimization: dict[str, Any]
    post_pass: dict[str, Any]
    post_codegen: dict[str, Any]
    graph_breaks: tuple[dict[str, Any], ...]
    fallbacks: tuple[dict[str, Any], ...]
    counters: dict[str, Any]
    probe_versions: dict[NonEmptyStr, NonEmptyStr]

    @model_validator(mode="after")
    def _validate_decision(self) -> DecisionObservation:
        canonical_bytes(self.to_record())
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class ProcessRep(SchemaRecord):
    RECORD_KIND = "performance_process_rep"

    proc: int = Field(ge=1)
    warmup_discarded: int = Field(ge=0)
    samples_us: tuple[float, ...]

    @model_validator(mode="after")
    def _validate_samples(self) -> ProcessRep:
        if len(self.samples_us) < 2 or any(sample < 0 for sample in self.samples_us):
            raise ValueError("performance reps require at least two non-negative samples")
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class PerformanceObservation(SchemaRecord):
    RECORD_KIND = "performance_observation"

    process_reps: tuple[ProcessRep, ...]
    compile_counters: dict[str, Any]
    clock: Literal["cuda_events", "perf_counter"]

    @model_validator(mode="after")
    def _validate_reps(self) -> PerformanceObservation:
        if len(self.process_reps) < 2:
            raise ValueError("performance observations require process-level replication")
        proc_ids = tuple(rep.proc for rep in self.process_reps)
        if len(proc_ids) != len(set(proc_ids)):
            raise ValueError("performance process IDs must be unique")
        canonical_bytes(self.compile_counters)
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class ObservationSpec(SchemaRecord):
    RECORD_KIND = "observation"

    schema_version: Literal["observation_v1"] = "observation_v1"
    obs_id: ObservationId
    run_id: RunId
    attempt: int = Field(ge=1)
    identity: ObservationIdentity
    provenance: ObservationProvenance
    status: ObservationStatus
    error: ObservationError | None
    semantic: SemanticObservation | None
    decision: DecisionObservation | None
    performance: PerformanceObservation | None

    @classmethod
    def create(cls, **fields: Any) -> ObservationSpec:
        payload = cls._payload(**fields)
        return cls(obs_id=content_id("obs", payload), **fields)

    @staticmethod
    def _payload(**fields: Any) -> dict[str, Any]:
        return {
            "run_id": fields["run_id"],
            "attempt": fields["attempt"],
            "identity": dump_value(fields["identity"]),
            "provenance": dump_value(fields["provenance"]),
            "status": (
                fields["status"].value
                if isinstance(fields["status"], ObservationStatus)
                else fields["status"]
            ),
            "error": dump_value(fields["error"]) if fields["error"] else None,
            "semantic": (
                dump_value(fields["semantic"]) if fields["semantic"] else None
            ),
            "decision": (
                dump_value(fields["decision"]) if fields["decision"] else None
            ),
            "performance": (
                dump_value(fields["performance"]) if fields["performance"] else None
            ),
        }

    def to_hash_payload(self) -> dict[str, Any]:
        return self._payload(
            run_id=self.run_id,
            attempt=self.attempt,
            identity=self.identity,
            provenance=self.provenance,
            status=self.status,
            error=self.error,
            semantic=self.semantic,
            decision=self.decision,
            performance=self.performance,
        )

    @model_validator(mode="after")
    def _validate_observation(self) -> ObservationSpec:
        if self.status is ObservationStatus.OK and self.error is not None:
            raise ValueError("successful observations cannot define an error")
        if self.status is not ObservationStatus.OK and self.error is None:
            raise ValueError("failed observations require an error")
        if self.identity.run_kind is RunKind.PAIRED_SEMANTIC:
            if self.status is ObservationStatus.OK and self.semantic is None:
                raise ValueError("successful paired runs require semantic observations")
            if self.performance is not None:
                raise ValueError("paired semantic observations cannot contain performance")
        elif self.semantic is not None:
            raise ValueError("only paired semantic observations may contain semantic data")
        if self.identity.run_kind is RunKind.PERFORMANCE:
            if self.status is ObservationStatus.OK and self.performance is None:
                raise ValueError("successful performance runs require performance data")
        elif self.performance is not None:
            raise ValueError("only performance observations may contain performance data")
        expected = content_id("obs", self.to_hash_payload())
        if self.obs_id != expected:
            raise ValueError("obs_id does not match observation payload")
        return self
