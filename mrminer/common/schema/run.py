"""Execution, relation-check, and attempt-projection schemas."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import Field, StringConstraints, model_validator
from pydantic.dataclasses import dataclass

from mrminer.common.canonical import canonical_bytes, content_id
from mrminer.common.schema.base import SCHEMA_CONFIG, SchemaRecord, dump_value

NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]
RunId = Annotated[str, StringConstraints(pattern=r"^run:sha256:[0-9a-f]{64}$")]
CheckId = Annotated[str, StringConstraints(pattern=r"^chk:sha256:[0-9a-f]{64}$")]
ProgramId = Annotated[
    str, StringConstraints(pattern=r"^prog:sha256:[0-9a-f]{64}$")
]
InputId = Annotated[str, StringConstraints(pattern=r"^inp:sha256:[0-9a-f]{64}$")]
ArtifactId = Annotated[str, StringConstraints(pattern=r"^art:[^\s/]+$")]
OracleId = Annotated[
    str, StringConstraints(pattern=r"^orc:sha256:[0-9a-f]{64}$")
]
CaseId = Annotated[
    str, StringConstraints(pattern=r"^case:sha256:[0-9a-f]{64}$")
]
ObservationRef = Annotated[
    str, StringConstraints(pattern=r"^(obs|aggregate):sha256:[0-9a-f]{64}$")
]


class RunKind(str, Enum):
    SINGLE = "single"
    PAIRED_SEMANTIC = "paired_semantic"
    PERFORMANCE = "performance"


class InputPolicy(str, Enum):
    DEEP_CLONE = "deep_clone_preserve_layout_alias"
    SERIALIZED_FIXTURE = "serialized_fixture"


class ModulePolicy(str, Enum):
    RECONSTRUCT_TWICE = "reconstruct_twice"
    DEEPCOPY = "deepcopy"
    STATE_DICT_RESTORE = "state_dict_restore"


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class GlobalContextPolicy(SchemaRecord):
    RECORD_KIND = "global_context_policy"

    grad_mode: Literal["same"] = "same"
    autocast: Literal["same"] = "same"
    deterministic_algorithms: Literal[True] = True


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class BackendCachePolicy(SchemaRecord):
    RECORD_KIND = "backend_cache_policy"

    dynamo_reset: bool
    inductor_cache: Literal["fresh", "reuse"]


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class StateIsolationPolicy(SchemaRecord):
    RECORD_KIND = "state_isolation_policy"

    input_policy: InputPolicy
    module_policy: ModulePolicy
    rng_policy: Literal["snapshot_restore"] = "snapshot_restore"
    global_context: GlobalContextPolicy
    backend_cache: BackendCachePolicy
    candidate_compile_timing: Literal["after_restore"] = "after_restore"
    reference_replay: int = Field(ge=2)


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class PairedSemanticPolicy(SchemaRecord):
    RECORD_KIND = "paired_semantic_policy"

    reference_mode: Literal["eager"] = "eager"
    candidate_mode: Literal["compiled"] = "compiled"
    same_process: Literal[True] = True
    state_isolation: StateIsolationPolicy


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class ProbeBundle(SchemaRecord):
    RECORD_KIND = "probe_bundle"

    version: NonEmptyStr
    probes: tuple[NonEmptyStr, ...]

    @model_validator(mode="after")
    def _validate_probes(self) -> ProbeBundle:
        if not self.probes or self.probes != tuple(sorted(set(self.probes))):
            raise ValueError("probe bundle must be non-empty, sorted, and unique")
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class PerformanceRepetitions(SchemaRecord):
    RECORD_KIND = "performance_repetitions"

    processes: int = Field(ge=2)
    in_proc_samples: int = Field(ge=2)
    warmup: int = Field(ge=0)


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class ExecutionPolicy(SchemaRecord):
    RECORD_KIND = "execution_policy"

    isolation: Literal["subprocess"] = "subprocess"
    timeout_s: int = Field(gt=0)
    perf_reps: PerformanceRepetitions | None = None


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class RunSpec(SchemaRecord):
    RECORD_KIND = "run_spec"

    schema_version: Literal["run_spec_v1"] = "run_spec_v1"
    run_id: RunId
    run_kind: RunKind
    program_id: ProgramId
    input_id: InputId
    artifact_id: ArtifactId
    compile_config: dict[str, Any]
    runtime_config: dict[str, Any]
    device_class: NonEmptyStr
    phase: NonEmptyStr
    paired_semantic: PairedSemanticPolicy | None
    probe_bundle: ProbeBundle
    execution_policy: ExecutionPolicy

    @classmethod
    def create(cls, **fields: Any) -> RunSpec:
        payload = cls._payload(**fields)
        return cls(run_id=content_id("run", payload), **fields)

    @staticmethod
    def _payload(**fields: Any) -> dict[str, Any]:
        return {
            "run_kind": (
                fields["run_kind"].value
                if isinstance(fields["run_kind"], RunKind)
                else fields["run_kind"]
            ),
            "program_id": fields["program_id"],
            "input_id": fields["input_id"],
            "artifact_id": fields["artifact_id"],
            "compile_config": fields["compile_config"],
            "runtime_config": fields["runtime_config"],
            "device_class": fields["device_class"],
            "phase": fields["phase"],
            "paired_semantic": (
                dump_value(fields["paired_semantic"])
                if fields["paired_semantic"] is not None
                else None
            ),
            "probe_bundle": dump_value(fields["probe_bundle"]),
            "execution_policy": dump_value(fields["execution_policy"]),
        }

    def to_hash_payload(self) -> dict[str, Any]:
        return self._payload(
            run_kind=self.run_kind,
            program_id=self.program_id,
            input_id=self.input_id,
            artifact_id=self.artifact_id,
            compile_config=dict(self.compile_config),
            runtime_config=dict(self.runtime_config),
            device_class=self.device_class,
            phase=self.phase,
            paired_semantic=self.paired_semantic,
            probe_bundle=self.probe_bundle,
            execution_policy=self.execution_policy,
        )

    @model_validator(mode="after")
    def _validate_run(self) -> RunSpec:
        canonical_bytes(
            {
                "compile_config": self.compile_config,
                "runtime_config": self.runtime_config,
            }
        )
        if self.run_kind is RunKind.PAIRED_SEMANTIC:
            if self.paired_semantic is None:
                raise ValueError("paired_semantic runs require paired policy")
            if self.execution_policy.perf_reps is not None:
                raise ValueError("paired_semantic runs cannot define perf_reps")
        elif self.paired_semantic is not None:
            raise ValueError("only paired_semantic runs may define paired policy")
        if self.run_kind is RunKind.PERFORMANCE:
            if self.execution_policy.perf_reps is None:
                raise ValueError("performance runs require perf_reps")
        elif self.execution_policy.perf_reps is not None:
            raise ValueError("only performance runs may define perf_reps")
        expected = content_id("run", self.to_hash_payload())
        if self.run_id != expected:
            raise ValueError("run_id does not match semantic payload")
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class RoleBinding(SchemaRecord):
    RECORD_KIND = "relation_role_binding"

    run_id: RunId
    use: Literal["canonical"] | ObservationRef


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class RelationCheckSpec(SchemaRecord):
    RECORD_KIND = "relation_check"

    schema_version: Literal["relation_check_v1"] = "relation_check_v1"
    check_id: CheckId
    oracle_id: OracleId
    case_id: CaseId
    role_bindings: dict[NonEmptyStr, RoleBinding]
    checker_id: NonEmptyStr
    checker_version: NonEmptyStr
    params_snapshot: dict[str, Any]

    @classmethod
    def create(cls, **fields: Any) -> RelationCheckSpec:
        payload = cls._payload(**fields)
        return cls(check_id=content_id("chk", payload), **fields)

    @staticmethod
    def _payload(**fields: Any) -> dict[str, Any]:
        return {
            "oracle_id": fields["oracle_id"],
            "case_id": fields["case_id"],
            "role_bindings": {
                role: dump_value(binding)
                for role, binding in fields["role_bindings"].items()
            },
            "checker_id": fields["checker_id"],
            "checker_version": fields["checker_version"],
            "params_snapshot": fields["params_snapshot"],
        }

    def to_hash_payload(self) -> dict[str, Any]:
        return self._payload(
            oracle_id=self.oracle_id,
            case_id=self.case_id,
            role_bindings=dict(self.role_bindings),
            checker_id=self.checker_id,
            checker_version=self.checker_version,
            params_snapshot=dict(self.params_snapshot),
        )

    @model_validator(mode="after")
    def _validate_check(self) -> RelationCheckSpec:
        if not self.role_bindings:
            raise ValueError("relation checks require at least one role binding")
        canonical_bytes(self.params_snapshot)
        expected = content_id("chk", self.to_hash_payload())
        if self.check_id != expected:
            raise ValueError("check_id does not match semantic payload")
        return self


class SelectionRule(str, Enum):
    ENV_RETRY_OK = "env_retry_ok"
    MERGED_PERF = "merged_perf"
    INCONCLUSIVE_FLAKY = "inconclusive_flaky"


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class FlakinessSpec(SchemaRecord):
    RECORD_KIND = "flakiness"

    flaky: bool
    pattern: NonEmptyStr | None = None

    @model_validator(mode="after")
    def _validate_pattern(self) -> FlakinessSpec:
        if self.flaky != (self.pattern is not None):
            raise ValueError("flaky and pattern must be set together")
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class RunResultSet(SchemaRecord):
    RECORD_KIND = "run_result_set"

    schema_version: Literal["run_result_set_v1"] = "run_result_set_v1"
    run_id: RunId
    attempts: tuple[Annotated[str, StringConstraints(pattern=r"^obs:sha256:[0-9a-f]{64}$")], ...]
    canonical_result_ref: ObservationRef | None
    selection_rule: SelectionRule
    flakiness: FlakinessSpec

    @model_validator(mode="after")
    def _validate_selection(self) -> RunResultSet:
        if not self.attempts or len(self.attempts) != len(set(self.attempts)):
            raise ValueError("attempt refs must be non-empty and unique")
        if self.selection_rule is SelectionRule.INCONCLUSIVE_FLAKY:
            if self.canonical_result_ref is not None or not self.flakiness.flaky:
                raise ValueError("inconclusive flaky results require null canonical ref")
        elif self.canonical_result_ref is None:
            raise ValueError("non-flaky selection requires a canonical result")
        if self.selection_rule is SelectionRule.MERGED_PERF and (
            self.canonical_result_ref is None
            or not self.canonical_result_ref.startswith("aggregate:")
        ):
            raise ValueError("merged performance results require an aggregate ref")
        if self.selection_rule is SelectionRule.ENV_RETRY_OK and (
            self.canonical_result_ref is None
            or not self.canonical_result_ref.startswith("obs:")
        ):
            raise ValueError("environment retry selection requires an observation ref")
        return self
