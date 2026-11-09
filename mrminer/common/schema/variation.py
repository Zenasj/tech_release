"""Parameterized variation templates and concrete variation instances."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import Field, StringConstraints, model_validator
from pydantic.dataclasses import dataclass

from mrminer.common.canonical import canonical_bytes, canonical_json, content_id
from mrminer.common.schema.base import SCHEMA_CONFIG, SchemaRecord, dump_value

NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]
ProgramId = Annotated[
    str, StringConstraints(pattern=r"^prog:sha256:[0-9a-f]{64}$")
]
InputId = Annotated[str, StringConstraints(pattern=r"^inp:sha256:[0-9a-f]{64}$")]
ArtifactId = Annotated[str, StringConstraints(pattern=r"^art:[^\s/]+$")]
VariationTemplateId = Annotated[
    str, StringConstraints(pattern=r"^vart:sha256:[0-9a-f]{64}$")
]
VariationId = Annotated[
    str, StringConstraints(pattern=r"^var:sha256:[0-9a-f]{64}$")
]
EntityId = Annotated[str, StringConstraints(pattern=r"^[a-z]+:[^\s/]+$")]
JsonScalar = bool | int | float | str


class ExecutionMode(str, Enum):
    EAGER = "eager"
    COMPILED = "compiled"


class GuardCheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class ProgramTransformOperation(SchemaRecord):
    RECORD_KIND = "program_transform_operation"

    kind: Literal["program_transform"] = "program_transform"
    transform_id: NonEmptyStr
    params: dict[str, Any]

    @model_validator(mode="after")
    def _validate_params(self) -> ProgramTransformOperation:
        canonical_bytes(self.params)
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class InputTransformOperation(SchemaRecord):
    RECORD_KIND = "input_transform_operation"

    kind: Literal["input_transform"] = "input_transform"
    transform_id: NonEmptyStr
    params: dict[str, Any]

    @model_validator(mode="after")
    def _validate_params(self) -> InputTransformOperation:
        canonical_bytes(self.params)
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class ConfigOverrideOperation(SchemaRecord):
    RECORD_KIND = "config_override_operation"

    kind: Literal["config_override"] = "config_override"
    path: NonEmptyStr
    from_value: Any = Field(alias="from")
    to_value: Any = Field(alias="to")

    @model_validator(mode="after")
    def _validate_values(self) -> ConfigOverrideOperation:
        canonical_bytes({"from": self.from_value, "to": self.to_value})
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class ArtifactSelectOperation(SchemaRecord):
    RECORD_KIND = "artifact_select_operation"

    kind: Literal["artifact_select"] = "artifact_select"
    from_artifact: ArtifactId = Field(alias="from")
    to_artifact: ArtifactId = Field(alias="to")

    @model_validator(mode="after")
    def _validate_artifacts(self) -> ArtifactSelectOperation:
        if self.from_artifact == self.to_artifact:
            raise ValueError("artifact selection requires two distinct artifacts")
        return self


VariationOperation = (
    ProgramTransformOperation
    | InputTransformOperation
    | ConfigOverrideOperation
    | ArtifactSelectOperation
)


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class ParameterDefinition(SchemaRecord):
    RECORD_KIND = "variation_parameter"

    enum: tuple[JsonScalar, ...]

    @model_validator(mode="after")
    def _validate_enum(self) -> ParameterDefinition:
        encoded = tuple(canonical_json(value) for value in self.enum)
        if not encoded or encoded != tuple(sorted(set(encoded))):
            raise ValueError("parameter enum must be non-empty, canonical-sorted, and unique")
        return self


def _collect_placeholders(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value} if value.startswith("$") else set()
    if isinstance(value, dict):
        placeholders: set[str] = set()
        for key, item in value.items():
            placeholders.update(_collect_placeholders(key))
            placeholders.update(_collect_placeholders(item))
        return placeholders
    if isinstance(value, (list, tuple)):
        placeholders = set()
        for item in value:
            placeholders.update(_collect_placeholders(item))
        return placeholders
    return set()


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class VariationTemplateSpec(SchemaRecord):
    RECORD_KIND = "variation_template"

    schema_version: Literal["variation_template_v1"] = "variation_template_v1"
    variation_template_id: VariationTemplateId
    operations: tuple[VariationOperation, ...]
    param_schema: dict[NonEmptyStr, ParameterDefinition]
    roles_produced: tuple[NonEmptyStr, ...]

    @classmethod
    def create(
        cls,
        *,
        operations: tuple[VariationOperation, ...],
        param_schema: dict[str, ParameterDefinition],
        roles_produced: tuple[str, ...],
    ) -> VariationTemplateSpec:
        payload = {
            "operations": [dump_value(operation) for operation in operations],
            "param_schema": {
                name: dump_value(definition)
                for name, definition in param_schema.items()
            },
            "roles_produced": list(roles_produced),
        }
        return cls(
            variation_template_id=content_id("vart", payload),
            operations=operations,
            param_schema=param_schema,
            roles_produced=roles_produced,
        )

    def to_hash_payload(self) -> dict[str, Any]:
        return {
            "operations": [dump_value(operation) for operation in self.operations],
            "param_schema": {
                name: dump_value(definition)
                for name, definition in self.param_schema.items()
            },
            "roles_produced": list(self.roles_produced),
        }

    @model_validator(mode="after")
    def _validate_template(self) -> VariationTemplateSpec:
        if not self.operations:
            raise ValueError("variation templates require at least one operation")
        if self.roles_produced != tuple(sorted(set(self.roles_produced))):
            raise ValueError("roles_produced must be sorted and unique")
        if any(not name.startswith("$") for name in self.param_schema):
            raise ValueError("variation parameter names must start with $")
        placeholders = _collect_placeholders(
            [dump_value(operation) for operation in self.operations]
        )
        if placeholders != set(self.param_schema):
            raise ValueError("operation placeholders must exactly match param_schema")
        expected = content_id("vart", self.to_hash_payload())
        if self.variation_template_id != expected:
            raise ValueError("variation_template_id does not match semantic payload")
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class ScenarioSpec(SchemaRecord):
    RECORD_KIND = "scenario"

    program_id: ProgramId
    input_id: InputId
    artifact_id: ArtifactId
    compile_config: dict[str, Any]
    runtime_config: dict[str, Any]
    execution_mode: ExecutionMode

    @model_validator(mode="after")
    def _validate_configs(self) -> ScenarioSpec:
        canonical_bytes(
            {
                "compile_config": self.compile_config,
                "runtime_config": self.runtime_config,
            }
        )
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class GuardCheck(SchemaRecord):
    RECORD_KIND = "variation_guard_check"

    status: GuardCheckStatus
    reason: NonEmptyStr | None = None

    @model_validator(mode="after")
    def _validate_reason(self) -> GuardCheck:
        if self.status is GuardCheckStatus.FAIL and self.reason is None:
            raise ValueError("failed guard checks require a reason")
        if self.status is GuardCheckStatus.PASS and self.reason is not None:
            raise ValueError("passed guard checks cannot define a reason")
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class ExpectedTrigger(SchemaRecord):
    RECORD_KIND = "expected_trigger"

    entity: EntityId
    probe: NonEmptyStr


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class VariationFailure(SchemaRecord):
    RECORD_KIND = "variation_failure"

    code: NonEmptyStr
    detail: NonEmptyStr


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class VariationSpec(SchemaRecord):
    RECORD_KIND = "variation"

    schema_version: Literal["variation_v1"] = "variation_v1"
    variation_id: VariationId
    variation_template_id: VariationTemplateId
    parameter_assignment: dict[NonEmptyStr, JsonScalar]
    base_scenario: ScenarioSpec
    operations: tuple[VariationOperation, ...]
    result_roles: dict[NonEmptyStr, ScenarioSpec]
    input_correspondence: dict[str, Any]
    boundary_map: dict[NonEmptyStr, NonEmptyStr] | None
    guard_check: GuardCheck
    expected_trigger: ExpectedTrigger | None
    static_trigger_possible: bool
    failure: VariationFailure | None

    @classmethod
    def create(cls, **fields: Any) -> VariationSpec:
        payload = cls._payload(**fields)
        return cls(variation_id=content_id("var", payload), **fields)

    @staticmethod
    def _payload(**fields: Any) -> dict[str, Any]:
        return {
            "variation_template_id": fields["variation_template_id"],
            "parameter_assignment": fields["parameter_assignment"],
            "base_scenario": dump_value(fields["base_scenario"]),
            "operations": [
                dump_value(operation) for operation in fields["operations"]
            ],
            "result_roles": {
                role: dump_value(scenario)
                for role, scenario in fields["result_roles"].items()
            },
            "input_correspondence": fields["input_correspondence"],
            "boundary_map": fields["boundary_map"],
            "guard_check": dump_value(fields["guard_check"]),
            "expected_trigger": (
                dump_value(fields["expected_trigger"])
                if fields["expected_trigger"] is not None
                else None
            ),
            "static_trigger_possible": fields["static_trigger_possible"],
            "failure": (
                dump_value(fields["failure"])
                if fields["failure"] is not None
                else None
            ),
        }

    def to_hash_payload(self) -> dict[str, Any]:
        return self._payload(
            variation_template_id=self.variation_template_id,
            parameter_assignment=dict(self.parameter_assignment),
            base_scenario=self.base_scenario,
            operations=self.operations,
            result_roles=dict(self.result_roles),
            input_correspondence=dict(self.input_correspondence),
            boundary_map=(
                dict(self.boundary_map) if self.boundary_map is not None else None
            ),
            guard_check=self.guard_check,
            expected_trigger=self.expected_trigger,
            static_trigger_possible=self.static_trigger_possible,
            failure=self.failure,
        )

    @model_validator(mode="after")
    def _validate_variation(self) -> VariationSpec:
        if any(not name.startswith("$") for name in self.parameter_assignment):
            raise ValueError("parameter assignment names must start with $")
        concrete_payload = {
            "operations": [dump_value(operation) for operation in self.operations],
            "result_roles": {
                role: dump_value(scenario)
                for role, scenario in self.result_roles.items()
            },
        }
        if _collect_placeholders(concrete_payload):
            raise ValueError("concrete variations cannot contain unresolved placeholders")
        if not self.result_roles:
            raise ValueError("concrete variations require result roles")
        canonical_bytes(self.to_hash_payload())
        expected = content_id("var", self.to_hash_payload())
        if self.variation_id != expected:
            raise ValueError("variation_id does not match semantic payload")
        return self
