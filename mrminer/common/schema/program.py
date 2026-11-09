"""Executable program identity and append-only provenance."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import StringConstraints, model_validator
from pydantic.dataclasses import dataclass

from mrminer.common.canonical import content_id
from mrminer.common.schema.base import SCHEMA_CONFIG, SchemaRecord, dump_value

NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]
ProgramId = Annotated[
    str,
    StringConstraints(pattern=r"^prog:sha256:[0-9a-f]{64}$"),
]
EntityId = Annotated[str, StringConstraints(pattern=r"^[a-z]+:[^\s/]+$")]


class ProgramKind(str, Enum):
    PYTEST_NODE = "pytest_node"
    PYTHON_CALLABLE = "python_callable"
    FX_GRAPH = "fx_graph"
    MODEL = "model"


class ProgramPhase(str, Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    TRAIN_STEP = "train_step"


class ProvenanceOrigin(str, Enum):
    PR_REGRESSION_TEST = "pr_regression_test"
    TEMPLATE_MIN_TRIGGER = "template_min_trigger"
    CONTAINER_POOL = "container_pool"
    TRANSFORM_OUTPUT = "transform_output"


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class ProgramSourceRef(SchemaRecord):
    RECORD_KIND = "program_source_ref"

    repo: NonEmptyStr
    commit: NonEmptyStr
    path: NonEmptyStr
    node_id: NonEmptyStr | None = None


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class InputInterface(SchemaRecord):
    RECORD_KIND = "input_interface"

    arity: int
    roles: tuple[NonEmptyStr, ...]

    @model_validator(mode="after")
    def _validate_arity(self) -> InputInterface:
        if self.arity < 0 or self.arity != len(self.roles):
            raise ValueError("input arity must equal the number of roles")
        if len(set(self.roles)) != len(self.roles):
            raise ValueError("input roles must be unique")
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class ProgramManifest(SchemaRecord):
    RECORD_KIND = "program_manifest"

    schema_version: Literal["program_manifest_v1"] = "program_manifest_v1"
    program_id: ProgramId
    program_kind: ProgramKind
    source_ref: ProgramSourceRef | None
    entrypoint: NonEmptyStr | None
    dependencies: dict[NonEmptyStr, NonEmptyStr]
    input_interface: InputInterface
    required_device_class: NonEmptyStr
    phase: ProgramPhase
    bound_entities: tuple[EntityId, ...]

    @classmethod
    def create(
        cls,
        *,
        program_kind: ProgramKind,
        source_ref: ProgramSourceRef | None,
        entrypoint: str | None,
        dependencies: dict[str, str],
        input_interface: InputInterface,
        required_device_class: str,
        phase: ProgramPhase,
        bound_entities: tuple[str, ...],
    ) -> ProgramManifest:
        payload: dict[str, Any] = {
            "program_kind": program_kind.value,
            "source_ref": dump_value(source_ref) if source_ref is not None else None,
            "entrypoint": entrypoint,
            "dependencies": dependencies,
            "input_interface": dump_value(input_interface),
            "required_device_class": required_device_class,
            "phase": phase.value,
            "bound_entities": list(bound_entities),
        }
        return cls(program_id=content_id("prog", payload), **payload)

    def to_hash_payload(self) -> dict[str, Any]:
        return {
            "program_kind": self.program_kind.value,
            "source_ref": (
                dump_value(self.source_ref) if self.source_ref is not None else None
            ),
            "entrypoint": self.entrypoint,
            "dependencies": dict(self.dependencies),
            "input_interface": dump_value(self.input_interface),
            "required_device_class": self.required_device_class,
            "phase": self.phase.value,
            "bound_entities": list(self.bound_entities),
        }

    @model_validator(mode="after")
    def _validate_manifest(self) -> ProgramManifest:
        if self.program_kind is ProgramKind.PYTEST_NODE:
            if self.source_ref is None or self.source_ref.node_id is None:
                raise ValueError("pytest_node programs require source_ref.node_id")
        elif self.entrypoint is None:
            raise ValueError("non-pytest programs require an entrypoint")
        if self.bound_entities != tuple(sorted(set(self.bound_entities))):
            raise ValueError("bound_entities must be sorted and unique")
        expected = content_id("prog", self.to_hash_payload())
        if self.program_id != expected:
            raise ValueError("program_id does not match semantic payload")
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class ProvenanceAnnotation(SchemaRecord):
    RECORD_KIND = "program_provenance"

    annotation_version: Literal["v1"] = "v1"
    program_id: ProgramId
    origin: ProvenanceOrigin
    origin_ref: NonEmptyStr
    recorded_at: NonEmptyStr
