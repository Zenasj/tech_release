"""Deterministic input manifests and the supported v1 pytree."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import Field, StringConstraints, model_validator
from pydantic.dataclasses import dataclass

from mrminer.common.canonical import canonical_bytes, content_id
from mrminer.common.schema.base import SCHEMA_CONFIG, SchemaRecord, dump_value

NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]
Sha256Ref = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
InputId = Annotated[
    str,
    StringConstraints(pattern=r"^inp:sha256:[0-9a-f]{64}$"),
]
JsonScalar = bool | int | float | str


class InputNodeKind(str, Enum):
    TENSOR = "tensor_spec"
    SCALAR = "scalar"
    NONE = "none"
    TUPLE = "tuple"
    LIST = "list"
    DICT = "dict"


class MaterializationMode(str, Enum):
    REGENERATE = "regenerate"
    SERIALIZED_FIXTURE = "serialized_fixture"


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class TensorSpec(SchemaRecord):
    RECORD_KIND = "tensor_spec"

    shape: tuple[int, ...]
    dtype: NonEmptyStr
    device_class: NonEmptyStr
    layout: NonEmptyStr
    requires_grad: bool = False

    @model_validator(mode="after")
    def _validate_shape(self) -> TensorSpec:
        if any(dim < 0 for dim in self.shape):
            raise ValueError("tensor dimensions cannot be negative")
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class InputTree(SchemaRecord):
    RECORD_KIND = "input_tree"

    kind: InputNodeKind
    tensor_spec: TensorSpec | None = None
    scalar: JsonScalar | None = None
    children: tuple[InputTree, ...] = ()
    entries: tuple[tuple[NonEmptyStr, InputTree], ...] = ()

    @model_validator(mode="after")
    def _validate_node(self) -> InputTree:
        populated = {
            "tensor": self.tensor_spec is not None,
            "scalar": self.scalar is not None,
            "children": bool(self.children),
            "entries": bool(self.entries),
        }
        if self.kind is InputNodeKind.TENSOR:
            if not populated["tensor"] or any(
                populated[name] for name in ("scalar", "children", "entries")
            ):
                raise ValueError("tensor nodes require only tensor_spec")
        elif self.kind is InputNodeKind.SCALAR:
            if not populated["scalar"] or any(
                populated[name] for name in ("tensor", "children", "entries")
            ):
                raise ValueError("scalar nodes require only a non-None scalar")
        elif self.kind is InputNodeKind.NONE:
            if any(populated.values()):
                raise ValueError("none nodes cannot contain values")
        elif self.kind in {InputNodeKind.TUPLE, InputNodeKind.LIST}:
            if populated["tensor"] or populated["scalar"] or populated["entries"]:
                raise ValueError("sequence nodes may only contain children")
        else:
            if populated["tensor"] or populated["scalar"] or populated["children"]:
                raise ValueError("dict nodes may only contain entries")
            keys = tuple(key for key, _ in self.entries)
            if keys != tuple(sorted(set(keys))):
                raise ValueError("dict input keys must be sorted and unique")
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class GeneratorSpec(SchemaRecord):
    RECORD_KIND = "input_generator"

    generator_id: NonEmptyStr
    algo_version: NonEmptyStr
    params: dict[str, Any]

    @model_validator(mode="after")
    def _validate_params(self) -> GeneratorSpec:
        canonical_bytes(self.params)
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class FixtureSpec(SchemaRecord):
    RECORD_KIND = "input_fixture"

    fixture_ref: NonEmptyStr
    format: NonEmptyStr
    serializer_version: NonEmptyStr
    fixture_hash: Sha256Ref


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class MaterializationSpec(SchemaRecord):
    RECORD_KIND = "input_materialization"

    mode: MaterializationMode
    fixture: FixtureSpec | None = None

    @model_validator(mode="after")
    def _validate_fixture(self) -> MaterializationSpec:
        if self.mode is MaterializationMode.SERIALIZED_FIXTURE and self.fixture is None:
            raise ValueError("serialized_fixture mode requires fixture metadata")
        if self.mode is MaterializationMode.REGENERATE and self.fixture is not None:
            raise ValueError("regenerate mode cannot define a fixture")
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class InputManifest(SchemaRecord):
    RECORD_KIND = "input_manifest"

    schema_version: Literal["input_manifest_v1"] = "input_manifest_v1"
    input_id: InputId
    input_tree: InputTree
    generator: GeneratorSpec
    seed: int
    materialization: MaterializationSpec

    @classmethod
    def create(
        cls,
        *,
        input_tree: InputTree,
        generator: GeneratorSpec,
        seed: int,
        materialization: MaterializationSpec,
    ) -> InputManifest:
        payload = {
            "input_tree": dump_value(input_tree),
            "generator": dump_value(generator),
            "seed": seed,
            "materialization": dump_value(materialization),
        }
        return cls(input_id=content_id("inp", payload), **payload)

    def to_hash_payload(self) -> dict[str, Any]:
        return {
            "input_tree": dump_value(self.input_tree),
            "generator": dump_value(self.generator),
            "seed": self.seed,
            "materialization": dump_value(self.materialization),
        }

    @model_validator(mode="after")
    def _validate_content_id(self) -> InputManifest:
        expected = content_id("inp", self.to_hash_payload())
        if self.input_id != expected:
            raise ValueError("input_id does not match semantic payload")
        return self
