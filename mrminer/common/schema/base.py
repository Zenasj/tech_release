"""Shared serialization and validation helpers for schema dataclasses."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, ClassVar, cast

from pydantic import ConfigDict, TypeAdapter
from typing_extensions import Self

SCHEMA_CONFIG = ConfigDict(
    extra="forbid",
    populate_by_name=True,
    validate_default=True,
)


@lru_cache(maxsize=None)
def _adapter(record_type: type[Any]) -> TypeAdapter[Any]:
    return TypeAdapter(record_type)


def dump_value(value: Any) -> Any:
    return _adapter(type(value)).dump_python(value, mode="json", by_alias=True)


class SchemaRecord:
    """Mixin for Pydantic dataclasses with stable record conversion."""

    RECORD_KIND: ClassVar[str]

    def to_record(self) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            _adapter(type(self)).dump_python(self, mode="json", by_alias=True),
        )

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Self:
        return cast(Self, _adapter(cls).validate_python(record))

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        return _adapter(cls).json_schema()
