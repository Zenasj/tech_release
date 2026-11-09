"""Deterministic JSON and content-address helpers."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from collections.abc import Mapping
from enum import Enum
from typing import Any, Protocol, runtime_checkable
from urllib.parse import quote, unquote

_ID_PREFIX_RE = re.compile(r"[a-z][a-z0-9_]*\Z")


class CanonicalizationError(ValueError):
    """Raised when a value cannot be represented as stable JSON."""


@runtime_checkable
class HashProjectable(Protocol):
    def to_hash_payload(self) -> Mapping[str, Any]:
        """Return the exact semantic payload used for content addressing."""


def _to_json_value(value: Any, active: set[int]) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalizationError("non-finite floats are not valid canonical JSON")
        return value
    if isinstance(value, Enum):
        return _to_json_value(value.value, active)

    track_identity = dataclasses.is_dataclass(value) or isinstance(
        value, (Mapping, list, tuple)
    )
    identity = id(value)
    if track_identity:
        if identity in active:
            raise CanonicalizationError("cyclic values are not valid canonical JSON")
        active.add(identity)

    try:
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {
                field.name: _to_json_value(getattr(value, field.name), active)
                for field in dataclasses.fields(value)
            }
        if isinstance(value, Mapping):
            if not all(isinstance(key, str) for key in value):
                raise CanonicalizationError("canonical JSON mapping keys must be strings")
            return {key: _to_json_value(item, active) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_to_json_value(item, active) for item in value]
        if isinstance(value, (set, frozenset)):
            raise CanonicalizationError(
                "sets are unordered; convert them to a deterministically sorted list"
            )
    finally:
        if track_identity:
            active.remove(identity)

    raise CanonicalizationError(
        f"unsupported canonical JSON value: {type(value).__name__}"
    )


def canonical_json(value: Any) -> str:
    """Serialize a supported value to deterministic, whitespace-free JSON."""

    normalized = _to_json_value(value, set())
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def sha256_digest(value: Any) -> str:
    return f"sha256:{hashlib.sha256(canonical_bytes(value)).hexdigest()}"


def content_id(prefix: str, payload: Any) -> str:
    if not _ID_PREFIX_RE.fullmatch(prefix):
        raise ValueError(
            "content ID prefix must match [a-z][a-z0-9_]*"
        )
    return f"{prefix}:{sha256_digest(payload)}"


def projectable_id(prefix: str, record: HashProjectable) -> str:
    return content_id(prefix, record.to_hash_payload())


def fs_safe(value: str) -> str:
    """Encode an opaque ID as one reversible filesystem path component."""

    if not value:
        raise ValueError("filesystem-safe IDs cannot be empty")
    encoded = quote(value, safe="-._~", encoding="utf-8", errors="strict")
    if encoded in {".", ".."}:
        raise ValueError("filesystem-safe IDs cannot be dot path components")
    return encoded


def fs_restore(value: str) -> str:
    if not value or "/" in value or "\\" in value or value in {".", ".."}:
        raise ValueError("value is not one filesystem-safe path component")
    return unquote(value, encoding="utf-8", errors="strict")
