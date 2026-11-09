"""Evidence records shared by claims, graph edges, and lifecycle events."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import StringConstraints
from pydantic.dataclasses import dataclass

from mrminer.common.schema.base import SCHEMA_CONFIG, SchemaRecord

Sha256Ref = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]


class EvidenceStrength(str, Enum):
    EXPLICIT_CLAIM = "explicit_claim"
    TEST_ASSERTION = "test_assertion"
    INFERRED = "inferred"


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class EvidenceSpec(SchemaRecord):
    RECORD_KIND = "evidence"

    evidence_version: Literal["v1"] = "v1"
    artifact_ref: NonEmptyStr
    locator: NonEmptyStr
    quote: NonEmptyStr
    quote_hash: Sha256Ref
    strength: EvidenceStrength
    supports_field: NonEmptyStr
    extractor: NonEmptyStr
