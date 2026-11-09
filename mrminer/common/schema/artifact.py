"""Compiler version artifacts and PR landing facts."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import StringConstraints, model_validator
from pydantic.dataclasses import dataclass

from mrminer.common.canonical import content_id
from mrminer.common.schema.base import SCHEMA_CONFIG, SchemaRecord
from mrminer.common.schema.evidence import EvidenceSpec

NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]
ArtifactId = Annotated[str, StringConstraints(pattern=r"^art:[^\s/]+$")]
CatalogEntryId = Annotated[str, StringConstraints(pattern=r"^vce:sha256:[0-9a-f]{64}$")]
CommitSha = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{7,64}$")]
Sha256Ref = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]


class BuildKind(str, Enum):
    NIGHTLY = "nightly"
    RELEASE = "release"
    SOURCE = "source"


class LandingKind(str, Enum):
    MERGE = "merge"
    RELAND = "reland"
    REVERT = "revert_landing"


class LandingState(str, Enum):
    ABSENT = "absent"
    PARTIAL = "partial"
    COMPLETE = "complete"


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class CommitInterval(SchemaRecord):
    RECORD_KIND = "commit_interval"

    branch: NonEmptyStr
    from_exclusive: CommitSha
    to_inclusive: CommitSha

    @model_validator(mode="after")
    def _validate_bounds(self) -> CommitInterval:
        if self.from_exclusive == self.to_inclusive:
            raise ValueError("commit interval bounds must differ")
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class EnvironmentConstraints(SchemaRecord):
    RECORD_KIND = "environment_constraints"

    cuda: NonEmptyStr | None = None
    python: NonEmptyStr


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class VersionArtifact(SchemaRecord):
    RECORD_KIND = "version_artifact"

    schema_version: Literal["version_artifact_v1"] = "version_artifact_v1"
    artifact_id: ArtifactId
    compiler: NonEmptyStr
    build_kind: BuildKind
    source_commit: CommitSha
    base_branch: NonEmptyStr
    package_version: NonEmptyStr
    package_hash: Sha256Ref
    built_at: NonEmptyStr
    commit_interval: CommitInterval
    environment_constraints: EnvironmentConstraints


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class VersionCatalogEntry(SchemaRecord):
    """A lightweight, offline attribution boundary on a base branch.

    Unlike :class:`VersionArtifact` this record carries no ``package_hash`` and
    is never executed: it only partitions branch history into ordered intervals
    so that PR landings can be attributed to a version window. Executable
    artifacts are provisioned on demand (see ``artifact_catalog``) and linked
    back through ``source_commit``. This separation keeps the frozen
    VersionArtifact schema untouched (see journal Step M6.2).
    """

    RECORD_KIND = "version_catalog_entry"

    schema_version: Literal["version_catalog_entry_v1"] = "version_catalog_entry_v1"
    entry_id: CatalogEntryId
    compiler: NonEmptyStr
    version_label: NonEmptyStr
    boundary_kind: Literal["release_branch_point", "window_bound"]
    boundary_ref: NonEmptyStr
    source_commit: CommitSha
    base_branch: NonEmptyStr
    commit_interval: CommitInterval

    @classmethod
    def create(cls, **fields: Any) -> VersionCatalogEntry:
        payload = cls._payload(**fields)
        return cls(entry_id=content_id("vce", payload), **fields)

    @staticmethod
    def _payload(**fields: Any) -> dict[str, Any]:
        interval: CommitInterval = fields["commit_interval"]
        return {
            "compiler": fields["compiler"],
            "version_label": fields["version_label"],
            "boundary_kind": fields["boundary_kind"],
            "boundary_ref": fields["boundary_ref"],
            "source_commit": fields["source_commit"],
            "base_branch": fields["base_branch"],
            "commit_interval": {
                "branch": interval.branch,
                "from_exclusive": interval.from_exclusive,
                "to_inclusive": interval.to_inclusive,
            },
        }

    def to_hash_payload(self) -> dict[str, Any]:
        return self._payload(
            compiler=self.compiler,
            version_label=self.version_label,
            boundary_kind=self.boundary_kind,
            boundary_ref=self.boundary_ref,
            source_commit=self.source_commit,
            base_branch=self.base_branch,
            commit_interval=self.commit_interval,
        )


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class ChangeLanding(SchemaRecord):
    RECORD_KIND = "change_landing"

    landing_version: Literal["v1"] = "v1"
    pr: int | None
    branch: NonEmptyStr
    landed_commits: tuple[CommitSha, ...]
    landed_at: NonEmptyStr
    kind: LandingKind
    reverts_pr: int | None = None
    evidence: tuple[EvidenceSpec, ...]
    confidence: float

    @model_validator(mode="after")
    def _validate_landing(self) -> ChangeLanding:
        if self.pr is not None and self.pr <= 0:
            raise ValueError("pr must be positive when present")
        if not self.landed_commits:
            raise ValueError("a landing requires at least one commit")
        if self.landed_commits != tuple(dict.fromkeys(self.landed_commits)):
            raise ValueError("landed commits must be unique and ordered")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between zero and one")
        if self.kind is LandingKind.REVERT:
            if self.reverts_pr is None or self.reverts_pr <= 0:
                raise ValueError("revert landings require a positive reverts_pr")
        else:
            if self.pr is None:
                raise ValueError("merge/reland landings require a source PR")
            if self.reverts_pr is not None:
                raise ValueError("only revert landings may define reverts_pr")
        return self
