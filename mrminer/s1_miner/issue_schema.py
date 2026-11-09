"""Repository-scoped Issue corpus and fetch queue records."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator
from pydantic.dataclasses import dataclass

from mrminer.common.canonical import canonical_bytes, sha256_digest
from mrminer.common.schema.base import SCHEMA_CONFIG, SchemaRecord
from mrminer.s1_miner.pr_schema import (
    IssueFetchFailure,
    NonEmptyStr,
    RepositoryName,
    Sha256Ref,
    TextSource,
)

OpaqueCursor = Annotated[str, StringConstraints(min_length=1)]


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class IssueCommentCollection(SchemaRecord):
    RECORD_KIND = "issue_comment_collection"

    items: tuple[TextSource, ...]
    total_count: int = Field(ge=0)
    retained_count: int = Field(ge=0)
    collection_truncated: bool
    next_cursor: OpaqueCursor | None

    @model_validator(mode="after")
    def _validate_collection(self) -> IssueCommentCollection:
        if self.retained_count != len(self.items):
            raise ValueError("retained issue comment count must match items")
        if self.retained_count > self.total_count:
            raise ValueError("retained issue comments exceed total count")
        if self.collection_truncated != (self.retained_count < self.total_count):
            raise ValueError("issue comment truncation disagrees with counts")
        if self.collection_truncated != (self.next_cursor is not None):
            raise ValueError("truncated issue comments require a next cursor")
        source_ids = tuple(item.source_id for item in self.items)
        if source_ids != tuple(sorted(set(source_ids))):
            raise ValueError("issue comments must have sorted unique source IDs")
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class IssueClosingRef(SchemaRecord):
    RECORD_KIND = "issue_closing_ref"

    pr_number: int = Field(gt=0)
    source: Literal["graphql_closingIssuesReferences"]


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class IssueCollectionMetadata(SchemaRecord):
    RECORD_KIND = "issue_collection"

    config_sha: Sha256Ref
    collected_at: NonEmptyStr
    max_comments: int = Field(ge=0)
    max_comment_chars: int = Field(gt=0)


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class IssueCorpusV1(SchemaRecord):
    RECORD_KIND = "issue_corpus"

    schema_version: Literal["issue_v1"] = "issue_v1"
    repository: RepositoryName
    issue_number: int = Field(gt=0)
    html_url: NonEmptyStr
    state: Literal["open", "closed"]
    title: str
    body: TextSource
    labels: tuple[str, ...]
    author: str | None
    created_at: NonEmptyStr
    updated_at: NonEmptyStr
    closed_at: str | None
    comments: IssueCommentCollection
    closing_refs: tuple[IssueClosingRef, ...]
    collection: IssueCollectionMetadata

    @model_validator(mode="after")
    def _validate_issue(self) -> IssueCorpusV1:
        if self.body.source_id != "issue_body":
            raise ValueError("issue body source_id must be issue_body")
        if self.body.truncation.value != "complete":
            raise ValueError("issue body must preserve the complete API text")
        if self.labels != tuple(sorted(set(self.labels))):
            raise ValueError("issue labels must be sorted and unique")
        refs = tuple((ref.pr_number, ref.source) for ref in self.closing_refs)
        if refs != tuple(sorted(set(refs))):
            raise ValueError("issue closing refs must be sorted and unique")
        return self

    @property
    def identity(self) -> tuple[str, int]:
        return self.repository, self.issue_number


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class IssueFetchEvent(SchemaRecord):
    RECORD_KIND = "issue_fetch_event"

    schema_version: Literal["issue_fetch_event_v1"] = "issue_fetch_event_v1"
    event_id: Sha256Ref
    repository: RepositoryName
    pr_number: int = Field(gt=0)
    issue_number: int = Field(gt=0)
    outcome: Literal["failed", "resolved"]
    failure: IssueFetchFailure | None
    occurred_at: NonEmptyStr

    @classmethod
    def create(cls, **fields) -> IssueFetchEvent:
        payload = {
            key: (
                value.to_record()
                if isinstance(value, SchemaRecord)
                else value
            )
            for key, value in fields.items()
        }
        return cls(event_id=sha256_digest(payload), **fields)

    @model_validator(mode="after")
    def _validate_event(self) -> IssueFetchEvent:
        if (self.outcome == "failed") != (self.failure is not None):
            raise ValueError("failed event requires failure; resolved event forbids it")
        payload = {
            "repository": self.repository,
            "pr_number": self.pr_number,
            "issue_number": self.issue_number,
            "outcome": self.outcome,
            "failure": self.failure.to_record() if self.failure else None,
            "occurred_at": self.occurred_at,
        }
        if self.event_id != sha256_digest(payload):
            raise ValueError("issue fetch event_id does not match payload")
        canonical_bytes(payload)
        return self
