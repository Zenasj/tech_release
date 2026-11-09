"""Versioned S1 corpus records and v1 compatibility conversion."""

from __future__ import annotations

from enum import Enum
import re
from typing import Annotated, Any, Literal

from pydantic import Field, StringConstraints, model_validator
from pydantic.dataclasses import dataclass

from mrminer.common.canonical import canonical_bytes
from mrminer.common.schema.base import SCHEMA_CONFIG, SchemaRecord

NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]
Sha256Ref = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
RepositoryName = Annotated[
    str,
    StringConstraints(pattern=r"^[^/\s]+/[^/\s]+$"),
]


class TruncationStatus(str, Enum):
    COMPLETE = "complete"
    TRUNCATED = "truncated"
    UNKNOWN = "unknown"


class RevertEvidenceKind(str, Enum):
    TITLE_PATTERN = "title_pattern"
    BOT_COMMENT = "bot_comment"
    MANUAL = "manual"


class IssueLinkKind(str, Enum):
    FIXES_KEYWORD = "fixes_keyword"
    CLOSING_REFERENCE = "closing_reference"
    COMMENT_MENTION = "comment_mention"
    LEGACY_IMPORT = "legacy_import"


class IssueFetchStatus(str, Enum):
    STORED = "stored"
    NOT_FOUND = "not_found"
    FORBIDDEN = "forbidden"
    TRANSIENT_ERROR = "transient_error"
    LEGACY_INLINE = "legacy_inline"


class IssueFetchFailureKind(str, Enum):
    NOT_FOUND = "not_found"
    FORBIDDEN = "forbidden"
    RATE_LIMITED = "rate_limited"
    NETWORK = "network"
    MALFORMED = "malformed"


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class TextSource(SchemaRecord):
    RECORD_KIND = "pr_text_source"

    source_id: NonEmptyStr
    text: str
    char_start: Literal[0] = 0
    char_end: int
    truncation: TruncationStatus
    author: str | None = None
    created_at: str | None = None
    html_url: str | None = None

    @model_validator(mode="after")
    def _validate_offsets(self) -> TextSource:
        if self.char_end != len(self.text):
            raise ValueError("text char_end must equal the retained text length")
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class TouchedFilePatch(SchemaRecord):
    RECORD_KIND = "touched_file_patch"

    path: NonEmptyStr
    patch: str | None
    truncated: bool

    @model_validator(mode="after")
    def _validate_patch(self) -> TouchedFilePatch:
        if self.patch is None and not self.truncated:
            raise ValueError("missing patches must be marked truncated")
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class IssueLinkEvidence(SchemaRecord):
    RECORD_KIND = "issue_link_evidence"

    kind: IssueLinkKind
    source_id: NonEmptyStr
    locator: NonEmptyStr


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class IssueFetchFailure(SchemaRecord):
    RECORD_KIND = "issue_fetch_failure"

    kind: IssueFetchFailureKind
    attempts: int = Field(ge=1)
    last_attempted_at: NonEmptyStr


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class LinkedIssueRef(SchemaRecord):
    RECORD_KIND = "linked_issue_ref"

    repository: RepositoryName
    number: int = Field(gt=0)
    issue_ref: str | None
    fetch_status: IssueFetchStatus
    link_evidence: tuple[IssueLinkEvidence, ...]
    fetch_failure: IssueFetchFailure | None
    legacy_payload: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _validate_reference(self) -> LinkedIssueRef:
        evidence_keys = tuple(
            (item.kind.value, item.source_id, item.locator)
            for item in self.link_evidence
        )
        if not evidence_keys or evidence_keys != tuple(sorted(set(evidence_keys))):
            raise ValueError(
                "linked issue evidence must be non-empty, sorted, and unique"
            )
        if self.fetch_status is IssueFetchStatus.STORED:
            expected = f"issues/{self.number}.json"
            if self.issue_ref != expected or self.fetch_failure is not None:
                raise ValueError(
                    "stored issue references require the canonical path and no failure"
                )
            if self.legacy_payload is not None:
                raise ValueError("stored issue references cannot define legacy payload")
        elif self.fetch_status is IssueFetchStatus.LEGACY_INLINE:
            if self.issue_ref is not None or self.fetch_failure is not None:
                raise ValueError("legacy inline issue cannot define path or failure")
            if self.legacy_payload is None:
                raise ValueError("legacy inline issue requires its original payload")
        else:
            if self.issue_ref is not None or self.fetch_failure is None:
                raise ValueError(
                    "failed issue references require failure and no issue path"
                )
            if self.legacy_payload is not None:
                raise ValueError("failed issue references cannot define legacy payload")
            expected_failure = {
                IssueFetchStatus.NOT_FOUND: IssueFetchFailureKind.NOT_FOUND,
                IssueFetchStatus.FORBIDDEN: IssueFetchFailureKind.FORBIDDEN,
            }.get(self.fetch_status)
            if (
                expected_failure is not None
                and self.fetch_failure.kind is not expected_failure
            ):
                raise ValueError("issue fetch status and failure kind disagree")
        canonical_bytes(self.legacy_payload)
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class DefSpan(SchemaRecord):
    RECORD_KIND = "test_def_span"

    start_line: int | None
    end_line: int | None
    complete: bool

    @model_validator(mode="after")
    def _validate_span(self) -> DefSpan:
        if (self.start_line is None) != (self.end_line is None):
            raise ValueError("test span bounds must be both present or both absent")
        if (
            self.start_line is not None
            and self.end_line is not None
            and (self.start_line <= 0 or self.end_line < self.start_line)
        ):
            raise ValueError("test span bounds are invalid")
        if self.complete and self.start_line is None:
            raise ValueError("complete test spans require line bounds")
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class AddedTest(SchemaRecord):
    RECORD_KIND = "added_test"

    path: NonEmptyStr
    test_node_id: NonEmptyStr
    test_name: NonEmptyStr
    def_span: DefSpan
    added_code: str


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class CommitRefs(SchemaRecord):
    RECORD_KIND = "pr_commit_refs"

    merge_commit_sha: str | None
    head_sha: str | None
    base_sha: str | None
    base_ref: str | None


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class CollectionMetadata(SchemaRecord):
    RECORD_KIND = "pr_collection"

    schema_version: Literal["collection_v1"] = "collection_v1"
    config_sha: Sha256Ref
    collected_at: NonEmptyStr
    files_expected_count: int | None = Field(default=None, ge=0)
    files_expected_count_source: Literal[
        "github_detail",
        "git_base_head",
    ] = "github_detail"
    files_github_reported_count: int | None = Field(default=None, ge=0)
    files_retained_count: int | None = Field(default=None, ge=0)
    files_collection_truncated: bool | None = None
    files_fetch_complete: bool | None = None
    conversation_comments_total_count: int | None = Field(default=None, ge=0)
    conversation_comments_retained_count: int | None = Field(default=None, ge=0)
    conversation_comments_collection_truncated: bool | None = None
    conversation_comments_fetch_complete: bool | None = None
    review_comments_total_count: int | None = Field(default=None, ge=0)
    review_comments_retained_count: int | None = Field(default=None, ge=0)
    review_comments_collection_truncated: bool | None = None
    review_comments_fetch_complete: bool | None = None
    closing_refs_collection_complete: bool | None = None

    @model_validator(mode="after")
    def _validate_counts(self) -> CollectionMetadata:
        if (
            self.files_expected_count_source == "git_base_head"
            and self.files_github_reported_count is None
        ):
            raise ValueError(
                "git_base_head file count requires GitHub's reported count"
            )
        if (
            self.files_expected_count_source == "github_detail"
            and self.files_github_reported_count is not None
            and self.files_expected_count is not None
            and self.files_github_reported_count
            != self.files_expected_count
        ):
            raise ValueError(
                "GitHub file counts disagree without a git_base_head fallback"
            )
        for expected, retained, truncated, fetch_complete, label in (
            (
                self.files_expected_count,
                self.files_retained_count,
                self.files_collection_truncated,
                self.files_fetch_complete,
                "files",
            ),
            (
                self.conversation_comments_total_count,
                self.conversation_comments_retained_count,
                self.conversation_comments_collection_truncated,
                self.conversation_comments_fetch_complete,
                "conversation comments",
            ),
            (
                self.review_comments_total_count,
                self.review_comments_retained_count,
                self.review_comments_collection_truncated,
                self.review_comments_fetch_complete,
                "review comments",
            ),
        ):
            present = (
                expected is not None,
                retained is not None,
                truncated is not None,
                fetch_complete is not None,
            )
            if any(present) and not all(present):
                raise ValueError(f"{label} collection audit must be all present")
            if (
                expected is not None
                and retained is not None
                and retained > expected
            ):
                raise ValueError(f"{label} retained count exceeds expected count")
            if (
                expected is not None
                and retained is not None
                and fetch_complete is not None
                and truncated != (
                    retained < expected or not fetch_complete
                )
            ):
                raise ValueError(f"{label} truncation disagrees with counts")
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class RevertEvidence(SchemaRecord):
    RECORD_KIND = "revert_evidence"

    kind: RevertEvidenceKind
    target_pr: int
    source_id: NonEmptyStr
    quote: NonEmptyStr
    # Extra PR numbers reverted alongside target_pr (multi-PR reverts). Empty for
    # the common single-target case; additive so existing single-target records
    # still load and their derived revert edge is unchanged.
    additional_target_prs: tuple[int, ...] = ()

    @model_validator(mode="after")
    def _validate_target(self) -> RevertEvidence:
        if self.target_pr <= 0:
            raise ValueError("revert target PR must be positive")
        if any(number <= 0 for number in self.additional_target_prs):
            raise ValueError("revert target PR must be positive")
        if self.target_pr in self.additional_target_prs:
            raise ValueError("additional_target_prs must not repeat target_pr")
        if len(set(self.additional_target_prs)) != len(
            self.additional_target_prs
        ):
            raise ValueError("additional_target_prs must be unique")
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class PrCorpusV2(SchemaRecord):
    RECORD_KIND = "pr_corpus"

    schema_version: Literal["pr_corpus_v2"] = "pr_corpus_v2"
    source_schema: Literal[
        "pr_corpus_v1",
        "pr_corpus_v2",
        "pr_corpus_v2_legacy_inline",
    ]
    repository: RepositoryName
    pr_number: int
    html_url: str | None
    author: str | None
    merged: bool
    merged_at: str | None
    closed_at: str | None
    labels: tuple[str, ...]
    title: str
    body: TextSource
    comments: tuple[TextSource, ...]
    linked_issues: tuple[LinkedIssueRef, ...]
    diff_stats: dict[str, Any]
    touched_files_patch: tuple[TouchedFilePatch, ...]
    subsystem_labels: tuple[str, ...]
    change_kind: NonEmptyStr
    tests_added: tuple[AddedTest, ...]
    commit_refs: CommitRefs
    revert_evidence: RevertEvidence | None
    reverted_by: int | None
    info_richness: dict[str, Any]
    collection: CollectionMetadata

    @model_validator(mode="after")
    def _validate_record(self) -> PrCorpusV2:
        if self.pr_number <= 0:
            raise ValueError("PR number must be positive")
        if self.reverted_by is not None and self.reverted_by <= 0:
            raise ValueError("reverted_by must be positive")
        if len({source.source_id for source in self.comments}) != len(self.comments):
            raise ValueError("comment source IDs must be unique")
        if len({test.test_node_id for test in self.tests_added}) != len(
            self.tests_added
        ):
            raise ValueError("test node IDs must be unique")
        issue_keys = tuple(
            (issue.repository, issue.number) for issue in self.linked_issues
        )
        if issue_keys != tuple(sorted(set(issue_keys))):
            raise ValueError("linked issues must be sorted and unique")
        if any(issue.repository != self.repository for issue in self.linked_issues):
            raise ValueError("linked issue repository must match PR repository")
        if self.source_schema == "pr_corpus_v2":
            if self.body.source_id != "body":
                raise ValueError("new v2 PR body source_id must be body")
            if not (
                self.commit_refs.head_sha
                and self.commit_refs.base_sha
                and self.commit_refs.base_ref
            ):
                raise ValueError(
                    "new v2 records require head/base commit references"
                )
            for value, label in (
                (self.commit_refs.head_sha, "head_sha"),
                (self.commit_refs.base_sha, "base_sha"),
                (self.commit_refs.merge_commit_sha, "merge_commit_sha"),
            ):
                if value is not None and not re.fullmatch(
                    r"[0-9a-f]{40,64}",
                    value,
                ):
                    raise ValueError(f"new v2 {label} is not a commit SHA")
            if any(
                issue.fetch_status is IssueFetchStatus.LEGACY_INLINE
                for issue in self.linked_issues
            ):
                raise ValueError("new v2 records cannot emit legacy inline issues")
            required_audit = (
                self.collection.files_expected_count,
                self.collection.files_fetch_complete,
                self.collection.conversation_comments_total_count,
                self.collection.conversation_comments_fetch_complete,
                self.collection.review_comments_total_count,
                self.collection.review_comments_fetch_complete,
                self.collection.closing_refs_collection_complete,
            )
            if any(value is None for value in required_audit):
                raise ValueError("new v2 records require complete collection audit")
            file_stats = self.diff_stats.get("files")
            if not isinstance(file_stats, list):
                raise ValueError("new v2 diff_stats.files must be an array")
            file_paths = tuple(
                item.get("path") if isinstance(item, dict) else None
                for item in file_stats
            )
            if (
                any(not isinstance(path, str) or not path for path in file_paths)
                or len(set(file_paths)) != len(file_paths)
            ):
                raise ValueError("new v2 diff file paths must be non-empty and unique")
            if len(file_stats) != self.collection.files_retained_count:
                raise ValueError("diff file count disagrees with collection audit")
            patch_paths = tuple(item.path for item in self.touched_files_patch)
            if patch_paths != tuple(sorted(set(patch_paths))):
                raise ValueError("touched file patches must be sorted and unique")
            if not set(patch_paths) <= set(file_paths):
                raise ValueError("touched file patch is absent from diff stats")
            conversation_count = sum(
                item.source_id.startswith("pr_comment:")
                for item in self.comments
            )
            review_count = sum(
                item.source_id.startswith("pr_review_comment:")
                for item in self.comments
            )
            if (
                conversation_count + review_count != len(self.comments)
                or conversation_count
                != self.collection.conversation_comments_retained_count
                or review_count
                != self.collection.review_comments_retained_count
            ):
                raise ValueError("PR comment counts disagree with collection audit")
        canonical_bytes(self.diff_stats)
        canonical_bytes(self.info_richness)
        return self

    @classmethod
    def from_v1(
        cls,
        record: dict[str, Any],
        *,
        config_sha: str,
        repository: str,
    ) -> PrCorpusV2:
        tests = tuple(
            AddedTest(
                path=item["path"],
                test_node_id=f"{item['path']}::{item['test_name']}",
                test_name=item["test_name"],
                def_span=DefSpan(
                    start_line=None,
                    end_line=None,
                    complete=False,
                ),
                added_code=item.get("code", ""),
            )
            for item in record.get("tests_added", [])
        )
        body_text = record.get("body") or ""
        comments = tuple(
            TextSource(
                source_id=f"comment:{index}",
                text=text or "",
                char_end=len(text or ""),
                truncation=TruncationStatus.UNKNOWN,
            )
            for index, text in enumerate(record.get("comments", []))
        )
        revert_of = record.get("revert_of")
        revert_evidence = (
            RevertEvidence(
                kind=RevertEvidenceKind.TITLE_PATTERN,
                target_pr=revert_of,
                source_id="title",
                quote=record.get("title") or f"revert #{revert_of}",
            )
            if revert_of
            else None
        )
        legacy_issues = tuple(
            LinkedIssueRef(
                repository=repository,
                number=int(item["number"]),
                issue_ref=None,
                fetch_status=IssueFetchStatus.LEGACY_INLINE,
                link_evidence=(
                    IssueLinkEvidence(
                        kind=IssueLinkKind.LEGACY_IMPORT,
                        source_id="legacy:inline",
                        locator="legacy:inline",
                    ),
                ),
                fetch_failure=None,
                legacy_payload=dict(item),
            )
            for item in record.get("linked_issues", [])
            if isinstance(item, dict) and int(item.get("number", 0)) > 0
        )
        return cls(
            source_schema="pr_corpus_v1",
            repository=repository,
            pr_number=record["pr_number"],
            html_url=record.get("html_url"),
            author=record.get("author"),
            merged=bool(record.get("merged")),
            merged_at=record.get("merged_at"),
            closed_at=record.get("closed_at"),
            labels=tuple(record.get("labels", [])),
            title=record.get("title") or "",
            body=TextSource(
                source_id="body",
                text=body_text,
                char_end=len(body_text),
                truncation=TruncationStatus.UNKNOWN,
            ),
            comments=comments,
            linked_issues=legacy_issues,
            diff_stats=dict(record.get("diff_stats", {})),
            touched_files_patch=(),
            subsystem_labels=tuple(record.get("subsystem_labels", [])),
            change_kind=record.get("change_kind") or "unknown",
            tests_added=tests,
            commit_refs=CommitRefs(
                merge_commit_sha=None,
                head_sha=None,
                base_sha=None,
                base_ref=None,
            ),
            revert_evidence=revert_evidence,
            reverted_by=record.get("reverted_by"),
            info_richness=dict(record.get("info_richness", {})),
            collection=CollectionMetadata(
                config_sha=config_sha,
                collected_at=record.get("collected_at") or "unknown",
            ),
        )

    @classmethod
    def from_legacy_v2(
        cls,
        record: dict[str, Any],
        *,
        repository: str,
    ) -> PrCorpusV2:
        payload = dict(record)
        embedded = payload.get("linked_issues", [])
        legacy_issues = []
        for item in embedded:
            if not isinstance(item, dict):
                continue
            if "fetch_status" in item:
                legacy_issues.append(item)
                continue
            number = int(item.get("number", 0))
            if number <= 0:
                continue
            legacy_issues.append(
                LinkedIssueRef(
                    repository=repository,
                    number=number,
                    issue_ref=None,
                    fetch_status=IssueFetchStatus.LEGACY_INLINE,
                    link_evidence=(
                        IssueLinkEvidence(
                            kind=IssueLinkKind.LEGACY_IMPORT,
                            source_id="legacy:inline",
                            locator="legacy:inline",
                        ),
                    ),
                    fetch_failure=None,
                    legacy_payload=dict(item),
                ).to_record()
            )
        payload["source_schema"] = "pr_corpus_v2_legacy_inline"
        payload["repository"] = repository
        payload["linked_issues"] = legacy_issues
        return cls.from_record(payload)


def load_pr_record(
    record: dict[str, Any],
    *,
    config_sha: str,
    repository: str | None = None,
) -> PrCorpusV2:
    version = record.get("schema_version")
    if version == "pr_corpus_v2":
        collection = record.get("collection")
        audit_keys = (
            set(collection)
            if isinstance(collection, dict)
            else set()
        )
        is_legacy = (
            "repository" not in record
            or any(
                isinstance(item, dict) and "fetch_status" not in item
                for item in record.get("linked_issues", [])
            )
            or not {
                "files_expected_count",
                "files_fetch_complete",
                "conversation_comments_total_count",
                "conversation_comments_fetch_complete",
                "review_comments_total_count",
                "review_comments_fetch_complete",
                "closing_refs_collection_complete",
            }.issubset(audit_keys)
        )
        if not is_legacy:
            return PrCorpusV2.from_record(record)
        resolved_repository = record.get("repository") or repository
        if resolved_repository is None:
            raise ValueError(
                "repository is required to load legacy pr_corpus_v2"
            )
        if repository is not None and resolved_repository != repository:
            raise ValueError("PR record repository does not match caller")
        return PrCorpusV2.from_legacy_v2(
            record,
            repository=resolved_repository,
        )
    if version == "pr_corpus_v1":
        if repository is None:
            raise ValueError("repository is required to load pr_corpus_v1")
        return PrCorpusV2.from_v1(
            record,
            config_sha=config_sha,
            repository=repository,
        )
    raise ValueError(f"unsupported PR corpus schema: {version!r}")
