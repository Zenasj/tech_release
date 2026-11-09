"""Linked-Issue discovery, side storage, retry, and integrity helpers."""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from mrminer.common.github_client import (
    GitHubClient,
    GitHubNotFound,
    GitHubRequestError,
)
from mrminer.common.storage import JsonlStore
from mrminer.s1_miner import triage
from mrminer.s1_miner.issue_schema import (
    IssueClosingRef,
    IssueCollectionMetadata,
    IssueCommentCollection,
    IssueCorpusV1,
    IssueFetchEvent,
)
from mrminer.s1_miner.pr_schema import (
    IssueFetchFailure,
    IssueFetchFailureKind,
    IssueFetchStatus,
    IssueLinkEvidence,
    IssueLinkKind,
    LinkedIssueRef,
    PrCorpusV2,
    TextSource,
    TruncationStatus,
)


_CLOSING_ISSUES_QUERY = """
query ClosingIssues($owner: String!, $name: String!, $number: Int!, $after: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      closingIssuesReferences(first: 100, after: $after) {
        nodes { number }
        pageInfo { hasNextPage endCursor }
      }
    }
  }
}
"""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _repo_parts(repository: str) -> tuple[str, str]:
    parts = repository.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"invalid GitHub repository: {repository!r}")
    return parts[0], parts[1]


def closing_issue_evidence(
    client: GitHubClient,
    *,
    repository: str,
    pr_number: int,
    max_pages: int = 30,
) -> tuple[dict[int, tuple[IssueLinkEvidence, ...]], bool]:
    if max_pages <= 0:
        raise ValueError("max_pages must be positive")
    owner, name = _repo_parts(repository)
    after = None
    discovered: dict[int, set[IssueLinkEvidence]] = {}
    for _page in range(max_pages):
        data = client.graphql(
            _CLOSING_ISSUES_QUERY,
            {
                "owner": owner,
                "name": name,
                "number": pr_number,
                "after": after,
            },
        )
        repository_data = data.get("repository")
        pull_request = (
            repository_data.get("pullRequest")
            if isinstance(repository_data, dict)
            else None
        )
        connection = (
            pull_request.get("closingIssuesReferences")
            if isinstance(pull_request, dict)
            else None
        )
        if not isinstance(connection, dict):
            raise ValueError("GraphQL closingIssuesReferences is missing")
        nodes = connection.get("nodes")
        page_info = connection.get("pageInfo")
        if not isinstance(nodes, list) or not isinstance(page_info, dict):
            raise ValueError("GraphQL closing issue connection is malformed")
        for node in nodes:
            number = node.get("number") if isinstance(node, dict) else None
            if not isinstance(number, int) or number <= 0:
                raise ValueError("GraphQL closing issue has invalid number")
            discovered.setdefault(number, set()).add(
                IssueLinkEvidence(
                    kind=IssueLinkKind.CLOSING_REFERENCE,
                    source_id="graphql",
                    locator="graphql:closingIssuesReferences",
                )
            )
        if not page_info.get("hasNextPage"):
            return _sorted_evidence(discovered), True
        after = page_info.get("endCursor")
        if not isinstance(after, str) or not after:
            raise ValueError("GraphQL connection has no end cursor")
    return _sorted_evidence(discovered), False


def _sorted_evidence(
    discovered: dict[int, set[IssueLinkEvidence]],
) -> dict[int, tuple[IssueLinkEvidence, ...]]:
    return {
        number: tuple(
            sorted(
                evidence,
                key=lambda item: (
                    item.kind.value,
                    item.source_id,
                    item.locator,
                ),
            )
        )
        for number, evidence in sorted(discovered.items())
    }


def discover_linked_issues(
    *,
    title: str,
    body: str,
    conversation_comments: tuple[TextSource, ...],
    review_comments: tuple[TextSource, ...],
    closing: dict[int, tuple[IssueLinkEvidence, ...]],
) -> dict[int, tuple[IssueLinkEvidence, ...]]:
    collections = [
        triage.issue_link_evidence(
            title,
            source_id="title",
            locator_prefix="title",
            kind=IssueLinkKind.FIXES_KEYWORD,
        ),
        triage.issue_link_evidence(
            body,
            source_id="body",
            locator_prefix="body",
            kind=IssueLinkKind.FIXES_KEYWORD,
        ),
        closing,
    ]
    for source in (*conversation_comments, *review_comments):
        if source.author and source.author.casefold().endswith(("bot", "[bot]")):
            continue
        collections.append(
            triage.issue_link_evidence(
                source.text,
                source_id=source.source_id,
                locator_prefix=source.source_id,
                kind=IssueLinkKind.COMMENT_MENTION,
            )
        )
    return triage.merge_issue_link_evidence(*collections)


def _comment_source(
    raw: dict[str, Any],
    *,
    prefix: str,
    max_chars: int | None = None,
) -> TextSource:
    raw_id = raw.get("id")
    if not isinstance(raw_id, int) or raw_id <= 0:
        raise ValueError(f"{prefix} comment has no stable numeric id")
    full_text = raw.get("body") or ""
    text = full_text if max_chars is None else full_text[:max_chars]
    return TextSource(
        source_id=f"{prefix}:{raw_id}",
        text=text,
        char_end=len(text),
        truncation=(
            TruncationStatus.TRUNCATED
            if len(text) < len(full_text)
            else TruncationStatus.COMPLETE
        ),
        author=(raw.get("user") or {}).get("login"),
        created_at=raw.get("created_at"),
        html_url=raw.get("html_url"),
    )


def pr_comment_sources(
    raw_comments: tuple[dict, ...],
    *,
    prefix: str,
    retained_limit: int,
) -> tuple[TextSource, ...]:
    if retained_limit < 0:
        raise ValueError("retained_limit cannot be negative")
    retained = raw_comments[:retained_limit]
    return tuple(
        sorted(
            (_comment_source(raw, prefix=prefix) for raw in retained),
            key=lambda source: source.source_id,
        )
    )


def fetch_issue_corpus(
    client: GitHubClient,
    *,
    repository: str,
    issue_number: int,
    config_sha: str,
    max_comments: int,
    max_comment_chars: int,
    closing_pr_numbers: tuple[int, ...],
    collected_at: str,
) -> IssueCorpusV1 | None:
    if max_comments < 0:
        raise ValueError("max_comments cannot be negative")
    if max_comment_chars <= 0:
        raise ValueError("max_comment_chars must be positive")
    detail = client.get(f"/repos/{repository}/issues/{issue_number}")
    if detail.get("number") != issue_number:
        raise ValueError("GitHub Issue response identity mismatch")
    if "pull_request" in detail:
        return None
    total_comments = detail.get("comments")
    if type(total_comments) is not int or total_comments < 0:
        raise ValueError("GitHub Issue response has invalid comments count")
    for field in ("html_url", "state", "created_at", "updated_at"):
        if not detail.get(field):
            raise ValueError(f"GitHub Issue response is missing {field}")
    pages = max(1, math.ceil(max_comments / 100)) if max_comments else 0
    if pages:
        paginated = client.paginate_with_metadata(
            f"/repos/{repository}/issues/{issue_number}/comments",
            max_pages=pages,
            expected_count=total_comments,
        )
        raw_comments = paginated.items[:max_comments]
    else:
        raw_comments = ()
    comments = tuple(
        sorted(
            (
                _comment_source(
                    raw,
                    prefix="issue_comment",
                    max_chars=max_comment_chars,
                )
                for raw in raw_comments
            ),
            key=lambda source: source.source_id,
        )
    )
    retained_count = len(comments)
    truncated = retained_count < total_comments
    if truncated:
        next_page = retained_count // 100 + 1
        next_offset = retained_count % 100
        next_cursor = (
            f"rest:page={next_page};per_page=100;offset={next_offset}"
        )
    else:
        next_cursor = None
    body = detail.get("body") or ""
    labels = tuple(
        sorted(
            {
                label.get("name") or ""
                for label in detail.get("labels", [])
                if label.get("name")
            }
        )
    )
    return IssueCorpusV1(
        repository=repository,
        issue_number=issue_number,
        html_url=detail["html_url"],
        state=detail["state"],
        title=detail.get("title") or "",
        body=TextSource(
            source_id="issue_body",
            text=body,
            char_end=len(body),
            truncation=TruncationStatus.COMPLETE,
        ),
        labels=labels,
        author=(detail.get("user") or {}).get("login"),
        created_at=detail["created_at"],
        updated_at=detail["updated_at"],
        closed_at=detail.get("closed_at"),
        comments=IssueCommentCollection(
            items=comments,
            total_count=total_comments,
            retained_count=retained_count,
            collection_truncated=truncated,
            next_cursor=next_cursor,
        ),
        closing_refs=tuple(
            IssueClosingRef(
                pr_number=number,
                source="graphql_closingIssuesReferences",
            )
            for number in sorted(set(closing_pr_numbers))
        ),
        collection=IssueCollectionMetadata(
            config_sha=config_sha,
            collected_at=collected_at,
            max_comments=max_comments,
            max_comment_chars=max_comment_chars,
        ),
    )


def atomic_write_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(record, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=".tmp-",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def put_issue_record(path: Path, issue: IssueCorpusV1) -> bool:
    if path.exists():
        existing = IssueCorpusV1.from_record(json.loads(path.read_text()))
        if existing.identity != issue.identity:
            raise ValueError("issue side-store identity conflict")
        existing_refs = {
            (ref.pr_number, ref.source) for ref in existing.closing_refs
        }
        new_refs = {
            (ref.pr_number, ref.source) for ref in issue.closing_refs
        }
        if new_refs <= existing_refs:
            return False
        record = existing.to_record()
        record["closing_refs"] = [
            {"pr_number": number, "source": source}
            for number, source in sorted(existing_refs | new_refs)
        ]
        atomic_write_record(path, IssueCorpusV1.from_record(record).to_record())
        return True
    atomic_write_record(path, issue.to_record())
    return True


def _failure_from_error(
    error: BaseException,
    *,
    attempts: int,
    attempted_at: str,
) -> tuple[IssueFetchStatus, IssueFetchFailure]:
    if isinstance(error, GitHubNotFound):
        kind = IssueFetchFailureKind.NOT_FOUND
        status = IssueFetchStatus.NOT_FOUND
    elif isinstance(error, GitHubRequestError) and error.kind == "forbidden":
        kind = IssueFetchFailureKind.FORBIDDEN
        status = IssueFetchStatus.FORBIDDEN
    elif isinstance(error, GitHubRequestError) and error.kind == "rate_limited":
        kind = IssueFetchFailureKind.RATE_LIMITED
        status = IssueFetchStatus.TRANSIENT_ERROR
    elif isinstance(error, GitHubRequestError) and error.kind == "network":
        kind = IssueFetchFailureKind.NETWORK
        status = IssueFetchStatus.TRANSIENT_ERROR
    else:
        kind = IssueFetchFailureKind.MALFORMED
        status = IssueFetchStatus.TRANSIENT_ERROR
    return status, IssueFetchFailure(
        kind=kind,
        attempts=attempts,
        last_attempted_at=attempted_at,
    )


def _append_fetch_event(
    path: Path,
    event: IssueFetchEvent,
) -> None:
    JsonlStore(path).append(
        event.to_record(),
        idempotency_key=event.event_id,
    )


def collect_linked_issue_refs(
    client: GitHubClient,
    *,
    repository: str,
    pr_number: int,
    evidence_by_issue: dict[int, tuple[IssueLinkEvidence, ...]],
    repository_dir: Path,
    config_sha: str,
    max_comments: int,
    max_comment_chars: int,
    attempted_at: str,
    prior_attempts: dict[int, int] | None = None,
) -> tuple[LinkedIssueRef, ...]:
    references: list[LinkedIssueRef] = []
    failure_log = repository_dir / "issue_fetch_failures.jsonl"
    for issue_number, evidence in sorted(evidence_by_issue.items()):
        issue_path = repository_dir / "issues" / f"{issue_number}.json"
        attempts = (prior_attempts or {}).get(issue_number, 0) + 1
        closing_prs = (
            (pr_number,)
            if any(
                item.kind is IssueLinkKind.CLOSING_REFERENCE
                for item in evidence
            )
            else ()
        )
        if issue_path.exists():
            issue = IssueCorpusV1.from_record(json.loads(issue_path.read_text()))
            if issue.identity != (repository, issue_number):
                raise ValueError("existing issue side-store identity mismatch")
            if closing_prs:
                record = issue.to_record()
                refs = {
                    (item["pr_number"], item["source"])
                    for item in record["closing_refs"]
                }
                refs.add((pr_number, "graphql_closingIssuesReferences"))
                record["closing_refs"] = [
                    {"pr_number": number, "source": source}
                    for number, source in sorted(refs)
                ]
                put_issue_record(issue_path, IssueCorpusV1.from_record(record))
            references.append(
                LinkedIssueRef(
                    repository=repository,
                    number=issue_number,
                    issue_ref=f"issues/{issue_number}.json",
                    fetch_status=IssueFetchStatus.STORED,
                    link_evidence=evidence,
                    fetch_failure=None,
                )
            )
            if attempts > 1:
                _append_fetch_event(
                    failure_log,
                    IssueFetchEvent.create(
                        repository=repository,
                        pr_number=pr_number,
                        issue_number=issue_number,
                        outcome="resolved",
                        failure=None,
                        occurred_at=attempted_at,
                    ),
                )
            continue
        try:
            issue = fetch_issue_corpus(
                client,
                repository=repository,
                issue_number=issue_number,
                config_sha=config_sha,
                max_comments=max_comments,
                max_comment_chars=max_comment_chars,
                closing_pr_numbers=closing_prs,
                collected_at=attempted_at,
            )
        except (GitHubNotFound, GitHubRequestError, ValueError) as error:
            status, failure = _failure_from_error(
                error,
                attempts=attempts,
                attempted_at=attempted_at,
            )
            references.append(
                LinkedIssueRef(
                    repository=repository,
                    number=issue_number,
                    issue_ref=None,
                    fetch_status=status,
                    link_evidence=evidence,
                    fetch_failure=failure,
                )
            )
            _append_fetch_event(
                failure_log,
                IssueFetchEvent.create(
                    repository=repository,
                    pr_number=pr_number,
                    issue_number=issue_number,
                    outcome="failed",
                    failure=failure,
                    occurred_at=attempted_at,
                ),
            )
            continue
        if issue is None:
            continue
        put_issue_record(issue_path, issue)
        references.append(
            LinkedIssueRef(
                repository=repository,
                number=issue_number,
                issue_ref=f"issues/{issue_number}.json",
                fetch_status=IssueFetchStatus.STORED,
                link_evidence=evidence,
                fetch_failure=None,
            )
        )
        if attempts > 1:
            _append_fetch_event(
                failure_log,
                IssueFetchEvent.create(
                    repository=repository,
                    pr_number=pr_number,
                    issue_number=issue_number,
                    outcome="resolved",
                    failure=None,
                    occurred_at=attempted_at,
                ),
            )
    return tuple(
        sorted(references, key=lambda reference: reference.number)
    )


def retry_transient_issue_failures(
    client: GitHubClient,
    *,
    repository: str,
    repository_dir: Path,
    config_sha: str,
    max_comments: int,
    max_comment_chars: int,
    attempted_at: str | None = None,
) -> tuple[int, int]:
    if max_comments < 0:
        raise ValueError("max_comments cannot be negative")
    if max_comment_chars <= 0:
        raise ValueError("max_comment_chars must be positive")
    attempted_at = attempted_at or utc_now()
    retried = resolved = 0
    for pr_path in sorted(repository_dir.glob("*.json")):
        raw = json.loads(pr_path.read_text())
        if raw.get("schema_version") != "pr_corpus_v2":
            continue
        record = PrCorpusV2.from_record(raw)
        transient = {
            ref.number: ref
            for ref in record.linked_issues
            if ref.fetch_status is IssueFetchStatus.TRANSIENT_ERROR
        }
        if not transient:
            continue
        retried += len(transient)
        refreshed = collect_linked_issue_refs(
            client,
            repository=repository,
            pr_number=record.pr_number,
            evidence_by_issue={
                number: ref.link_evidence for number, ref in transient.items()
            },
            repository_dir=repository_dir,
            config_sha=config_sha,
            max_comments=max_comments,
            max_comment_chars=max_comment_chars,
            attempted_at=attempted_at,
            prior_attempts={
                number: ref.fetch_failure.attempts
                for number, ref in transient.items()
                if ref.fetch_failure is not None
            },
        )
        refreshed_by_number = {ref.number: ref for ref in refreshed}
        merged = tuple(
            sorted(
                [
                    refreshed_by_number.get(ref.number, ref)
                    if ref.number in transient
                    else ref
                    for ref in record.linked_issues
                ],
                key=lambda ref: ref.number,
            )
        )
        resolved += sum(
            ref.fetch_status is IssueFetchStatus.STORED
            for ref in refreshed
        )
        updated = record.to_record()
        updated["linked_issues"] = [ref.to_record() for ref in merged]
        atomic_write_record(pr_path, PrCorpusV2.from_record(updated).to_record())
    return retried, resolved
