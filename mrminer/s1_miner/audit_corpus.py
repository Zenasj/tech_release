"""Audit an S1 corpus without changing collected records.

Run example (from MR_Oracle_Mining/):

  python3 -m mrminer.s1_miner.audit_corpus \
      --repo pytorch/pytorch \
      --corpus-root data/pr_corpus \
      --min-prs 50 --require-complete \
      --output reports/s1_pytorch_audit.json

Input : data/pr_corpus/{owner}__{repo}/*.json and issues/*.json
Output: JSON report on stdout or at --output
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from mrminer.common.canonical import sha256_digest
from mrminer.common.fsck import FsckCatalog, fsck
from mrminer.common.schema import ChangeLanding, LandingKind
from mrminer.s1_miner.issue_collector import atomic_write_record, utc_now
from mrminer.s1_miner.issue_schema import IssueCorpusV1
from mrminer.s1_miner import triage
from mrminer.s1_miner.pr_schema import (
    IssueFetchStatus,
    IssueLinkKind,
    PrCorpusV2,
    load_pr_record,
)
from mrminer.s1_miner.run_s1 import (
    collection_ready_for_completion,
    derive_change_landing,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


def _p95(values: list[int]) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(len(ordered) * 0.95) - 1]


def _short_error(error: BaseException) -> str:
    text = " ".join(str(error).split())
    return text[:1000]


def audit_repository(
    *,
    repository_dir: Path,
    repository: str,
    config_sha: str,
    subsystem_rules: dict[str, str] | None = None,
    compile_stack: set[str] | None = None,
    min_prs: int = 0,
    require_complete: bool = False,
    expected_candidates: int | None = None,
    max_patch_p95_bytes: int = 300 * 1024,
) -> dict[str, Any]:
    if min_prs < 0:
        raise ValueError("min_prs cannot be negative")
    if max_patch_p95_bytes <= 0:
        raise ValueError("max_patch_p95_bytes must be positive")
    if expected_candidates is not None and expected_candidates < 0:
        raise ValueError("expected_candidates cannot be negative")

    pr_records: list[PrCorpusV2] = []
    issue_records: list[IssueCorpusV1] = []
    schema_errors: list[dict[str, str]] = []

    for path in sorted(repository_dir.glob("*.json")):
        try:
            raw = json.loads(path.read_text())
            pr_records.append(
                load_pr_record(
                    raw,
                    config_sha=config_sha,
                    repository=repository,
                )
            )
        except (OSError, json.JSONDecodeError, ValueError) as error:
            schema_errors.append(
                {"path": str(path), "error": _short_error(error)}
            )

    issues_dir = repository_dir / "issues"
    for path in sorted(issues_dir.glob("*.json")):
        try:
            raw = json.loads(path.read_text())
            issue_records.append(IssueCorpusV1.from_record(raw))
        except (OSError, json.JSONDecodeError, ValueError) as error:
            schema_errors.append(
                {"path": str(path), "error": _short_error(error)}
            )

    fsck_issues = fsck(
        FsckCatalog(
            pr_records=tuple(pr_records),
            issue_records=tuple(issue_records),
        )
    )
    new_v2 = [
        record
        for record in pr_records
        if record.source_schema == "pr_corpus_v2"
    ]
    endpoint_complete = [
        record
        for record in new_v2
        if collection_ready_for_completion(record)
    ]
    config_mismatch = [
        record.pr_number
        for record in new_v2
        if record.collection.config_sha != config_sha
    ]
    patches_by_key = {
        (record.pr_number, patch.path): patch
        for record in new_v2
        for patch in record.touched_files_patch
    }
    if subsystem_rules is not None and compile_stack is not None:
        expected_patch_keys = {
            (record.pr_number, item["path"])
            for record in new_v2
            for item in record.diff_stats["files"]
            if set(
                triage.subsystem_labels(
                    [{"filename": item["path"]}],
                    subsystem_rules,
                )
            )
            & compile_stack
        }
    else:
        expected_patch_keys = set(patches_by_key)
    unexpected_patch_keys = set(patches_by_key) - expected_patch_keys
    patch_accounted_keys = {
        key
        for key in expected_patch_keys
        if (
            key in patches_by_key
            and (
                patches_by_key[key].patch is not None
                or patches_by_key[key].truncated
            )
        )
    }
    patch_sizes = [
        len(patch.patch.encode("utf-8"))
        for patch in patches_by_key.values()
        if patch.patch is not None
    ]
    patch_p95 = _p95(patch_sizes)
    linked = [
        reference
        for record in new_v2
        for reference in record.linked_issues
    ]
    status_counts = Counter(reference.fetch_status.value for reference in linked)
    evidence_kind_counts = Counter(
        evidence.kind.value
        for reference in linked
        for evidence in reference.link_evidence
    )
    bot_comment_evidence = []
    unreproducible_comment_evidence = []
    for record in new_v2:
        sources = {
            source.source_id: source
            for source in record.comments
        }
        for reference in record.linked_issues:
            for evidence in reference.link_evidence:
                if evidence.kind is not IssueLinkKind.COMMENT_MENTION:
                    continue
                source = sources.get(evidence.source_id)
                if (
                    source is not None
                    and source.author
                    and source.author.casefold().endswith(("bot", "[bot]"))
                ):
                    bot_comment_evidence.append(
                        {
                            "pr_number": record.pr_number,
                            "issue_number": reference.number,
                            "source_id": evidence.source_id,
                        }
                    )
                reproduced = (
                    triage.issue_link_evidence(
                        source.text,
                        source_id=evidence.source_id,
                        locator_prefix=evidence.source_id,
                        kind=IssueLinkKind.COMMENT_MENTION,
                    ).get(reference.number, ())
                    if source is not None
                    else ()
                )
                if evidence not in reproduced:
                    unreproducible_comment_evidence.append(
                        {
                            "pr_number": record.pr_number,
                            "issue_number": reference.number,
                            "source_id": evidence.source_id,
                            "locator": evidence.locator,
                        }
                    )
    issue_ids = {
        (record.repository, record.issue_number)
        for record in issue_records
    }
    stored = [
        reference
        for reference in linked
        if reference.fetch_status is IssueFetchStatus.STORED
    ]
    stored_resolved = [
        reference
        for reference in stored
        if (reference.repository, reference.number) in issue_ids
    ]
    failure_count = len(linked) - len(stored)
    body_complete = sum(
        issue.body.truncation.value == "complete"
        for issue in issue_records
    )
    comment_items = [
        comment
        for issue in issue_records
        for comment in issue.comments.items
    ]
    completed_path = repository_dir / "completed.txt"
    completed_lines = (
        [line for line in completed_path.read_text().splitlines() if line.strip()]
        if completed_path.exists()
        else []
    )
    completed_numbers = {
        int(line.split()[0])
        for line in completed_lines
    }
    failed_path = repository_dir / "failed.txt"
    failure_lines = (
        [line for line in failed_path.read_text().splitlines() if line.strip()]
        if failed_path.exists()
        else []
    )
    failed_numbers = {
        int(line.split()[0])
        for line in failure_lines
    }
    unresolved_failure_numbers = sorted(failed_numbers - completed_numbers)
    count_sources = Counter(
        record.collection.files_expected_count_source
        for record in new_v2
    )
    git_count_overrides = [
        {
            "pr_number": record.pr_number,
            "github_reported_count": (
                record.collection.files_github_reported_count
            ),
            "git_base_head_count": record.collection.files_expected_count,
        }
        for record in new_v2
        if record.collection.files_expected_count_source == "git_base_head"
    ]
    records_by_number = {
        record.pr_number: record
        for record in pr_records
    }
    revert_evidence_counts = Counter(
        record.revert_evidence.kind.value
        for record in new_v2
        if record.revert_evidence is not None
    )
    unreliable_bot_reverts = [
        record.pr_number
        for record in new_v2
        if (
            record.revert_evidence is not None
            and record.revert_evidence.kind.value == "bot_comment"
            and triage.explicit_bot_revert_target(
                record.revert_evidence.quote
            )
            != record.revert_evidence.target_pr
        )
    ]
    unreliable_title_reverts = [
        record.pr_number
        for record in new_v2
        if (
            record.revert_evidence is not None
            and record.revert_evidence.kind.value == "title_pattern"
            and triage.revert_of(
                record.title,
                record.body.text,
                current_pr=record.pr_number,
            )
            != record.revert_evidence.target_pr
        )
    ]
    invalid_reverted_by = []
    for record in new_v2:
        if record.reverted_by is None:
            continue
        reverter = records_by_number.get(record.reverted_by)
        if (
            reverter is None
            or not reverter.merged
            or reverter.revert_evidence is None
            or reverter.revert_evidence.target_pr != record.pr_number
        ):
            invalid_reverted_by.append(
                {
                    "pr_number": record.pr_number,
                    "reverted_by": record.reverted_by,
                }
            )

    expected_landings = [
        landing
        for record in pr_records
        if (landing := derive_change_landing(record)) is not None
    ]
    landings_path = repository_dir.parent / "landings.jsonl"
    landing_schema_errors = []
    actual_landings = []
    if landings_path.exists():
        for line_number, line in enumerate(
            landings_path.read_text().splitlines(),
            1,
        ):
            if not line.strip():
                continue
            try:
                actual_landings.append(
                    ChangeLanding.from_record(json.loads(line))
                )
            except (json.JSONDecodeError, ValueError) as error:
                landing_schema_errors.append(
                    {
                        "line": line_number,
                        "error": _short_error(error),
                    }
                )
    expected_landing_records = {
        json.dumps(item.to_record(), sort_keys=True)
        for item in expected_landings
    }
    actual_landing_records = {
        json.dumps(item.to_record(), sort_keys=True)
        for item in actual_landings
    }

    integrity_errors = [
        *(
            f"schema:{item['path']}: {item['error']}"
            for item in schema_errors
        ),
        *(
            f"fsck:{item.code}:{item.source}->{item.target}"
            for item in fsck_issues
        ),
    ]
    gate_failures = list(integrity_errors)
    if len(new_v2) < min_prs:
        gate_failures.append(
            f"corpus has {len(new_v2)} new v2 PRs, requires at least {min_prs}"
        )
    if config_mismatch:
        gate_failures.append(
            "one or more new v2 PRs use a different collection config"
        )
    if len(patch_accounted_keys) != len(expected_patch_keys):
        gate_failures.append("compile-stack patch accounting is below 100%")
    if unexpected_patch_keys:
        gate_failures.append("non-compile-stack patches were retained")
    if patch_p95 is not None and patch_p95 >= max_patch_p95_bytes:
        gate_failures.append(
            f"retained patch P95 is {patch_p95} bytes, must be below "
            f"{max_patch_p95_bytes}"
        )
    if len(stored_resolved) != len(stored):
        gate_failures.append("stored Issue side-record coverage is below 100%")
    if bot_comment_evidence:
        gate_failures.append("one or more Issue links came from bot comments")
    if unreproducible_comment_evidence:
        gate_failures.append(
            "one or more comment Issue locators cannot be reproduced"
        )
    if require_complete and len(endpoint_complete) != len(new_v2):
        gate_failures.append("one or more PR discovery endpoints are incomplete")
    if require_complete and unresolved_failure_numbers:
        gate_failures.append("one or more collection failures remain unresolved")
    if (
        expected_candidates is not None
        and len(completed_numbers) != expected_candidates
    ):
        gate_failures.append(
            f"completed candidate count is {len(completed_numbers)}, "
            f"expected {expected_candidates}"
        )
    if unreliable_bot_reverts:
        gate_failures.append(
            "one or more bot-comment revert claims lack explicit direction"
        )
    if unreliable_title_reverts:
        gate_failures.append(
            "one or more title revert claims lack an explicit target"
        )
    if invalid_reverted_by:
        gate_failures.append(
            "one or more reverted_by links lack a merged matching reverter"
        )
    if landing_schema_errors:
        gate_failures.append("one or more ChangeLanding rows are invalid")
    if landings_path.exists() and (
        actual_landing_records != expected_landing_records
        or len(actual_landings) != len(actual_landing_records)
    ):
        gate_failures.append(
            "ChangeLanding index does not exactly match current PR records"
        )

    return {
        "report_version": "s1_audit_v1",
        "repository": repository,
        "repository_dir": str(repository_dir),
        "generated_at": utc_now(),
        "pr": {
            "total": len(pr_records),
            "new_v2_total": len(new_v2),
            "legacy_compatible_total": len(pr_records) - len(new_v2),
            "endpoint_complete_count": len(endpoint_complete),
            "endpoint_complete_rate": _ratio(
                len(endpoint_complete),
                len(new_v2),
            ),
            "files_fetch_complete_count": sum(
                bool(record.collection.files_fetch_complete)
                for record in new_v2
            ),
            "conversation_comments_fetch_complete_count": sum(
                bool(
                    record.collection.conversation_comments_fetch_complete
                )
                for record in new_v2
            ),
            "review_comments_fetch_complete_count": sum(
                bool(record.collection.review_comments_fetch_complete)
                for record in new_v2
            ),
            "closing_refs_fetch_complete_count": sum(
                bool(record.collection.closing_refs_collection_complete)
                for record in new_v2
            ),
            "config_mismatch_pr_numbers": sorted(config_mismatch),
            "files_expected_count_source_counts": dict(
                sorted(count_sources.items())
            ),
            "git_base_head_count_overrides": git_count_overrides,
            "compile_stack_candidate_rate": _ratio(
                len(new_v2),
                len(completed_numbers),
            ),
            "merged_count": sum(record.merged for record in new_v2),
            "merged_rate": _ratio(
                sum(record.merged for record in new_v2),
                len(new_v2),
            ),
            "with_linked_issues_count": sum(
                bool(record.linked_issues)
                for record in new_v2
            ),
            "with_linked_issues_rate": _ratio(
                sum(bool(record.linked_issues) for record in new_v2),
                len(new_v2),
            ),
            "with_added_tests_count": sum(
                bool(record.tests_added)
                for record in new_v2
            ),
            "with_added_tests_rate": _ratio(
                sum(bool(record.tests_added) for record in new_v2),
                len(new_v2),
            ),
            "added_test_total": sum(
                len(record.tests_added)
                for record in new_v2
            ),
            "change_kind_counts": dict(
                sorted(
                    Counter(
                        record.change_kind
                        for record in new_v2
                    ).items()
                )
            ),
            "subsystem_label_counts": dict(
                sorted(
                    Counter(
                        label
                        for record in new_v2
                        for label in record.subsystem_labels
                    ).items()
                )
            ),
        },
        "state": {
            "completed_line_count": len(completed_lines),
            "completed_unique_count": len(completed_numbers),
            "duplicate_completed_line_count": (
                len(completed_lines) - len(completed_numbers)
            ),
            "failure_history_event_count": len(failure_lines),
            "failure_history_pr_count": len(failed_numbers),
            "resolved_failure_pr_count": len(
                failed_numbers & completed_numbers
            ),
            "unresolved_failure_pr_numbers": unresolved_failure_numbers,
            "expected_candidate_count": expected_candidates,
        },
        "diff": {
            "compile_stack_file_count": len(expected_patch_keys),
            "retained_patch_count": len(patches_by_key),
            "patch_accounted_count": len(patch_accounted_keys),
            "patch_accounted_rate": _ratio(
                len(patch_accounted_keys),
                len(expected_patch_keys),
            ),
            "retained_patch_p95_bytes": patch_p95,
            "patch_text_present_count": sum(
                patch.patch is not None
                for patch in patches_by_key.values()
            ),
            "explicitly_truncated_count": sum(
                patch.truncated
                for patch in patches_by_key.values()
            ),
            "patch_text_missing_count": sum(
                patch.patch is None
                for patch in patches_by_key.values()
            ),
            "enumerated_file_count": sum(
                len(record.diff_stats["files"])
                for record in new_v2
            ),
            "max_patch_p95_bytes": max_patch_p95_bytes,
            "unaccounted_files": [
                {"pr_number": number, "path": path}
                for number, path in sorted(
                    expected_patch_keys - patch_accounted_keys
                )
            ],
            "unexpected_retained_files": [
                {"pr_number": number, "path": path}
                for number, path in sorted(unexpected_patch_keys)
            ],
        },
        "linked_issues": {
            "reference_total": len(linked),
            "stored_count": len(stored),
            "explicit_failure_count": failure_count,
            "fetch_status_counts": dict(sorted(status_counts.items())),
            "link_evidence_kind_counts": dict(
                sorted(evidence_kind_counts.items())
            ),
            "stored_side_record_resolved_count": len(stored_resolved),
            "stored_side_record_coverage_rate": _ratio(
                len(stored_resolved),
                len(stored),
            ),
            "fetch_success_rate": _ratio(len(stored), len(linked)),
            "pr_with_reference_count": sum(
                bool(record.linked_issues)
                for record in new_v2
            ),
            "bot_comment_evidence_count": len(bot_comment_evidence),
            "bot_comment_evidence": bot_comment_evidence,
            "unreproducible_comment_evidence_count": len(
                unreproducible_comment_evidence
            ),
            "unreproducible_comment_evidence": (
                unreproducible_comment_evidence
            ),
        },
        "issues": {
            "side_record_total": len(issue_records),
            "body_complete_count": body_complete,
            "comment_item_count": len(comment_items),
            "comment_text_truncated_count": sum(
                item.truncation.value == "truncated"
                for item in comment_items
            ),
            "comment_collection_truncated_count": sum(
                issue.comments.collection_truncated
                for issue in issue_records
            ),
        },
        "reverts": {
            "evidence_kind_counts": dict(
                sorted(revert_evidence_counts.items())
            ),
            "unreliable_bot_comment_pr_numbers": sorted(
                unreliable_bot_reverts
            ),
            "unreliable_title_pr_numbers": sorted(
                unreliable_title_reverts
            ),
            "reverted_by_count": sum(
                record.reverted_by is not None
                for record in new_v2
            ),
            "invalid_reverted_by": invalid_reverted_by,
            "revert_landing_count": sum(
                landing.kind is LandingKind.REVERT
                for landing in expected_landings
            ),
        },
        "landings": {
            "path": str(landings_path),
            "expected_count": len(expected_landings),
            "actual_count": len(actual_landings),
            "unique_actual_count": len(actual_landing_records),
            "missing_count": len(
                expected_landing_records - actual_landing_records
            ),
            "unexpected_count": len(
                actual_landing_records - expected_landing_records
            ),
            "schema_errors": landing_schema_errors,
        },
        "schema_errors": schema_errors,
        "fsck_issues": [
            {
                "code": item.code,
                "source": item.source,
                "target": item.target,
            }
            for item in fsck_issues
        ],
        "gate": {
            "min_prs": min_prs,
            "require_complete": require_complete,
            "max_patch_p95_bytes": max_patch_p95_bytes,
            "passed": not gate_failures,
            "failures": gate_failures,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="pytorch/pytorch")
    parser.add_argument(
        "--corpus-root",
        default=str(PROJECT_ROOT / "data" / "pr_corpus"),
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "s1_pytorch.json"),
    )
    parser.add_argument("--min-prs", type=int, default=0)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--expected-candidates", type=int)
    parser.add_argument(
        "--max-patch-p95-bytes",
        type=int,
        default=300 * 1024,
    )
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    config = json.loads(Path(args.config).read_text())
    report = audit_repository(
        repository_dir=(
            Path(args.corpus_root) / args.repo.replace("/", "__")
        ),
        repository=args.repo,
        config_sha=sha256_digest(config),
        subsystem_rules=config.get("subsystem_rules"),
        compile_stack=set(config.get("compile_stack", [])),
        min_prs=args.min_prs,
        require_complete=args.require_complete,
        expected_candidates=args.expected_candidates,
        max_patch_p95_bytes=args.max_patch_p95_bytes,
    )
    if args.output:
        atomic_write_record(Path(args.output), report)
        print(f"[s1-audit] report: {args.output}", flush=True)
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if report["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
