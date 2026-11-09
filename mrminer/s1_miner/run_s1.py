"""S1 Change Miner — collect, enrich and triage compiler PRs into pr_corpus_v2 JSON.

Run examples (from MR_Oracle_Mining/):

  # small smoke run, no token needed (slow rate limits apply)
  python -m mrminer.s1_miner.run_s1 --repo pytorch/pytorch \
      --since 2025-04-10 --until 2025-04-12 --max-prs 3

  # real run (export GITHUB_TOKEN first), compile-stack PRs only
  export GITHUB_TOKEN=...   # fine-grained read-only token
  python -m mrminer.s1_miner.run_s1 --repo pytorch/pytorch \
      --since 2025-01-01 --until 2025-06-30 --compile-stack-only

  # post-hoc pass linking Revert PRs back to their targets
  python -m mrminer.s1_miner.run_s1 --repo pytorch/pytorch --link-reverts-only

Input : GitHub REST API + GraphQL (token via GITHUB_TOKEN or .env.local)
Config: configs/s1_pytorch.json (subsystem path rules)
Output: data/pr_corpus/{owner}__{repo}/{pr_number}.json  (schema pr_corpus_v2)
        data/pr_corpus/{owner}__{repo}/issues/{issue_number}.json (issue_v1)
        data/pr_corpus/{owner}__{repo}/issue_fetch_failures.jsonl
        data/pr_corpus/landings.jsonl  (ChangeLanding)
        data/pr_corpus/{owner}__{repo}/completed.txt / failed.txt (resume state)
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

from mrminer.common.canonical import sha256_digest
from mrminer.common.github_client import GitHubClient, GitHubRequestError
from mrminer.common.schema import (
    ChangeLanding,
    EvidenceSpec,
    EvidenceStrength,
    LandingKind,
)
from mrminer.s1_miner import triage
from mrminer.s1_miner.issue_collector import (
    atomic_write_record,
    closing_issue_evidence,
    collect_linked_issue_refs,
    discover_linked_issues,
    pr_comment_sources,
    retry_transient_issue_failures,
    utc_now,
)
from mrminer.s1_miner.diff_fallback import (
    parse_full_pull_diff,
    recover_git_commit_diff,
)
from mrminer.s1_miner.pr_schema import (
    CollectionMetadata,
    CommitRefs,
    PrCorpusV2,
    TextSource,
    TruncationStatus,
    load_pr_record,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "pr_corpus_v2"


# ----------------------------------------------------------------- collection

def search_pr_numbers(client: GitHubClient, repo: str, since: str, until: str) -> list[int]:
    """All closed-PR numbers in [since, until] (dates, inclusive), splitting
    windows that exceed the Search API's 1000-result cap.

    Bisection is on datetime, not just dates: some repos (e.g. tensorflow, via
    copybara automation) close >1000 PRs in a single calendar day, which a
    day-granularity split cannot break up. We therefore recurse down to
    sub-day (second) granularity. At date granularity the results are identical
    to a day-level split, so the PyTorch line is unaffected."""
    lo = dt.datetime.combine(dt.date.fromisoformat(since), dt.time.min, tzinfo=dt.timezone.utc)
    hi = dt.datetime.combine(dt.date.fromisoformat(until), dt.time(23, 59, 59), tzinfo=dt.timezone.utc)
    return _search_window(client, repo, lo, hi)


def _search_window(client: GitHubClient, repo: str, lo: dt.datetime, hi: dt.datetime) -> list[int]:
    lo = lo.replace(microsecond=0)
    hi = hi.replace(microsecond=0)
    query = f"repo:{repo} is:pr is:closed closed:{lo.isoformat()}..{hi.isoformat()}"
    first = client.search_issues(query, page=1)
    if first.get("incomplete_results"):
        raise GitHubRequestError(
            f"GitHub Search returned incomplete results for {lo.isoformat()}..{hi.isoformat()}",
            kind="incomplete_search",
        )
    total = _required_count(
        first,
        "total_count",
        artifact="GitHub Search response",
    )
    if total > 1000:
        if hi - lo <= dt.timedelta(seconds=1):
            raise GitHubRequestError(
                f"GitHub Search exceeds the 1000-item cap even at {lo.isoformat()} "
                f"(1-second window); cannot split further",
                kind="incomplete_search",
            )
        mid = lo + (hi - lo) / 2
        mid = mid.replace(microsecond=0)
        left = _search_window(client, repo, lo, mid)
        right = _search_window(client, repo, mid + dt.timedelta(seconds=1), hi)
        return sorted(set(left) | set(right))
    numbers = [it["number"] for it in first.get("items", [])]
    pages = (min(total, 1000) + 99) // 100
    for page in range(2, pages + 1):
        resp = client.search_issues(query, page=page)
        if resp.get("incomplete_results"):
            raise GitHubRequestError(
                f"GitHub Search returned incomplete page {page}",
                kind="incomplete_search",
            )
        numbers.extend(it["number"] for it in resp.get("items", []))
    expected = min(int(total), 1000)
    if len(set(numbers)) != expected:
        raise GitHubRequestError(
            f"GitHub Search expected {expected} unique PRs, got "
            f"{len(set(numbers))}",
            kind="incomplete_search",
        )
    print(f"[s1] window {lo.isoformat()}..{hi.isoformat()}: {len(numbers)} PRs", flush=True)
    return sorted(set(numbers))


# ----------------------------------------------------------------- enrichment

def _required_count(record: dict, field: str, *, artifact: str) -> int:
    value = record.get(field)
    if type(value) is not int or value < 0:
        raise ValueError(f"{artifact} has invalid {field}")
    return value


def _validate_collection_limits(
    *,
    max_comments: int,
    max_issue_comments: int,
    max_issue_comment_chars: int,
    max_file_pages: int,
    max_link_pages: int,
) -> None:
    if max_comments < 0 or max_issue_comments < 0:
        raise ValueError("comment limits cannot be negative")
    if max_issue_comment_chars <= 0:
        raise ValueError("max_issue_comment_chars must be positive")
    if max_file_pages <= 0 or max_link_pages <= 0:
        raise ValueError("pagination limits must be positive")


def fetch_pr_record(client: GitHubClient, repo: str, number: int,
                    subsystem_rules: dict, compile_stack: set[str],
                    max_comments: int, max_issue_comments: int,
                    max_issue_comment_chars: int, max_file_pages: int,
                    max_link_pages: int, config_sha: str,
                    repository_dir: Path,
                    compile_stack_only: bool = False) -> dict | None:
    _validate_collection_limits(
        max_comments=max_comments,
        max_issue_comments=max_issue_comments,
        max_issue_comment_chars=max_issue_comment_chars,
        max_file_pages=max_file_pages,
        max_link_pages=max_link_pages,
    )
    detail = client.get(f"/repos/{repo}/pulls/{number}")
    if detail.get("number") != number:
        raise ValueError("GitHub PR response identity mismatch")
    github_reported_files = _required_count(
        detail,
        "changed_files",
        artifact="GitHub PR response",
    )
    expected_files = github_reported_files
    files_expected_count_source = "github_detail"
    rest_file_capacity = min(max_file_pages * 100, 3000)
    if expected_files > rest_file_capacity:
        files = []
        files_fetch_complete = False
    else:
        file_page = client.paginate_with_metadata(
            f"/repos/{repo}/pulls/{number}/files",
            max_pages=max_file_pages,
            expected_count=expected_files,
        )
        files = list(file_page.items)
        files_fetch_complete = (
            file_page.complete and len(files) == expected_files
        )
    if not files_fetch_complete:
        print(
            f"[s1] PR #{number}: REST files incomplete "
            f"({len(files)}/{expected_files}); using full diff fallback",
            flush=True,
        )
        try:
            files = parse_full_pull_diff(
                client.get_pull_diff(repo, number),
                expected_count=expected_files,
            )
        except (GitHubRequestError, TimeoutError, ValueError) as error:
            print(
                f"[s1] PR #{number}: full diff is inconsistent "
                f"({type(error).__name__}: {error}); using exact "
                "base/head git diff",
                flush=True,
            )
            files = recover_git_commit_diff(
                repository=repo,
                base_sha=(detail.get("base") or {}).get("sha") or "",
                head_sha=(detail.get("head") or {}).get("sha") or "",
                retain_patch=lambda path: bool(
                    set(
                        triage.subsystem_labels(
                            [{"filename": path}],
                            subsystem_rules,
                        )
                    )
                    & compile_stack
                ),
            )
            expected_files = len(files)
            files_expected_count_source = "git_base_head"
        files_fetch_complete = True
    file_paths = [item.get("filename") for item in files]
    if any(not isinstance(path, str) or not path for path in file_paths):
        raise ValueError("GitHub file response has an invalid filename")
    if len(file_paths) != len(set(file_paths)):
        raise ValueError("GitHub file response has duplicate filenames")

    labels_by_subsystem = tuple(
        triage.subsystem_labels(files, subsystem_rules)
    )
    if (
        compile_stack_only
        and not (set(labels_by_subsystem) & compile_stack)
    ):
        return None

    expected_comments = _required_count(
        detail,
        "comments",
        artifact="GitHub PR response",
    )
    conversation_page = client.paginate_with_metadata(
        f"/repos/{repo}/issues/{number}/comments",
        max_pages=max_link_pages,
        expected_count=expected_comments,
    )
    expected_review_comments = _required_count(
        detail,
        "review_comments",
        artifact="GitHub PR response",
    )
    review_page = client.paginate_with_metadata(
        f"/repos/{repo}/pulls/{number}/comments",
        max_pages=max_link_pages,
        expected_count=expected_review_comments,
    )
    all_conversation = pr_comment_sources(
        conversation_page.items,
        prefix="pr_comment",
        retained_limit=len(conversation_page.items),
    )
    all_review = pr_comment_sources(
        review_page.items,
        prefix="pr_review_comment",
        retained_limit=len(review_page.items),
    )

    title = detail.get("title") or ""
    body = detail.get("body") or ""
    labels = [l["name"] for l in detail.get("labels", [])]
    try:
        closing, closing_complete = closing_issue_evidence(
            client,
            repository=repo,
            pr_number=number,
            max_pages=max_link_pages,
        )
    except (GitHubRequestError, ValueError):
        closing = {}
        closing_complete = False
    evidence_by_issue = discover_linked_issues(
        title=title,
        body=body,
        conversation_comments=all_conversation,
        review_comments=all_review,
        closing=closing,
    )
    evidence_source_ids = {
        evidence.source_id
        for evidence_group in evidence_by_issue.values()
        for evidence in evidence_group
    }

    def retained_sources(
        sources: tuple[TextSource, ...],
    ) -> tuple[TextSource, ...]:
        selected = {
            source.source_id: source
            for source in sources[:max_comments]
        }
        selected.update(
            {
                source.source_id: source
                for source in sources
                if source.source_id in evidence_source_ids
            }
        )
        return tuple(selected[key] for key in sorted(selected))

    conversation = retained_sources(all_conversation)
    review = retained_sources(all_review)
    linked = collect_linked_issue_refs(
        client,
        repository=repo,
        pr_number=number,
        evidence_by_issue=evidence_by_issue,
        repository_dir=repository_dir,
        config_sha=config_sha,
        max_comments=max_issue_comments,
        max_comment_chars=max_issue_comment_chars,
        attempted_at=utc_now(),
    )
    comment_texts = [source.text for source in all_conversation]

    tests_added = triage.extract_tests_added_v2(files)
    file_stats = [{"path": f.get("filename", ""), "status": f.get("status", ""),
                   "additions": f.get("additions", 0), "deletions": f.get("deletions", 0)}
                  for f in files]
    collected_at = utc_now()
    if files_expected_count_source == "git_base_head":
        insertions = sum(item.get("additions", 0) for item in files)
        deletions = sum(item.get("deletions", 0) for item in files)
    else:
        insertions = detail.get("additions", 0)
        deletions = detail.get("deletions", 0)
    record = PrCorpusV2(
        source_schema="pr_corpus_v2",
        repository=repo,
        pr_number=number,
        html_url=detail.get("html_url"),
        author=(detail.get("user") or {}).get("login"),
        merged=triage.is_merged(detail, labels, comment_texts),
        merged_at=detail.get("merged_at"),
        closed_at=detail.get("closed_at"),
        labels=tuple(labels),
        title=title,
        body=TextSource(
            source_id="body",
            text=body,
            char_end=len(body),
            truncation=TruncationStatus.COMPLETE,
        ),
        comments=tuple(sorted((*conversation, *review), key=lambda item: item.source_id)),
        linked_issues=linked,
        diff_stats={
            "files": file_stats,
            "insertions": insertions,
            "deletions": deletions,
        },
        touched_files_patch=triage.touched_file_patches(
            files,
            subsystem_rules=subsystem_rules,
            compile_stack=compile_stack,
        ),
        subsystem_labels=labels_by_subsystem,
        change_kind=triage.classify_change_kind(title, body, files),
        tests_added=tests_added,
        commit_refs=CommitRefs(
            merge_commit_sha=detail.get("merge_commit_sha"),
            head_sha=(detail.get("head") or {}).get("sha"),
            base_sha=(detail.get("base") or {}).get("sha"),
            base_ref=(detail.get("base") or {}).get("ref"),
        ),
        revert_evidence=triage.structured_revert_evidence(
            title,
            body,
            list(conversation_page.items),
            current_pr=number,
        ),
        reverted_by=None,
        info_richness=triage.info_richness(
            body,
            [test.to_record() for test in tests_added],
        ),
        collection=CollectionMetadata(
            config_sha=config_sha,
            collected_at=collected_at,
            files_expected_count=expected_files,
            files_expected_count_source=files_expected_count_source,
            files_github_reported_count=github_reported_files,
            files_retained_count=len(files),
            files_collection_truncated=(
                len(files) < expected_files or not files_fetch_complete
            ),
            files_fetch_complete=files_fetch_complete,
            conversation_comments_total_count=expected_comments,
            conversation_comments_retained_count=len(conversation),
            conversation_comments_collection_truncated=(
                len(conversation) < expected_comments
                or not conversation_page.complete
            ),
            conversation_comments_fetch_complete=conversation_page.complete,
            review_comments_total_count=expected_review_comments,
            review_comments_retained_count=len(review),
            review_comments_collection_truncated=(
                len(review) < expected_review_comments
                or not review_page.complete
            ),
            review_comments_fetch_complete=review_page.complete,
            closing_refs_collection_complete=closing_complete,
        ),
    )
    return record.to_record()


def _quote_hash(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def derive_change_landing(record: PrCorpusV2) -> ChangeLanding | None:
    merge_sha = record.commit_refs.merge_commit_sha
    if not record.merged or not merge_sha:
        return None
    if record.revert_evidence is not None:
        kind = LandingKind.REVERT
        reverts_pr = record.revert_evidence.target_pr
    elif "reland" in record.title.lower():
        kind = LandingKind.RELAND
        reverts_pr = None
    else:
        kind = LandingKind.MERGE
        reverts_pr = None
    quote = record.title or f"PR #{record.pr_number}"
    evidence = EvidenceSpec(
        artifact_ref=f"pr:{record.pr_number}",
        locator=f"title@chars 0-{len(record.title)}",
        quote=quote,
        quote_hash=_quote_hash(quote),
        strength=EvidenceStrength.EXPLICIT_CLAIM,
        supports_field="change_landing",
        extractor="s1.change_landing/v1",
    )
    return ChangeLanding(
        pr=record.pr_number,
        branch=record.commit_refs.base_ref or "unknown",
        landed_commits=(merge_sha,),
        landed_at=record.merged_at or record.closed_at or record.collection.collected_at,
        kind=kind,
        reverts_pr=reverts_pr,
        evidence=(evidence,),
        confidence=0.95,
    )


def append_landing(path: Path, landing: ChangeLanding) -> bool:
    key = (
        landing.pr,
        landing.branch,
        landing.landed_commits,
        landing.kind.value,
        landing.reverts_pr,
    )
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            current = ChangeLanding.from_record(json.loads(line))
            current_key = (
                current.pr,
                current.branch,
                current.landed_commits,
                current.kind.value,
                current.reverts_pr,
            )
            if current_key == key:
                return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(landing.to_record(), ensure_ascii=False) + "\n")
    return True


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=".tmp-",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def collection_ready_for_completion(record: PrCorpusV2) -> bool:
    collection = record.collection
    return all(
        (
            collection.files_fetch_complete,
            collection.conversation_comments_fetch_complete,
            collection.review_comments_fetch_complete,
            collection.closing_refs_collection_complete,
        )
    )


# --------------------------------------------------------------- revert links

def link_reverts(out_dir: Path) -> int:
    """Second pass: for every revert PR in the corpus, mark its target."""
    by_number: dict[int, tuple[Path, dict]] = {}
    candidates: dict[int, list[dict]] = {}
    for path in sorted(out_dir.glob("*.json")):
        try:
            rec = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        by_number[rec["pr_number"]] = (path, rec)
        if not rec.get("merged"):
            continue
        if rec.get("schema_version") == "pr_corpus_v2":
            evidence = rec.get("revert_evidence")
            target = evidence.get("target_pr") if evidence else None
        elif rec.get("revert_of"):
            target = rec["revert_of"]
        else:
            target = None
        if (
            isinstance(target, int)
            and target > 0
            and target != rec["pr_number"]
        ):
            candidates.setdefault(target, []).append(rec)

    desired = {
        target: min(
            records,
            key=lambda record: (
                record.get("merged_at")
                or record.get("closed_at")
                or (record.get("collection") or {}).get("collected_at")
                or "9999",
                record["pr_number"],
            ),
        )["pr_number"]
        for target, records in candidates.items()
    }
    linked = 0
    for target, (target_path, rec) in by_number.items():
        reverter = desired.get(target)
        if rec.get("reverted_by") != reverter:
            rec["reverted_by"] = reverter
            atomic_write_record(target_path, rec)
            linked += 1
    eligible_reverts = sum(len(records) for records in candidates.values())
    print(
        f"[s1] revert linking: {eligible_reverts} eligible revert PRs, "
        f"{linked} targets updated",
        flush=True,
    )
    return linked


def rebuild_landings(out_dir: Path, landings_path: Path) -> int:
    landings = []
    for path in sorted(out_dir.glob("*.json")):
        record = PrCorpusV2.from_record(json.loads(path.read_text()))
        landing = derive_change_landing(record)
        if landing is not None:
            landings.append(landing)
    payload = "".join(
        json.dumps(
            landing.to_record(),
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
        for landing in landings
    ).encode("utf-8")
    _atomic_write_bytes(landings_path, payload)
    return len(landings)


def repair_revert_metadata(
    out_dir: Path,
    landings_path: Path,
) -> tuple[int, int, int]:
    sanitized = 0
    for path in sorted(out_dir.glob("*.json")):
        record = json.loads(path.read_text())
        evidence = record.get("revert_evidence")
        replacement = evidence
        if evidence and evidence.get("kind") == "bot_comment":
            if (
                triage.explicit_bot_revert_target(
                    evidence.get("quote") or ""
                )
                != evidence.get("target_pr")
            ):
                replacement = None
        elif evidence and evidence.get("kind") == "title_pattern":
            target = triage.revert_of(
                record.get("title") or "",
                (record.get("body") or {}).get("text") or "",
                current_pr=record["pr_number"],
            )
            if target is None:
                replacement = None
            elif target != evidence.get("target_pr"):
                replacement = {
                    **evidence,
                    "target_pr": target,
                    "quote": record.get("title") or "",
                }
        if replacement != evidence:
            record["revert_evidence"] = replacement
            atomic_write_record(path, record)
            sanitized += 1

    linked = link_reverts(out_dir)
    backup_path = landings_path.with_name(
        "landings_before_revert_precision_fix.jsonl"
    )
    if landings_path.exists() and not backup_path.exists():
        _atomic_write_bytes(backup_path, landings_path.read_bytes())
    rebuilt = rebuild_landings(out_dir, landings_path)
    print(
        f"[s1] revert repair: sanitized={sanitized} "
        f"landings_rebuilt={rebuilt} backup={backup_path}",
        flush=True,
    )
    return sanitized, linked, rebuilt


# --------------------------------------------------------------------- driver

def load_state(out_dir: Path, name: str) -> set[int]:
    f = out_dir / name
    if not f.exists():
        return set()
    return {int(line.split()[0]) for line in f.read_text().splitlines() if line.strip()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="pytorch/pytorch")
    ap.add_argument("--since", help="closed-date window start, YYYY-MM-DD")
    ap.add_argument("--until", help="closed-date window end, YYYY-MM-DD")
    ap.add_argument("--max-prs", type=int, default=0, help="cap for pilot runs (0 = no cap)")
    ap.add_argument("--max-comments", type=int)
    ap.add_argument("--max-issue-comments", type=int)
    ap.add_argument("--max-issue-comment-chars", type=int)
    ap.add_argument("--max-file-pages", type=int)
    ap.add_argument("--max-link-pages", type=int)
    ap.add_argument("--request-timeout", type=float)
    ap.add_argument("--request-retries", type=int)
    ap.add_argument("--compile-stack-only", action="store_true",
                    help="keep only PRs touching subsystems listed in config compile_stack")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "s1_pytorch.json"))
    ap.add_argument("--out-root", default=str(PROJECT_ROOT / "data" / "pr_corpus"))
    ap.add_argument("--link-reverts-only", action="store_true",
                    help="only run the revert-linking pass over an existing corpus")
    ap.add_argument(
        "--repair-reverts-only",
        action="store_true",
        help="sanitize directional revert evidence and rebuild derived landings",
    )
    ap.add_argument(
        "--retry-issue-failures",
        action="store_true",
        help="retry transient linked-Issue failures without recollecting PRs",
    )
    args = ap.parse_args(argv)

    cfg = json.loads(Path(args.config).read_text())
    out_dir = Path(args.out_root) / args.repo.replace("/", "__")
    out_dir.mkdir(parents=True, exist_ok=True)
    landings_path = Path(args.out_root) / "landings.jsonl"
    issue_config = cfg.get("issue_collection", {})
    max_issue_comments = (
        args.max_issue_comments
        if args.max_issue_comments is not None
        else int(issue_config.get("max_comments", 30))
    )
    max_issue_comment_chars = (
        args.max_issue_comment_chars
        if args.max_issue_comment_chars is not None
        else int(issue_config.get("max_comment_chars", 20000))
    )
    max_file_pages = (
        args.max_file_pages
        if args.max_file_pages is not None
        else int(cfg.get("max_file_pages", 30))
    )
    max_link_pages = (
        args.max_link_pages
        if args.max_link_pages is not None
        else int(issue_config.get("max_link_pages", 30))
    )
    max_comments = (
        args.max_comments
        if args.max_comments is not None
        else int(cfg.get("max_pr_comments", 30))
    )
    request_timeout = (
        args.request_timeout
        if args.request_timeout is not None
        else float(cfg.get("request_timeout_seconds", 30))
    )
    if request_timeout <= 0:
        ap.error("request timeout must be positive")
    request_retries = (
        args.request_retries
        if args.request_retries is not None
        else int(cfg.get("request_retries", 3))
    )
    if request_retries < 0:
        ap.error("request retries cannot be negative")
    try:
        _validate_collection_limits(
            max_comments=max_comments,
            max_issue_comments=max_issue_comments,
            max_issue_comment_chars=max_issue_comment_chars,
            max_file_pages=max_file_pages,
            max_link_pages=max_link_pages,
        )
    except ValueError as error:
        ap.error(str(error))
    effective_config = dict(cfg)
    effective_config["max_pr_comments"] = max_comments
    effective_config["max_file_pages"] = max_file_pages
    effective_config["request_timeout_seconds"] = request_timeout
    effective_config["request_retries"] = request_retries
    effective_config["issue_collection"] = {
        **issue_config,
        "max_comments": max_issue_comments,
        "max_comment_chars": max_issue_comment_chars,
        "max_link_pages": max_link_pages,
    }
    config_sha = sha256_digest(effective_config)

    if args.link_reverts_only:
        link_reverts(out_dir)
        return 0
    if args.repair_reverts_only:
        repair_revert_metadata(out_dir, landings_path)
        return 0

    client = GitHubClient(
        request_timeout=request_timeout,
        request_retries=request_retries,
    )
    if args.retry_issue_failures:
        retried, resolved = retry_transient_issue_failures(
            client,
            repository=args.repo,
            repository_dir=out_dir,
            config_sha=config_sha,
            max_comments=max_issue_comments,
            max_comment_chars=max_issue_comment_chars,
        )
        print(
            f"[s1] issue retries: attempted={retried} resolved={resolved}",
            flush=True,
        )
        return 0

    if not args.since or not args.until:
        ap.error(
            "--since and --until are required unless a maintenance mode is used"
        )

    if not client.token:
        print("[s1] no GITHUB_TOKEN set: running unauthenticated (low rate limits)", flush=True)

    numbers = search_pr_numbers(client, args.repo, args.since, args.until)
    completed = load_state(out_dir, "completed.txt")
    todo = [n for n in numbers if n not in completed]
    if args.max_prs:
        todo = todo[: args.max_prs]
    print(f"[s1] {len(numbers)} PRs in window, {len(todo)} to process "
          f"({len(completed)} already done)", flush=True)

    rules = cfg["subsystem_rules"]
    compile_stack = set(cfg.get("compile_stack", []))
    kept = skipped = failed = 0
    for i, number in enumerate(todo, 1):
        try:
            should_mark_complete = True
            rec = fetch_pr_record(
                client,
                args.repo,
                number,
                rules,
                compile_stack,
                max_comments,
                max_issue_comments,
                max_issue_comment_chars,
                max_file_pages,
                max_link_pages,
                config_sha,
                out_dir,
                compile_stack_only=args.compile_stack_only,
            )
            if rec is None:
                skipped += 1
            else:
                atomic_write_record(out_dir / f"{number}.json", rec)
                parsed = load_pr_record(
                    rec,
                    config_sha=config_sha,
                    repository=args.repo,
                )
                landing = derive_change_landing(
                    parsed
                )
                if landing is not None:
                    append_landing(landings_path, landing)
                if collection_ready_for_completion(parsed):
                    kept += 1
                else:
                    failed += 1
                    should_mark_complete = False
                    with (out_dir / "failed.txt").open("a") as f:
                        f.write(
                            f"{number} IncompleteCollection: "
                            "one or more discovery endpoints were not exhausted\n"
                        )
            if should_mark_complete:
                with (out_dir / "completed.txt").open("a") as f:
                    f.write(f"{number}\n")
        except Exception as e:  # noqa: BLE001 - record and continue
            failed += 1
            with (out_dir / "failed.txt").open("a") as f:
                f.write(f"{number} {type(e).__name__}: {e}\n")
        if i % 20 == 0 or i == len(todo):
            print(f"[s1] progress {i}/{len(todo)} kept={kept} "
                  f"skipped={skipped} failed={failed}", flush=True)

    link_reverts(out_dir)
    print(f"[s1] done. corpus dir: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
