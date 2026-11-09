"""Rebuild ChangeLanding facts from canonical first-parent Git history."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from pathlib import Path

from mrminer.common.schema import (
    ChangeLanding,
    EvidenceSpec,
    EvidenceStrength,
    LandingKind,
)
from mrminer.s1_miner.issue_collector import atomic_write_record
from mrminer.s1_miner.pr_schema import PrCorpusV2


_PR_SUFFIX = re.compile(r"\(#(?P<number>[1-9][0-9]*)\)$")


class CanonicalLandingStatus(str, Enum):
    CANONICAL_PR_SUFFIX = "canonical_pr_suffix"
    ORIGINAL_SHA_ANCESTOR = "original_sha_ancestor"
    GITHUB_NATIVE_MERGE = "github_native_merge"
    UNRESOLVED = "unresolved"
    NOT_MERGED = "not_merged"


@dataclass(frozen=True)
class CanonicalCommit:
    branch: str
    commit: str
    committed_at: str
    subject: str
    pr_number: int | None

    def to_record(self) -> dict:
        return {
            "branch": self.branch,
            "commit": self.commit,
            "committed_at": self.committed_at,
            "subject": self.subject,
            "pr_number": self.pr_number,
        }


@dataclass(frozen=True)
class BranchHistory:
    branch: str
    git_ref: str
    commits: tuple[CanonicalCommit, ...]

    @cached_property
    def by_sha(self) -> dict[str, CanonicalCommit]:
        return {item.commit: item for item in self.commits}

    @cached_property
    def by_pr(self) -> dict[int, tuple[CanonicalCommit, ...]]:
        grouped: dict[int, list[CanonicalCommit]] = {}
        for item in self.commits:
            if item.pr_number is not None:
                grouped.setdefault(item.pr_number, []).append(item)
        return {
            number: tuple(items)
            for number, items in grouped.items()
        }


@dataclass(frozen=True)
class CanonicalLandingResolution:
    pr_number: int
    status: CanonicalLandingStatus
    matches: tuple[CanonicalCommit, ...]
    title_mismatch_count: int
    reason: str

    def to_record(self) -> dict:
        return {
            "pr_number": self.pr_number,
            "status": self.status.value,
            "matches": [item.to_record() for item in self.matches],
            "title_mismatch_count": self.title_mismatch_count,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class _GitResult:
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool
    native_signal: str | None


def _signal_name(returncode: int | None) -> str | None:
    if returncode is None or returncode >= 0:
        return None
    try:
        return signal.Signals(-returncode).name
    except ValueError:
        return f"SIG{-returncode}"


class CanonicalGitHistoryReader:
    def __init__(
        self,
        repository: Path,
        *,
        timeout_s: float = 300.0,
        termination_grace_s: float = 0.5,
    ):
        repository = repository.resolve()
        if timeout_s <= 0 or termination_grace_s <= 0:
            raise ValueError("Git timeout values must be positive")
        if not (repository / ".git").exists():
            raise ValueError(f"not a Git worktree: {repository}")
        self.repository = repository
        self.timeout_s = timeout_s
        self.termination_grace_s = termination_grace_s

    def _terminate(self, process: subprocess.Popen[str]) -> None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=self.termination_grace_s)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait()

    def _run(self, arguments: list[str]) -> _GitResult:
        try:
            process = subprocess.Popen(
                ["git", "-C", str(self.repository), *arguments],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                start_new_session=True,
            )
        except OSError as error:
            return _GitResult(
                returncode=None,
                stdout="",
                stderr=f"{type(error).__name__}: {error}",
                timed_out=False,
                native_signal=None,
            )
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=self.timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            self._terminate(process)
            stdout, stderr = process.communicate()
        return _GitResult(
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            native_signal=_signal_name(process.returncode),
        )

    def resolve_branch(self, branch: str) -> str | None:
        candidates = (branch, f"origin/{branch}")
        for candidate in candidates:
            result = self._run(
                ["rev-parse", "--verify", f"{candidate}^{{commit}}"]
            )
            if result.returncode == 0 and not result.timed_out:
                return candidate
        return None

    def scan(
        self,
        branch: str,
        git_ref: str,
        *,
        exclude_ref: str | None = None,
    ) -> BranchHistory:
        arguments = [
            "log",
            "--first-parent",
            "--format=%H%x00%cI%x00%s%x00",
            git_ref,
        ]
        if exclude_ref is not None:
            arguments.extend(["--not", exclude_ref])
        result = self._run(
            arguments
        )
        if result.timed_out:
            raise TimeoutError(
                f"Git history scan for {git_ref} exceeded {self.timeout_s:g}s"
            )
        if result.native_signal is not None:
            raise RuntimeError(
                f"Git history scan for {git_ref} ended by "
                f"{result.native_signal}"
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"Git history scan for {git_ref} failed with "
                f"{result.returncode}: {result.stderr[-1000:]}"
            )
        fields = result.stdout.split("\0")
        while fields and not fields[-1].strip():
            fields.pop()
        if len(fields) % 3:
            raise ValueError(
                f"Git history scan for {git_ref} returned malformed records"
            )
        commits = []
        for index in range(0, len(fields), 3):
            commit = fields[index].strip()
            committed_at = fields[index + 1].strip()
            subject = fields[index + 2].strip()
            match = _PR_SUFFIX.search(subject)
            commits.append(
                CanonicalCommit(
                    branch=branch,
                    commit=commit,
                    committed_at=committed_at,
                    subject=subject,
                    pr_number=(
                        int(match.group("number"))
                        if match is not None
                        else None
                    ),
                )
            )
        return BranchHistory(
            branch=branch,
            git_ref=git_ref,
            commits=tuple(commits),
        )


def _normalized_title(value: str) -> str:
    return " ".join(value.split()).casefold()


def _subject_title(subject: str) -> str:
    return _PR_SUFFIX.sub("", subject).strip()


def _landing_kind(record: PrCorpusV2) -> tuple[LandingKind, int | None]:
    if record.revert_evidence is not None:
        return LandingKind.REVERT, record.revert_evidence.target_pr
    if "reland" in record.title.casefold():
        return LandingKind.RELAND, None
    return LandingKind.MERGE, None


def _evidence(commit: CanonicalCommit) -> EvidenceSpec:
    quote = commit.subject
    return EvidenceSpec(
        artifact_ref=f"git:{commit.commit}",
        locator=f"commit_subject@chars 0-{len(quote)}",
        quote=quote,
        quote_hash=(
            f"sha256:{hashlib.sha256(quote.encode('utf-8')).hexdigest()}"
        ),
        strength=EvidenceStrength.EXPLICIT_CLAIM,
        supports_field="change_landing.landed_commits",
        extractor="s1.canonical_git_landing/v1",
    )


def _native_merge_evidence(
    record: PrCorpusV2,
    commit: CanonicalCommit,
) -> EvidenceSpec:
    quote = record.title or f"PR #{record.pr_number}"
    return EvidenceSpec(
        artifact_ref=f"pr:{record.pr_number}",
        locator=f"title@chars 0-{len(record.title)}",
        quote=quote,
        quote_hash=(
            f"sha256:{hashlib.sha256(quote.encode('utf-8')).hexdigest()}"
        ),
        strength=EvidenceStrength.EXPLICIT_CLAIM,
        supports_field="change_landing.landed_commits",
        extractor="s1.github_native_merge/v1",
    )


def resolve_record(
    record: PrCorpusV2,
    histories: tuple[BranchHistory, ...],
) -> tuple[tuple[ChangeLanding, ...], CanonicalLandingResolution]:
    if not record.merged:
        return (), CanonicalLandingResolution(
            pr_number=record.pr_number,
            status=CanonicalLandingStatus.NOT_MERGED,
            matches=(),
            title_mismatch_count=0,
            reason="PR is not marked merged in the S1 corpus",
        )
    by_pr_matches = [
        commit
        for history in histories
        for commit in history.by_pr.get(record.pr_number, ())
    ]
    status = CanonicalLandingStatus.CANONICAL_PR_SUFFIX
    reason = "canonical first-parent commit subject ends with the PR number"
    matches = by_pr_matches
    if not matches and record.commit_refs.merge_commit_sha is not None:
        matches = [
            commit
            for history in histories
            if (
                commit := history.by_sha.get(
                    record.commit_refs.merge_commit_sha
                )
            )
            is not None
        ]
        if matches:
            status = CanonicalLandingStatus.ORIGINAL_SHA_ANCESTOR
            reason = (
                "original merge_commit_sha is a canonical first-parent commit"
            )
    if (
        not matches
        and record.merged_at is not None
        and record.commit_refs.merge_commit_sha is not None
    ):
        matches = [
            CanonicalCommit(
                branch=record.commit_refs.base_ref or "unknown",
                commit=record.commit_refs.merge_commit_sha,
                committed_at=record.merged_at,
                subject=record.title,
                pr_number=record.pr_number,
            )
        ]
        status = CanonicalLandingStatus.GITHUB_NATIVE_MERGE
        reason = (
            "GitHub reports an exact native merge time, commit, and base branch"
        )
    if not matches:
        return (), CanonicalLandingResolution(
            pr_number=record.pr_number,
            status=CanonicalLandingStatus.UNRESOLVED,
            matches=(),
            title_mismatch_count=0,
            reason=(
                "no canonical (#PR) commit or first-parent merge SHA was found"
            ),
        )
    title_mismatches = sum(
        _normalized_title(_subject_title(commit.subject))
        != _normalized_title(record.title)
        for commit in matches
    )
    initial_kind, reverts_pr = _landing_kind(record)
    ordered_matches = sorted(
        matches,
        key=lambda item: (
            item.branch,
            item.committed_at,
            item.commit,
        ),
    )
    branch_positions: Counter[str] = Counter()
    landings = []
    for commit in ordered_matches:
        position = branch_positions[commit.branch]
        branch_positions[commit.branch] += 1
        kind = (
            LandingKind.RELAND
            if position > 0 and initial_kind is not LandingKind.REVERT
            else initial_kind
        )
        landings.append(
            ChangeLanding(
                pr=record.pr_number,
                branch=commit.branch,
                landed_commits=(commit.commit,),
                landed_at=commit.committed_at,
                kind=kind,
                reverts_pr=reverts_pr,
                evidence=(
                    (
                        _native_merge_evidence(record, commit)
                        if status is CanonicalLandingStatus.GITHUB_NATIVE_MERGE
                        else _evidence(commit)
                    ),
                ),
                confidence=1.0,
            )
        )
    return tuple(landings), CanonicalLandingResolution(
        pr_number=record.pr_number,
        status=status,
        matches=tuple(matches),
        title_mismatch_count=title_mismatches,
        reason=reason,
    )


def _atomic_write(path: Path, payload: bytes) -> None:
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


def _load_records(repository_dir: Path) -> tuple[PrCorpusV2, ...]:
    return tuple(
        PrCorpusV2.from_record(json.loads(path.read_bytes()))
        for path in sorted(repository_dir.glob("*.json"))
    )


def _requested_branches(records: tuple[PrCorpusV2, ...]) -> tuple[str, ...]:
    release = {
        branch
        for record in records
        if (
            (branch := record.commit_refs.base_ref) is not None
            and branch.startswith("release/")
        )
    }
    return ("main", *sorted(release))


def rebuild_canonical_landings(
    *,
    corpus_root: Path,
    repository: str,
    source_root: Path,
    output_report: Path,
    apply: bool,
    git_timeout_s: float,
) -> dict:
    repository_dir = corpus_root / repository.replace("/", "__")
    if not repository_dir.is_dir():
        raise ValueError(f"corpus repository directory is absent: {repository_dir}")
    records = _load_records(repository_dir)
    reader = CanonicalGitHistoryReader(
        source_root,
        timeout_s=git_timeout_s,
    )
    histories = []
    unavailable_branches = []
    main_ref = reader.resolve_branch("main")
    if main_ref is None:
        raise RuntimeError("canonical main branch is unavailable")
    for branch in _requested_branches(records):
        git_ref = reader.resolve_branch(branch)
        if git_ref is None:
            unavailable_branches.append(branch)
            continue
        histories.append(
            reader.scan(
                branch,
                git_ref,
                exclude_ref=main_ref if branch != "main" else None,
            )
        )

    landings: list[ChangeLanding] = []
    resolutions = []
    for record in records:
        resolved, resolution = resolve_record(record, tuple(histories))
        landings.extend(resolved)
        resolutions.append(resolution)
    landings.sort(
        key=lambda item: (
            item.branch,
            item.landed_at,
            item.pr,
            item.landed_commits,
        )
    )
    status_counts = Counter(item.status.value for item in resolutions)
    landing_path = corpus_root / "landings.jsonl"
    old_count = (
        sum(bool(line.strip()) for line in landing_path.read_text().splitlines())
        if landing_path.exists()
        else 0
    )
    payload = "".join(
        json.dumps(
            landing.to_record(),
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
        for landing in landings
    ).encode("utf-8")
    backup_path = corpus_root / "landings_before_canonical_git_fix.jsonl"
    if apply:
        if landing_path.exists() and not backup_path.exists():
            shutil.copy2(landing_path, backup_path)
        _atomic_write(landing_path, payload)
    report = {
        "report_version": "canonical_landing_repair_v1",
        "corpus_root": str(corpus_root.resolve()),
        "repository": repository,
        "source_root": str(source_root.resolve()),
        "applied": apply,
        "record_count": len(records),
        "merged_record_count": sum(record.merged for record in records),
        "old_landing_count": old_count,
        "new_landing_count": len(landings),
        "unique_landed_pr_count": len({item.pr for item in landings}),
        "status_counts": dict(sorted(status_counts.items())),
        "title_mismatch_count": sum(
            item.title_mismatch_count for item in resolutions
        ),
        "requested_branches": list(_requested_branches(records)),
        "scanned_branches": [
            {
                "branch": history.branch,
                "git_ref": history.git_ref,
                "first_parent_commit_count": len(history.commits),
                "pr_suffix_commit_count": sum(
                    item.pr_number is not None for item in history.commits
                ),
            }
            for history in histories
        ],
        "unavailable_branches": unavailable_branches,
        "landing_path": str(landing_path),
        "backup_path": str(backup_path) if apply else None,
        "landings_sha256": (
            f"sha256:{hashlib.sha256(payload).hexdigest()}"
        ),
        "resolutions": [item.to_record() for item in resolutions],
    }
    atomic_write_record(output_report, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mrminer-canonical-landings")
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--repository", default="pytorch/pytorch")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git-timeout", type=float, default=300.0)
    parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = rebuild_canonical_landings(
        corpus_root=args.corpus_root,
        repository=args.repository,
        source_root=args.source_root,
        output_report=args.output,
        apply=args.apply,
        git_timeout_s=args.git_timeout,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "applied": report["applied"],
                "new_landing_count": report["new_landing_count"],
                "status": "ok",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
