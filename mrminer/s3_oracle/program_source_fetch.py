"""Fetch exact PR head objects without modifying the source worktree."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from mrminer.s3_oracle.artifact_resolution import GitFetchOperation


class ProgramSourceFetchStatus(str, Enum):
    ALREADY_PRESENT = "already_present"
    FETCHED = "fetched"
    FETCHED_REF_MISMATCH = "fetched_ref_mismatch_expected_present"
    REF_NOT_FOUND = "ref_not_found"
    SHA_MISMATCH = "sha_mismatch"
    TIMEOUT = "timeout"
    NATIVE_SIGNAL = "native_signal"
    FETCH_FAILED = "fetch_failed"


class FetchGit(Protocol):
    def commit_exists(self, commit: str) -> bool | None:
        """Return exact object presence."""

    def fetch_ref(
        self,
        *,
        remote: str,
        remote_ref: str,
        local_ref: str,
    ) -> GitFetchOperation:
        """Fetch one remote ref to a namespaced local ref."""

    def resolve_ref(self, ref: str) -> str | None:
        """Resolve a local ref to its commit."""


@dataclass(frozen=True)
class ProgramSourceFetchResult:
    pr_number: int
    expected_head_sha: str
    remote: str
    remote_ref: str
    local_ref: str
    status: ProgramSourceFetchStatus
    observed_head_sha: str | None
    expected_commit_present: bool | None
    fetch: GitFetchOperation | None
    reason: str

    @property
    def source_available(self) -> bool:
        return self.expected_commit_present is True

    def to_record(self) -> dict:
        return {
            "fetch_version": "program_source_fetch_v1",
            "pr_number": self.pr_number,
            "expected_head_sha": self.expected_head_sha,
            "remote": self.remote,
            "remote_ref": self.remote_ref,
            "local_ref": self.local_ref,
            "status": self.status.value,
            "observed_head_sha": self.observed_head_sha,
            "expected_commit_present": self.expected_commit_present,
            "source_available": self.source_available,
            "fetch": self.fetch.to_record() if self.fetch is not None else None,
            "reason": self.reason,
        }


class ProgramSourceFetcher:
    def __init__(self, git: FetchGit, *, remote: str = "origin"):
        self.git = git
        self.remote = remote

    def fetch(
        self,
        *,
        pr_number: int,
        expected_head_sha: str,
    ) -> ProgramSourceFetchResult:
        if pr_number <= 0:
            raise ValueError("pr_number must be positive")
        remote_ref = f"refs/pull/{pr_number}/head"
        local_ref = f"refs/mrminer/pr/{pr_number}/head"
        before = self.git.commit_exists(expected_head_sha)
        if before is True:
            return ProgramSourceFetchResult(
                pr_number=pr_number,
                expected_head_sha=expected_head_sha,
                remote=self.remote,
                remote_ref=remote_ref,
                local_ref=local_ref,
                status=ProgramSourceFetchStatus.ALREADY_PRESENT,
                observed_head_sha=None,
                expected_commit_present=True,
                fetch=None,
                reason="expected PR head commit is already present",
            )
        operation = self.git.fetch_ref(
            remote=self.remote,
            remote_ref=remote_ref,
            local_ref=local_ref,
        )
        if operation.timed_out:
            status = ProgramSourceFetchStatus.TIMEOUT
            reason = "git fetch exceeded the configured hard timeout"
        elif operation.native_signal is not None:
            status = ProgramSourceFetchStatus.NATIVE_SIGNAL
            reason = f"git fetch terminated by {operation.native_signal}"
        elif not operation.succeeded:
            not_found = "couldn't find remote ref" in operation.stderr.lower()
            status = (
                ProgramSourceFetchStatus.REF_NOT_FOUND
                if not_found
                else ProgramSourceFetchStatus.FETCH_FAILED
            )
            reason = (
                "GitHub pull head ref was not found"
                if not_found
                else f"git fetch failed with return code {operation.returncode}"
            )
        else:
            observed = self.git.resolve_ref(local_ref)
            after = self.git.commit_exists(expected_head_sha)
            if observed == expected_head_sha:
                return ProgramSourceFetchResult(
                    pr_number=pr_number,
                    expected_head_sha=expected_head_sha,
                    remote=self.remote,
                    remote_ref=remote_ref,
                    local_ref=local_ref,
                    status=ProgramSourceFetchStatus.FETCHED,
                    observed_head_sha=observed,
                    expected_commit_present=after,
                    fetch=operation,
                    reason="fetched PR head matches the S1 commit reference",
                )
            if after is True:
                return ProgramSourceFetchResult(
                    pr_number=pr_number,
                    expected_head_sha=expected_head_sha,
                    remote=self.remote,
                    remote_ref=remote_ref,
                    local_ref=local_ref,
                    status=ProgramSourceFetchStatus.FETCHED_REF_MISMATCH,
                    observed_head_sha=observed,
                    expected_commit_present=True,
                    fetch=operation,
                    reason=(
                        "current pull ref differs, but the recorded head was "
                        "included in the fetched object graph"
                    ),
                )
            return ProgramSourceFetchResult(
                pr_number=pr_number,
                expected_head_sha=expected_head_sha,
                remote=self.remote,
                remote_ref=remote_ref,
                local_ref=local_ref,
                status=ProgramSourceFetchStatus.SHA_MISMATCH,
                observed_head_sha=observed,
                expected_commit_present=after,
                fetch=operation,
                reason=(
                    "fetched pull ref does not match the recorded S1 head SHA"
                ),
            )
        return ProgramSourceFetchResult(
            pr_number=pr_number,
            expected_head_sha=expected_head_sha,
            remote=self.remote,
            remote_ref=remote_ref,
            local_ref=local_ref,
            status=status,
            observed_head_sha=None,
            expected_commit_present=self.git.commit_exists(expected_head_sha),
            fetch=operation,
            reason=reason,
        )
