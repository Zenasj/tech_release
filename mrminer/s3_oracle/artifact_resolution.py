"""Conservative resolution of PR landings to immutable build artifacts."""

from __future__ import annotations

import os
import re
import signal
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Protocol

from mrminer.common.schema import (
    ChangeLanding,
    LandingKind,
    LandingState,
    VersionArtifact,
)


class ArtifactResolutionStatus(str, Enum):
    RESOLVED = "resolved"
    NO_LANDING = "no_landing"
    NO_ARTIFACT_CATALOG = "no_artifact_catalog"
    NO_COMPATIBLE_BRANCH = "no_compatible_branch"
    NO_COMPLETE_ARTIFACT = "no_complete_artifact"
    ANCESTRY_UNAVAILABLE = "ancestry_unavailable"
    AMBIGUOUS_ARTIFACT = "ambiguous_artifact"
    INVALID_ARTIFACT_CATALOG = "invalid_artifact_catalog"


class ContainmentState(str, Enum):
    ABSENT = LandingState.ABSENT.value
    PARTIAL = LandingState.PARTIAL.value
    COMPLETE = LandingState.COMPLETE.value
    ANCESTRY_UNAVAILABLE = "ancestry_unavailable"
    INCOMPARABLE_BRANCH = "incomparable_branch"


class ChangeActivityStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE_REVERTED = "inactive_reverted"
    INACTIVE_NOT_YET_LANDED = "inactive_not_yet_landed"
    ANCESTRY_UNAVAILABLE = "ancestry_unavailable"
    AMBIGUOUS_HISTORY = "ambiguous_history"
    ARTIFACT_UNKNOWN = "artifact_unknown"
    INVALID_ARTIFACT = "invalid_artifact"


class AncestryOracle(Protocol):
    def commit_exists(self, commit: str) -> bool | None:
        """Return None when the repository query itself was unavailable."""

    def is_ancestor(
        self,
        branch: str,
        older: str,
        newer: str,
    ) -> bool | None:
        """Return None when either commit or ancestry cannot be queried."""


@dataclass(frozen=True)
class GitCommandAudit:
    operation: str
    returncode: int | None
    timed_out: bool
    native_signal: str | None
    stderr: str

    def to_record(self) -> dict:
        return {
            "operation": self.operation,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "native_signal": self.native_signal,
            "stderr": self.stderr,
        }


class GitBlobStatus(str, Enum):
    OK = "ok"
    COMMIT_ABSENT = "commit_absent"
    COMMIT_UNAVAILABLE = "commit_unavailable"
    PATH_ABSENT = "path_absent"
    QUERY_UNAVAILABLE = "query_unavailable"


@dataclass(frozen=True)
class GitBlobRead:
    status: GitBlobStatus
    text: str | None
    detail: str


@dataclass(frozen=True)
class GitFetchOperation:
    returncode: int | None
    timed_out: bool
    native_signal: str | None
    stderr: str

    @property
    def succeeded(self) -> bool:
        return (
            self.returncode == 0
            and not self.timed_out
            and self.native_signal is None
        )

    def to_record(self) -> dict:
        return {
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "native_signal": self.native_signal,
            "stderr": self.stderr,
        }


@dataclass(frozen=True)
class _CommandResult:
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


class SubprocessGitAncestry:
    """Git ancestry adapter with process-group timeout and signal auditing."""

    def __init__(
        self,
        repository: Path,
        *,
        timeout_s: float = 10.0,
        termination_grace_s: float = 0.5,
        git_executable: str = "git",
        environment: Mapping[str, str] | None = None,
    ):
        if timeout_s <= 0 or termination_grace_s <= 0:
            raise ValueError("git timeout values must be positive")
        repository = repository.resolve()
        is_worktree = (repository / ".git").exists()
        is_bare = (
            (repository / "HEAD").is_file()
            and (repository / "objects").is_dir()
        )
        if not is_worktree and not is_bare:
            raise ValueError(f"not a Git worktree or bare repository: {repository}")
        self.repository = repository
        self.timeout_s = timeout_s
        self.termination_grace_s = termination_grace_s
        self.git_executable = git_executable
        self.environment = (
            {**os.environ, **environment}
            if environment is not None
            else None
        )
        self._commit_cache: dict[str, bool | None] = {}
        self._ancestor_cache: dict[tuple[str, str], bool | None] = {}
        self._blob_cache: dict[tuple[str, str], GitBlobRead] = {}
        self._audits: list[GitCommandAudit] = []

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
        try:
            process.wait(timeout=self.termination_grace_s)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    def _run(
        self,
        operation: str,
        arguments: list[str],
        *,
        stdin: str | None = None,
    ) -> _CommandResult:
        try:
            process = subprocess.Popen(
                [self.git_executable, "-C", str(self.repository), *arguments],
                stdin=subprocess.PIPE if stdin is not None else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                start_new_session=True,
                env=self.environment,
            )
        except OSError as error:
            self._audits.append(
                GitCommandAudit(
                    operation=operation,
                    returncode=None,
                    timed_out=False,
                    native_signal=None,
                    stderr=f"{type(error).__name__}: {error}",
                )
            )
            return _CommandResult(
                returncode=None,
                stdout="",
                stderr=str(error),
                timed_out=False,
                native_signal=None,
            )
        timed_out = False
        try:
            stdout, stderr = process.communicate(
                input=stdin,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired:
            timed_out = True
            self._terminate(process)
            stdout, stderr = process.communicate()
        native_signal = _signal_name(process.returncode)
        result = _CommandResult(
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            native_signal=native_signal,
        )
        if timed_out or native_signal is not None or process.returncode not in (0, 1):
            self._audits.append(
                GitCommandAudit(
                    operation=operation,
                    returncode=process.returncode,
                    timed_out=timed_out,
                    native_signal=native_signal,
                    stderr=stderr[-2_000:],
                )
            )
        return result

    def prefetch_commits(self, commits: set[str]) -> None:
        pending = sorted(set(commits) - self._commit_cache.keys())
        if not pending:
            return
        result = self._run(
            "cat-file --batch-check",
            ["cat-file", "--batch-check=%(objectname) %(objecttype)"],
            stdin="".join(f"{commit}\n" for commit in pending),
        )
        if result.timed_out or result.native_signal is not None or result.returncode != 0:
            self._commit_cache.update({commit: None for commit in pending})
            return
        lines = result.stdout.splitlines()
        if len(lines) != len(pending):
            self._audits.append(
                GitCommandAudit(
                    operation="cat-file --batch-check parse",
                    returncode=result.returncode,
                    timed_out=False,
                    native_signal=None,
                    stderr=(
                        f"expected {len(pending)} lines, received {len(lines)}"
                    ),
                )
            )
            self._commit_cache.update({commit: None for commit in pending})
            return
        for commit, line in zip(pending, lines, strict=True):
            fields = line.split()
            self._commit_cache[commit] = (
                len(fields) == 2 and fields[1] == "commit"
            )

    def commit_exists(self, commit: str) -> bool | None:
        self.prefetch_commits({commit})
        return self._commit_cache[commit]

    def is_ancestor(
        self,
        branch: str,
        older: str,
        newer: str,
    ) -> bool | None:
        del branch  # Git commit ancestry is branch-name independent.
        if older == newer:
            return self.commit_exists(older)
        key = (older, newer)
        if key in self._ancestor_cache:
            return self._ancestor_cache[key]
        self.prefetch_commits({older, newer})
        if (
            self._commit_cache[older] is not True
            or self._commit_cache[newer] is not True
        ):
            self._ancestor_cache[key] = None
            return None
        result = self._run(
            "merge-base --is-ancestor",
            ["merge-base", "--is-ancestor", older, newer],
        )
        value = (
            result.returncode == 0
            if not result.timed_out
            and result.native_signal is None
            and result.returncode in (0, 1)
            else None
        )
        self._ancestor_cache[key] = value
        return value

    def is_between(
        self,
        branch: str,
        low_exclusive: str,
        candidate: str,
        high_inclusive: str,
    ) -> bool | None:
        """Return whether ``low_exclusive < candidate <= high_inclusive`` in Git
        ancestry, or None when any referenced commit is unavailable.

        Satisfies the ``s2_graph.graph_store.CommitAncestry`` protocol so a real
        Git mirror can back ``EntityGraph.prs_between``.
        """

        lower = self.is_ancestor(branch, low_exclusive, candidate)
        if lower is None:
            return None
        upper = self.is_ancestor(branch, candidate, high_inclusive)
        if upper is None:
            return None
        strict_lower = lower and low_exclusive != candidate
        return strict_lower and upper

    def diagnostics(self) -> tuple[GitCommandAudit, ...]:
        return tuple(self._audits)

    @staticmethod
    def _validate_ref(ref: str) -> None:
        if (
            re.fullmatch(r"refs/[A-Za-z0-9._/-]+", ref) is None
            or ".." in ref
            or "//" in ref
            or "@{" in ref
            or ref.endswith(("/", ".", ".lock"))
        ):
            raise ValueError(f"invalid Git ref: {ref!r}")

    @staticmethod
    def _validate_branch(branch: str) -> None:
        if (
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", branch) is None
            or ".." in branch
            or "//" in branch
            or "@{" in branch
            or branch.endswith(("/", ".", ".lock"))
        ):
            raise ValueError(f"invalid Git branch: {branch!r}")

    def fetch_ref(
        self,
        *,
        remote: str,
        remote_ref: str,
        local_ref: str,
    ) -> GitFetchOperation:
        if not remote or remote.startswith("-") or any(
            character.isspace() or ord(character) < 32
            for character in remote
        ):
            raise ValueError("remote must be a non-option argument without whitespace")
        self._validate_ref(remote_ref)
        self._validate_ref(local_ref)
        result = self._run(
            "fetch ref",
            [
                "fetch",
                "--no-tags",
                "--",
                remote,
                f"+{remote_ref}:{local_ref}",
            ],
        )
        if result.returncode == 0:
            self._commit_cache = {
                commit: present
                for commit, present in self._commit_cache.items()
                if present is True
            }
            self._ancestor_cache.clear()
        return GitFetchOperation(
            returncode=result.returncode,
            timed_out=result.timed_out,
            native_signal=result.native_signal,
            stderr=result.stderr[-2_000:],
        )

    def resolve_ref(self, ref: str) -> str | None:
        self._validate_ref(ref)
        presence = self._run(
            "show-ref presence",
            ["show-ref", "--verify", "--quiet", ref],
        )
        if presence.returncode != 0 or presence.timed_out:
            return None
        result = self._run(
            "show-ref",
            ["show-ref", "--verify", "--hash", ref],
        )
        if result.returncode != 0 or result.timed_out:
            return None
        commit = result.stdout.strip()
        if re.fullmatch(r"[0-9a-f]{40,64}", commit) is None:
            return None
        return commit

    def resolve_branch_tip(self, branch: str) -> str | None:
        self._validate_branch(branch)
        for ref in (
            f"refs/heads/{branch}",
            f"refs/remotes/origin/{branch}",
        ):
            commit = self.resolve_ref(ref)
            if commit is not None:
                return commit
        return None

    def merge_base(self, commit_a: str, commit_b: str) -> str | None:
        """Return the best common ancestor of two commits, or None fail-closed.

        Both arguments must be already-resolved hex commit shas (resolve refs to
        commits via ``resolve_branch_tip``/``resolve_ref`` first) so no ref is
        interpolated into the command.
        """

        for commit in (commit_a, commit_b):
            if re.fullmatch(r"[0-9a-f]{7,64}", commit) is None:
                raise ValueError(f"merge_base requires hex commit shas: {commit!r}")
        result = self._run("merge-base", ["merge-base", commit_a, commit_b])
        if (
            result.timed_out
            or result.native_signal is not None
            or result.returncode != 0
        ):
            return None
        base = result.stdout.strip()
        if re.fullmatch(r"[0-9a-f]{40,64}", base) is None:
            return None
        return base

    def commits_in_range(
        self, low_exclusive: str, high_inclusive: str
    ) -> frozenset[str] | None:
        """Return every commit in ``(low_exclusive, high_inclusive]`` as full
        shas, or None fail-closed. One ``git rev-list`` covers a whole interval,
        so attribution over many landings needs a handful of Git calls instead
        of one per commit.
        """

        for commit in (low_exclusive, high_inclusive):
            if re.fullmatch(r"[0-9a-f]{7,64}", commit) is None:
                raise ValueError(
                    f"commits_in_range requires hex commit shas: {commit!r}"
                )
        result = self._run(
            "rev-list range",
            ["rev-list", f"{low_exclusive}..{high_inclusive}"],
        )
        if (
            result.timed_out
            or result.native_signal is not None
            or result.returncode != 0
        ):
            return None
        commits = frozenset(
            line.strip()
            for line in result.stdout.splitlines()
            if re.fullmatch(r"[0-9a-f]{40,64}", line.strip()) is not None
        )
        return commits

    def read_text_file(self, commit: str, path: str) -> GitBlobRead:
        key = (commit, path)
        if key in self._blob_cache:
            return self._blob_cache[key]
        exists = self.commit_exists(commit)
        if exists is False:
            result = GitBlobRead(
                status=GitBlobStatus.COMMIT_ABSENT,
                text=None,
                detail=f"commit {commit} is absent from the local object database",
            )
            self._blob_cache[key] = result
            return result
        if exists is None:
            result = GitBlobRead(
                status=GitBlobStatus.COMMIT_UNAVAILABLE,
                text=None,
                detail=f"commit presence query for {commit} was unavailable",
            )
            self._blob_cache[key] = result
            return result
        result = self._run(
            "show commit:path",
            ["show", f"{commit}:{path}"],
        )
        if result.returncode == 0 and not result.timed_out:
            blob = GitBlobRead(
                status=GitBlobStatus.OK,
                text=result.stdout,
                detail="exact source blob loaded",
            )
        elif (
            result.returncode == 128
            and not result.timed_out
            and result.native_signal is None
        ):
            blob = GitBlobRead(
                status=GitBlobStatus.PATH_ABSENT,
                text=None,
                detail=f"{path} is absent at commit {commit}",
            )
        else:
            blob = GitBlobRead(
                status=GitBlobStatus.QUERY_UNAVAILABLE,
                text=None,
                detail=(
                    "git show timed out"
                    if result.timed_out
                    else f"git show failed with return code {result.returncode}"
                ),
            )
        self._blob_cache[key] = blob
        return blob

    def pr_landing_commits(self, pr_number: int) -> tuple[tuple[str, str], ...]:
        """Real landing commits (sha, subject) on HEAD's history for ``(#pr_number)``.

        Recovers the true landing when the graph's recorded merge_commit_sha is a
        GitHub-synthetic commit absent from real history. Newest-first, subject must
        literally reference this exact PR number; reachable-from-HEAD by construction
        so the returned commits are present locally.
        """
        if pr_number <= 0:
            return ()
        token = f"(#{pr_number})"
        result = self._run(
            "log --grep",
            [
                "log",
                "--grep",
                token,
                "--fixed-strings",
                "--format=%H%x1f%s",
                "-n",
                "50",
                "HEAD",
            ],
        )
        if result.returncode != 0 or result.timed_out:
            return ()
        commits: list[tuple[str, str]] = []
        for line in result.stdout.splitlines():
            if "\x1f" not in line:
                continue
            sha, subject = line.split("\x1f", 1)
            if token in subject:
                commits.append((sha, subject))
        return tuple(commits)

    def commit_presence(self) -> dict[str, int]:
        counts = {"present": 0, "absent": 0, "unavailable": 0}
        for value in self._commit_cache.values():
            if value is True:
                counts["present"] += 1
            elif value is False:
                counts["absent"] += 1
            else:
                counts["unavailable"] += 1
        return counts


@dataclass(frozen=True)
class LandingEvaluation:
    artifact_id: str
    landing_pr: int | None
    landing_kind: str
    branch: str
    state: ContainmentState
    contained_commits: tuple[str, ...]
    absent_commits: tuple[str, ...]
    unavailable_commits: tuple[str, ...]

    def to_record(self) -> dict:
        return {
            "artifact_id": self.artifact_id,
            "landing_pr": self.landing_pr,
            "landing_kind": self.landing_kind,
            "branch": self.branch,
            "state": self.state.value,
            "contained_commits": list(self.contained_commits),
            "absent_commits": list(self.absent_commits),
            "unavailable_commits": list(self.unavailable_commits),
        }


@dataclass(frozen=True)
class EffectiveArtifactResolution:
    pr_number: int
    requested_branch: str | None
    status: ArtifactResolutionStatus
    artifact_id: str | None
    candidate_artifact_ids: tuple[str, ...]
    reason: str
    evaluations: tuple[LandingEvaluation, ...]

    def to_record(self) -> dict:
        return {
            "resolution_version": "effective_artifact_resolution_v1",
            "pr_number": self.pr_number,
            "requested_branch": self.requested_branch,
            "status": self.status.value,
            "artifact_id": self.artifact_id,
            "candidate_artifact_ids": list(self.candidate_artifact_ids),
            "reason": self.reason,
            "evaluations": [
                evaluation.to_record() for evaluation in self.evaluations
            ],
        }


@dataclass(frozen=True)
class ChangeActivityEvaluation:
    pr_number: int
    artifact_id: str
    status: ChangeActivityStatus
    active: bool | None
    reason: str
    latest_event: ChangeLanding | None
    considered_events: tuple[ChangeLanding, ...]

    def to_record(self) -> dict:
        return {
            "evaluation_version": "change_activity_evaluation_v1",
            "pr_number": self.pr_number,
            "artifact_id": self.artifact_id,
            "status": self.status.value,
            "active": self.active,
            "reason": self.reason,
            "latest_event": (
                self.latest_event.to_record()
                if self.latest_event is not None
                else None
            ),
            "considered_events": [
                event.to_record() for event in self.considered_events
            ],
        }


def _landing_in_interval(
    landing: ChangeLanding,
    artifact: VersionArtifact,
    ancestry: AncestryOracle,
) -> LandingEvaluation:
    interval = artifact.commit_interval
    if landing.branch != interval.branch:
        return LandingEvaluation(
            artifact_id=artifact.artifact_id,
            landing_pr=landing.pr,
            landing_kind=landing.kind.value,
            branch=landing.branch,
            state=ContainmentState.INCOMPARABLE_BRANCH,
            contained_commits=(),
            absent_commits=landing.landed_commits,
            unavailable_commits=(),
        )
    contained: list[str] = []
    absent: list[str] = []
    unavailable: list[str] = []
    for commit in landing.landed_commits:
        if commit == interval.from_exclusive:
            absent.append(commit)
            continue
        after_low = ancestry.is_ancestor(
            landing.branch,
            interval.from_exclusive,
            commit,
        )
        before_high = ancestry.is_ancestor(
            landing.branch,
            commit,
            interval.to_inclusive,
        )
        if after_low is None or before_high is None:
            unavailable.append(commit)
        elif after_low and before_high:
            contained.append(commit)
        else:
            absent.append(commit)
    if unavailable:
        state = ContainmentState.ANCESTRY_UNAVAILABLE
    elif len(contained) == len(landing.landed_commits):
        state = ContainmentState.COMPLETE
    elif contained:
        state = ContainmentState.PARTIAL
    else:
        state = ContainmentState.ABSENT
    return LandingEvaluation(
        artifact_id=artifact.artifact_id,
        landing_pr=landing.pr,
        landing_kind=landing.kind.value,
        branch=landing.branch,
        state=state,
        contained_commits=tuple(contained),
        absent_commits=tuple(absent),
        unavailable_commits=tuple(unavailable),
    )


def _event_before_artifact(
    landing: ChangeLanding,
    artifact: VersionArtifact,
    ancestry: AncestryOracle,
) -> bool | None:
    if landing.branch != artifact.commit_interval.branch:
        return False
    answers = [
        ancestry.is_ancestor(
            landing.branch,
            commit,
            artifact.commit_interval.to_inclusive,
        )
        for commit in landing.landed_commits
    ]
    if any(answer is None for answer in answers):
        return None
    return all(answers)


def _latest_event(
    events: list[ChangeLanding],
    *,
    branch: str,
    ancestry: AncestryOracle,
) -> ChangeLanding | None:
    latest = events[0]
    for event in events[1:]:
        left = latest.landed_commits[-1]
        right = event.landed_commits[-1]
        if left == right:
            if (
                latest.pr,
                latest.kind,
                latest.reverts_pr,
            ) != (
                event.pr,
                event.kind,
                event.reverts_pr,
            ):
                return None
            continue
        left_before_right = ancestry.is_ancestor(branch, left, right)
        right_before_left = ancestry.is_ancestor(branch, right, left)
        if left_before_right is None or right_before_left is None:
            return None
        if left_before_right and not right_before_left:
            latest = event
        elif not left_before_right and not right_before_left:
            return None
    return latest


class EffectiveArtifactResolver:
    def __init__(
        self,
        *,
        artifacts: list[VersionArtifact],
        landings: list[ChangeLanding],
        ancestry: AncestryOracle,
    ):
        self.artifacts = tuple(sorted(artifacts, key=lambda item: item.artifact_id))
        self.landings = tuple(landings)
        self.ancestry = ancestry

    def evaluate_activity(
        self,
        pr_number: int,
        artifact_id: str,
    ) -> ChangeActivityEvaluation:
        matches = [
            artifact
            for artifact in self.artifacts
            if artifact.artifact_id == artifact_id
        ]
        if len(matches) != 1:
            return ChangeActivityEvaluation(
                pr_number=pr_number,
                artifact_id=artifact_id,
                status=ChangeActivityStatus.ARTIFACT_UNKNOWN,
                active=None,
                reason="artifact ID does not resolve to exactly one catalog record",
                latest_event=None,
                considered_events=(),
            )
        artifact = matches[0]
        if (
            artifact.base_branch != artifact.commit_interval.branch
            or artifact.source_commit
            != artifact.commit_interval.to_inclusive
        ):
            return ChangeActivityEvaluation(
                pr_number=pr_number,
                artifact_id=artifact_id,
                status=ChangeActivityStatus.INVALID_ARTIFACT,
                active=None,
                reason=(
                    "artifact source_commit/base_branch disagrees with its "
                    "commit interval"
                ),
                latest_event=None,
                considered_events=(),
            )

        branch = artifact.commit_interval.branch
        activations = [
            landing
            for landing in self.landings
            if landing.pr == pr_number
            and landing.branch == branch
            and landing.kind is not LandingKind.REVERT
        ]
        reverts = [
            landing
            for landing in self.landings
            if landing.kind is LandingKind.REVERT
            and landing.reverts_pr == pr_number
            and landing.branch == branch
        ]
        events = activations + reverts
        prior: list[ChangeLanding] = []
        for event in events:
            before = _event_before_artifact(event, artifact, self.ancestry)
            if before is None:
                return ChangeActivityEvaluation(
                    pr_number=pr_number,
                    artifact_id=artifact_id,
                    status=ChangeActivityStatus.ANCESTRY_UNAVAILABLE,
                    active=None,
                    reason=(
                        "commit ancestry was unavailable for a required "
                        "landing event"
                    ),
                    latest_event=None,
                    considered_events=tuple(events),
                )
            if before:
                prior.append(event)
        if not any(event.pr == pr_number for event in prior):
            return ChangeActivityEvaluation(
                pr_number=pr_number,
                artifact_id=artifact_id,
                status=ChangeActivityStatus.INACTIVE_NOT_YET_LANDED,
                active=False,
                reason=(
                    "no complete activation landing precedes the artifact "
                    "source commit"
                ),
                latest_event=None,
                considered_events=tuple(prior),
            )
        latest = _latest_event(
            prior,
            branch=branch,
            ancestry=self.ancestry,
        )
        if latest is None:
            return ChangeActivityEvaluation(
                pr_number=pr_number,
                artifact_id=artifact_id,
                status=ChangeActivityStatus.AMBIGUOUS_HISTORY,
                active=None,
                reason="landing history has no unique ancestry-latest event",
                latest_event=None,
                considered_events=tuple(prior),
            )
        if (
            latest.kind is LandingKind.REVERT
            and latest.reverts_pr == pr_number
        ):
            return ChangeActivityEvaluation(
                pr_number=pr_number,
                artifact_id=artifact_id,
                status=ChangeActivityStatus.INACTIVE_REVERTED,
                active=False,
                reason="the ancestry-latest event explicitly reverts the change",
                latest_event=latest,
                considered_events=tuple(prior),
            )
        if latest.pr == pr_number:
            return ChangeActivityEvaluation(
                pr_number=pr_number,
                artifact_id=artifact_id,
                status=ChangeActivityStatus.ACTIVE,
                active=True,
                reason="the ancestry-latest event activates the change",
                latest_event=latest,
                considered_events=tuple(prior),
            )
        return ChangeActivityEvaluation(
            pr_number=pr_number,
            artifact_id=artifact_id,
            status=ChangeActivityStatus.AMBIGUOUS_HISTORY,
            active=None,
            reason="ancestry-latest event has no activation/revert semantics",
            latest_event=latest,
            considered_events=tuple(prior),
        )

    def _earliest(
        self,
        artifacts: list[VersionArtifact],
    ) -> tuple[VersionArtifact | None, bool]:
        if len(artifacts) == 1:
            return artifacts[0], False
        earliest: list[VersionArtifact] = []
        unavailable = False
        for candidate in artifacts:
            candidate_is_earliest = True
            for other in artifacts:
                if candidate is other:
                    continue
                order = self.ancestry.is_ancestor(
                    candidate.commit_interval.branch,
                    candidate.commit_interval.to_inclusive,
                    other.commit_interval.to_inclusive,
                )
                if order is None:
                    unavailable = True
                    candidate_is_earliest = False
                    break
                if not order:
                    candidate_is_earliest = False
                    break
            if candidate_is_earliest:
                earliest.append(candidate)
        if len(earliest) == 1:
            return earliest[0], False
        return None, unavailable

    def resolve(
        self,
        pr_number: int,
        *,
        branch: str | None = None,
    ) -> EffectiveArtifactResolution:
        activations = [
            landing
            for landing in self.landings
            if landing.pr == pr_number
            and (branch is None or landing.branch == branch)
        ]
        if not activations:
            return EffectiveArtifactResolution(
                pr_number=pr_number,
                requested_branch=branch,
                status=ArtifactResolutionStatus.NO_LANDING,
                artifact_id=None,
                candidate_artifact_ids=(),
                reason="no ChangeLanding matches the PR and requested branch",
                evaluations=(),
            )
        if not self.artifacts:
            return EffectiveArtifactResolution(
                pr_number=pr_number,
                requested_branch=branch,
                status=ArtifactResolutionStatus.NO_ARTIFACT_CATALOG,
                artifact_id=None,
                candidate_artifact_ids=(),
                reason="the graph contains no VersionArtifact records",
                evaluations=(),
            )
        landing_branches = {landing.branch for landing in activations}
        compatible = [
            artifact
            for artifact in self.artifacts
            if artifact.commit_interval.branch in landing_branches
        ]
        if not compatible:
            return EffectiveArtifactResolution(
                pr_number=pr_number,
                requested_branch=branch,
                status=ArtifactResolutionStatus.NO_COMPATIBLE_BRANCH,
                artifact_id=None,
                candidate_artifact_ids=(),
                reason="no artifact interval is on a landing branch",
                evaluations=(),
            )
        invalid = [
            artifact.artifact_id
            for artifact in compatible
            if artifact.base_branch != artifact.commit_interval.branch
            or artifact.source_commit != artifact.commit_interval.to_inclusive
        ]
        if invalid:
            return EffectiveArtifactResolution(
                pr_number=pr_number,
                requested_branch=branch,
                status=ArtifactResolutionStatus.INVALID_ARTIFACT_CATALOG,
                artifact_id=None,
                candidate_artifact_ids=tuple(invalid),
                reason=(
                    "compatible artifact source_commit/base_branch disagrees "
                    "with its commit_interval"
                ),
                evaluations=(),
            )

        external_reverts = [
            landing
            for landing in self.landings
            if landing.kind is LandingKind.REVERT
            and landing.reverts_pr == pr_number
            and landing.branch in landing_branches
        ]
        all_events = activations + external_reverts
        evaluations: list[LandingEvaluation] = []
        candidates: list[VersionArtifact] = []
        ancestry_unavailable = False
        for artifact in compatible:
            artifact_activations = [
                landing
                for landing in activations
                if landing.branch == artifact.commit_interval.branch
            ]
            interval_evaluations = [
                _landing_in_interval(landing, artifact, self.ancestry)
                for landing in artifact_activations
            ]
            evaluations.extend(interval_evaluations)
            if any(
                evaluation.state is ContainmentState.ANCESTRY_UNAVAILABLE
                for evaluation in interval_evaluations
            ):
                ancestry_unavailable = True
            if not any(
                evaluation.state is ContainmentState.COMPLETE
                for evaluation in interval_evaluations
            ):
                continue
            prior_events: list[ChangeLanding] = []
            prior_unknown = False
            for event in all_events:
                before = _event_before_artifact(
                    event,
                    artifact,
                    self.ancestry,
                )
                if before is None:
                    prior_unknown = True
                elif before:
                    prior_events.append(event)
            if prior_unknown:
                ancestry_unavailable = True
                continue
            latest = (
                _latest_event(
                    prior_events,
                    branch=artifact.commit_interval.branch,
                    ancestry=self.ancestry,
                )
                if prior_events
                else None
            )
            if latest is None:
                ancestry_unavailable = True
                continue
            if latest.pr == pr_number:
                candidates.append(artifact)

        if not candidates:
            status = (
                ArtifactResolutionStatus.ANCESTRY_UNAVAILABLE
                if ancestry_unavailable
                else ArtifactResolutionStatus.NO_COMPLETE_ARTIFACT
            )
            reason = (
                "commit ancestry was unavailable for at least one required check"
                if ancestry_unavailable
                else "no artifact completely contains an active PR landing"
            )
            return EffectiveArtifactResolution(
                pr_number=pr_number,
                requested_branch=branch,
                status=status,
                artifact_id=None,
                candidate_artifact_ids=(),
                reason=reason,
                evaluations=tuple(evaluations),
            )
        candidate_branches = {
            artifact.commit_interval.branch for artifact in candidates
        }
        if len(candidate_branches) != 1:
            return EffectiveArtifactResolution(
                pr_number=pr_number,
                requested_branch=branch,
                status=ArtifactResolutionStatus.AMBIGUOUS_ARTIFACT,
                artifact_id=None,
                candidate_artifact_ids=tuple(
                    sorted(artifact.artifact_id for artifact in candidates)
                ),
                reason="complete landing artifacts span incomparable branches",
                evaluations=tuple(evaluations),
            )
        earliest, order_unavailable = self._earliest(candidates)
        if earliest is None:
            return EffectiveArtifactResolution(
                pr_number=pr_number,
                requested_branch=branch,
                status=(
                    ArtifactResolutionStatus.ANCESTRY_UNAVAILABLE
                    if order_unavailable
                    else ArtifactResolutionStatus.AMBIGUOUS_ARTIFACT
                ),
                artifact_id=None,
                candidate_artifact_ids=tuple(
                    sorted(artifact.artifact_id for artifact in candidates)
                ),
                reason=(
                    "candidate artifact ordering requires unavailable ancestry"
                    if order_unavailable
                    else "no unique earliest complete artifact"
                ),
                evaluations=tuple(evaluations),
            )
        return EffectiveArtifactResolution(
            pr_number=pr_number,
            requested_branch=branch,
            status=ArtifactResolutionStatus.RESOLVED,
            artifact_id=earliest.artifact_id,
            candidate_artifact_ids=(earliest.artifact_id,),
            reason="earliest artifact with a complete active landing",
            evaluations=tuple(evaluations),
        )
