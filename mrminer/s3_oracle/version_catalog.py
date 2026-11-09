"""M6.2: offline VersionCatalogEntry builder from release branch points.

Each release branch (e.g. ``release/2.7``) is cut from ``main`` at a single
commit; that merge-base is an exact, offline, main-line boundary. Consecutive
boundaries partition history into intervals ``(B_{k-1}, B_k]`` that attribute a
PR landing to the release window it first shipped in. No wheel download and no
schema change (see journal Step M6.2): executable artifacts remain the job of
``artifact_catalog`` and are linked back through ``source_commit``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

from mrminer.common.schema import (
    ChangeLanding,
    CommitInterval,
    VersionCatalogEntry,
)


class CatalogGit(Protocol):
    def resolve_branch_tip(self, branch: str) -> str | None: ...

    def merge_base(self, commit_a: str, commit_b: str) -> str | None: ...

    def is_ancestor(
        self, branch: str, older: str, newer: str
    ) -> bool | None: ...


class RangeGit(Protocol):
    def commits_in_range(
        self, low_exclusive: str, high_inclusive: str
    ) -> frozenset[str] | None: ...


class VersionCatalogBuildStatus(str, Enum):
    READY = "ready"
    NO_BOUNDARIES = "no_boundaries"
    BASE_TIP_UNRESOLVED = "base_tip_unresolved"
    NON_LINEAR_BOUNDARIES = "non_linear_boundaries"


class BoundaryStatus(str, Enum):
    RESOLVED = "resolved"
    BRANCH_TIP_UNRESOLVED = "branch_tip_unresolved"
    MERGE_BASE_UNAVAILABLE = "merge_base_unavailable"
    DUPLICATE_BOUNDARY = "duplicate_boundary"


@dataclass(frozen=True, kw_only=True)
class ReleaseBoundaryRequest:
    version_label: str
    branch: str


@dataclass(frozen=True, kw_only=True)
class VersionCatalogRequest:
    compiler: str
    base_branch: str
    releases: tuple[ReleaseBoundaryRequest, ...]
    window_end_label: str = "main-tip"


@dataclass(frozen=True, kw_only=True)
class SkippedBoundary:
    version_label: str
    branch: str
    status: BoundaryStatus

    def to_record(self) -> dict:
        return {
            "version_label": self.version_label,
            "branch": self.branch,
            "status": self.status.value,
        }


@dataclass(frozen=True, kw_only=True)
class ResolvedBoundary:
    version_label: str
    branch: str
    commit: str


@dataclass(kw_only=True)
class VersionCatalogBuildReport:
    status: VersionCatalogBuildStatus
    entries: tuple[VersionCatalogEntry, ...] = ()
    ordered_boundaries: tuple[ResolvedBoundary, ...] = ()
    skipped: tuple[SkippedBoundary, ...] = ()

    def to_record(self) -> dict:
        return {
            "status": self.status.value,
            "entry_count": len(self.entries),
            "entries": [entry.to_record() for entry in self.entries],
            "ordered_boundaries": [
                {
                    "version_label": item.version_label,
                    "branch": item.branch,
                    "commit": item.commit,
                }
                for item in self.ordered_boundaries
            ],
            "skipped": [item.to_record() for item in self.skipped],
        }


def _resolve_boundaries(
    request: VersionCatalogRequest,
    git: CatalogGit,
    base_tip: str,
) -> tuple[list[ResolvedBoundary], list[SkippedBoundary]]:
    resolved: list[ResolvedBoundary] = []
    skipped: list[SkippedBoundary] = []
    seen_commits: set[str] = set()
    for release in request.releases:
        tip = git.resolve_branch_tip(release.branch)
        if tip is None:
            skipped.append(
                SkippedBoundary(
                    version_label=release.version_label,
                    branch=release.branch,
                    status=BoundaryStatus.BRANCH_TIP_UNRESOLVED,
                )
            )
            continue
        boundary = git.merge_base(tip, base_tip)
        if boundary is None:
            skipped.append(
                SkippedBoundary(
                    version_label=release.version_label,
                    branch=release.branch,
                    status=BoundaryStatus.MERGE_BASE_UNAVAILABLE,
                )
            )
            continue
        if boundary in seen_commits:
            # Two release branches that share a main-line cut collapse to one
            # boundary; keep the first, record the collision explicitly.
            skipped.append(
                SkippedBoundary(
                    version_label=release.version_label,
                    branch=release.branch,
                    status=BoundaryStatus.DUPLICATE_BOUNDARY,
                )
            )
            continue
        seen_commits.add(boundary)
        resolved.append(
            ResolvedBoundary(
                version_label=release.version_label,
                branch=release.branch,
                commit=boundary,
            )
        )
    return resolved, skipped


def _order_boundaries(
    base_branch: str,
    boundaries: list[ResolvedBoundary],
    git: CatalogGit,
) -> list[ResolvedBoundary] | None:
    """Insertion-sort by Git ancestry; return None if any pair is incomparable."""

    ordered: list[ResolvedBoundary] = []
    for candidate in boundaries:
        position = len(ordered)
        for index, existing in enumerate(ordered):
            older_first = git.is_ancestor(
                base_branch, candidate.commit, existing.commit
            )
            if older_first is None:
                return None
            if older_first:
                position = index
                break
        # Confirm the candidate is a descendant of the element it follows so a
        # genuinely incomparable (divergent) pair fails closed instead of being
        # silently ordered.
        if position > 0:
            predecessor = ordered[position - 1]
            descends = git.is_ancestor(
                base_branch, predecessor.commit, candidate.commit
            )
            if descends is None or not descends:
                return None
        ordered.insert(position, candidate)
    return ordered


def build_version_catalog(
    request: VersionCatalogRequest,
    git: CatalogGit,
) -> VersionCatalogBuildReport:
    base_tip = git.resolve_branch_tip(request.base_branch)
    if base_tip is None:
        return VersionCatalogBuildReport(
            status=VersionCatalogBuildStatus.BASE_TIP_UNRESOLVED
        )
    resolved, skipped = _resolve_boundaries(request, git, base_tip)
    if not resolved:
        return VersionCatalogBuildReport(
            status=VersionCatalogBuildStatus.NO_BOUNDARIES,
            skipped=tuple(skipped),
        )
    ordered = _order_boundaries(request.base_branch, resolved, git)
    if ordered is None:
        return VersionCatalogBuildReport(
            status=VersionCatalogBuildStatus.NON_LINEAR_BOUNDARIES,
            skipped=tuple(skipped),
        )

    entries: list[VersionCatalogEntry] = []
    # Release intervals: (B_{k-1}, B_k] labelled by the release that closes them.
    for previous, current in zip(ordered, ordered[1:]):
        entries.append(
            VersionCatalogEntry.create(
                compiler=request.compiler,
                version_label=current.version_label,
                boundary_kind="release_branch_point",
                boundary_ref=(
                    f"merge-base({current.branch}, {request.base_branch})"
                ),
                source_commit=current.commit,
                base_branch=request.base_branch,
                commit_interval=CommitInterval(
                    branch=request.base_branch,
                    from_exclusive=previous.commit,
                    to_inclusive=current.commit,
                ),
            )
        )
    # Final open window: (last release cut, base tip].
    last = ordered[-1]
    if last.commit != base_tip:
        entries.append(
            VersionCatalogEntry.create(
                compiler=request.compiler,
                version_label=request.window_end_label,
                boundary_kind="window_bound",
                boundary_ref=f"tip({request.base_branch})",
                source_commit=base_tip,
                base_branch=request.base_branch,
                commit_interval=CommitInterval(
                    branch=request.base_branch,
                    from_exclusive=last.commit,
                    to_inclusive=base_tip,
                ),
            )
        )

    return VersionCatalogBuildReport(
        status=VersionCatalogBuildStatus.READY,
        entries=tuple(entries),
        ordered_boundaries=tuple(ordered),
        skipped=tuple(skipped),
    )


def persist_catalog(
    entries: Iterable[VersionCatalogEntry],
    path: Path,
) -> bytes:
    """Write catalog entries as deterministic JSONL and return the bytes.

    Entries are ordered by ``(to_inclusive interval, entry_id)`` and each line
    is canonical JSON, so a dry-run and a later apply over the same catalog
    produce byte-identical files (idempotent import).
    """

    ordered = sorted(
        entries,
        key=lambda e: (
            e.commit_interval.to_inclusive,
            e.entry_id,
        ),
    )
    payload = "".join(
        json.dumps(entry.to_record(), sort_keys=True, separators=(",", ":")) + "\n"
        for entry in ordered
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


# --- M6.3: landing attribution over catalog intervals ------------------------


class LandingAttributionStatus(str, Enum):
    RESOLVED = "resolved"
    STRADDLES_BOUNDARY = "straddles_boundary"
    OUTSIDE_CATALOG = "outside_catalog"
    INCOMPARABLE_BRANCH = "incomparable_branch"


@dataclass(frozen=True, kw_only=True)
class LandingAttribution:
    pr: int | None
    branch: str
    kind: str
    status: LandingAttributionStatus
    entry_id: str | None
    version_label: str | None


@dataclass(kw_only=True)
class AttributionReport:
    base_branch: str
    total: int
    status_counts: dict[str, int]
    label_counts: dict[str, int]
    attributions: tuple[LandingAttribution, ...]

    def to_record(self) -> dict:
        return {
            "base_branch": self.base_branch,
            "total": self.total,
            "status_counts": dict(sorted(self.status_counts.items())),
            "label_counts": dict(sorted(self.label_counts.items())),
        }


def build_interval_commit_sets(
    entries: Iterable[VersionCatalogEntry],
    git: RangeGit,
) -> dict[str, frozenset[str]] | None:
    """Materialize each interval's commit set with one rev-list per entry.

    Returns None (fail-closed) if any interval's range is unavailable, so a
    partial Git view never silently under-counts attribution coverage.
    """

    commit_sets: dict[str, frozenset[str]] = {}
    for entry in entries:
        commits = git.commits_in_range(
            entry.commit_interval.from_exclusive,
            entry.commit_interval.to_inclusive,
        )
        if commits is None:
            return None
        commit_sets[entry.entry_id] = commits
    return commit_sets


def attribute_landings(
    entries: Iterable[VersionCatalogEntry],
    commit_sets: Mapping[str, frozenset[str]],
    landings: Iterable[ChangeLanding],
    *,
    base_branch: str,
) -> AttributionReport:
    entries = tuple(entries)
    labels = {entry.entry_id: entry.version_label for entry in entries}
    attributions: list[LandingAttribution] = []
    status_counts: dict[str, int] = {
        status.value: 0 for status in LandingAttributionStatus
    }
    label_counts: dict[str, int] = {}

    for landing in landings:
        entry_id: str | None = None
        version_label: str | None = None
        if landing.branch != base_branch:
            status = LandingAttributionStatus.INCOMPARABLE_BRANCH
        else:
            hit_entries: set[str] = set()
            any_uncontained = False
            for commit in landing.landed_commits:
                found = next(
                    (
                        entry.entry_id
                        for entry in entries
                        if commit in commit_sets[entry.entry_id]
                    ),
                    None,
                )
                if found is None:
                    any_uncontained = True
                else:
                    hit_entries.add(found)
            if not hit_entries:
                status = LandingAttributionStatus.OUTSIDE_CATALOG
            elif len(hit_entries) == 1 and not any_uncontained:
                status = LandingAttributionStatus.RESOLVED
                entry_id = next(iter(hit_entries))
                version_label = labels[entry_id]
                label_counts[version_label] = label_counts.get(version_label, 0) + 1
            else:
                status = LandingAttributionStatus.STRADDLES_BOUNDARY
        status_counts[status.value] += 1
        attributions.append(
            LandingAttribution(
                pr=landing.pr,
                branch=landing.branch,
                kind=landing.kind.value,
                status=status,
                entry_id=entry_id,
                version_label=version_label,
            )
        )

    return AttributionReport(
        base_branch=base_branch,
        total=len(attributions),
        status_counts=status_counts,
        label_counts=label_counts,
        attributions=tuple(attributions),
    )
