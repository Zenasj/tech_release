"""M6.3 runner: build the release-boundary catalog from a real Git mirror and
attribute a landings corpus to release intervals, writing a coverage report.

Kept separate from the pure builder/attribution logic in ``version_catalog`` so
the IO boundary (mirror path, corpus file, report file) stays out of the
testable core.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mrminer.common.schema import ChangeLanding
from mrminer.s3_oracle.artifact_resolution import SubprocessGitAncestry
from mrminer.s3_oracle.version_catalog import (
    ReleaseBoundaryRequest,
    VersionCatalogBuildStatus,
    VersionCatalogRequest,
    attribute_landings,
    build_interval_commit_sets,
    build_version_catalog,
    persist_catalog,
)

DEFAULT_RELEASES = (
    "release/2.5",
    "release/2.6",
    "release/2.7",
    "release/2.8",
    "release/2.9",
)


def _load_landings(path: Path) -> list[ChangeLanding]:
    landings: list[ChangeLanding] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            landings.append(ChangeLanding.from_record(json.loads(line)))
    return landings


def run_audit(
    *,
    mirror: Path,
    landings_path: Path,
    out_path: Path,
    releases: tuple[str, ...] = DEFAULT_RELEASES,
    base_branch: str = "main",
    compiler: str = "pytorch",
    anchor_prs: tuple[int, ...] = (168157,),
) -> dict:
    git = SubprocessGitAncestry(mirror)
    request = VersionCatalogRequest(
        compiler=compiler,
        base_branch=base_branch,
        releases=tuple(
            ReleaseBoundaryRequest(version_label=name, branch=name)
            for name in releases
        ),
    )
    build = build_version_catalog(request, git)
    if build.status is not VersionCatalogBuildStatus.READY:
        report = {
            "status": "catalog_build_failed",
            "build": build.to_record(),
        }
        out_path.write_text(json.dumps(report, indent=1, sort_keys=True))
        return report

    commit_sets = build_interval_commit_sets(build.entries, git)
    if commit_sets is None:
        report = {"status": "interval_ranges_unavailable"}
        out_path.write_text(json.dumps(report, indent=1, sort_keys=True))
        return report

    catalog_path = out_path.parent / "m6_version_catalog.jsonl"
    persist_catalog(build.entries, catalog_path)

    landings = _load_landings(landings_path)
    attribution = attribute_landings(
        build.entries, commit_sets, landings, base_branch=base_branch
    )

    by_pr = {a.pr: a for a in attribution.attributions}
    anchors = {
        str(pr): (
            {
                "status": by_pr[pr].status.value,
                "version_label": by_pr[pr].version_label,
            }
            if pr in by_pr
            else {"status": "not_in_corpus"}
        )
        for pr in anchor_prs
    }

    report = {
        "status": "ok",
        "mirror": str(mirror),
        "landings_path": str(landings_path),
        "catalog": {
            "catalog_path": str(catalog_path),
            "entries": [entry.to_record() for entry in build.entries],
            "interval_sizes": {
                entry.version_label: len(commit_sets[entry.entry_id])
                for entry in build.entries
            },
            "skipped_boundaries": [s.to_record() for s in build.skipped],
        },
        "attribution": attribution.to_record(),
        "coverage_rate_over_base_branch": _coverage(attribution),
        "anchor_prs": anchors,
    }
    out_path.write_text(json.dumps(report, indent=1, sort_keys=True))
    return report


def _coverage(attribution) -> float:
    on_branch = attribution.total - attribution.status_counts.get(
        "incomparable_branch", 0
    )
    if on_branch == 0:
        return 0.0
    resolved = attribution.status_counts.get("resolved", 0)
    return round(resolved / on_branch, 6)


def main() -> None:
    parser = argparse.ArgumentParser(description="M6.3 landing attribution audit")
    parser.add_argument("--mirror", required=True, type=Path)
    parser.add_argument("--landings", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report = run_audit(
        mirror=args.mirror,
        landings_path=args.landings,
        out_path=args.out,
    )
    print(json.dumps(report.get("attribution", report), indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
