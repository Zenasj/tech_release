"""Evidence and relationship quality audit for a production EntityGraph."""

from __future__ import annotations

from collections import Counter, defaultdict
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from mrminer.common.canonical import canonical_json
from mrminer.common.storage import JsonlStore
from mrminer.s1_miner.issue_schema import IssueCorpusV1
from mrminer.s1_miner.pr_schema import PrCorpusV2
from mrminer.s2_graph.graph_store import EntityGraph
from mrminer.s2_graph.linker import is_implementation_path
from mrminer.s2_graph.schema import GraphEdge, GraphEdgeKind

_PATCH_LOCATOR = re.compile(
    r"^patch:(?P<path>.+)@chars (?P<start>\d+)-(?P<end>\d+)$"
)
_TEST_LOCATOR = re.compile(
    r"^test:(?P<node>.+)@chars (?P<start>\d+)-(?P<end>\d+)$"
)
_TEXT_LOCATOR = re.compile(
    r"^(?P<source>.+)@chars (?P<start>\d+)-(?P<end>\d+)$"
)


def _slice_matches(
    text: str,
    *,
    start: int,
    end: int,
    quote: str,
) -> bool:
    return 0 <= start < end <= len(text) and text[start:end] == quote


def _pr_source(pr: PrCorpusV2, source_id: str) -> str | None:
    if source_id == "title":
        return pr.title
    if source_id == pr.body.source_id:
        return pr.body.text
    return next(
        (
            source.text
            for source in pr.comments
            if source.source_id == source_id
        ),
        None,
    )


def _issue_source(issue: IssueCorpusV1, source_id: str) -> str | None:
    if source_id == issue.body.source_id:
        return issue.body.text
    return next(
        (
            source.text
            for source in issue.comments.items
            if source.source_id == source_id
        ),
        None,
    )


def _sample_key(seed: int, edge_id: str) -> str:
    return hashlib.sha256(f"{seed}:{edge_id}".encode()).hexdigest()


def audit_graph_quality(
    *,
    graph_root: Path,
    corpus_root: Path,
    output_path: Path | None = None,
    sample_size: int = 100,
    seed: int = 1337,
) -> dict[str, Any]:
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    graph_root = graph_root.resolve()
    corpus_root = corpus_root.resolve()
    repository_dirs = sorted(
        path
        for path in corpus_root.iterdir()
        if path.is_dir() and "__" in path.name
    )
    if len(repository_dirs) != 1:
        raise ValueError("quality audit requires a single-repository corpus")
    repository_root = repository_dirs[0]
    pr_paths = {
        int(path.stem): path
        for path in repository_root.glob("*.json")
        if path.stem.isdigit()
    }
    issue_paths = {
        int(path.stem): path
        for path in (repository_root / "issues").glob("*.json")
    }
    pr_cache: dict[int, PrCorpusV2] = {}
    issue_cache: dict[int, IssueCorpusV1] = {}

    def pr(number: int) -> PrCorpusV2:
        if number not in pr_paths:
            raise ValueError(
                f"graph references PR #{number}, but it is absent from "
                f"corpus {repository_root}"
            )
        if number not in pr_cache:
            pr_cache.clear()
            pr_cache[number] = PrCorpusV2.from_record(
                json.loads(pr_paths[number].read_bytes())
            )
        return pr_cache[number]

    def issue(number: int) -> IssueCorpusV1:
        if number not in issue_paths:
            raise ValueError(
                f"graph references issue #{number}, but it is absent from "
                f"corpus {repository_root / 'issues'}"
            )
        if number not in issue_cache:
            issue_cache.clear()
            issue_cache[number] = IssueCorpusV1.from_record(
                json.loads(issue_paths[number].read_bytes())
            )
        return issue_cache[number]

    edge_counts: Counter[str] = Counter()
    signal_counts: Counter[tuple[str, ...]] = Counter()
    connected_prs: set[int] = set()
    operator_linked_prs: set[int] = set()
    locator_checks = 0
    locator_failures: list[dict[str, Any]] = []
    relationship_failures: list[dict[str, Any]] = []
    operator_candidates: list[dict[str, Any]] = []
    non_implementation_operator_evidence = 0

    for record in JsonlStore(graph_root / "edges.jsonl").iter_records():
        edge = GraphEdge.from_record(record)
        edge_counts[edge.kind.value] += 1
        source_number = (
            int(edge.source.key.removeprefix("pr:"))
            if edge.source.key.startswith("pr:")
            else None
        )
        issue_number = (
            int(edge.source.key.removeprefix("issue:"))
            if edge.source.key.startswith("issue:")
            else None
        )
        if edge.kind is GraphEdgeKind.IMPLEMENTATION_TOUCHES:
            assert source_number is not None
            connected_prs.add(source_number)
            signals = tuple(edge.context.get("signals", ()))
            signal_counts[signals] += 1
            if edge.target.key.startswith("op:"):
                operator_linked_prs.add(source_number)
                operator_candidates.append(
                    {
                        "edge_id": edge.edge_id,
                        "pr": source_number,
                        "entity_id": edge.target.key,
                        "signals": list(signals),
                        "confidence": edge.confidence,
                        "evidence": [
                            item.to_record() for item in edge.evidence
                        ],
                    }
                )

        for evidence in edge.evidence:
            locator_checks += 1
            valid = False
            reason = "unsupported locator"
            if edge.kind is GraphEdgeKind.IMPLEMENTATION_TOUCHES:
                assert source_number is not None
                source_pr = pr(source_number)
                patch_match = _PATCH_LOCATOR.fullmatch(evidence.locator)
                if patch_match is not None:
                    if (
                        edge.target.key.startswith("op:")
                        and not is_implementation_path(
                            patch_match.group("path")
                        )
                    ):
                        non_implementation_operator_evidence += 1
                    patch = next(
                        (
                            item.patch
                            for item in source_pr.touched_files_patch
                            if item.path == patch_match.group("path")
                        ),
                        None,
                    )
                    if patch is not None:
                        valid = _slice_matches(
                            patch,
                            start=int(patch_match.group("start")),
                            end=int(patch_match.group("end")),
                            quote=evidence.quote,
                        )
                        reason = "patch slice mismatch"
                    else:
                        reason = "patch unavailable"
                elif evidence.locator == "diff_stats.files[0].path":
                    paths = [
                        item.get("path")
                        for item in source_pr.diff_stats.get("files", ())
                        if isinstance(item, dict)
                    ]
                    valid = bool(paths) and paths[0] == evidence.quote
                    reason = "diff_stats path mismatch"
                elif evidence.locator == "subsystem_labels":
                    valid = evidence.quote in source_pr.subsystem_labels
                    reason = "subsystem label mismatch"
            elif edge.kind is GraphEdgeKind.STATICALLY_MENTIONS:
                assert source_number is not None
                source_pr = pr(source_number)
                match = _TEST_LOCATOR.fullmatch(evidence.locator)
                if match is not None:
                    test = next(
                        (
                            item
                            for item in source_pr.tests_added
                            if item.test_node_id == match.group("node")
                        ),
                        None,
                    )
                    if test is not None:
                        valid = _slice_matches(
                            test.added_code,
                            start=int(match.group("start")),
                            end=int(match.group("end")),
                            quote=evidence.quote,
                        )
                        reason = "test slice mismatch"
                    else:
                        reason = "test node missing"
            elif edge.kind is GraphEdgeKind.SYMPTOM_MENTIONS:
                assert issue_number is not None
                source_issue = issue(issue_number)
                match = _TEXT_LOCATOR.fullmatch(evidence.locator)
                if match is not None:
                    text = _issue_source(
                        source_issue,
                        match.group("source"),
                    )
                    if text is not None:
                        valid = _slice_matches(
                            text,
                            start=int(match.group("start")),
                            end=int(match.group("end")),
                            quote=evidence.quote,
                        )
                        reason = "issue text slice mismatch"
                    else:
                        reason = "issue text source missing"
            elif edge.kind is GraphEdgeKind.FIXES:
                assert source_number is not None
                source_pr = pr(source_number)
                if evidence.locator == "graphql:closingIssuesReferences":
                    valid = evidence.quote.startswith(
                        "closingIssuesReferences(issue:"
                    )
                    reason = "closing reference quote mismatch"
                else:
                    match = _TEXT_LOCATOR.fullmatch(evidence.locator)
                    if match is not None:
                        text = _pr_source(
                            source_pr,
                            match.group("source"),
                        )
                        if text is not None:
                            valid = _slice_matches(
                                text,
                                start=int(match.group("start")),
                                end=int(match.group("end")),
                                quote=evidence.quote,
                            )
                            reason = "PR text slice mismatch"
                        else:
                            reason = "PR text source missing"
            elif edge.kind is GraphEdgeKind.REVERTS:
                assert source_number is not None
                text = _pr_source(pr(source_number), evidence.locator)
                valid = text is not None and evidence.quote in text
                reason = "revert quote missing from source"
            else:
                valid = True
                reason = ""
            if not valid:
                locator_failures.append(
                    {
                        "edge_id": edge.edge_id,
                        "kind": edge.kind.value,
                        "artifact_ref": evidence.artifact_ref,
                        "locator": evidence.locator,
                        "reason": reason,
                    }
                )

        if edge.kind is GraphEdgeKind.FIXES:
            assert source_number is not None
            target = int(edge.target.key.removeprefix("issue:"))
            source_pr = pr(source_number)
            linked = next(
                (
                    item
                    for item in source_pr.linked_issues
                    if item.number == target
                ),
                None,
            )
            if linked is None or not set(edge.context["link_kinds"]) <= {
                "fixes_keyword",
                "closing_reference",
            }:
                relationship_failures.append(
                    {
                        "edge_id": edge.edge_id,
                        "reason": "fixes edge is not backed by a qualifying link",
                    }
                )
        elif edge.kind is GraphEdgeKind.REVERTS:
            assert source_number is not None
            target = int(edge.target.key.removeprefix("pr:"))
            revert = pr(source_number).revert_evidence
            if revert is None or revert.target_pr != target:
                relationship_failures.append(
                    {
                        "edge_id": edge.edge_id,
                        "reason": "revert edge disagrees with PR evidence",
                    }
                )

    graph = EntityGraph(graph_root)
    stats = graph.stats()
    build_report = json.loads(
        (graph_root / "build_report.json").read_text()
    )
    expected_stats = {
        "nodes": build_report["index"]["counts"]["nodes"],
        "edges": build_report["index"]["counts"]["edges"],
        "landings": build_report["index"]["counts"]["landings"],
    }
    builder_partitions_match = (
        all(
            stats["nodes"].get(kind, 0) == count
            for kind, count in expected_stats["nodes"].items()
        )
        and all(
            stats["edges"].get(kind, 0) == count
            for kind, count in expected_stats["edges"].items()
        )
        and stats["landings"] == expected_stats["landings"]
    )
    append_only_node_counts = {
        kind: count
        for kind, count in stats["nodes"].items()
        if kind not in expected_stats["nodes"]
    }
    append_only_edge_counts = {
        kind: count
        for kind, count in stats["edges"].items()
        if kind not in expected_stats["edges"]
    }
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in operator_candidates:
        groups[tuple(item["signals"])].append(item)
    mandatory = [
        min(items, key=lambda item: _sample_key(seed, item["edge_id"]))
        for _, items in sorted(groups.items())
    ]
    selected_ids = {item["edge_id"] for item in mandatory}
    remainder = sorted(
        (
            item
            for item in operator_candidates
            if item["edge_id"] not in selected_ids
        ),
        key=lambda item: _sample_key(seed, item["edge_id"]),
    )
    sample = (mandatory + remainder)[:sample_size]

    gates = {
        "build_gate_passed": bool(build_report["gates"]["passed"]),
        "all_evidence_locators_resolve": not locator_failures,
        "all_prs_have_implementation_context": (
            connected_prs == set(pr_paths)
        ),
        "relationship_sources_match": not relationship_failures,
        "graph_counts_match_build_report": builder_partitions_match,
        "no_non_implementation_operator_evidence": (
            non_implementation_operator_evidence == 0
        ),
    }
    report = {
        "schema_version": "entity_graph_quality_audit_v1",
        "audited_at": dt.datetime.now(dt.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "graph_root": str(graph_root),
        "corpus_root": str(corpus_root),
        "counts": {
            "prs": len(pr_paths),
            "issues": len(issue_paths),
            "edges": dict(sorted(edge_counts.items())),
            "implementation_signal_sets": {
                "+".join(key): value
                for key, value in sorted(signal_counts.items())
            },
            "operator_linked_prs": len(operator_linked_prs),
            "operator_linked_pr_coverage": (
                len(operator_linked_prs) / len(pr_paths)
            ),
            "connected_prs": len(connected_prs),
            "evidence_locators_checked": locator_checks,
            "evidence_locator_failures": len(locator_failures),
            "relationship_failures": len(relationship_failures),
            "non_implementation_operator_evidence": (
                non_implementation_operator_evidence
            ),
            "graph": stats,
            "append_only_nodes": append_only_node_counts,
            "append_only_edges": append_only_edge_counts,
        },
        "failures": {
            "locator": locator_failures,
            "relationship": relationship_failures,
        },
        "manual_precision_sample": {
            "seed": seed,
            "requested_size": sample_size,
            "actual_size": len(sample),
            "status": "pending_manual_label",
            "edges": sample,
        },
        "gates": {**gates, "passed": all(gates.values())},
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(canonical_json(report) + "\n")
    return report
