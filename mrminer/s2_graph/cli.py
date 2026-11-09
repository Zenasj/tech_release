"""Command-line interface for building and querying EntityGraph."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from mrminer.common.canonical import canonical_json
from mrminer.s2_graph.builder import build_entity_graph
from mrminer.s2_graph.graph_store import EntityGraph
from mrminer.s2_graph.quality_audit import audit_graph_quality
from mrminer.s2_graph.schema import GraphNodeKind, NodeRef


def _compact_pr(record) -> dict[str, Any]:
    return {
        "pr_number": record.pr_number,
        "title": record.title,
        "html_url": record.html_url,
        "merged": record.merged,
        "merged_at": record.merged_at,
        "change_kind": record.change_kind,
        "subsystem_labels": list(record.subsystem_labels),
        "linked_issues": [
            {
                "number": issue.number,
                "fetch_status": issue.fetch_status.value,
                "link_kinds": [
                    evidence.kind.value
                    for evidence in issue.link_evidence
                ],
            }
            for issue in record.linked_issues
        ],
    }


def _compact_edge(edge) -> dict[str, Any]:
    return {
        "edge_id": edge.edge_id,
        "kind": edge.kind.value,
        "source": edge.source.to_record(),
        "target": edge.target.to_record(),
        "confidence": edge.confidence,
        "context": edge.context,
        "at": edge.at,
        "evidence": [item.to_record() for item in edge.evidence],
    }


def _write(value: Any) -> None:
    print(canonical_json(value))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build, audit, and query the MRMiner EntityGraph."
    )
    parser.add_argument(
        "--graph-root",
        type=Path,
        default=Path("data/entity_graph_canonical"),
        help="Graph directory (default: data/entity_graph_canonical, a symlink "
        "to the current canonical graph).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build and publish a graph.")
    build.add_argument(
        "--corpus-root",
        type=Path,
        default=Path("data/pr_corpus_h1_2025_clean"),
    )
    build.add_argument(
        "--pytorch-source-root",
        type=Path,
        required=True,
    )
    build.add_argument("--fanout-limit", type=int, default=15)
    build.add_argument(
        "--force",
        action="store_true",
        help="Preserve the existing graph as a timestamped backup.",
    )

    subparsers.add_parser(
        "audit",
        help="Rebuild the SQLite index and report consistency.",
    )
    subparsers.add_parser("stats", help="Show graph node/edge counts.")
    quality = subparsers.add_parser(
        "quality-audit",
        help="Resolve all evidence locators and audit relationship sources.",
    )
    quality.add_argument(
        "--corpus-root",
        type=Path,
        default=Path("data/pr_corpus_h1_2025_clean"),
    )
    quality.add_argument("--output", type=Path)
    quality.add_argument("--sample-size", type=int, default=100)
    quality.add_argument("--seed", type=int, default=1337)

    search = subparsers.add_parser("search", help="Search entity IDs/names.")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=20)

    entity = subparsers.add_parser(
        "entity",
        help="Show one entity, its edge timeline, and related PR summaries.",
    )
    entity.add_argument("entity_id")
    entity.add_argument("--limit", type=int, default=100)

    pr = subparsers.add_parser("pr", help="Show one PR and graph edges.")
    pr.add_argument("pr_number", type=int)

    issue = subparsers.add_parser(
        "issue",
        help="Show one Issue node and graph edges.",
    )
    issue.add_argument("issue_number", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        if args.fanout_limit < 1:
            raise SystemExit("--fanout-limit must be positive")
        report = build_entity_graph(
            corpus_root=args.corpus_root,
            graph_root=args.graph_root,
            pytorch_source_root=args.pytorch_source_root,
            fanout_limit=args.fanout_limit,
            force=args.force,
        )
        _write(report)
        return 0

    graph = EntityGraph(args.graph_root)
    if args.command == "audit":
        report = graph.rebuild_index()
        _write(
            {
                "checksum_ok": report.checksum_ok,
                "records": report.records,
                "rebuilt_at": report.rebuilt_at,
                "errors": list(graph.last_rebuild_errors),
                "stats": graph.stats(),
            }
        )
        return 0 if report.checksum_ok else 2
    if args.command == "stats":
        _write(graph.stats())
        return 0
    if args.command == "quality-audit":
        report = audit_graph_quality(
            graph_root=args.graph_root,
            corpus_root=args.corpus_root,
            output_path=args.output,
            sample_size=args.sample_size,
            seed=args.seed,
        )
        _write(report)
        return 0 if report["gates"]["passed"] else 2
    if args.command == "search":
        _write(
            [
                entity.to_record()
                for entity in graph.search_entities(
                    args.query,
                    limit=args.limit,
                )
            ]
        )
        return 0
    if args.command == "entity":
        entity = graph.entity(args.entity_id)
        edges = graph.timeline(args.entity_id)
        prs = graph.related_pr_summaries(
            args.entity_id,
            limit=args.limit,
        )
        _write(
            {
                "entity": entity.to_record(),
                "edge_counts": dict(
                    sorted(Counter(edge.kind.value for edge in edges).items())
                ),
                "timeline": [_compact_edge(edge) for edge in edges],
                "related_prs": prs,
            }
        )
        return 0
    if args.command == "pr":
        pr = graph.pr(args.pr_number)
        edges = graph.edges_for_node(
            NodeRef(
                kind=GraphNodeKind.PR,
                key=f"pr:{args.pr_number}",
            )
        )
        _write(
            {
                "pr": _compact_pr(pr),
                "edges": [_compact_edge(edge) for edge in edges],
            }
        )
        return 0
    if args.command == "issue":
        issue = graph.issue(args.issue_number)
        edges = graph.edges_for_node(
            NodeRef(
                kind=GraphNodeKind.ISSUE,
                key=f"issue:{args.issue_number}",
            )
        )
        _write(
            {
                "issue": issue.to_record(),
                "edges": [_compact_edge(edge) for edge in edges],
            }
        )
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
