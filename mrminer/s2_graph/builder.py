"""Production EntityGraph construction from an S1 corpus."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Any
import uuid

from mrminer.common.canonical import canonical_json, sha256_digest
from mrminer.common.schema import (
    ChangeLanding,
    EvidenceSpec,
    EvidenceStrength,
)
from mrminer.s1_miner.issue_schema import IssueCorpusV1
from mrminer.s1_miner.pr_schema import (
    IssueFetchStatus,
    IssueLinkKind,
    PrCorpusV2,
)
from mrminer.s2_graph.graph_store import EntityGraph
from mrminer.common.schema import EntitySpec
from mrminer.s2_graph.linker import (
    DeterministicLinker,
    subsystem_entity,
)
from mrminer.s2_graph.registry_loader import (
    RegistrySnapshot,
    load_operator_registry,
)
from mrminer.s2_graph.schema import (
    GraphEdge,
    GraphEdgeKind,
    GraphNodeKind,
    IssueNode,
    NodeRef,
)

_CHAR_LOCATOR = re.compile(r"@chars (?P<start>\d+)-(?P<end>\d+)$")


class GraphBuildError(RuntimeError):
    pass


def _pr_time(pr: PrCorpusV2) -> str:
    value = pr.merged_at or pr.closed_at or pr.collection.collected_at
    if value == "unknown":
        raise GraphBuildError(f"PR #{pr.pr_number} has no ISO edge timestamp")
    return value


def _evidence(
    *,
    artifact_ref: str,
    locator: str,
    quote: str,
    supports_field: str,
    extractor: str,
    strength: EvidenceStrength,
) -> EvidenceSpec:
    return EvidenceSpec(
        artifact_ref=artifact_ref,
        locator=locator,
        quote=quote,
        quote_hash=sha256_digest(quote),
        strength=strength,
        supports_field=supports_field,
        extractor=extractor,
    )


def _source_text(pr: PrCorpusV2, source_id: str) -> str | None:
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


def _fix_edges(
    pr: PrCorpusV2,
    *,
    available_issues: frozenset[int],
    counters: Counter[str],
) -> Iterable[GraphEdge]:
    for linked in pr.linked_issues:
        if (
            linked.fetch_status is not IssueFetchStatus.STORED
            or linked.number not in available_issues
        ):
            counters["issue_reference_unavailable"] += 1
            continue
        evidence: list[EvidenceSpec] = []
        for link in linked.link_evidence:
            if link.kind is IssueLinkKind.COMMENT_MENTION:
                counters["comment_only_link_evidence"] += 1
                continue
            if link.kind is IssueLinkKind.CLOSING_REFERENCE:
                quote = (
                    f"closingIssuesReferences(issue:{linked.number})"
                )
            elif link.kind is IssueLinkKind.FIXES_KEYWORD:
                source = _source_text(pr, link.source_id)
                match = _CHAR_LOCATOR.search(link.locator)
                if source is None or match is None:
                    counters["invalid_fix_locator"] += 1
                    continue
                start, end = int(match.group("start")), int(match.group("end"))
                if not 0 <= start < end <= len(source):
                    counters["invalid_fix_locator"] += 1
                    continue
                quote = source[start:end]
            else:
                continue
            evidence.append(
                _evidence(
                    artifact_ref=f"pr:{pr.pr_number}",
                    locator=link.locator,
                    quote=quote,
                    supports_field="fixes",
                    extractor="s2.issue_relation/v1",
                    strength=EvidenceStrength.EXPLICIT_CLAIM,
                )
            )
        if not evidence:
            counters["linked_without_fix_semantics"] += 1
            continue
        unique = {
            (item.locator, item.quote_hash): item for item in evidence
        }
        yield GraphEdge.create(
            kind=GraphEdgeKind.FIXES,
            source=NodeRef(
                kind=GraphNodeKind.PR,
                key=f"pr:{pr.pr_number}",
            ),
            target=NodeRef(
                kind=GraphNodeKind.ISSUE,
                key=f"issue:{linked.number}",
            ),
            evidence=tuple(unique[key] for key in sorted(unique)),
            confidence=0.98,
            context={
                "link_kinds": sorted(
                    {
                        link.kind.value
                        for link in linked.link_evidence
                        if link.kind is not IssueLinkKind.COMMENT_MENTION
                    }
                )
            },
            at=_pr_time(pr),
        )


def _revert_edges(
    pr: PrCorpusV2,
    *,
    available_prs: frozenset[int],
    counters: Counter[str],
) -> list[GraphEdge]:
    revert = pr.revert_evidence
    if revert is None:
        return []
    edges: list[GraphEdge] = []
    for target_pr in (revert.target_pr, *revert.additional_target_prs):
        if target_pr not in available_prs:
            counters["revert_target_outside_corpus"] += 1
            continue
        edges.append(
            GraphEdge.create(
                kind=GraphEdgeKind.REVERTS,
                source=NodeRef(
                    kind=GraphNodeKind.PR,
                    key=f"pr:{pr.pr_number}",
                ),
                target=NodeRef(
                    kind=GraphNodeKind.PR,
                    key=f"pr:{target_pr}",
                ),
                evidence=(
                    _evidence(
                        artifact_ref=f"pr:{pr.pr_number}",
                        locator=revert.source_id,
                        quote=revert.quote,
                        supports_field="reverts",
                        extractor="s2.revert_relation/v1",
                        strength=EvidenceStrength.EXPLICIT_CLAIM,
                    ),
                ),
                confidence=0.95,
                context={"evidence_kind": revert.kind.value},
                at=_pr_time(pr),
            )
        )
    return edges


def _sqlite_counts(path: Path) -> dict[str, Any]:
    with sqlite3.connect(path) as connection:
        nodes = {
            kind: count
            for kind, count in connection.execute(
                "SELECT kind, COUNT(*) FROM nodes GROUP BY kind ORDER BY kind"
            )
        }
        edges = {
            kind: count
            for kind, count in connection.execute(
                "SELECT kind, COUNT(*) FROM edges GROUP BY kind ORDER BY kind"
            )
        }
        extras = {
            table: connection.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]
            for table in ("landings", "oracle_annotations", "projections")
        }
    return {"nodes": nodes, "edges": edges, **extras}


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as stream:
        for line_no, line in enumerate(stream, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise GraphBuildError(
                    f"{path}:{line_no} is invalid JSON"
                ) from error
            if not isinstance(value, dict):
                raise GraphBuildError(f"{path}:{line_no} is not an object")
            yield value


def build_entity_graph(
    *,
    corpus_root: Path,
    graph_root: Path,
    pytorch_source_root: Path,
    fanout_limit: int = 15,
    force: bool = False,
    registry_snapshot: RegistrySnapshot | None = None,
    api_registry: tuple[EntitySpec, ...] | None = None,
    fold_issues_into_pr: bool = False,
    max_pr_record_bytes: int = 100 * 1024 * 1024,
) -> dict[str, Any]:
    """Build a complete deterministic graph and atomically publish it."""
    corpus_root = corpus_root.resolve()
    graph_root = graph_root.resolve()
    if graph_root.exists() and not force:
        raise FileExistsError(
            f"graph root already exists; pass force to preserve it as a backup: "
            f"{graph_root}"
        )
    repository_dirs = sorted(
        path
        for path in corpus_root.iterdir()
        if path.is_dir() and "__" in path.name
    )
    if len(repository_dirs) != 1:
        raise GraphBuildError(
            "v1 IssueNode identity supports exactly one repository per graph"
        )
    repository_root = repository_dirs[0]
    pr_paths = sorted(
        (
            path
            for path in repository_root.glob("*.json")
            if path.stem.isdigit()
        ),
        key=lambda path: int(path.stem),
    )
    issue_paths = sorted(
        (repository_root / "issues").glob("*.json"),
        key=lambda path: int(path.stem),
    )
    landings_path = corpus_root / "landings.jsonl"
    if not pr_paths or not landings_path.exists():
        raise GraphBuildError("corpus has no PR records or landings.jsonl")

    snapshot = registry_snapshot or load_operator_registry(
        pytorch_source_root
    )
    api_entities = tuple(api_registry or ())
    api_ids = frozenset(entity.entity_id for entity in api_entities)
    linker = DeterministicLinker(
        operator_ids=snapshot.operator_ids,
        dispatch_symbols=snapshot.dispatch_symbols,
        fanout_limit=fanout_limit,
        api_ids=api_ids,
    )
    staging = graph_root.with_name(
        f".{graph_root.name}.build-{uuid.uuid4().hex[:12]}"
    )
    staging.mkdir(parents=True)
    graph = EntityGraph(staging)
    counters: Counter[str] = Counter()
    unknown_tokens: Counter[str] = Counter()
    subsystem_first_seen: dict[str, int] = {}
    # Mass-refactor / codemod PRs (thousands of changed files) are not
    # operator-oracle candidates, and their full record can exceed SQLite's
    # per-value size limit and break index storage. Skip them from both node
    # import and linking, recording which were dropped. Guard by on-disk record
    # size so we never have to parse a multi-GB record just to reject it.
    oversized_prs = frozenset(
        int(path.stem)
        for path in pr_paths
        if path.stat().st_size > max_pr_record_bytes
    )
    counters["skipped_oversized_prs"] = len(oversized_prs)
    available_prs = (
        frozenset(int(path.stem) for path in pr_paths) - oversized_prs
    )
    digest = hashlib.sha256()

    def prs() -> Iterable[PrCorpusV2]:
        for path in pr_paths:
            if int(path.stem) in oversized_prs:
                continue
            payload = path.read_bytes()
            digest.update(path.name.encode())
            digest.update(hashlib.sha256(payload).digest())
            pr = PrCorpusV2.from_record(json.loads(payload))
            counters["input_prs"] += 1
            for label in pr.subsystem_labels:
                subsystem_first_seen[label] = min(
                    pr.pr_number,
                    subsystem_first_seen.get(label, pr.pr_number),
                )
            yield pr

    imported_prs = graph.bulk_add_prs(prs())
    entities = list(snapshot.entities)
    entities.extend(api_entities)
    entities.extend(
        subsystem_entity(label, first_seen=f"pr:{number}")
        for label, number in sorted(subsystem_first_seen.items())
    )
    imported_entities = graph.bulk_add_entities(entities)

    issue_records: list[IssueCorpusV1] = []
    for path in issue_paths:
        payload = path.read_bytes()
        digest.update(f"issues/{path.name}".encode())
        digest.update(hashlib.sha256(payload).digest())
        issue_records.append(
            IssueCorpusV1.from_record(json.loads(payload))
        )
    available_issues = frozenset(
        issue.issue_number for issue in issue_records
    )
    # graph_model_v1 §4: an issue folds into the PR that fixes it (the PR record
    # already carries linked_issues), so the folded graph has no Issue node nor
    # fixes/symptom_mentions edge. The schema enum values are kept so existing
    # graphs still load; R2.5 removes them when the H1 graph is rebuilt.
    if fold_issues_into_pr:
        imported_issues = 0
    else:
        imported_issues = graph.bulk_add_issues(
            IssueNode(
                issue_number=issue.issue_number,
                title=issue.title,
                html_url=issue.html_url,
                created_at=issue.created_at,
            )
            for issue in issue_records
        )

    landing_payload = landings_path.read_bytes()
    digest.update(b"landings.jsonl")
    digest.update(hashlib.sha256(landing_payload).digest())
    # Drop landings whose PR was skipped as oversized: without its node the
    # landing would be a dangling reference and fail the graph gates.
    landings = tuple(
        landing
        for record in _iter_jsonl(landings_path)
        if (landing := ChangeLanding.from_record(record)).pr
        not in oversized_prs
    )
    imported_landings = graph.bulk_add_landings(landings)

    def edges() -> Iterable[GraphEdge]:
        for path in pr_paths:
            if int(path.stem) in oversized_prs:
                continue
            pr = PrCorpusV2.from_record(json.loads(path.read_bytes()))
            result = linker.link_pr(pr)
            counters["linked_prs"] += bool(result.exact_operator_count)
            counters["operator_touch_edges"] += result.exact_operator_count
            counters["broad_change_prs"] += result.broad_change
            unknown_tokens.update(result.unknown_operator_tokens)
            for edge in result.edges:
                counters[f"edge:{edge.kind.value}"] += 1
                yield edge
            if not fold_issues_into_pr:
                for edge in _fix_edges(
                    pr,
                    available_issues=available_issues,
                    counters=counters,
                ):
                    counters[f"edge:{edge.kind.value}"] += 1
                    yield edge
            for revert in _revert_edges(
                pr,
                available_prs=available_prs,
                counters=counters,
            ):
                counters[f"edge:{revert.kind.value}"] += 1
                yield revert
        if not fold_issues_into_pr:
            for issue in issue_records:
                for edge in linker.link_issue(issue):
                    counters[f"edge:{edge.kind.value}"] += 1
                    yield edge

    imported_edges = graph.bulk_add_edges(edges())
    index_report = graph.rebuild_index()
    sqlite_counts = _sqlite_counts(graph.sqlite_path)
    gates = {
        "all_prs_imported": (
            imported_prs == len(pr_paths) - len(oversized_prs)
        ),
        "all_issues_imported": (
            fold_issues_into_pr or imported_issues == len(issue_paths)
        ),
        "all_landings_imported": imported_landings == len(landings),
        "no_invalid_fix_locators": counters["invalid_fix_locator"] == 0,
        "index_checksum_ok": index_report.checksum_ok,
        "no_dangling_references": not graph.last_rebuild_errors,
    }
    report = {
        "schema_version": "entity_graph_build_report_v1",
        "built_at": dt.datetime.now(dt.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "corpus_root": str(corpus_root),
        "corpus_digest": f"sha256:{digest.hexdigest()}",
        "graph_root": str(graph_root),
        "registry": snapshot.to_manifest(),
        "fanout_limit": fanout_limit,
        "imports": {
            "entities": imported_entities,
            "prs": imported_prs,
            "issues": imported_issues,
            "landings": imported_landings,
            "edges": imported_edges,
        },
        "linking": {
            "operator_linked_prs": counters["linked_prs"],
            "operator_linked_pr_coverage": (
                counters["linked_prs"] / len(pr_paths)
            ),
            "operator_touch_edges": counters["operator_touch_edges"],
            "broad_change_prs": counters["broad_change_prs"],
            "skipped_oversized_prs": sorted(oversized_prs),
            "unknown_operator_token_count": sum(unknown_tokens.values()),
            "unknown_operator_tokens": dict(
                unknown_tokens.most_common(100)
            ),
            "relation_audit": {
                key: value
                for key, value in sorted(counters.items())
                if key.startswith("edge:")
                or key.endswith("_locator")
                or key.endswith("_corpus")
                or "link_evidence" in key
                or key == "linked_without_fix_semantics"
            },
        },
        "index": {
            "records": index_report.records,
            "rebuilt_at": index_report.rebuilt_at,
            "counts": sqlite_counts,
            "errors": list(graph.last_rebuild_errors),
        },
        "gates": {**gates, "passed": all(gates.values())},
    }
    (staging / "registry_manifest.json").write_text(
        canonical_json(snapshot.to_manifest()) + "\n"
    )
    (staging / "build_report.json").write_text(
        canonical_json(report) + "\n"
    )
    if not report["gates"]["passed"]:
        raise GraphBuildError(
            f"graph gates failed; staging data retained at {staging}"
        )

    if graph_root.exists():
        stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        backup = graph_root.with_name(f"{graph_root.name}.backup-{stamp}")
        if backup.exists():
            raise FileExistsError(f"backup path already exists: {backup}")
        os.replace(graph_root, backup)
    os.replace(staging, graph_root)
    return report
