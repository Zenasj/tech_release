"""Bounded, locator-preserving evidence bundles for one graph entity."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Literal

from mrminer.s1_miner.issue_schema import IssueCorpusV1
from mrminer.s1_miner.pr_schema import IssueFetchStatus, PrCorpusV2, TextSource
from mrminer.s2_graph import EntityGraph, GraphEdge, GraphEdgeKind

_CHAR_LOCATOR = re.compile(
    r"^(?P<source>.+)@chars (?P<start>\d+)-(?P<end>\d+)$"
)


@dataclass(frozen=True, kw_only=True)
class EvidenceBudget:
    max_prs: int = 100
    max_sources_per_pr: int = 32
    max_chars_per_source: int = 6_000
    max_total_chars_per_pr: int = 80_000
    max_issue_comments_per_issue: int = 8
    max_graph_evidence_per_pr: int = 64
    context_chars: int = 800

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.max_total_chars_per_pr < self.max_chars_per_source:
            raise ValueError(
                "max_total_chars_per_pr cannot be smaller than "
                "max_chars_per_source"
            )


@dataclass(frozen=True, kw_only=True)
class EvidenceSource:
    key: str
    artifact_ref: str
    source_id: str
    source_kind: Literal["prose", "patch", "test"]
    text: str
    char_start: int
    char_end: int
    source_truncated: bool

    def __post_init__(self) -> None:
        if not self.key or not self.artifact_ref or not self.source_id:
            raise ValueError("evidence source identity fields must be non-empty")
        if self.char_start < 0 or self.char_end <= self.char_start:
            raise ValueError("evidence source character range is invalid")
        if self.char_end - self.char_start != len(self.text):
            raise ValueError("evidence source range must match retained text length")

    def to_record(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "artifact_ref": self.artifact_ref,
            "source_id": self.source_id,
            "source_kind": self.source_kind,
            "text": self.text,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "source_truncated": self.source_truncated,
        }


@dataclass(frozen=True, kw_only=True)
class PrEvidenceBundle:
    entity_id: str
    pr_number: int
    title: str
    merged: bool
    merged_at: str | None
    change_kind: str
    subsystem_labels: tuple[str, ...]
    sources: tuple[EvidenceSource, ...]
    graph_evidence: tuple[dict[str, Any], ...]
    total_source_count: int
    omitted_source_count: int
    total_graph_evidence_count: int = 0
    omitted_graph_evidence_count: int = 0

    def source_map(self) -> dict[str, EvidenceSource]:
        return {source.key: source for source in self.sources}

    def to_record(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "pr_number": self.pr_number,
            "title": self.title,
            "merged": self.merged,
            "merged_at": self.merged_at,
            "change_kind": self.change_kind,
            "subsystem_labels": list(self.subsystem_labels),
            "sources": [source.to_record() for source in self.sources],
            "graph_evidence": list(self.graph_evidence),
            "total_source_count": self.total_source_count,
            "omitted_source_count": self.omitted_source_count,
            "total_graph_evidence_count": self.total_graph_evidence_count,
            "omitted_graph_evidence_count": self.omitted_graph_evidence_count,
        }

    def to_prompt_payload(self) -> dict[str, Any]:
        return self.to_record()


@dataclass(frozen=True, kw_only=True)
class OperatorEvidenceBundle:
    entity: dict[str, Any]
    prs: tuple[PrEvidenceBundle, ...]
    total_pr_count: int
    omitted_pr_count: int
    budget: EvidenceBudget

    def to_record(self) -> dict[str, Any]:
        return {
            "entity": self.entity,
            "prs": [pr.to_record() for pr in self.prs],
            "total_pr_count": self.total_pr_count,
            "omitted_pr_count": self.omitted_pr_count,
            "budget": vars(self.budget),
        }


def _window(
    *,
    artifact_ref: str,
    source_id: str,
    source_kind: Literal["prose", "patch", "test"],
    text: str,
    start: int,
    end: int,
    original_truncated: bool,
) -> EvidenceSource:
    retained = text[start:end]
    key = f"{artifact_ref}/{source_id}@{start}-{end}"
    return EvidenceSource(
        key=key,
        artifact_ref=artifact_ref,
        source_id=source_id,
        source_kind=source_kind,
        text=retained,
        char_start=start,
        char_end=end,
        source_truncated=original_truncated or start > 0 or end < len(text),
    )


def _front_window(
    *,
    artifact_ref: str,
    source_id: str,
    source_kind: Literal["prose", "patch", "test"],
    text: str,
    max_chars: int,
    original_truncated: bool = False,
) -> EvidenceSource | None:
    if not text:
        return None
    return _window(
        artifact_ref=artifact_ref,
        source_id=source_id,
        source_kind=source_kind,
        text=text,
        start=0,
        end=min(len(text), max_chars),
        original_truncated=original_truncated,
    )


def _text_source_window(
    artifact_ref: str,
    source: TextSource,
    *,
    max_chars: int,
) -> EvidenceSource | None:
    return _front_window(
        artifact_ref=artifact_ref,
        source_id=source.source_id,
        source_kind="prose",
        text=source.text,
        max_chars=max_chars,
        original_truncated=source.truncation.value != "complete",
    )


def _context_windows(
    *,
    artifact_ref: str,
    source_id: str,
    source_kind: Literal["patch", "test"],
    text: str,
    intervals: set[tuple[int, int]],
    budget: EvidenceBudget,
) -> list[EvidenceSource]:
    windows: list[EvidenceSource] = []
    for evidence_start, evidence_end in sorted(intervals):
        if not 0 <= evidence_start < evidence_end <= len(text):
            continue
        extra = max(0, budget.max_chars_per_source - (evidence_end - evidence_start))
        left_context = min(budget.context_chars, extra // 2)
        start = max(0, evidence_start - left_context)
        end = min(len(text), start + budget.max_chars_per_source)
        if end < evidence_end:
            end = evidence_end
            start = max(0, end - budget.max_chars_per_source)
        windows.append(
            _window(
                artifact_ref=artifact_ref,
                source_id=source_id,
                source_kind=source_kind,
                text=text,
                start=start,
                end=end,
                original_truncated=False,
            )
        )
    return windows


class EvidenceRetriever:
    def __init__(
        self,
        graph: EntityGraph,
        *,
        corpus_root: Path | None = None,
        budget: EvidenceBudget | None = None,
    ):
        self.graph = graph
        self.corpus_root = corpus_root
        self.budget = budget or EvidenceBudget()

    def retrieve(self, entity_id: str, *, pr_numbers: set[int] | None = None) -> OperatorEvidenceBundle:
        entity = self.graph.entity(entity_id)
        relevant = [
            edge
            for edge in self.graph.timeline(entity_id)
            if edge.kind
            in {
                GraphEdgeKind.IMPLEMENTATION_TOUCHES,
                GraphEdgeKind.STATICALLY_MENTIONS,
            }
            and edge.source.key.startswith("pr:")
        ]
        by_pr: dict[int, list[GraphEdge]] = {}
        for edge in relevant:
            number = int(edge.source.key.removeprefix("pr:"))
            by_pr.setdefault(number, []).append(edge)
        if pr_numbers is not None:
            by_pr = {number: edges for number, edges in by_pr.items()
                     if number in pr_numbers}
        def priority(number: int) -> tuple[float, int]:
            edges = by_pr[number]
            focused = [edge for edge in edges
                       if edge.kind == GraphEdgeKind.IMPLEMENTATION_TOUCHES
                       and not edge.context.get("broad_change")]
            return (max((edge.confidence for edge in focused), default=0.0), number)
        all_numbers = sorted(by_pr, key=priority, reverse=True)
        retained_numbers = all_numbers[: self.budget.max_prs]
        bundles = tuple(
            self._pr_bundle(
                entity_id=entity_id,
                pr=self.graph.pr(number),
                edges=by_pr[number],
            )
            for number in retained_numbers
        )
        return OperatorEvidenceBundle(
            entity=entity.to_record(),
            prs=bundles,
            total_pr_count=len(all_numbers),
            omitted_pr_count=len(all_numbers) - len(retained_numbers),
            budget=self.budget,
        )

    def _pr_bundle(
        self,
        *,
        entity_id: str,
        pr: PrCorpusV2,
        edges: list[GraphEdge],
    ) -> PrEvidenceBundle:
        artifact_ref = f"pr:{pr.pr_number}"
        sources: list[EvidenceSource] = []
        title = _front_window(
            artifact_ref=artifact_ref,
            source_id="title",
            source_kind="prose",
            text=pr.title,
            max_chars=self.budget.max_chars_per_source,
        )
        body = _text_source_window(
            artifact_ref,
            pr.body,
            max_chars=self.budget.max_chars_per_source,
        )
        sources.extend(item for item in (title, body) if item is not None)
        intervals: dict[str, set[tuple[int, int]]] = {}
        for edge in edges:
            for evidence in edge.evidence:
                match = _CHAR_LOCATOR.fullmatch(evidence.locator)
                if match is None:
                    continue
                intervals.setdefault(match.group("source"), set()).add(
                    (int(match.group("start")), int(match.group("end")))
                )

        tests = {f"test:{item.test_node_id}": item.added_code for item in pr.tests_added}
        patches = {
            f"patch:{item.path}": item.patch
            for item in pr.touched_files_patch
            if item.patch is not None
        }
        for source_id in sorted(set(intervals) & set(tests)):
            sources.extend(
                _context_windows(
                    artifact_ref=artifact_ref,
                    source_id=source_id,
                    source_kind="test",
                    text=tests[source_id],
                    intervals=intervals[source_id],
                    budget=self.budget,
                )
            )
        for source_id in sorted(set(intervals) & set(patches)):
            sources.extend(
                _context_windows(
                    artifact_ref=artifact_ref,
                    source_id=source_id,
                    source_kind="patch",
                    text=patches[source_id],
                    intervals=intervals[source_id],
                    budget=self.budget,
                )
            )
        sources.extend(self._issue_sources(pr))
        sources.extend(
            item
            for source in pr.comments
            if (
                item := _text_source_window(
                    artifact_ref,
                    source,
                    max_chars=self.budget.max_chars_per_source,
                )
            )
            is not None
        )

        deduplicated = {source.key: source for source in sources}
        ordered = list(deduplicated.values())
        total_source_count = len(ordered)
        retained_items: list[EvidenceSource] = []
        retained_chars = 0
        for source in ordered:
            if len(retained_items) >= self.budget.max_sources_per_pr:
                break
            if (
                retained_chars + len(source.text)
                > self.budget.max_total_chars_per_pr
            ):
                continue
            retained_items.append(source)
            retained_chars += len(source.text)
        retained = tuple(retained_items)
        all_graph_evidence = tuple(
            evidence.to_record()
            for edge in sorted(edges, key=lambda item: item.edge_id)
            for evidence in edge.evidence
        )
        graph_evidence = all_graph_evidence[
            : self.budget.max_graph_evidence_per_pr
        ]
        return PrEvidenceBundle(
            entity_id=entity_id,
            pr_number=pr.pr_number,
            title=pr.title,
            merged=pr.merged,
            merged_at=pr.merged_at,
            change_kind=pr.change_kind,
            subsystem_labels=pr.subsystem_labels,
            sources=retained,
            graph_evidence=graph_evidence,
            total_source_count=total_source_count,
            omitted_source_count=total_source_count - len(retained),
            total_graph_evidence_count=len(all_graph_evidence),
            omitted_graph_evidence_count=(
                len(all_graph_evidence) - len(graph_evidence)
            ),
        )

    def _issue_sources(self, pr: PrCorpusV2) -> list[EvidenceSource]:
        if self.corpus_root is None:
            return []
        sources: list[EvidenceSource] = []
        for linked in pr.linked_issues:
            if linked.fetch_status is not IssueFetchStatus.STORED:
                continue
            issue = self._load_issue(pr.repository, linked.number)
            if issue is None:
                continue
            artifact_ref = f"issue:{issue.issue_number}"
            body = _text_source_window(
                artifact_ref,
                issue.body,
                max_chars=self.budget.max_chars_per_source,
            )
            if body is not None:
                sources.append(body)
            for comment in issue.comments.items[
                : self.budget.max_issue_comments_per_issue
            ]:
                window = _text_source_window(
                    artifact_ref,
                    comment,
                    max_chars=self.budget.max_chars_per_source,
                )
                if window is not None:
                    sources.append(window)
        return sources

    def _load_issue(
        self,
        repository: str,
        issue_number: int,
    ) -> IssueCorpusV1 | None:
        assert self.corpus_root is not None
        repository_dir = self.corpus_root / repository.replace("/", "__")
        if not repository_dir.is_dir() and self.corpus_root.name == repository.replace(
            "/", "__"
        ):
            repository_dir = self.corpus_root
        path = repository_dir / "issues" / f"{issue_number}.json"
        if not path.is_file():
            return None
        return IssueCorpusV1.from_record(json.loads(path.read_bytes()))
