"""Derive auditable pytest ProgramManifest candidates from graph test links."""

from __future__ import annotations

import ast
import re
import textwrap
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

from mrminer.common.canonical import content_id
from mrminer.common.schema import (
    ChangeLanding,
    InputInterface,
    LandingKind,
    ProgramKind,
    ProgramManifest,
    ProgramPhase,
    ProgramSourceRef,
    ProvenanceAnnotation,
    ProvenanceOrigin,
)
from mrminer.common.storage import ContentAddressedFileStore, JsonlStore
from mrminer.s1_miner.pr_schema import AddedTest, PrCorpusV2
from mrminer.s2_graph import EntityGraph, GraphEdge
from mrminer.s3_oracle.artifact_resolution import GitBlobRead, GitBlobStatus


class SeedReadiness(str, Enum):
    SOURCE_VALIDATED = "source_validated"
    NEEDS_SOURCE_VALIDATION = "needs_source_validation"
    SELECTOR_UNRESOLVED = "selector_unresolved"
    MISSING_HEAD_COMMIT = "missing_head_commit"
    INVALID_GRAPH_LINK = "invalid_graph_link"
    TEST_CODE_UNRESOLVED = "test_code_unresolved"
    SOURCE_UNAVAILABLE = "source_unavailable"
    UNMERGED_PR = "unmerged_pr"


class SourceReader(Protocol):
    def read_text_file(self, commit: str, path: str) -> GitBlobRead:
        """Read an exact historical source blob or return a typed failure."""


class LandingLogReader(Protocol):
    def pr_landing_commits(self, pr_number: int) -> tuple[tuple[str, str], ...]:
        """Real landing commits (sha, subject) on HEAD's history for a PR."""


def resolve_real_landing_commits(
    pr_number: int,
    log_reader: LandingLogReader,
) -> tuple[str, ...]:
    """Recover a PR's true landing commit(s) via git log, reverts skipped.

    The graph frequently records the GitHub ``merge_commit_sha``, which for
    rebase/squash-merged PyTorch PRs is absent from real history (survey:
    98.4% of in-range landings). ``git log --grep "(#PR)"`` recovers the actual
    landing commit that is reachable from HEAD. git log is newest-first, so a
    reland precedes the reverted first-land; skipping pure ``Revert "..."``
    subjects leaves the live landing first.
    """
    ordered: list[str] = []
    seen: set[str] = set()
    for sha, subject in log_reader.pr_landing_commits(pr_number):
        if subject.lstrip().startswith('Revert "'):
            continue
        if sha not in seen:
            seen.add(sha)
            ordered.append(sha)
    return tuple(ordered)


@dataclass(frozen=True)
class SeedCheck:
    name: str
    status: str
    detail: str

    def to_record(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class TestCodeSlice:
    code: str
    definition_indent: int
    reconstructed_start_line: int | None
    reconstructed_end_line: int | None


@dataclass(frozen=True)
class DeviceInference:
    device_class: str
    source: str


@dataclass(frozen=True)
class PhaseInference:
    phase: ProgramPhase
    source: str


@dataclass(frozen=True)
class SourceCommitCandidate:
    commit: str
    source: str
    priority: int
    landed_at: str | None


def _landing_source_name(landing: ChangeLanding) -> str:
    extractors = {
        evidence.extractor
        for evidence in landing.evidence
    }
    if "s1.canonical_git_landing/v1" in extractors:
        fact_source = "canonical_git_landing"
    elif "s1.github_native_merge/v1" in extractors:
        fact_source = "github_native_merge"
    else:
        fact_source = "change_landing"
    return (
        f"{fact_source}:{landing.branch}:{landing.kind.value}"
    )


def source_commit_candidates(
    graph: EntityGraph,
    pr: PrCorpusV2,
    *,
    landing_log: LandingLogReader | None = None,
) -> tuple[SourceCommitCandidate, ...]:
    """Order exact source refs without treating a PR head as a landing.

    When ``landing_log`` is supplied, git-log-recovered real landing commits are
    tried first (they are reachable from HEAD, unlike the frequently-wrong recorded
    merge_commit_sha). Omitting it preserves the exact prior behaviour.
    """

    options: list[SourceCommitCandidate] = []
    if landing_log is not None:
        for commit in resolve_real_landing_commits(pr.pr_number, landing_log):
            options.append(
                SourceCommitCandidate(
                    commit=commit,
                    source="git_log_landing",
                    priority=-1,
                    landed_at=None,
                )
            )
    landings = sorted(
        graph.landings(pr_number=pr.pr_number),
        key=lambda landing: (
            0 if landing.branch == "main" else 1,
            0 if landing.kind is LandingKind.MERGE else 1,
            landing.landed_at,
            landing.branch,
            landing.landed_commits,
        ),
    )
    for landing in landings:
        source = _landing_source_name(landing)
        for commit in landing.landed_commits:
            options.append(
                SourceCommitCandidate(
                    commit=commit,
                    source=source,
                    priority=(
                        0 if landing.branch == "main" else 1
                    ),
                    landed_at=landing.landed_at,
                )
            )
    if pr.commit_refs.head_sha is not None:
        options.append(
            SourceCommitCandidate(
                commit=pr.commit_refs.head_sha,
                source="pr_head",
                priority=2,
                landed_at=None,
            )
        )
    unique: list[SourceCommitCandidate] = []
    seen: set[str] = set()
    for option in options:
        if option.commit in seen:
            continue
        seen.add(option.commit)
        unique.append(option)
    return tuple(unique)


@dataclass(frozen=True)
class ProgramSeedCandidate:
    candidate_id: str
    entity_id: str
    pr_number: int
    edge_id: str
    test_index: int | None
    evidence_test_node_id: str | None
    pytest_node_selector: str | None
    test_code: str | None
    reconstructed_span: tuple[int, int] | None
    source_commit: str | None
    source_commit_source: str | None
    device_class: str
    device_inference: str
    phase: str
    phase_inference: str
    readiness: SeedReadiness
    checks: tuple[SeedCheck, ...]
    manifest: ProgramManifest | None

    @classmethod
    def create(
        cls,
        *,
        entity_id: str,
        pr_number: int,
        edge_id: str,
        test_index: int | None,
        evidence_test_node_id: str | None,
        pytest_node_selector: str | None,
        test_code: str | None,
        reconstructed_span: tuple[int, int] | None,
        source_commit: str | None,
        source_commit_source: str | None,
        device: DeviceInference,
        phase: PhaseInference,
        readiness: SeedReadiness,
        checks: tuple[SeedCheck, ...],
        manifest: ProgramManifest | None,
    ) -> ProgramSeedCandidate:
        payload = {
            "entity_id": entity_id,
            "pr_number": pr_number,
            "edge_id": edge_id,
            "test_index": test_index,
            "evidence_test_node_id": evidence_test_node_id,
            "pytest_node_selector": pytest_node_selector,
            "test_code": test_code,
            "reconstructed_span": (
                list(reconstructed_span)
                if reconstructed_span is not None
                else None
            ),
            "source_commit": source_commit,
            "source_commit_source": source_commit_source,
            "device_class": device.device_class,
            "device_inference": device.source,
            "phase": phase.phase.value,
            "phase_inference": phase.source,
            "readiness": readiness.value,
            "checks": [check.to_record() for check in checks],
            "manifest": manifest.to_record() if manifest is not None else None,
        }
        return cls(
            candidate_id=content_id("seed", payload),
            **{
                **payload,
                "reconstructed_span": reconstructed_span,
                "source_commit": source_commit,
                "source_commit_source": source_commit_source,
                "device_class": device.device_class,
                "device_inference": device.source,
                "phase": phase.phase.value,
                "phase_inference": phase.source,
                "readiness": readiness,
                "checks": checks,
                "manifest": manifest,
            },
        )

    def to_record(self) -> dict:
        return {
            "seed_version": "program_seed_v2",
            "candidate_id": self.candidate_id,
            "entity_id": self.entity_id,
            "pr_number": self.pr_number,
            "edge_id": self.edge_id,
            "test_index": self.test_index,
            "evidence_test_node_id": self.evidence_test_node_id,
            "pytest_node_selector": self.pytest_node_selector,
            "test_code": self.test_code,
            "reconstructed_span": (
                list(self.reconstructed_span)
                if self.reconstructed_span is not None
                else None
            ),
            "source_commit": self.source_commit,
            "source_commit_source": self.source_commit_source,
            "device_class": self.device_class,
            "device_inference": self.device_inference,
            "phase": self.phase,
            "phase_inference": self.phase_inference,
            "readiness": self.readiness.value,
            "checks": [check.to_record() for check in self.checks],
            "manifest": (
                self.manifest.to_record()
                if self.manifest is not None
                else None
            ),
        }


@dataclass(frozen=True)
class ProgramSeedPersistReport:
    candidate_paths: tuple[str, ...]
    manifest_paths: tuple[str, ...]
    annotations_appended: int
    skipped_manifest_candidate_ids: tuple[str, ...]

    def to_record(self) -> dict:
        return {
            "candidate_paths": list(self.candidate_paths),
            "manifest_paths": list(self.manifest_paths),
            "annotations_appended": self.annotations_appended,
            "skipped_manifest_candidate_ids": list(
                self.skipped_manifest_candidate_ids
            ),
        }


_TEST_DEF = re.compile(
    r"^(?P<indent>[ \t]*)(?:async\s+)?def\s+(?P<name>test[A-Za-z0-9_]*)\s*\("
)
_CUDA = re.compile(
    r"(?:torch\.cuda|\.cuda\s*\(|device\s*=\s*[\"']cuda|[\"']cuda(?::\d+)?[\"'])"
)
_CPU = re.compile(r"(?:\.cpu\s*\(|device\s*=\s*[\"']cpu[\"']|[\"']cpu[\"'])")
_XPU = re.compile(
    r"(?:torch\.xpu|\.xpu\s*\(|device\s*=\s*[\"']xpu|[\"']xpu(?::\d+)?[\"'])"
)
_BACKWARD = re.compile(
    r"(?:\.backward\s*\(|\bbackward\b|\bgrad(?:ient)?s?\b|aot_autograd)"
)
_TRAIN = re.compile(r"(?:optimizer\.|zero_grad\s*\(|train_step|\btraining\b)")


def _leading_width(value: str) -> int:
    return len(value.expandtabs(8)) - len(value.lstrip(" \t").expandtabs(8))


def extract_test_code(test: AddedTest) -> TestCodeSlice | None:
    lines = test.added_code.splitlines()
    matches: list[tuple[int, int]] = []
    for index, line in enumerate(lines):
        match = _TEST_DEF.match(line)
        if match is not None and match.group("name") == test.test_name:
            matches.append((index, _leading_width(match.group("indent"))))
    if len(matches) != 1:
        return None
    start, indent = matches[0]
    end = len(lines)
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if not line.strip():
            continue
        if _leading_width(line) <= indent:
            end = index
            break
    code = "\n".join(lines[start:end]).rstrip() + "\n"
    start_line = test.def_span.start_line
    end_line = (
        start_line + end - start - 1
        if start_line is not None
        else None
    )
    return TestCodeSlice(
        code=code,
        definition_indent=indent,
        reconstructed_start_line=start_line,
        reconstructed_end_line=end_line,
    )


def infer_device(code: str) -> DeviceInference:
    excluded = _skip_guard_devices(code)
    matches = [
        name
        for name, pattern in (
            ("cuda", _CUDA),
            ("cpu", _CPU),
            ("xpu", _XPU),
        )
        if pattern.search(code) and name not in excluded
    ]
    if len(matches) == 1:
        return DeviceInference(matches[0], f"explicit_{matches[0]}_token")
    if len(matches) > 1:
        return DeviceInference("multi_device", "multiple_explicit_device_tokens")
    if excluded:
        devices = "_".join(sorted(excluded))
        return DeviceInference(
            "unknown",
            f"explicit_{devices}_skip_guard_is_not_a_requirement",
        )
    return DeviceInference("unknown", "no_explicit_device_token")


def _skip_guard_devices(code: str) -> set[str]:
    """Return device literals used only as explicit test-exclusion guards."""

    try:
        tree = ast.parse(textwrap.dedent(code))
    except SyntaxError:
        return set()

    def compared_device(node: ast.expr) -> str | None:
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            return None
        if not isinstance(node.ops[0], (ast.Eq, ast.Is)):
            return None
        values = (node.left, *node.comparators)
        has_device_ref = any(
            isinstance(value, ast.Name) and value.id == "device"
            or isinstance(value, ast.Attribute) and value.attr == "device"
            for value in values
        )
        if not has_device_ref:
            return None
        literals = {
            value.value
            for value in values
            if isinstance(value, ast.Constant)
            and isinstance(value.value, str)
            and value.value in {"cpu", "cuda", "xpu"}
        }
        return next(iter(literals)) if len(literals) == 1 else None

    def is_skip_call(node: ast.AST) -> bool:
        if not isinstance(node, ast.Call):
            return False
        target = node.func
        return (
            isinstance(target, ast.Name) and target.id in {"SkipTest", "skip"}
            or isinstance(target, ast.Attribute)
            and target.attr in {"SkipTest", "skip", "skipTest"}
        )

    excluded: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        device = compared_device(node.test)
        if device is None:
            continue
        skips = any(
            isinstance(child, ast.Raise)
            and child.exc is not None
            and is_skip_call(child.exc)
            or isinstance(child, ast.Expr) and is_skip_call(child.value)
            for statement in node.body
            for child in ast.walk(statement)
        )
        if skips:
            excluded.add(device)
    return excluded


def infer_phase(code: str) -> PhaseInference:
    if _TRAIN.search(code):
        return PhaseInference(ProgramPhase.TRAIN_STEP, "explicit_training_token")
    if _BACKWARD.search(code):
        return PhaseInference(ProgramPhase.BACKWARD, "explicit_backward_token")
    return PhaseInference(ProgramPhase.FORWARD, "default_forward_no_phase_token")


def _selector_from_source(
    source: str,
    *,
    test_name: str,
    expected_line: int | None,
) -> str | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    matches: list[tuple[int, str]] = []

    def visit(nodes: list[ast.stmt], classes: tuple[str, ...]) -> None:
        for node in nodes:
            if isinstance(node, ast.ClassDef):
                visit(node.body, (*classes, node.name))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == test_name:
                    matches.append((node.lineno, "::".join((*classes, node.name))))

    visit(tree.body, ())
    selected: str | None = None
    if expected_line is not None:
        exact = [selector for line, selector in matches if line == expected_line]
        if len(exact) == 1:
            selected = exact[0]
    if selected is None and len(matches) == 1:
        selected = matches[0][1]
    if selected is None or "::" not in selected:
        return selected
    root_class = selected.split("::", 1)[0]
    dynamic_factories = {
        "instantiate_device_type_tests",
        "instantiate_parametrized_tests",
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        function_name = (
            node.func.id
            if isinstance(node.func, ast.Name)
            else (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else None
            )
        )
        class_argument = node.args[0]
        if (
            function_name in dynamic_factories
            and isinstance(class_argument, ast.Name)
            and class_argument.id == root_class
        ):
            return None
    return selected


class ProgramSeedMiner:
    def __init__(
        self,
        graph: EntityGraph,
        *,
        source_reader: SourceReader | None = None,
    ):
        self.graph = graph
        self.source_reader = source_reader
        # reuse the source reader as the landing-log reader when it can search git
        # history (SubprocessGitAncestry), so recorded-landing gaps self-heal.
        self.landing_log: LandingLogReader | None = (
            source_reader
            if hasattr(source_reader, "pr_landing_commits")
            else None
        )

    def _invalid(
        self,
        *,
        entity_id: str,
        pr: PrCorpusV2,
        edge: GraphEdge,
        test_index: int | None,
        evidence_test_node_id: str | None,
        readiness: SeedReadiness,
        detail: str,
    ) -> ProgramSeedCandidate:
        return ProgramSeedCandidate.create(
            entity_id=entity_id,
            pr_number=pr.pr_number,
            edge_id=edge.edge_id,
            test_index=test_index,
            evidence_test_node_id=evidence_test_node_id,
            pytest_node_selector=None,
            test_code=None,
            reconstructed_span=None,
            source_commit=None,
            source_commit_source=None,
            device=DeviceInference("unknown", "test_code_unavailable"),
            phase=PhaseInference(
                ProgramPhase.FORWARD,
                "test_code_unavailable",
            ),
            readiness=readiness,
            checks=(SeedCheck("candidate", "failed", detail),),
            manifest=None,
        )

    def _mine_link(
        self,
        entity_id: str,
        pr: PrCorpusV2,
        edge: GraphEdge,
    ) -> ProgramSeedCandidate:
        raw_index = edge.context.get("test_index")
        evidence_node = edge.context.get("test_node_id")
        if (
            type(raw_index) is not int
            or not 0 <= raw_index < len(pr.tests_added)
            or not isinstance(evidence_node, str)
        ):
            return self._invalid(
                entity_id=entity_id,
                pr=pr,
                edge=edge,
                test_index=raw_index if type(raw_index) is int else None,
                evidence_test_node_id=(
                    evidence_node if isinstance(evidence_node, str) else None
                ),
                readiness=SeedReadiness.INVALID_GRAPH_LINK,
                detail="edge test_index/test_node_id does not address tests_added",
            )
        test = pr.tests_added[raw_index]
        if test.test_node_id != evidence_node:
            return self._invalid(
                entity_id=entity_id,
                pr=pr,
                edge=edge,
                test_index=raw_index,
                evidence_test_node_id=evidence_node,
                readiness=SeedReadiness.INVALID_GRAPH_LINK,
                detail="edge test_node_id disagrees with tests_added[test_index]",
            )
        sliced = extract_test_code(test)
        if sliced is None:
            return self._invalid(
                entity_id=entity_id,
                pr=pr,
                edge=edge,
                test_index=raw_index,
                evidence_test_node_id=evidence_node,
                readiness=SeedReadiness.TEST_CODE_UNRESOLVED,
                detail="added_code does not contain one exact matching test definition",
            )

        checks = [
            SeedCheck(
                "graph_link",
                "passed",
                "edge exactly addresses tests_added[test_index]",
            ),
            SeedCheck(
                "test_code_slice",
                "passed",
                "test body reconstructed from its definition indentation",
            ),
        ]
        selector = test.test_name if sliced.definition_indent == 0 else None
        source_options = source_commit_candidates(
            self.graph, pr, landing_log=self.landing_log
        )
        selected_source = source_options[0] if source_options else None
        source_ast_available = False
        if self.source_reader is not None:
            selected_source = None
            for option in source_options:
                blob = self.source_reader.read_text_file(
                    option.commit,
                    test.path,
                )
                if blob.status is not GitBlobStatus.OK or blob.text is None:
                    checks.append(
                        SeedCheck(
                            "historical_source_candidate",
                            "unavailable",
                            (
                                f"{option.source}@{option.commit}: "
                                f"{blob.detail}"
                            ),
                        )
                    )
                    continue
                source_ast_available = True
                resolved_selector = _selector_from_source(
                    blob.text,
                    test_name=test.test_name,
                    expected_line=test.def_span.start_line,
                )
                checks.append(
                    SeedCheck(
                        "historical_source_candidate",
                        (
                            "passed"
                            if resolved_selector is not None
                            else "failed"
                        ),
                        (
                            f"{option.source}@{option.commit}: "
                            "pytest selector resolved from exact source AST"
                            if resolved_selector is not None
                            else (
                                f"{option.source}@{option.commit}: test "
                                "definition was not unique"
                            )
                        ),
                    )
                )
                if resolved_selector is None:
                    continue
                selector = resolved_selector
                selected_source = option
                break
        else:
            checks.append(
                SeedCheck(
                    "historical_source",
                    "not_checked",
                    "no source reader was configured",
                )
            )

        device = infer_device(sliced.code)
        phase = infer_phase(sliced.code)
        if not pr.merged:
            readiness = SeedReadiness.UNMERGED_PR
        elif selected_source is None and not source_options:
            readiness = SeedReadiness.MISSING_HEAD_COMMIT
        elif (
            self.source_reader is not None
            and selected_source is not None
            and selector is not None
        ):
            readiness = SeedReadiness.SOURCE_VALIDATED
        elif self.source_reader is not None and source_ast_available:
            readiness = SeedReadiness.SELECTOR_UNRESOLVED
        elif self.source_reader is not None:
            readiness = SeedReadiness.SOURCE_UNAVAILABLE
        elif selector is None:
            readiness = SeedReadiness.SELECTOR_UNRESOLVED
        else:
            readiness = SeedReadiness.NEEDS_SOURCE_VALIDATION

        manifest = None
        if selected_source is not None and selector is not None and pr.merged:
            manifest = ProgramManifest.create(
                program_kind=ProgramKind.PYTEST_NODE,
                source_ref=ProgramSourceRef(
                    repo=pr.repository,
                    commit=selected_source.commit,
                    path=test.path,
                    node_id=selector,
                ),
                entrypoint=None,
                dependencies={
                    "pytest": "environment-provided",
                    "torch": "artifact-provided",
                },
                input_interface=InputInterface(arity=0, roles=()),
                required_device_class=device.device_class,
                phase=phase.phase,
                bound_entities=(entity_id,),
            )
        checks.extend(
            (
                SeedCheck(
                    "device_inference",
                    (
                        "passed"
                        if device.device_class != "unknown"
                        else "unresolved"
                    ),
                    device.source,
                ),
                SeedCheck("phase_inference", "passed", phase.source),
            )
        )
        return ProgramSeedCandidate.create(
            entity_id=entity_id,
            pr_number=pr.pr_number,
            edge_id=edge.edge_id,
            test_index=raw_index,
            evidence_test_node_id=evidence_node,
            pytest_node_selector=selector,
            test_code=sliced.code,
            reconstructed_span=(
                (
                    sliced.reconstructed_start_line,
                    sliced.reconstructed_end_line,
                )
                if sliced.reconstructed_start_line is not None
                and sliced.reconstructed_end_line is not None
                else None
            ),
            source_commit=(
                selected_source.commit
                if selected_source is not None
                else None
            ),
            source_commit_source=(
                selected_source.source
                if selected_source is not None
                else None
            ),
            device=device,
            phase=phase,
            readiness=readiness,
            checks=tuple(checks),
            manifest=manifest,
        )

    def mine(
        self,
        entity_id: str,
        *,
        links: list[tuple[PrCorpusV2, GraphEdge]] | None = None,
        include_unmerged: bool = False,
    ) -> tuple[ProgramSeedCandidate, ...]:
        self.graph.entity(entity_id)
        selected_links = (
            self.graph.static_test_mentions(entity_id)
            if links is None
            else links
        )
        if not include_unmerged:
            selected_links = [
                (pr, edge) for pr, edge in selected_links if pr.merged
            ]
        candidates = [
            self._mine_link(entity_id, pr, edge)
            for pr, edge in selected_links
        ]
        by_id = {candidate.candidate_id: candidate for candidate in candidates}
        return tuple(
            sorted(
                by_id.values(),
                key=lambda candidate: (
                    candidate.pr_number,
                    candidate.test_index
                    if candidate.test_index is not None
                    else -1,
                    candidate.candidate_id,
                ),
            )
        )


def persist_program_seeds(
    candidates: tuple[ProgramSeedCandidate, ...],
    *,
    output_root: Path,
    recorded_at: str,
) -> ProgramSeedPersistReport:
    candidate_store = ContentAddressedFileStore(
        output_root / "candidates",
        id_field="candidate_id",
    )
    manifest_store = ContentAddressedFileStore(
        output_root / "manifests",
        id_field="program_id",
    )
    annotations = JsonlStore(output_root / "annotations.jsonl")
    candidate_paths: list[str] = []
    manifest_paths: set[str] = set()
    skipped_manifest_candidate_ids: list[str] = []
    annotations_appended = 0
    for candidate in candidates:
        candidate_paths.append(
            str(candidate_store.put(candidate.to_record()))
        )
        if candidate.manifest is None:
            continue
        if candidate.readiness is not SeedReadiness.SOURCE_VALIDATED:
            skipped_manifest_candidate_ids.append(candidate.candidate_id)
            continue
        manifest_paths.add(
            str(manifest_store.put(candidate.manifest.to_record()))
        )
        annotation = ProvenanceAnnotation(
            program_id=candidate.manifest.program_id,
            origin=ProvenanceOrigin.PR_REGRESSION_TEST,
            origin_ref=f"pr:{candidate.pr_number}",
            recorded_at=recorded_at,
        )
        annotations_appended += int(
            annotations.append(
                annotation.to_record(),
                idempotency_key=content_id(
                    "program_provenance",
                    annotation.to_record(),
                ),
            )
        )
    return ProgramSeedPersistReport(
        candidate_paths=tuple(sorted(candidate_paths)),
        manifest_paths=tuple(sorted(manifest_paths)),
        annotations_appended=annotations_appended,
        skipped_manifest_candidate_ids=tuple(
            sorted(skipped_manifest_candidate_ids)
        ),
    )
