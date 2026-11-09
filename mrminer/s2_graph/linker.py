"""Deterministic PR/Issue to entity linking with exact source locators."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import re
from typing import Any, Iterable

from mrminer.common.canonical import sha256_digest
from mrminer.common.schema import (
    EntityKind,
    EntitySource,
    EntitySpec,
    EvidenceSpec,
    EvidenceStrength,
    SubsystemIntrinsic,
)
from mrminer.s1_miner.issue_schema import IssueCorpusV1
from mrminer.s1_miner.pr_schema import PrCorpusV2, TouchedFilePatch
from mrminer.s2_graph.api_resolution import resolve_aten_to_api
from mrminer.s2_graph.registry_loader import operator_entity_id
from mrminer.s2_graph.schema import (
    GraphEdge,
    GraphEdgeKind,
    GraphNodeKind,
    NodeRef,
)

_ATEN_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_])(?:torch\.ops\.)?aten(?:::|\.)"
    r"(?P<base>[A-Za-z_][A-Za-z0-9_]*)(?:\.[A-Za-z0-9_]+)?"
)
_FUNC_SCHEMA = re.compile(
    r"(?:^|\s)-\s*func:\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)?)"
)
_CPP_CALL = re.compile(
    r"\b(?P<name>[A-Za-z_][A-Za-z0-9_]{3,})"
    r"\s*(?:<[^;{}()]*>)?\s*\("
)
_TORCH_FUNC_MACRO = re.compile(
    r"\bTORCH_(?:IMPL|META)_FUNC(?:2)?\(\s*"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]{3,})"
)
_YAML_DISPATCH_VALUE = re.compile(
    r":\s*&?(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*$"
)
# Python operator-registration decorators (torch/_decomp, torch/_refs,
# torch/_meta_registrations, torch/_inductor/lowering): the operator association
# lives on the `@register_decomposition/meta/lowering(aten.X)` line, which is usually
# unchanged *context* around a change to the body it registers -- so the changed-line
# aten-token scan misses it. Match the decorator so the enclosing operator is still
# attributed. (register_lowering names 113 operators whose compiler work lives in
# inductor lowerings rather than native_functions.yaml / C++.)
_PY_OP_REGISTER = re.compile(
    r"(?:register_(?:decomposition|meta|lowering)|make_fallback)\s*\(")
# torch/_meta_registrations.py convention `def meta_<op>(...)`: catches meta PRs
# whose change is deep in the function body, past the `@register_meta(aten.X)`
# decorator (so the decorator line is outside the diff window and _PY_OP_REGISTER
# cannot see it), while the `def meta_<op>` signature is still shown as context.
_PY_META_FUNC = re.compile(r"\bdef\s+meta_(?P<op>[A-Za-z_][A-Za-z0-9_]*)\s*\(")
# In torch/_decomp, torch/_refs and torch/_inductor/{decomposition,lowering}.py a
# decomposition/lowering function is named after the operator it implements (`def mm`,
# `def addmm`). When a change sits deep in the body (past the decorator and def, so the
# only place the function name appears is the hunk's `@@ ... @@ def mm(` section
# heading), resolve the enclosing function to the operator -- but only as a *fallback*
# when the hunk produced no operator by any other signal, so a boundary heading naming a
# neighbouring function cannot override a real in-hunk decorator match.
_PY_DEF_HEADER = re.compile(r"\bdef\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)")


def _is_op_named_fn_file(path: str) -> bool:
    return (
        path.endswith(("decomposition.py", "decompositions.py", "lowering.py"))
        or "/_refs/" in path
    )
_CPP_SUFFIXES = frozenset({".c", ".cc", ".cpp", ".cxx", ".cu", ".h", ".hpp"})

# "affects" test signal (graph_model_v1 §2.1): public Python API references in
# test code. Modules are CamelCase after `nn.`; functionals are lowercase after
# `F.` / `functional.` / `torch.nn.functional.`.
_NN_MODULE_REF = re.compile(r"\bnn\.(?P<name>[A-Z][A-Za-z0-9_]*)")
_NN_FUNCTIONAL_REF = re.compile(
    r"(?:\bF\.|\bfunctional\.|\btorch\.nn\.functional\.)(?P<name>[a-z][A-Za-z0-9_]*)"
)
# Base container/utility classes are not operators.
_API_TEST_EXCLUDE = frozenset(
    {
        "Module",
        "Sequential",
        "ModuleList",
        "ModuleDict",
        "Parameter",
        "ParameterList",
        "ParameterDict",
        "Identity",
    }
)


@dataclass(frozen=True)
class LinkResult:
    edges: tuple[GraphEdge, ...]
    exact_operator_count: int
    broad_change: bool
    unknown_operator_tokens: tuple[str, ...]
    deferred: bool = False
    deferred_reason: str | None = None


def subsystem_entity(label: str, *, first_seen: str) -> EntitySpec:
    normalized = re.sub(r"[\s/]+", ".", label.strip()).strip(".")
    if not normalized:
        raise ValueError("subsystem label cannot normalize to an empty ID")
    return EntitySpec(
        entity_kind=EntityKind.SUBSYSTEM,
        entity_id=f"sys:{normalized}",
        display_name=label,
        intrinsic=SubsystemIntrinsic(label=label),
        first_seen=first_seen,
        source=EntitySource.LINKER,
    )


def is_implementation_path(path: str) -> bool:
    parts = path.split("/")
    filename = parts[-1]
    excluded_parts = {
        "test",
        "tests",
        "testing",
        "doc",
        "docs",
        "benchmark",
        "benchmarks",
        "example",
        "examples",
    }
    return not (
        any(part in excluded_parts for part in parts[:-1])
        or filename.startswith(("test_", "benchmark_"))
    )


def _evidence(
    *,
    artifact_ref: str,
    locator: str,
    quote: str,
    supports_field: str,
    extractor: str,
    strength: EvidenceStrength = EvidenceStrength.INFERRED,
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


def _edge_time(pr: PrCorpusV2) -> str:
    value = pr.merged_at or pr.closed_at or pr.collection.collected_at
    if value == "unknown":
        raise ValueError(f"PR #{pr.pr_number} has no usable edge timestamp")
    return value


def _changed_lines(patch: str) -> Iterable[tuple[int, str]]:
    offset = 0
    for line in patch.splitlines(keepends=True):
        if (
            line[:1] in {"+", "-"}
            and not line.startswith(("+++", "---"))
        ):
            yield offset + 1, line[1:].rstrip("\r\n")
        offset += len(line)


class DeterministicLinker:
    EXTRACTOR = "s2.deterministic_linker/v1"

    def __init__(
        self,
        *,
        operator_ids: Iterable[str],
        dispatch_symbols: dict[str, tuple[str, ...]] | None = None,
        fanout_limit: int = 15,
        api_ids: Iterable[str] | None = None,
        defer_limit: int | None = None,
    ):
        self.operator_ids = frozenset(operator_ids)
        self.dispatch_symbols = dispatch_symbols or {}
        self.fanout_limit = fanout_limit
        # When a PR touches more than defer_limit operators it is deferred: no
        # operator/affects/subsystem edges are emitted (graph_model_v1 §2.2), to
        # keep hub PRs from smearing the graph. None disables (backward compat).
        self.defer_limit = defer_limit
        # When an API registry is supplied, aten operator edges are retargeted
        # to the namesake public Python API (graph_model_v1 §5); empty means the
        # legacy aten-only behaviour (backward compatible).
        self.api_ids = frozenset(api_ids or ())
        # Short-name -> API operator ID maps, for matching Python API references
        # in test code (the "affects" test signal, graph_model_v1 §2.1).
        self._api_functional: dict[str, str] = {}
        self._api_module: dict[str, str] = {}
        for api_id in self.api_ids:
            qual = api_id.removeprefix("api:")
            short = qual.rsplit(".", 1)[-1]
            if short in _API_TEST_EXCLUDE:
                continue
            if qual.startswith("torch.nn.functional."):
                self._api_functional[short] = api_id
            elif qual.startswith("torch.nn."):
                self._api_module[short] = api_id

    def _operator_mentions(
        self,
        *,
        text: str,
        artifact_ref: str,
        locator_prefix: str,
        supports_field: str,
    ) -> tuple[dict[str, list[EvidenceSpec]], set[str]]:
        evidence: dict[str, list[EvidenceSpec]] = defaultdict(list)
        unknown: set[str] = set()
        for match in _ATEN_TOKEN.finditer(text):
            entity_id = operator_entity_id(match.group("base"))
            quote = match.group(0)
            if entity_id not in self.operator_ids:
                unknown.add(quote)
                continue
            evidence[entity_id].append(
                _evidence(
                    artifact_ref=artifact_ref,
                    locator=(
                        f"{locator_prefix}@chars "
                        f"{match.start()}-{match.end()}"
                    ),
                    quote=quote,
                    supports_field=supports_field,
                    extractor=self.EXTRACTOR,
                )
            )
        return evidence, unknown

    def _api_test_mentions(
        self,
        *,
        text: str,
        artifact_ref: str,
        locator_prefix: str,
    ) -> dict[str, list[EvidenceSpec]]:
        """Match public Python API references in test code to API operators.

        This is the "affects" test signal: a PR whose tests exercise
        ``nn.Linear`` / ``F.relu`` is telling us it affects those operators, even
        when its diff touches no aten symbol (graph_model_v1 §2.1).
        """

        evidence: dict[str, list[EvidenceSpec]] = defaultdict(list)
        if not self._api_functional and not self._api_module:
            return evidence
        for pattern, table in (
            (_NN_MODULE_REF, self._api_module),
            (_NN_FUNCTIONAL_REF, self._api_functional),
        ):
            for match in pattern.finditer(text):
                api_id = table.get(match.group("name"))
                if api_id is None:
                    continue
                evidence[api_id].append(
                    _evidence(
                        artifact_ref=artifact_ref,
                        locator=(
                            f"{locator_prefix}@chars "
                            f"{match.start()}-{match.end()}"
                        ),
                        quote=match.group(0),
                        supports_field="statically_mentions",
                        extractor=self.EXTRACTOR,
                    )
                )
        return evidence

    def _patch_mentions(
        self,
        patch_record: TouchedFilePatch,
        *,
        pr_number: int,
    ) -> tuple[dict[str, list[EvidenceSpec]], set[str], dict[str, set[str]]]:
        patch = patch_record.patch
        if patch is None or not is_implementation_path(patch_record.path):
            return {}, set(), {}
        artifact_ref = f"pr:{pr_number}"
        evidence: dict[str, list[EvidenceSpec]] = defaultdict(list)
        signals: dict[str, set[str]] = defaultdict(set)
        unknown: set[str] = set()
        current_native: tuple[str, int, int] | None = None
        current_cpp_owners: tuple[tuple[str, int, int], ...] = ()

        offset = 0
        hunk_hdr: tuple[str, int] | None = None   # (enclosing fn name, char start)
        hunk_base = 0                             # evidence-item count at hunk start

        def _ev_count() -> int:
            return sum(len(items) for items in evidence.values())

        def _finalize_hunk() -> None:
            # enclosing-fn fallback: only if the hunk matched no operator otherwise
            if hunk_hdr is None or _ev_count() != hunk_base:
                return
            name, hstart = hunk_hdr
            entity_id = operator_entity_id(name)
            if entity_id not in self.operator_ids:
                return
            evidence[entity_id].append(
                _evidence(
                    artifact_ref=artifact_ref,
                    locator=f"patch:{patch_record.path}@chars {hstart}-{hstart + len(name)}",
                    quote=f"def {name}",
                    supports_field="implementation_touches",
                    extractor=self.EXTRACTOR,
                )
            )
            signals[entity_id].add("enclosing_fn")

        for raw_line in patch.splitlines(keepends=True):
            if raw_line.startswith("@@"):
                _finalize_hunk()
                current_native = None
                header = raw_line.rstrip("\r\n")
                hunk_hdr = None
                if _is_op_named_fn_file(patch_record.path):
                    heading = header.split("@@")[-1]
                    fn = _PY_DEF_HEADER.search(heading)
                    if fn is not None:
                        hstart = offset + header.rfind(heading) + fn.start("name")
                        hunk_hdr = (fn.group("name"), hstart)
                hunk_base = _ev_count()
                current_cpp_owners = tuple(
                    (
                        symbol,
                        offset + start,
                        offset + end,
                    )
                    for symbol, start, end in self._hunk_owner_mentions(
                        path=patch_record.path,
                        header=header,
                    )
                )
            content = raw_line[1:].rstrip("\r\n") if raw_line else ""
            schema = _FUNC_SCHEMA.search(content)
            if schema is not None:
                base = schema.group("name").split(".", 1)[0]
                start = offset + 1 + schema.start("name")
                current_native = (base, start, start + len(schema.group("name")))

            is_changed = (
                raw_line[:1] in {"+", "-"}
                and not raw_line.startswith(("+++", "---"))
            )
            if is_changed:
                for match in _ATEN_TOKEN.finditer(content):
                    entity_id = operator_entity_id(match.group("base"))
                    quote = match.group(0)
                    if entity_id not in self.operator_ids:
                        unknown.add(quote)
                        continue
                    start = offset + 1 + match.start()
                    evidence[entity_id].append(
                        _evidence(
                            artifact_ref=artifact_ref,
                            locator=(
                                f"patch:{patch_record.path}@chars "
                                f"{start}-{start + len(quote)}"
                            ),
                            quote=quote,
                            supports_field="implementation_touches",
                            extractor=self.EXTRACTOR,
                        )
                    )
                    signals[entity_id].add("aten_symbol")

                for symbol, symbol_start, symbol_end in self._dispatch_mentions(
                    path=patch_record.path,
                    content=content,
                ):
                    for entity_id in self.dispatch_symbols[symbol]:
                        evidence[entity_id].append(
                            _evidence(
                                artifact_ref=artifact_ref,
                                locator=(
                                    f"patch:{patch_record.path}@chars "
                                    f"{offset + 1 + symbol_start}-"
                                    f"{offset + 1 + symbol_end}"
                                ),
                                quote=symbol,
                                supports_field="implementation_touches",
                                extractor=self.EXTRACTOR,
                            )
                        )
                        signals[entity_id].add("dispatch_symbol")

                if (
                    patch_record.path.endswith("native_functions.yaml")
                    and current_native is not None
                ):
                    base, start, end = current_native
                    entity_id = operator_entity_id(base)
                    if entity_id in self.operator_ids:
                        quote = patch[start:end]
                        evidence[entity_id].append(
                            _evidence(
                                artifact_ref=artifact_ref,
                                locator=(
                                    f"patch:{patch_record.path}@chars "
                                    f"{start}-{end}"
                                ),
                                quote=quote,
                                supports_field="implementation_touches",
                                extractor=self.EXTRACTOR,
                            )
                        )
                        signals[entity_id].add("native_schema")
                for symbol, start, end in current_cpp_owners:
                    for entity_id in self.dispatch_symbols[symbol]:
                        evidence[entity_id].append(
                            _evidence(
                                artifact_ref=artifact_ref,
                                locator=(
                                    f"patch:{patch_record.path}@chars "
                                    f"{start}-{end}"
                                ),
                                quote=patch[start:end],
                                supports_field="implementation_touches",
                                extractor=self.EXTRACTOR,
                            )
                        )
                        signals[entity_id].add("dispatch_symbol")
            # Operator-registration decorators (scanned regardless of is_changed:
            # the decorator line is typically unchanged context beside a change to
            # the decomposition/meta body it registers). Skip hunk/file headers so
            # the `@@ ... @@ def meta_<x>` enclosing-function context is not misread.
            is_body_line = not raw_line.startswith(("@@", "+++", "---"))
            if is_body_line and _PY_OP_REGISTER.search(content):
                for match in _ATEN_TOKEN.finditer(content):
                    entity_id = operator_entity_id(match.group("base"))
                    if entity_id not in self.operator_ids:
                        unknown.add(match.group(0))
                        continue
                    start = offset + 1 + match.start()
                    evidence[entity_id].append(
                        _evidence(
                            artifact_ref=artifact_ref,
                            locator=(
                                f"patch:{patch_record.path}@chars "
                                f"{start}-{start + len(match.group(0))}"
                            ),
                            quote=match.group(0),
                            supports_field="implementation_touches",
                            extractor=self.EXTRACTOR,
                        )
                    )
                    signals[entity_id].add("decomp_meta_register")
            if is_body_line and patch_record.path.endswith("_meta_registrations.py"):
                meta = _PY_META_FUNC.search(content)
                if meta is not None:
                    entity_id = operator_entity_id(meta.group("op"))
                    if entity_id in self.operator_ids:
                        start = offset + 1 + meta.start("op")
                        evidence[entity_id].append(
                            _evidence(
                                artifact_ref=artifact_ref,
                                locator=(
                                    f"patch:{patch_record.path}@chars "
                                    f"{start}-{start + len(meta.group('op'))}"
                                ),
                                quote=meta.group(0),
                                supports_field="implementation_touches",
                                extractor=self.EXTRACTOR,
                            )
                        )
                        signals[entity_id].add("meta_registration")
            offset += len(raw_line)
        _finalize_hunk()  # last hunk
        return evidence, unknown, signals

    @staticmethod
    def _deduplicate(
        items: Iterable[EvidenceSpec],
    ) -> tuple[EvidenceSpec, ...]:
        by_key = {
            (
                item.artifact_ref,
                item.locator,
                item.quote_hash,
                item.supports_field,
            ): item
            for item in items
        }
        return tuple(by_key[key] for key in sorted(by_key))

    def _dispatch_mentions(
        self,
        *,
        path: str,
        content: str,
    ) -> tuple[tuple[str, int, int], ...]:
        matches: dict[tuple[str, int, int], None] = {}
        if path.endswith("native_functions.yaml"):
            match = _YAML_DISPATCH_VALUE.search(content)
            if match is not None:
                symbol = match.group("name")
                if symbol in self.dispatch_symbols:
                    matches[
                        (symbol, match.start("name"), match.end("name"))
                    ] = None
        elif any(path.endswith(suffix) for suffix in _CPP_SUFFIXES):
            for match in _TORCH_FUNC_MACRO.finditer(content):
                symbol = match.group("name")
                if symbol in self.dispatch_symbols:
                    matches[
                        (symbol, match.start("name"), match.end("name"))
                    ] = None
        return tuple(sorted(matches))

    def _hunk_owner_mentions(
        self,
        *,
        path: str,
        header: str,
    ) -> tuple[tuple[str, int, int], ...]:
        if not any(path.endswith(suffix) for suffix in _CPP_SUFFIXES):
            return ()
        separator = header.find("@@", 2)
        if separator < 0:
            return ()
        context_start = separator + 2
        context = header[context_start:]
        matches: dict[tuple[str, int, int], None] = {}
        for match in _CPP_CALL.finditer(context):
            symbol = match.group("name")
            if symbol in self.dispatch_symbols:
                matches[
                    (
                        symbol,
                        context_start + match.start("name"),
                        context_start + match.end("name"),
                    )
                ] = None
        return tuple(sorted(matches))

    def link_pr(self, pr: PrCorpusV2) -> LinkResult:
        evidence_by_entity: dict[str, list[EvidenceSpec]] = defaultdict(list)
        signals_by_entity: dict[str, set[str]] = defaultdict(set)
        unknown: set[str] = set()
        for patch_record in pr.touched_files_patch:
            found, missed, signals = self._patch_mentions(
                patch_record,
                pr_number=pr.pr_number,
            )
            unknown.update(missed)
            for entity_id, items in found.items():
                evidence_by_entity[entity_id].extend(items)
            for entity_id, names in signals.items():
                signals_by_entity[entity_id].update(names)

        broad_change = len(evidence_by_entity) > self.fanout_limit
        deferred = (
            self.defer_limit is not None
            and len(evidence_by_entity) > self.defer_limit
        )
        deferred_reason = (
            f"{len(evidence_by_entity)} operators exceed defer_limit "
            f"{self.defer_limit}"
            if deferred
            else None
        )
        edges: list[GraphEdge] = []
        at = _edge_time(pr)
        for entity_id in ([] if deferred else sorted(evidence_by_entity)):
            resolved_api = resolve_aten_to_api(entity_id, self.api_ids)
            target_key = resolved_api or entity_id
            context: dict[str, Any] = {
                "broad_change": broad_change,
                "signals": sorted(signals_by_entity[entity_id]),
            }
            if resolved_api is not None:
                # Evidence still points at the aten diff symbol; record which
                # aten operator this API node was resolved from.
                context["resolved_from"] = entity_id
            edges.append(
                GraphEdge.create(
                    kind=GraphEdgeKind.IMPLEMENTATION_TOUCHES,
                    source=NodeRef(
                        kind=GraphNodeKind.PR,
                        key=f"pr:{pr.pr_number}",
                    ),
                    target=NodeRef(
                        kind=GraphNodeKind.ENTITY,
                        key=target_key,
                    ),
                    evidence=self._deduplicate(evidence_by_entity[entity_id]),
                    confidence=0.55 if broad_change else 0.95,
                    context=context,
                    at=at,
                )
            )

        for test_index, test in (
            [] if deferred else enumerate(pr.tests_added)
        ):
            found, missed = self._operator_mentions(
                text=test.added_code,
                artifact_ref=f"pr:{pr.pr_number}",
                locator_prefix=f"test:{test.test_node_id}",
                supports_field="statically_mentions",
            )
            unknown.update(missed)
            for entity_id, items in self._api_test_mentions(
                text=test.added_code,
                artifact_ref=f"pr:{pr.pr_number}",
                locator_prefix=f"test:{test.test_node_id}",
            ).items():
                found.setdefault(entity_id, []).extend(items)
            for entity_id, items in sorted(found.items()):
                edges.append(
                    GraphEdge.create(
                        kind=GraphEdgeKind.STATICALLY_MENTIONS,
                        source=NodeRef(
                            kind=GraphNodeKind.PR,
                            key=f"pr:{pr.pr_number}",
                        ),
                        target=NodeRef(
                            kind=GraphNodeKind.ENTITY,
                            key=entity_id,
                        ),
                        evidence=self._deduplicate(items),
                        confidence=0.8,
                        context={
                            "test_index": test_index,
                            "test_node_id": test.test_node_id,
                        },
                        at=at,
                    )
                )

        if not deferred and (not evidence_by_entity or broad_change):
            first_path = next(
                (
                    item.get("path")
                    for item in pr.diff_stats.get("files", ())
                    if isinstance(item, dict) and item.get("path")
                ),
                None,
            )
            for label in sorted(set(pr.subsystem_labels)):
                entity = subsystem_entity(
                    label,
                    first_seen=f"pr:{pr.pr_number}",
                )
                quote = first_path or label
                locator = (
                    "diff_stats.files[0].path"
                    if first_path is not None
                    else "subsystem_labels"
                )
                edges.append(
                    GraphEdge.create(
                        kind=GraphEdgeKind.IMPLEMENTATION_TOUCHES,
                        source=NodeRef(
                            kind=GraphNodeKind.PR,
                            key=f"pr:{pr.pr_number}",
                        ),
                        target=NodeRef(
                            kind=GraphNodeKind.ENTITY,
                            key=entity.entity_id,
                        ),
                        evidence=(
                            _evidence(
                                artifact_ref=f"pr:{pr.pr_number}",
                                locator=locator,
                                quote=quote,
                                supports_field="implementation_touches",
                                extractor=self.EXTRACTOR,
                            ),
                        ),
                        confidence=0.45 if broad_change else 0.6,
                        context={
                            "broad_change": broad_change,
                            "signals": ["subsystem_fallback"],
                        },
                        at=at,
                    )
                )

        return LinkResult(
            edges=tuple(sorted(edges, key=lambda edge: edge.edge_id)),
            exact_operator_count=len(evidence_by_entity),
            broad_change=broad_change,
            unknown_operator_tokens=tuple(sorted(unknown)),
            deferred=deferred,
            deferred_reason=deferred_reason,
        )

    def link_issue(self, issue: IssueCorpusV1) -> tuple[GraphEdge, ...]:
        edges: list[GraphEdge] = []
        sources = (issue.body, *issue.comments.items)
        for source in sources:
            found, _ = self._operator_mentions(
                text=source.text,
                artifact_ref=f"issue:{issue.issue_number}",
                locator_prefix=source.source_id,
                supports_field="symptom_mentions",
            )
            for entity_id, items in sorted(found.items()):
                edges.append(
                    GraphEdge.create(
                        kind=GraphEdgeKind.SYMPTOM_MENTIONS,
                        source=NodeRef(
                            kind=GraphNodeKind.ISSUE,
                            key=f"issue:{issue.issue_number}",
                        ),
                        target=NodeRef(
                            kind=GraphNodeKind.ENTITY,
                            key=entity_id,
                        ),
                        evidence=self._deduplicate(items),
                        confidence=0.7,
                        context={"source_id": source.source_id},
                        at=source.created_at or issue.created_at,
                    )
                )
        return tuple(sorted(edges, key=lambda edge: edge.edge_id))
