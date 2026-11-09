"""CLI for operator evidence retrieval, claim extraction, and oracle induction."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from mrminer.common.schema import (
    ChangeLanding,
    LlmAudit,
    ProvenanceOrigin,
    TriggerSpec,
)
from mrminer.llm import (
    DEFAULT_DEEPSEEK_BASE_URL,
    DEFAULT_DEEPSEEK_MODEL,
    DeepSeekConfig,
    DeepSeekProvider,
    FixtureLlmProvider,
    StructuredLlmProvider,
)
from mrminer.s1_miner.issue_collector import atomic_write_record
from mrminer.s2_graph import ClaimCandidate, EntityGraph
from mrminer.s3_oracle.claim_extractor import ClaimExtractor
from mrminer.s3_oracle.context_assembler import (
    AssembledContext,
    ContextGap,
    assemble_context,
    select_program_ids,
)
from mrminer.s3_oracle.artifact_catalog import (
    ArtifactImportStatus,
    apply_artifact_results,
    inspect_artifact_manifest,
)
from mrminer.s3_oracle.artifact_resolution import (
    EffectiveArtifactResolver,
    SubprocessGitAncestry,
)
from mrminer.s3_oracle.evidence_bundle import (
    EvidenceBudget,
    EvidenceRetriever,
)
from mrminer.s3_oracle.inducer import (
    InductionContext,
    LlmLineageJudge,
    OracleInducer,
)
from mrminer.s3_oracle.persistence import OracleActionApplier
from mrminer.s3_oracle.program_seeds import (
    ProgramSeedMiner,
    persist_program_seeds,
    source_commit_candidates,
)
from mrminer.s3_oracle.program_source_fetch import ProgramSourceFetcher

# Stable pointer to the current canonical graph (a symlink under data/); keep in
# step with the s2 CLI default so the whole pipeline follows one switch.
DEFAULT_GRAPH_ROOT = Path("data/entity_graph_canonical")


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_bytes())
    if type(payload) is not dict:
        raise ValueError(f"{path} must contain one JSON object")
    return payload


def _write_output(path: Path, record: dict[str, Any]) -> None:
    atomic_write_record(path, record)
    print(json.dumps({"output": str(path), "status": "ok"}, sort_keys=True))


def _budget(args: argparse.Namespace) -> EvidenceBudget:
    return EvidenceBudget(
        max_prs=args.max_prs,
        max_sources_per_pr=args.max_sources_per_pr,
        max_chars_per_source=args.max_chars_per_source,
        max_total_chars_per_pr=args.max_total_chars_per_pr,
        max_issue_comments_per_issue=args.max_issue_comments_per_issue,
        max_graph_evidence_per_pr=args.max_graph_evidence_per_pr,
        context_chars=args.context_chars,
    )


def _retriever(args: argparse.Namespace) -> EvidenceRetriever:
    return EvidenceRetriever(
        EntityGraph(args.graph_root),
        corpus_root=args.corpus_root,
        budget=_budget(args),
    )


def _fixture_provider(path: Path | None) -> FixtureLlmProvider | None:
    if path is None:
        return None
    payload = _read_object(path)
    if payload.get("schema_version") in {
        "claim_extraction_v1",
        "claim_extraction_v2",
    }:
        return FixtureLlmProvider(
            responses={"operator_claim_extraction": payload}
        )
    responses = payload.get("responses", payload)
    if not isinstance(responses, dict):
        raise ValueError("fixture response mapping must be a JSON object")

    def handler(request):
        pr_number = request.input_payload.get("pr_number")
        response = responses.get(str(pr_number))
        if not isinstance(response, dict):
            raise KeyError(f"no fixture response for PR #{pr_number}")
        return response

    return FixtureLlmProvider(handler=handler)


def _claim_provider(
    args: argparse.Namespace,
) -> tuple[StructuredLlmProvider | None, str]:
    if args.llm_fixture is not None:
        return _fixture_provider(args.llm_fixture), "fixture"
    if args.llm_provider == "deepseek":
        return (
            DeepSeekProvider(
                DeepSeekConfig(
                    base_url=args.llm_base_url,
                    model=args.llm_model,
                    read_timeout_s=args.llm_timeout,
                    max_retries=args.llm_max_retries,
                    max_tokens=args.llm_max_tokens,
                    thinking=args.llm_thinking,
                )
            ),
            "deepseek",
        )
    return None, "unconfigured"


def _retrieve(args: argparse.Namespace) -> int:
    bundle = _retriever(args).retrieve(args.entity_id)
    _write_output(args.output, bundle.to_record())
    return 0


def _extract(args: argparse.Namespace) -> int:
    graph = EntityGraph(args.graph_root)
    if args.reviewed_report is not None:
        return _persist_reviewed_claims(args, graph)
    bundle = EvidenceRetriever(
        graph,
        corpus_root=args.corpus_root,
        budget=_budget(args),
    ).retrieve(args.entity_id)
    selected = [
        item
        for item in bundle.prs
        if not args.pr_number or item.pr_number in set(args.pr_number)
    ]
    if not selected:
        raise ValueError("no PR evidence bundles matched the selection")
    provider, provider_name = _claim_provider(args)
    extractor = ClaimExtractor(provider)
    reports = [
        extractor.extract_pr(
            item,
            effective_artifact=args.effective_artifact,
        )
        for item in selected
    ]
    persisted = 0
    if args.persist_claims:
        if args.at is None:
            raise ValueError("--persist-claims requires --at")
        for report in reports:
            for claim in report.claims:
                persisted += int(graph.add_claim(claim, at=args.at))
        index = graph.rebuild_index()
        if not index.checksum_ok:
            raise RuntimeError(
                f"graph rebuild failed: {graph.last_rebuild_errors}"
            )
    output = {
        "entity_id": args.entity_id,
        "effective_artifact": args.effective_artifact,
        "selected_prs": [item.pr_number for item in selected],
        "reports": [report.to_record() for report in reports],
        "accepted_claims": sum(len(report.claims) for report in reports),
        "rejected_claims": sum(len(report.rejected) for report in reports),
        "persisted_claims": persisted,
        "llm_provider": provider_name,
        "llm_calls": [
            item.to_record()
            for item in getattr(provider, "call_audits", ())
        ],
    }
    _write_output(args.output, output)
    return 0


def _persist_reviewed_claims(
    args: argparse.Namespace,
    graph: EntityGraph,
) -> int:
    if not args.persist_claims or args.at is None:
        raise ValueError(
            "--reviewed-report requires --persist-claims and --at"
        )
    if args.output.resolve() == args.reviewed_report.resolve():
        raise ValueError("--output must not overwrite --reviewed-report")

    report_bytes = args.reviewed_report.read_bytes()
    reviewed = _read_object(args.reviewed_report)
    if reviewed.get("persisted_claims") != 0:
        raise ValueError(
            "reviewed report must be an unpersisted dry-run"
        )
    if reviewed.get("entity_id") != args.entity_id:
        raise ValueError("reviewed report entity_id does not match --entity-id")
    if reviewed.get("effective_artifact") != args.effective_artifact:
        raise ValueError(
            "reviewed report effective_artifact does not match "
            "--effective-artifact"
        )

    selected_prs = reviewed.get("selected_prs")
    if (
        not isinstance(selected_prs, list)
        or not selected_prs
        or any(type(item) is not int or item <= 0 for item in selected_prs)
        or len(selected_prs) != len(set(selected_prs))
    ):
        raise ValueError("reviewed report selected_prs is invalid")
    requested_prs = sorted(set(args.pr_number or selected_prs))
    if sorted(selected_prs) != requested_prs:
        raise ValueError(
            "reviewed report selected_prs does not match --pr-number"
        )
    graph.entity(args.entity_id)
    for pr_number in selected_prs:
        graph.pr(pr_number)

    report_items = reviewed.get("reports")
    if not isinstance(report_items, list):
        raise ValueError("reviewed report reports must be a list")
    claims: list[ClaimCandidate] = []
    report_prs: list[int] = []
    for item in report_items:
        if not isinstance(item, dict) or type(item.get("pr_number")) is not int:
            raise ValueError("reviewed report contains an invalid PR report")
        report_pr = item["pr_number"]
        report_prs.append(report_pr)
        records = item.get("claims")
        if not isinstance(records, list):
            raise ValueError("reviewed PR report claims must be a list")
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("reviewed claim must be a JSON object")
            claim = ClaimCandidate.from_record(record)
            if claim.source_pr != report_pr or report_pr not in selected_prs:
                raise ValueError("reviewed claim source_pr is inconsistent")
            if claim.effective_artifact != args.effective_artifact:
                raise ValueError(
                    "reviewed claim effective_artifact is inconsistent"
                )
            if args.entity_id not in claim.target_entities:
                raise ValueError("reviewed claim does not target --entity-id")
            claims.append(claim)

    if sorted(report_prs) != sorted(selected_prs):
        raise ValueError(
            "reviewed report PR records are incomplete or duplicated"
        )
    if reviewed.get("accepted_claims") != len(claims):
        raise ValueError("reviewed report accepted_claims count is inconsistent")

    persisted = sum(graph.add_claim(claim, at=args.at) for claim in claims)
    index = graph.rebuild_index()
    if not index.checksum_ok:
        raise RuntimeError(
            f"graph rebuild failed: {graph.last_rebuild_errors}"
        )
    output = {
        "entity_id": args.entity_id,
        "effective_artifact": args.effective_artifact,
        "selected_prs": selected_prs,
        "reviewed_report": str(args.reviewed_report.resolve()),
        "reviewed_report_sha256": (
            "sha256:" + hashlib.sha256(report_bytes).hexdigest()
        ),
        "reviewed_claim_ids": [claim.claim_id for claim in claims],
        "accepted_claims": len(claims),
        "rejected_claims": reviewed.get("rejected_claims", 0),
        "persisted_claims": persisted,
        "llm_provider": "reviewed_report",
        "llm_calls": reviewed.get("llm_calls", []),
    }
    _write_output(args.output, output)
    return 0


def _induction_context(record: dict[str, Any]) -> InductionContext:
    origins = tuple(
        sorted(
            ProvenanceOrigin(value)
            for value in record.get(
                "source_origins",
                [ProvenanceOrigin.PR_REGRESSION_TEST.value],
            )
        )
    )
    trigger_record = record.get("dynamic_trigger")
    llm_record = record.get("llm_audit")
    return InductionContext(
        validity_artifact=record["validity_artifact"],
        program_ids=tuple(sorted(set(record.get("program_ids", [])))),
        created_at=record["created_at"],
        pipeline_version=record.get("pipeline_version", "s3.inducer/v1"),
        source_origins=origins,
        variation_template_id=record.get("variation_template_id"),
        param_grid={
            name: tuple(values)
            for name, values in record.get("param_grid", {}).items()
        },
        required_capabilities=tuple(
            sorted(set(record.get("required_capabilities", [])))
        ),
        dynamic_trigger=(
            TriggerSpec.from_record(trigger_record)
            if isinstance(trigger_record, dict)
            else None
        ),
        relation_params=dict(record.get("relation_params", {})),
        llm_audit=(
            LlmAudit.from_record(llm_record)
            if isinstance(llm_record, dict)
            else None
        ),
    )


def _induce(args: argparse.Namespace) -> int:
    graph = EntityGraph(args.graph_root)
    claims = graph.claims(args.entity_id)
    existing = graph.oracles(args.entity_id, state=None)
    context = _induction_context(_read_object(args.context))
    lineage_provider = None
    if args.lineage_fixture is not None:
        lineage_provider = FixtureLlmProvider(
            responses={
                "oracle_lineage_judge": _read_object(args.lineage_fixture)
            }
        )
    inducer = OracleInducer(
        lineage_judge=(
            LlmLineageJudge(lineage_provider)
            if lineage_provider is not None
            else None
        )
    )
    actions = inducer.induce(
        entity_id=args.entity_id,
        claims=claims,
        existing_oracles=existing,
        context=context,
    )
    ready = [action for action in actions if action.persistable]
    apply_report = None
    if args.apply_ready:
        if args.lifecycle_root is None or args.at is None:
            raise ValueError(
                "--apply-ready requires --lifecycle-root and --at"
            )
        if ready:
            apply_report = OracleActionApplier(
                graph,
                lifecycle_root=args.lifecycle_root,
            ).apply(
                ready,
                claims=claims,
                at=args.at,
            )
    output = {
        "entity_id": args.entity_id,
        "claim_count": len(claims),
        "existing_oracle_count": len(existing),
        "actions": [action.to_record() for action in actions],
        "ready_action_ids": [action.action_id for action in ready],
        "skipped_action_ids": [
            action.action_id for action in actions if not action.persistable
        ],
        "apply_report": (
            apply_report.to_record() if apply_report is not None else None
        ),
    }
    _write_output(args.output, output)
    return 0


def _assemble_context(args: argparse.Namespace) -> int:
    """Chain assemble + induce: extract report + seed report -> oracles, unattended.

    Reads standard tool outputs (an `extract` report and a `seed-programs` report),
    binds each accepted claim to its source_validated programs, assembles an
    InductionContext (or a typed ContextGap), and runs the real inducer. No
    hand-authored context.
    """
    extract = _read_object(args.extract)
    entity_id = args.entity_id or extract.get("entity_id")
    if not entity_id:
        raise ValueError("--entity-id is required when the extract lacks entity_id")
    seed = _read_object(args.seed_report)
    seed_candidates = seed.get("candidates", [])
    if not isinstance(seed_candidates, list):
        raise ValueError("seed report has no candidates list")
    inducer = OracleInducer()
    results: list[dict[str, Any]] = []
    for report in extract.get("reports", []):
        for record in report.get("claims", []):
            claim = ClaimCandidate.from_record(record)
            program_ids = select_program_ids(
                seed_candidates,
                pr_number=claim.source_pr,
                entity_id=entity_id,
            )
            outcome = assemble_context(claim, program_ids=program_ids, at=args.at)
            entry: dict[str, Any] = {
                "claim_id": claim.claim_id,
                "source_pr": claim.source_pr,
                "aspect": claim.aspect.value,
            }
            if isinstance(outcome, ContextGap):
                entry["outcome"] = outcome.to_record()
            else:
                assert isinstance(outcome, AssembledContext)
                actions = inducer.induce(
                    entity_id=entity_id,
                    claims=(claim,),
                    existing_oracles=(),
                    context=outcome.context,
                )
                specs = []
                for action in actions:
                    action_record = action.to_record()
                    spec = (action_record.get("new_specs") or [{}])[0]
                    specs.append(
                        {
                            "status": action_record.get("status"),
                            "action_kind": action_record.get("kind"),
                            "oracle_id": spec.get("oracle_id"),
                            "template_id": spec.get("template_id"),
                            "checker_id": (spec.get("relation") or {}).get(
                                "checker_id"
                            ),
                        }
                    )
                entry["outcome"] = {**outcome.to_record(), "actions": specs}
            results.append(entry)
    assembled = [
        item for item in results
        if item["outcome"].get("outcome") == "assembled"
    ]
    ready = sum(
        1
        for item in assembled
        for action in item["outcome"].get("actions", [])
        if action.get("status") == "ready"
    )
    output = {
        "report_version": "context_assembly_report_v1",
        "entity_id": entity_id,
        "extract": str(args.extract.resolve()),
        "seed_report": str(args.seed_report.resolve()),
        "claim_count": len(results),
        "assembled_count": len(assembled),
        "gap_count": len(results) - len(assembled),
        "ready_oracle_count": ready,
        "results": results,
    }
    _write_output(args.output, output)
    return 0


def _resolve_artifacts(args: argparse.Namespace) -> int:
    graph = EntityGraph(args.graph_root)
    artifacts = graph.artifacts()
    landings = graph.landings()
    ancestry = SubprocessGitAncestry(
        args.git_root,
        timeout_s=args.git_timeout,
    )
    commits = {
        commit
        for landing in landings
        for commit in landing.landed_commits
    }
    for artifact in artifacts:
        commits.update(
            {
                artifact.source_commit,
                artifact.commit_interval.from_exclusive,
                artifact.commit_interval.to_inclusive,
            }
        )
    ancestry.prefetch_commits(commits)
    requested = (
        sorted(set(args.pr_number))
        if args.pr_number
        else graph.pr_numbers()
    )
    resolver = EffectiveArtifactResolver(
        artifacts=artifacts,
        landings=landings,
        ancestry=ancestry,
    )
    resolutions = [
        resolver.resolve(pr_number, branch=args.branch)
        for pr_number in requested
    ]
    status_counts = Counter(item.status.value for item in resolutions)
    output = {
        "audit_version": "effective_artifact_audit_v1",
        "graph_root": str(args.graph_root.resolve()),
        "git_root": str(args.git_root.resolve()),
        "requested_branch": args.branch,
        "pr_count": len(requested),
        "artifact_count": len(artifacts),
        "landing_count": len(landings),
        "unique_landing_commit_count": len(
            {
                commit
                for landing in landings
                for commit in landing.landed_commits
            }
        ),
        "status_counts": dict(sorted(status_counts.items())),
        "git_commit_presence": ancestry.commit_presence(),
        "git_diagnostics": [
            item.to_record() for item in ancestry.diagnostics()
        ],
        "resolutions": [item.to_record() for item in resolutions],
    }
    _write_output(args.output, output)
    return 0


def _evaluate_change_activity(args: argparse.Namespace) -> int:
    graph = EntityGraph(args.graph_root)
    artifacts = graph.artifacts()
    graph_landings = graph.landings()
    supplemental_landings = []
    landing_manifests = tuple(args.landing_manifest or ())
    for path in landing_manifests:
        manifest = _read_object(path)
        if manifest.get("manifest_version") != (
            "supplemental_change_landings_v1"
        ):
            raise ValueError(f"unsupported landing manifest: {path}")
        raw_landings = manifest.get("landings")
        if not isinstance(raw_landings, list):
            raise ValueError(f"landing manifest has no landing list: {path}")
        if not all(isinstance(record, dict) for record in raw_landings):
            raise ValueError(f"landing manifest has a non-object entry: {path}")
        supplemental_landings.extend(
            ChangeLanding.from_record(record)
            for record in raw_landings
        )
    unique_landings: dict[str, ChangeLanding] = {}
    for landing in (*graph_landings, *supplemental_landings):
        record = landing.to_record()
        unique_landings[json.dumps(record, sort_keys=True)] = landing
    landings = list(unique_landings.values())
    ancestry = SubprocessGitAncestry(
        args.git_root,
        timeout_s=args.git_timeout,
    )
    commits = {
        commit
        for landing in landings
        for commit in landing.landed_commits
    }
    for artifact in artifacts:
        commits.update(
            {
                artifact.source_commit,
                artifact.commit_interval.from_exclusive,
                artifact.commit_interval.to_inclusive,
            }
        )
    ancestry.prefetch_commits(commits)
    evaluation = EffectiveArtifactResolver(
        artifacts=artifacts,
        landings=landings,
        ancestry=ancestry,
    ).evaluate_activity(args.pr_number, args.artifact_id)
    output = {
        "report_version": "change_activity_report_v1",
        "graph_root": str(args.graph_root.resolve()),
        "git_root": str(args.git_root.resolve()),
        "artifact_count": len(artifacts),
        "graph_landing_count": len(graph_landings),
        "supplemental_landing_count": len(supplemental_landings),
        "landing_manifests": [
            str(path.resolve()) for path in landing_manifests
        ],
        "landing_count": len(landings),
        "git_commit_presence": ancestry.commit_presence(),
        "git_diagnostics": [
            item.to_record() for item in ancestry.diagnostics()
        ],
        "evaluation": evaluation.to_record(),
    }
    _write_output(args.output, output)
    return 0


def _catalog_artifacts(args: argparse.Namespace) -> int:
    git_options: dict[str, Any] = {
        "timeout_s": args.git_timeout,
    }
    if args.proxy_url is not None:
        git_options["environment"] = {
            "http_proxy": args.proxy_url,
            "https_proxy": args.proxy_url,
        }
    git = SubprocessGitAncestry(args.git_root, **git_options)
    results = inspect_artifact_manifest(args.manifest, git=git)
    rejected = [
        item
        for item in results
        if item.status is not ArtifactImportStatus.READY
    ]
    apply_report = None
    apply_refused = bool(args.apply and rejected)
    if args.apply and not rejected:
        if args.graph_root is None:
            raise ValueError("--apply requires --graph-root")
        apply_report = apply_artifact_results(
            results,
            catalog_root=args.catalog_root,
            graph=EntityGraph(args.graph_root),
        )
    statuses = Counter(item.status.value for item in results)
    output = {
        "report_version": "version_artifact_import_report_v1",
        "manifest": str(args.manifest.resolve()),
        "git_root": str(args.git_root.resolve()),
        "catalog_root": str(args.catalog_root.resolve()),
        "graph_root": (
            str(args.graph_root.resolve())
            if args.graph_root is not None
            else None
        ),
        "proxy_configured": args.proxy_url is not None,
        "requested_apply": args.apply,
        "apply_refused": apply_refused,
        "entry_count": len(results),
        "ready_count": len(results) - len(rejected),
        "rejected_count": len(rejected),
        "status_counts": dict(sorted(statuses.items())),
        "git_commit_presence": git.commit_presence(),
        "git_diagnostics": [
            item.to_record() for item in git.diagnostics()
        ],
        "apply_report": (
            apply_report.to_record()
            if apply_report is not None
            else None
        ),
        "results": [item.to_record() for item in results],
    }
    _write_output(args.output, output)
    return 2 if apply_refused else 0


def _seed_programs(args: argparse.Namespace) -> int:
    graph = EntityGraph(args.graph_root)
    links = graph.static_test_mentions(args.entity_id)
    selected_links = (
        links
        if args.include_unmerged
        else [(pr, edge) for pr, edge in links if pr.merged]
    )
    source_reader = None
    if args.source_root is not None:
        source_reader = SubprocessGitAncestry(
            args.source_root,
            timeout_s=args.git_timeout,
        )
        source_reader.prefetch_commits(
            {
                option.commit
                for pr, _ in selected_links
                for option in source_commit_candidates(
                    graph, pr, landing_log=source_reader
                )
            }
        )
    candidates = ProgramSeedMiner(
        graph,
        source_reader=source_reader,
    ).mine(
        args.entity_id,
        links=selected_links,
        include_unmerged=args.include_unmerged,
    )
    persist_report = None
    if args.persist:
        if args.output_root is None or args.at is None:
            raise ValueError("--persist requires --output-root and --at")
        persist_report = persist_program_seeds(
            candidates,
            output_root=args.output_root,
            recorded_at=args.at,
        )
    readiness = Counter(item.readiness.value for item in candidates)
    devices = Counter(item.device_class for item in candidates)
    phases = Counter(item.phase for item in candidates)
    output = {
        "report_version": "program_seed_report_v2",
        "graph_root": str(args.graph_root.resolve()),
        "entity_id": args.entity_id,
        "source_root": (
            str(args.source_root.resolve())
            if args.source_root is not None
            else None
        ),
        "linked_test_count": len(links),
        "excluded_unmerged_link_count": len(links) - len(selected_links),
        "candidate_count": len(candidates),
        "manifest_candidate_count": sum(
            item.manifest is not None for item in candidates
        ),
        "persistable_manifest_count": sum(
            item.manifest is not None
            and item.readiness.value == "source_validated"
            for item in candidates
        ),
        "readiness_counts": dict(sorted(readiness.items())),
        "device_counts": dict(sorted(devices.items())),
        "phase_counts": dict(sorted(phases.items())),
        "source_commit_presence": (
            source_reader.commit_presence()
            if source_reader is not None
            else None
        ),
        "git_diagnostics": (
            [
                item.to_record()
                for item in source_reader.diagnostics()
            ]
            if source_reader is not None
            else []
        ),
        "persist_report": (
            persist_report.to_record()
            if persist_report is not None
            else None
        ),
        "candidates": [item.to_record() for item in candidates],
    }
    _write_output(args.output, output)
    return 0


def _fetch_program_sources(args: argparse.Namespace) -> int:
    graph = EntityGraph(args.graph_root)
    graph.entity(args.entity_id)
    links = graph.static_test_mentions(args.entity_id)
    selected = [
        (pr, edge)
        for pr, edge in links
        if (args.include_unmerged or pr.merged)
        and (not args.pr_number or pr.pr_number in set(args.pr_number))
    ]
    prs = {pr.pr_number: pr for pr, _ in selected}
    missing_head = sorted(
        pr.pr_number
        for pr in prs.values()
        if pr.commit_refs.head_sha is None
    )
    fetchable = [
        pr for pr in prs.values() if pr.commit_refs.head_sha is not None
    ]
    git_options: dict[str, Any] = {
        "timeout_s": args.fetch_timeout,
    }
    if args.proxy_url is not None:
        git_options["environment"] = {
            "http_proxy": args.proxy_url,
            "https_proxy": args.proxy_url,
        }
    git = SubprocessGitAncestry(args.source_root, **git_options)
    git.prefetch_commits(
        {
            pr.commit_refs.head_sha
            for pr in fetchable
            if pr.commit_refs.head_sha is not None
        }
    )
    fetcher = ProgramSourceFetcher(git, remote=args.remote)
    results = [
        fetcher.fetch(
            pr_number=pr.pr_number,
            expected_head_sha=pr.commit_refs.head_sha,
        )
        for pr in sorted(fetchable, key=lambda item: item.pr_number)
        if pr.commit_refs.head_sha is not None
    ]
    statuses = Counter(item.status.value for item in results)
    output = {
        "report_version": "program_source_fetch_report_v1",
        "graph_root": str(args.graph_root.resolve()),
        "entity_id": args.entity_id,
        "source_root": str(args.source_root.resolve()),
        "remote": args.remote,
        "proxy_configured": args.proxy_url is not None,
        "linked_test_count": len(links),
        "selected_link_count": len(selected),
        "selected_pr_count": len(prs),
        "excluded_unmerged_link_count": (
            sum(not pr.merged for pr, _ in links)
            if not args.include_unmerged
            else 0
        ),
        "missing_head_prs": missing_head,
        "status_counts": dict(sorted(statuses.items())),
        "source_available_count": sum(
            item.source_available for item in results
        ),
        "git_commit_presence": git.commit_presence(),
        "git_diagnostics": [
            item.to_record() for item in git.diagnostics()
        ],
        "results": [item.to_record() for item in results],
    }
    _write_output(args.output, output)
    return 0


def _add_common_retrieval_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--graph-root", type=Path, default=DEFAULT_GRAPH_ROOT)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--entity-id", required=True)
    parser.add_argument("--max-prs", type=int, default=100)
    parser.add_argument("--max-sources-per-pr", type=int, default=32)
    parser.add_argument("--max-chars-per-source", type=int, default=6_000)
    parser.add_argument("--max-total-chars-per-pr", type=int, default=80_000)
    parser.add_argument("--max-issue-comments-per-issue", type=int, default=8)
    parser.add_argument("--max-graph-evidence-per-pr", type=int, default=64)
    parser.add_argument("--context-chars", type=int, default=800)
    parser.add_argument("--output", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mrminer-s3")
    commands = parser.add_subparsers(dest="command", required=True)

    retrieve = commands.add_parser("retrieve")
    _add_common_retrieval_arguments(retrieve)
    retrieve.set_defaults(handler=_retrieve)

    extract = commands.add_parser("extract")
    _add_common_retrieval_arguments(extract)
    extract.add_argument("--effective-artifact", required=True)
    extract.add_argument("--pr-number", type=int, action="append")
    llm = extract.add_mutually_exclusive_group()
    llm.add_argument("--llm-fixture", type=Path)
    llm.add_argument("--llm-provider", choices=("deepseek",))
    llm.add_argument("--reviewed-report", type=Path)
    extract.add_argument(
        "--llm-base-url", default=DEFAULT_DEEPSEEK_BASE_URL
    )
    extract.add_argument("--llm-model", default=DEFAULT_DEEPSEEK_MODEL)
    extract.add_argument("--llm-timeout", type=float, default=120.0)
    extract.add_argument("--llm-max-retries", type=int, default=2)
    extract.add_argument("--llm-max-tokens", type=int, default=8_192)
    extract.add_argument("--llm-thinking", action="store_true")
    extract.add_argument("--persist-claims", action="store_true")
    extract.add_argument("--at")
    extract.set_defaults(handler=_extract)

    induce = commands.add_parser("induce")
    induce.add_argument("--graph-root", type=Path, default=DEFAULT_GRAPH_ROOT)
    induce.add_argument("--entity-id", required=True)
    induce.add_argument("--context", type=Path, required=True)
    induce.add_argument("--lineage-fixture", type=Path)
    induce.add_argument("--apply-ready", action="store_true")
    induce.add_argument("--lifecycle-root", type=Path)
    induce.add_argument("--at")
    induce.add_argument("--output", type=Path, required=True)
    induce.set_defaults(handler=_induce)

    assemble = commands.add_parser("assemble-context")
    assemble.add_argument("--extract", type=Path, required=True)
    assemble.add_argument("--seed-report", type=Path, required=True)
    assemble.add_argument("--entity-id")
    assemble.add_argument("--at", required=True)
    assemble.add_argument("--output", type=Path, required=True)
    assemble.set_defaults(handler=_assemble_context)

    resolve_artifacts = commands.add_parser("resolve-artifacts")
    resolve_artifacts.add_argument("--graph-root", type=Path, default=DEFAULT_GRAPH_ROOT)
    resolve_artifacts.add_argument("--git-root", type=Path, required=True)
    resolve_artifacts.add_argument("--git-timeout", type=float, default=10.0)
    resolve_artifacts.add_argument("--branch")
    resolve_artifacts.add_argument("--pr-number", type=int, action="append")
    resolve_artifacts.add_argument("--output", type=Path, required=True)
    resolve_artifacts.set_defaults(handler=_resolve_artifacts)

    evaluate_activity = commands.add_parser("evaluate-change-activity")
    evaluate_activity.add_argument("--graph-root", type=Path, default=DEFAULT_GRAPH_ROOT)
    evaluate_activity.add_argument("--git-root", type=Path, required=True)
    evaluate_activity.add_argument("--git-timeout", type=float, default=10.0)
    evaluate_activity.add_argument("--pr-number", type=int, required=True)
    evaluate_activity.add_argument("--artifact-id", required=True)
    evaluate_activity.add_argument(
        "--landing-manifest",
        type=Path,
        action="append",
    )
    evaluate_activity.add_argument("--output", type=Path, required=True)
    evaluate_activity.set_defaults(handler=_evaluate_change_activity)

    catalog_artifacts = commands.add_parser("catalog-artifacts")
    catalog_artifacts.add_argument("--manifest", type=Path, required=True)
    catalog_artifacts.add_argument("--git-root", type=Path, required=True)
    catalog_artifacts.add_argument(
        "--catalog-root",
        type=Path,
        required=True,
    )
    catalog_artifacts.add_argument(
        "--graph-root", type=Path, default=DEFAULT_GRAPH_ROOT
    )
    catalog_artifacts.add_argument("--git-timeout", type=float, default=10.0)
    catalog_artifacts.add_argument("--proxy-url")
    catalog_artifacts.add_argument("--apply", action="store_true")
    catalog_artifacts.add_argument("--output", type=Path, required=True)
    catalog_artifacts.set_defaults(handler=_catalog_artifacts)

    seed_programs = commands.add_parser("seed-programs")
    seed_programs.add_argument("--graph-root", type=Path, default=DEFAULT_GRAPH_ROOT)
    seed_programs.add_argument("--entity-id", required=True)
    seed_programs.add_argument("--source-root", type=Path)
    seed_programs.add_argument("--git-timeout", type=float, default=10.0)
    seed_programs.add_argument("--include-unmerged", action="store_true")
    seed_programs.add_argument("--persist", action="store_true")
    seed_programs.add_argument("--output-root", type=Path)
    seed_programs.add_argument("--at")
    seed_programs.add_argument("--output", type=Path, required=True)
    seed_programs.set_defaults(handler=_seed_programs)

    fetch_sources = commands.add_parser("fetch-program-sources")
    fetch_sources.add_argument("--graph-root", type=Path, default=DEFAULT_GRAPH_ROOT)
    fetch_sources.add_argument("--entity-id", required=True)
    fetch_sources.add_argument("--source-root", type=Path, required=True)
    fetch_sources.add_argument("--remote", default="origin")
    fetch_sources.add_argument("--proxy-url")
    fetch_sources.add_argument("--fetch-timeout", type=float, default=300.0)
    fetch_sources.add_argument("--include-unmerged", action="store_true")
    fetch_sources.add_argument("--pr-number", type=int, action="append")
    fetch_sources.add_argument("--output", type=Path, required=True)
    fetch_sources.set_defaults(handler=_fetch_program_sources)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
