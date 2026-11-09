"""Cross-store reference checks over already parsed records."""

from __future__ import annotations

from dataclasses import dataclass

from mrminer.common.schema import (
    ObservationSpec,
    OracleCaseSpec,
    OracleSpec,
    OracleStateProjection,
    RelationCheckSpec,
    RunSpec,
    VerdictSpec,
)
from mrminer.s1_miner.issue_schema import IssueCorpusV1
from mrminer.s1_miner.pr_schema import IssueFetchStatus, PrCorpusV2


@dataclass(frozen=True)
class FsckIssue:
    code: str
    source: str
    target: str


@dataclass(frozen=True)
class FsckCatalog:
    entity_ids: frozenset[str] = frozenset()
    claim_ids: frozenset[str] = frozenset()
    artifact_ids: frozenset[str] = frozenset()
    program_ids: frozenset[str] = frozenset()
    input_ids: frozenset[str] = frozenset()
    variation_template_ids: frozenset[str] = frozenset()
    variation_ids: frozenset[str] = frozenset()
    cell_ids: frozenset[str] = frozenset()
    event_ids: frozenset[str] = frozenset()
    oracles: tuple[OracleSpec, ...] = ()
    cases: tuple[OracleCaseSpec, ...] = ()
    runs: tuple[RunSpec, ...] = ()
    checks: tuple[RelationCheckSpec, ...] = ()
    observations: tuple[ObservationSpec, ...] = ()
    verdicts: tuple[VerdictSpec, ...] = ()
    projections: tuple[OracleStateProjection, ...] = ()
    pr_records: tuple[PrCorpusV2, ...] = ()
    issue_records: tuple[IssueCorpusV1, ...] = ()


def _check(
    issues: list[FsckIssue],
    *,
    exists: bool,
    code: str,
    source: str,
    target: str,
) -> None:
    if not exists:
        issues.append(FsckIssue(code=code, source=source, target=target))


def fsck(catalog: FsckCatalog) -> list[FsckIssue]:
    issues: list[FsckIssue] = []
    oracle_ids = {record.oracle_id for record in catalog.oracles}
    case_ids = {record.case_id for record in catalog.cases}
    run_ids = {record.run_id for record in catalog.runs}
    check_ids = {record.check_id for record in catalog.checks}
    obs_ids = {record.obs_id for record in catalog.observations}
    issue_ids = {
        (record.repository, record.issue_number)
        for record in catalog.issue_records
    }

    for pr in catalog.pr_records:
        source = f"pr:{pr.repository}#{pr.pr_number}"
        for issue in pr.linked_issues:
            if issue.fetch_status is not IssueFetchStatus.STORED:
                continue
            target = f"issue:{issue.repository}#{issue.number}"
            _check(
                issues,
                exists=(issue.repository, issue.number) in issue_ids,
                code="missing_issue_side_record",
                source=source,
                target=target,
            )

    for oracle in catalog.oracles:
        source = f"oracle:{oracle.oracle_id}"
        for binding in oracle.binding:
            _check(
                issues,
                exists=binding.entity_id in catalog.entity_ids,
                code="missing_entity",
                source=source,
                target=binding.entity_id,
            )
        for claim_id in oracle.claim.claim_refs:
            _check(
                issues,
                exists=claim_id in catalog.claim_ids,
                code="missing_claim",
                source=source,
                target=claim_id,
            )
        if oracle.variation is not None:
            target = oracle.variation.variation_template_id
            _check(
                issues,
                exists=target in catalog.variation_template_ids,
                code="missing_variation_template",
                source=source,
                target=target,
            )
        for artifact_id in (
            oracle.validity.from_artifact,
            oracle.validity.to_artifact,
        ):
            if artifact_id is not None:
                _check(
                    issues,
                    exists=artifact_id in catalog.artifact_ids,
                    code="missing_artifact",
                    source=source,
                    target=artifact_id,
                )

    for case in catalog.cases:
        source = f"case:{case.case_id}"
        for target, pool, code in (
            (case.oracle_id, oracle_ids, "missing_oracle"),
            (case.program_id, catalog.program_ids, "missing_program"),
            (case.base_input_id, catalog.input_ids, "missing_input"),
            (case.variation_id, catalog.variation_ids, "missing_variation"),
            (case.domain_cell, catalog.cell_ids, "missing_cell"),
        ):
            _check(issues, exists=target in pool, code=code, source=source, target=target)

    for run in catalog.runs:
        source = f"run:{run.run_id}"
        for target, pool, code in (
            (run.program_id, catalog.program_ids, "missing_program"),
            (run.input_id, catalog.input_ids, "missing_input"),
            (run.artifact_id, catalog.artifact_ids, "missing_artifact"),
        ):
            _check(issues, exists=target in pool, code=code, source=source, target=target)

    for check in catalog.checks:
        source = f"check:{check.check_id}"
        _check(
            issues,
            exists=check.oracle_id in oracle_ids,
            code="missing_oracle",
            source=source,
            target=check.oracle_id,
        )
        _check(
            issues,
            exists=check.case_id in case_ids,
            code="missing_case",
            source=source,
            target=check.case_id,
        )
        for binding in check.role_bindings.values():
            _check(
                issues,
                exists=binding.run_id in run_ids,
                code="missing_run",
                source=source,
                target=binding.run_id,
            )

    for observation in catalog.observations:
        _check(
            issues,
            exists=observation.run_id in run_ids,
            code="missing_run",
            source=f"observation:{observation.obs_id}",
            target=observation.run_id,
        )

    for verdict in catalog.verdicts:
        source = f"verdict:{verdict.verdict_id}"
        for target, pool, code in (
            (verdict.check_id, check_ids, "missing_check"),
            (verdict.oracle_id, oracle_ids, "missing_oracle"),
        ):
            _check(issues, exists=target in pool, code=code, source=source, target=target)
        for ref in verdict.obs_refs:
            if ref.startswith("obs:"):
                _check(
                    issues,
                    exists=ref in obs_ids,
                    code="missing_observation",
                    source=source,
                    target=ref,
                )

    for projection in catalog.projections:
        source = f"projection:{projection.oracle_id}"
        _check(
            issues,
            exists=projection.oracle_id in oracle_ids,
            code="missing_oracle",
            source=source,
            target=projection.oracle_id,
        )
        _check(
            issues,
            exists=projection.head_event_id in catalog.event_ids,
            code="missing_event",
            source=source,
            target=projection.head_event_id,
        )
        for cell_id in projection.active_support_cells:
            _check(
                issues,
                exists=cell_id in catalog.cell_ids,
                code="missing_cell",
                source=source,
                target=cell_id,
            )
    return issues
