"""Single-record and cross-record validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from pydantic import ValidationError

from mrminer.common.schema import (
    DomainCell,
    EntitySpec,
    EvidenceStrength,
    InputManifest,
    LifecycleEvent,
    ObservationSpec,
    OracleAspect,
    OracleCaseSpec,
    OracleSpec,
    OracleStateProjection,
    ProgramManifest,
    RelationCheckSpec,
    RunKind,
    RunResultSet,
    RunSpec,
    SpecClass,
    ValidationCellStatus,
    ValidationReport,
    VariationSpec,
    VariationTemplateSpec,
    VerdictOutcome,
    VerdictSpec,
    VersionArtifact,
)
from mrminer.common.schema.base import SchemaRecord
from mrminer.s1_miner.issue_schema import IssueCorpusV1, IssueFetchEvent
from mrminer.s1_miner.pr_schema import PrCorpusV2


class SchemaValidationError(ValueError):
    pass


SCHEMA_REGISTRY: dict[str, type[SchemaRecord]] = {
    "entity_spec_v1": EntitySpec,
    "version_artifact_v1": VersionArtifact,
    "program_manifest_v1": ProgramManifest,
    "input_manifest_v1": InputManifest,
    "domain_cell_v1": DomainCell,
    "variation_template_v1": VariationTemplateSpec,
    "variation_v1": VariationSpec,
    "oracle_spec_v1": OracleSpec,
    "oracle_case_v1": OracleCaseSpec,
    "run_spec_v1": RunSpec,
    "relation_check_v1": RelationCheckSpec,
    "run_result_set_v1": RunResultSet,
    "observation_v1": ObservationSpec,
    "verdict_v1": VerdictSpec,
    "pr_corpus_v2": PrCorpusV2,
    "issue_v1": IssueCorpusV1,
    "issue_fetch_event_v1": IssueFetchEvent,
}


class SchemaValidator:
    def __init__(
        self,
        registry: Mapping[str, type[SchemaRecord]] | None = None,
    ):
        self.registry = dict(registry or SCHEMA_REGISTRY)

    def validate(self, record: dict[str, Any]) -> SchemaRecord:
        version = record.get("schema_version")
        record_type = self.registry.get(version)
        if record_type is None:
            raise SchemaValidationError(f"unknown schema_version: {version!r}")
        try:
            return record_type.from_record(record)
        except ValidationError as error:
            raise SchemaValidationError(str(error)) from error


@dataclass(frozen=True)
class ContractIssue:
    code: str
    message: str
    path: str


@dataclass(frozen=True)
class ContractContext:
    oracle: OracleSpec
    case: OracleCaseSpec | None = None
    check: RelationCheckSpec | None = None
    runs: Mapping[str, RunSpec] | None = None
    verdicts: tuple[VerdictSpec, ...] = ()
    projection: OracleStateProjection | None = None
    validation_report: ValidationReport | None = None
    checker_roles: Mapping[tuple[str, str], frozenset[str]] | None = None


class ContractValidator:
    PERFORMANCE_PARAM_GROUPS = (
        ("margin_delta", "delta"),
        ("confidence_level",),
        ("process_reps", "processes"),
        ("estimator",),
        ("ci_method",),
        ("stopping_rule",),
        ("metric",),
        ("direction",),
    )

    def validate(self, context: ContractContext) -> list[ContractIssue]:
        oracle = context.oracle
        issues: list[ContractIssue] = []
        required = set(oracle.guard.environment.required_capabilities)
        requested_probes = {
            probe
            for observation in oracle.relation.required_observations
            for probe in observation.probes
        }

        # 1. Observation probes must be declared as capabilities.
        missing = requested_probes - required
        if missing:
            issues.append(
                ContractIssue(
                    "required_probe_not_capability",
                    f"required probes are missing capabilities: {sorted(missing)}",
                    "relation.required_observations",
                )
            )

        # 2. Dynamic trigger probe must be declared.
        trigger = oracle.guard.dynamic.trigger
        if trigger is not None and trigger.probe not in required:
            issues.append(
                ContractIssue(
                    "trigger_probe_not_capability",
                    f"trigger probe {trigger.probe!r} is not a required capability",
                    "guard.dynamic.trigger.probe",
                )
            )

        relation_roles = {
            observation.role for observation in oracle.relation.required_observations
        }

        # 3. Checker role contract must match the oracle relation.
        checker_roles = context.checker_roles or {}
        expected_roles = checker_roles.get(
            (oracle.relation.checker_id, oracle.relation.checker_version)
        )
        if expected_roles is not None and set(expected_roles) != relation_roles:
            issues.append(
                ContractIssue(
                    "checker_role_mismatch",
                    "checker roles do not match relation roles",
                    "relation.required_observations",
                )
            )

        # 4. A concrete case must produce every required role.
        if context.case is not None:
            missing_roles = relation_roles - set(context.case.result_roles)
            if missing_roles:
                issues.append(
                    ContractIssue(
                        "case_role_missing",
                        f"case is missing roles: {sorted(missing_roles)}",
                        "case.result_roles",
                    )
                )

        # 5. Bound runs and probe families must match each role's run kind.
        run_by_id = context.runs or {}
        required_by_role = {
            item.role: item for item in oracle.relation.required_observations
        }
        if context.check is not None:
            for role, binding in context.check.role_bindings.items():
                requirement = required_by_role.get(role)
                run = run_by_id.get(binding.run_id)
                if requirement is None or run is None:
                    continue
                if run.run_kind.value != requirement.run_kind:
                    issues.append(
                        ContractIssue(
                            "run_kind_mismatch",
                            f"role {role!r} requires {requirement.run_kind}",
                            f"check.role_bindings.{role}",
                        )
                    )
                has_perf_probe = any(
                    probe.startswith("performance.") for probe in requirement.probes
                )
                if has_perf_probe != (run.run_kind is RunKind.PERFORMANCE):
                    issues.append(
                        ContractIssue(
                            "probe_run_kind_mismatch",
                            f"role {role!r} has incompatible performance probes",
                            f"relation.required_observations.{role}.probes",
                        )
                    )

        # 6. Hypotheses may never produce bug candidates.
        if oracle.spec_class is SpecClass.HYPOTHESIS:
            for verdict in context.verdicts:
                if verdict.outcome is VerdictOutcome.BUG_CANDIDATE:
                    issues.append(
                        ContractIssue(
                            "hypothesis_bug_candidate",
                            "hypothesis verdict cannot be a bug candidate",
                            f"verdict.{verdict.verdict_id}.outcome",
                        )
                    )

        # 7. Performance relations require a complete statistical design.
        if oracle.aspect is OracleAspect.PERFORMANCE:
            params = oracle.relation.params
            absent = [
                aliases[0]
                for aliases in self.PERFORMANCE_PARAM_GROUPS
                if not any(alias in params for alias in aliases)
            ]
            if params.get("ci_method") == "bootstrap":
                absent.extend(
                    name
                    for name in ("bootstrap_seed", "bootstrap_resamples")
                    if name not in params
                )
            if absent:
                issues.append(
                    ContractIssue(
                        "performance_design_incomplete",
                        f"missing statistical parameters: {absent}",
                        "relation.params",
                    )
                )

        # 8. Effect/decision contracts require explicit evidence strength.
        if oracle.aspect in {OracleAspect.EFFECT, OracleAspect.DECISION} and (
            oracle.evidence_policy.min_strength
            is not EvidenceStrength.EXPLICIT_CLAIM
        ):
            issues.append(
                ContractIssue(
                    "evidence_strength_insufficient",
                    "effect/decision contracts require explicit_claim",
                    "evidence_policy.min_strength",
                )
            )

        # 9. Validity is the only artifact-range source.
        if "artifact_range" in oracle.guard.static.to_record():
            issues.append(
                ContractIssue(
                    "duplicate_artifact_range",
                    "guard cannot define artifact_range; use validity",
                    "guard.static",
                )
            )

        # 10. Active projections may reference only validated cells.
        if context.projection is not None and context.validation_report is not None:
            validated = {
                cell_id
                for cell_id, status in context.validation_report.cells.items()
                if status is ValidationCellStatus.VALIDATED
            }
            invalid_active = set(context.projection.active_support_cells) - validated
            if invalid_active:
                issues.append(
                    ContractIssue(
                        "active_cell_not_validated",
                        f"active cells are not validated: {sorted(invalid_active)}",
                        "projection.active_support_cells",
                    )
                )
        return issues

    def assert_valid(self, context: ContractContext) -> None:
        issues = self.validate(context)
        if issues:
            joined = "; ".join(f"{issue.code}: {issue.message}" for issue in issues)
            raise SchemaValidationError(joined)
