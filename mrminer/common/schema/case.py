"""Concrete Oracle × Program/Input × Variation cases."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass as runtime_dataclass
from itertools import islice, product
from typing import Annotated, Any, Literal

from pydantic import StringConstraints, model_validator
from pydantic.dataclasses import dataclass

from mrminer.common.canonical import canonical_json, content_id
from mrminer.common.schema.base import SCHEMA_CONFIG, SchemaRecord, dump_value
from mrminer.common.schema.domain_cell import DomainCell
from mrminer.common.schema.input import InputManifest
from mrminer.common.schema.oracle import OracleSpec
from mrminer.common.schema.program import (
    ProgramManifest,
    ProvenanceOrigin,
)
from mrminer.common.schema.variation import (
    JsonScalar,
    ScenarioSpec,
    VariationId,
    VariationSpec,
    VariationTemplateSpec,
)

NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]
CaseId = Annotated[
    str, StringConstraints(pattern=r"^case:sha256:[0-9a-f]{64}$")
]
OracleId = Annotated[
    str, StringConstraints(pattern=r"^orc:sha256:[0-9a-f]{64}$")
]
ProgramId = Annotated[
    str, StringConstraints(pattern=r"^prog:sha256:[0-9a-f]{64}$")
]
InputId = Annotated[str, StringConstraints(pattern=r"^inp:sha256:[0-9a-f]{64}$")]
CellId = Annotated[
    str, StringConstraints(pattern=r"^cell:sha256:[0-9a-f]{64}$")
]


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class CaseResultRole(SchemaRecord):
    RECORD_KIND = "case_result_role"

    run_kind: Literal["single", "paired_semantic", "performance"]
    scenario: ScenarioSpec


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class OracleCaseSpec(SchemaRecord):
    RECORD_KIND = "oracle_case"

    schema_version: Literal["oracle_case_v1"] = "oracle_case_v1"
    case_id: CaseId
    oracle_id: OracleId
    program_id: ProgramId
    base_input_id: InputId
    variation_id: VariationId
    parameter_assignment: dict[NonEmptyStr, JsonScalar]
    domain_cell: CellId
    result_roles: dict[NonEmptyStr, CaseResultRole]
    required_observation_roles: tuple[NonEmptyStr, ...]

    @classmethod
    def create(cls, **fields: Any) -> OracleCaseSpec:
        payload = cls._payload(**fields)
        return cls(case_id=content_id("case", payload), **fields)

    @staticmethod
    def _payload(**fields: Any) -> dict[str, Any]:
        return {
            "oracle_id": fields["oracle_id"],
            "program_id": fields["program_id"],
            "base_input_id": fields["base_input_id"],
            "variation_id": fields["variation_id"],
            "parameter_assignment": fields["parameter_assignment"],
            "domain_cell": fields["domain_cell"],
            "result_roles": {
                role: dump_value(value)
                for role, value in fields["result_roles"].items()
            },
            "required_observation_roles": list(
                fields["required_observation_roles"]
            ),
        }

    def to_hash_payload(self) -> dict[str, Any]:
        return self._payload(
            oracle_id=self.oracle_id,
            program_id=self.program_id,
            base_input_id=self.base_input_id,
            variation_id=self.variation_id,
            parameter_assignment=dict(self.parameter_assignment),
            domain_cell=self.domain_cell,
            result_roles=dict(self.result_roles),
            required_observation_roles=self.required_observation_roles,
        )

    @model_validator(mode="after")
    def _validate_case(self) -> OracleCaseSpec:
        if any(not name.startswith("$") for name in self.parameter_assignment):
            raise ValueError("case parameter assignment names must start with $")
        canonical_json(self.parameter_assignment)
        if self.required_observation_roles != tuple(
            sorted(set(self.required_observation_roles))
        ):
            raise ValueError(
                "required_observation_roles must be sorted and unique"
            )
        missing = set(self.required_observation_roles) - set(self.result_roles)
        if missing:
            raise ValueError(f"result_roles do not cover required roles: {sorted(missing)}")
        expected = content_id("case", self.to_hash_payload())
        if self.case_id != expected:
            raise ValueError("case_id does not match semantic payload")
        return self


class CaseExpansionError(ValueError):
    pass


@runtime_dataclass(frozen=True)
class CaseExpansionSeed:
    """One finite program/input/domain point eligible for selector matching."""

    program: ProgramManifest
    base_input: InputManifest
    origins: tuple[ProvenanceOrigin, ...]
    domain_cell: DomainCell
    base_scenario: ScenarioSpec


@runtime_dataclass(frozen=True)
class ExpandedOracleCase:
    variation: VariationSpec
    case: OracleCaseSpec


VariationMaterializer = Callable[
    [VariationTemplateSpec, dict[str, JsonScalar], ScenarioSpec],
    VariationSpec,
]


def _finite_seeds(
    seeds: Iterable[CaseExpansionSeed],
    *,
    max_sources: int,
) -> tuple[CaseExpansionSeed, ...]:
    if max_sources <= 0:
        raise CaseExpansionError("max_sources must be positive")
    bounded = tuple(islice(seeds, max_sources + 1))
    if len(bounded) > max_sources:
        raise CaseExpansionError(
            f"source inventory exceeds finite bound max_sources={max_sources}"
        )
    return bounded


def _validate_seed(seed: CaseExpansionSeed) -> None:
    if not seed.origins or seed.origins != tuple(sorted(set(seed.origins))):
        raise CaseExpansionError("seed origins must be non-empty, sorted, and unique")
    if seed.base_scenario.program_id != seed.program.program_id:
        raise CaseExpansionError("seed scenario program_id does not match program")
    if seed.base_scenario.input_id != seed.base_input.input_id:
        raise CaseExpansionError("seed scenario input_id does not match base input")
    observed = seed.domain_cell.observed_from
    if observed.program_id != seed.program.program_id:
        raise CaseExpansionError("domain cell program_id does not match seed")
    if observed.input_id != seed.base_input.input_id:
        raise CaseExpansionError("domain cell input_id does not match seed")
    if observed.artifact_id != seed.base_scenario.artifact_id:
        raise CaseExpansionError("domain cell artifact_id does not match seed scenario")


def _parameter_assignments(
    oracle: OracleSpec,
    template: VariationTemplateSpec,
) -> tuple[dict[str, JsonScalar], ...]:
    oracle_variation = oracle.variation
    if oracle_variation is None:
        raise CaseExpansionError("case expansion requires oracle variation")
    if oracle_variation.variation_template_id != template.variation_template_id:
        raise CaseExpansionError("oracle references a different variation template")
    if set(oracle_variation.param_grid) != set(template.param_schema):
        raise CaseExpansionError(
            "param_grid keys must exactly match variation template param_schema"
        )

    names = tuple(sorted(template.param_schema))
    grids: list[tuple[JsonScalar, ...]] = []
    for name in names:
        values = oracle_variation.param_grid[name]
        allowed = {
            canonical_json(value) for value in template.param_schema[name].enum
        }
        rejected = [
            value for value in values if canonical_json(value) not in allowed
        ]
        if rejected:
            raise CaseExpansionError(
                f"param_grid {name} contains values outside template enum: {rejected}"
            )
        grids.append(values)
    return tuple(
        dict(zip(names, values, strict=True))
        for values in product(*grids)
    )


def expand_oracle_cases(
    *,
    oracle: OracleSpec,
    template: VariationTemplateSpec,
    seeds: Iterable[CaseExpansionSeed],
    materialize_variation: VariationMaterializer,
    max_sources: int = 10_000,
    max_cases: int = 10_000,
) -> tuple[ExpandedOracleCase, ...]:
    """Deterministically expand a finite selector × param_grid inventory."""
    if max_cases <= 0:
        raise CaseExpansionError("max_cases must be positive")
    inventory = _finite_seeds(seeds, max_sources=max_sources)
    for seed in inventory:
        _validate_seed(seed)

    selector = oracle.source.selector
    eligible = [
        seed
        for seed in inventory
        if seed.program.program_id in selector.program_ids
        and set(seed.origins).intersection(selector.origin)
    ]
    if not eligible:
        raise CaseExpansionError("source selector matched no program/input seeds")

    deduplicated = {
        (
            seed.program.program_id,
            seed.base_input.input_id,
            seed.domain_cell.cell_id,
            canonical_json(seed.base_scenario),
        ): seed
        for seed in eligible
    }
    ordered_seeds = tuple(
        deduplicated[key] for key in sorted(deduplicated)
    )
    assignments = _parameter_assignments(oracle, template)
    projected_count = len(ordered_seeds) * len(assignments)
    if projected_count > max_cases:
        raise CaseExpansionError(
            f"case expansion size {projected_count} exceeds max_cases={max_cases}"
        )

    required = {
        item.role: item.run_kind for item in oracle.relation.required_observations
    }
    if set(required) != set(template.roles_produced):
        raise CaseExpansionError(
            "template roles_produced must exactly match required observation roles"
        )

    expanded: dict[str, ExpandedOracleCase] = {}
    for seed in ordered_seeds:
        for assignment in assignments:
            variation = materialize_variation(
                template,
                dict(assignment),
                seed.base_scenario,
            )
            if variation.variation_template_id != template.variation_template_id:
                raise CaseExpansionError("materializer returned a different template")
            if variation.parameter_assignment != assignment:
                raise CaseExpansionError("materializer returned a different assignment")
            if variation.base_scenario != seed.base_scenario:
                raise CaseExpansionError("materializer returned a different base scenario")
            if set(variation.result_roles) != set(required):
                raise CaseExpansionError(
                    "variation result roles do not match required observation roles"
                )
            if (
                variation.failure is not None
                or not variation.static_trigger_possible
                or variation.guard_check.status.value != "pass"
            ):
                raise CaseExpansionError(
                    "non-executable variation cannot produce an OracleCase"
                )

            result_roles = {
                role: CaseResultRole(
                    run_kind=required[role],
                    scenario=variation.result_roles[role],
                )
                for role in sorted(required)
            }
            case = OracleCaseSpec.create(
                oracle_id=oracle.oracle_id,
                program_id=seed.program.program_id,
                base_input_id=seed.base_input.input_id,
                variation_id=variation.variation_id,
                parameter_assignment=dict(assignment),
                domain_cell=seed.domain_cell.cell_id,
                result_roles=result_roles,
                required_observation_roles=tuple(sorted(required)),
            )
            existing = expanded.get(case.case_id)
            candidate = ExpandedOracleCase(variation=variation, case=case)
            if existing is not None and existing != candidate:
                raise CaseExpansionError(
                    f"case ID collision for {case.case_id}"
                )
            expanded[case.case_id] = candidate

    return tuple(expanded[case_id] for case_id in sorted(expanded))
