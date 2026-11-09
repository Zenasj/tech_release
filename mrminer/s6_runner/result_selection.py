"""Deterministic attempt selection that never hides non-environment conflicts."""

from __future__ import annotations

from mrminer.common.canonical import canonical_json
from mrminer.common.schema import (
    FlakinessSpec,
    ObservationSpec,
    ObservationStatus,
    RunResultSet,
    SelectionRule,
)


def _is_environment_error(observation: ObservationSpec) -> bool:
    return (
        observation.error is not None
        and observation.error.stage == "environment"
    )


def _outcome_signature(observation: ObservationSpec) -> str:
    if observation.status is ObservationStatus.OK:
        semantic = observation.semantic
        return canonical_json(
            {
                "status": observation.status.value,
                "semantic": (
                    {
                        "stability": semantic.stability.value,
                        "allclose": semantic.pairwise_compare.allclose,
                        "structure_equal": (
                            semantic.pairwise_compare.structure_equal
                        ),
                        "mutation_equal": semantic.pairwise_compare.mutation_equal,
                        "alias_equal": semantic.pairwise_compare.alias_equal,
                    }
                    if semantic is not None
                    else None
                ),
                "decision": (
                    observation.decision.to_record()
                    if observation.decision is not None
                    else None
                ),
                "performance": (
                    observation.performance.to_record()
                    if observation.performance is not None
                    else None
                ),
            }
        )
    return canonical_json(
        {
            "status": observation.status.value,
            "stage": observation.error.stage if observation.error else None,
            "exception_type": (
                observation.error.exception_type
                if observation.error
                else None
            ),
        }
    )


def select_run_results(
    run_id: str,
    observations: tuple[ObservationSpec, ...],
) -> RunResultSet:
    if not observations:
        raise ValueError("attempt selection requires observations")
    ordered = tuple(
        sorted(observations, key=lambda item: (item.attempt, item.obs_id))
    )
    if any(observation.run_id != run_id for observation in ordered):
        raise ValueError("all observations must reference the selected run")
    if len({observation.attempt for observation in ordered}) != len(ordered):
        raise ValueError("attempt numbers must be unique per run")

    substantive = tuple(
        observation
        for observation in ordered
        if not _is_environment_error(observation)
    )
    signatures = {
        _outcome_signature(observation) for observation in substantive
    }
    if not substantive:
        return RunResultSet(
            run_id=run_id,
            attempts=tuple(item.obs_id for item in ordered),
            canonical_result_ref=None,
            selection_rule=SelectionRule.INCONCLUSIVE_FLAKY,
            flakiness=FlakinessSpec(
                flaky=True,
                pattern="environment_only_no_canonical_result",
            ),
        )
    if len(signatures) > 1:
        statuses = sorted(
            {observation.status.value for observation in substantive}
        )
        return RunResultSet(
            run_id=run_id,
            attempts=tuple(item.obs_id for item in ordered),
            canonical_result_ref=None,
            selection_rule=SelectionRule.INCONCLUSIVE_FLAKY,
            flakiness=FlakinessSpec(
                flaky=True,
                pattern=f"non_environment_conflict:{','.join(statuses)}",
            ),
        )

    canonical = substantive[0]
    return RunResultSet(
        run_id=run_id,
        attempts=tuple(item.obs_id for item in ordered),
        canonical_result_ref=canonical.obs_id,
        selection_rule=SelectionRule.ENV_RETRY_OK,
        flakiness=FlakinessSpec(flaky=False),
    )
