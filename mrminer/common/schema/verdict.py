"""Template-arbitrated verdict records."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import Field, StringConstraints, model_validator
from pydantic.dataclasses import dataclass

from mrminer.common.canonical import canonical_bytes, content_id
from mrminer.common.schema.base import SCHEMA_CONFIG, SchemaRecord, dump_value
from mrminer.common.schema.run import CheckId, ObservationRef

NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]
VerdictId = Annotated[
    str, StringConstraints(pattern=r"^ver:sha256:[0-9a-f]{64}$")
]
OracleId = Annotated[
    str, StringConstraints(pattern=r"^orc:sha256:[0-9a-f]{64}$")
]


class VerdictOutcome(str, Enum):
    PASS = "pass"
    BUG_CANDIDATE = "bug_candidate"
    NOT_APPLICABLE = "not_applicable"
    UNSUPPORTED = "unsupported"
    INCONCLUSIVE = "inconclusive"
    ORACLE_INVALID = "oracle_invalid"
    ORACLE_EXPIRED = "oracle_expired"
    ENVIRONMENT_ERROR = "environment_error"
    HARNESS_ERROR = "harness_error"
    COMPILER_CRASH = "compiler_crash"
    TIMEOUT = "timeout"


class StatisticalVerdict(str, Enum):
    NON_INFERIOR = "non_inferior"
    REGRESSED = "regressed"
    INCONCLUSIVE = "inconclusive"
    IMPROVED = "improved"


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class StatisticalResult(SchemaRecord):
    RECORD_KIND = "statistical_result"

    metric: NonEmptyStr
    direction: Literal["lower_better", "higher_better"]
    margin_delta: float = Field(ge=0)
    confidence_level: float = Field(gt=0, lt=1)
    ci: tuple[float, float]
    estimator: NonEmptyStr
    verdict: StatisticalVerdict

    @model_validator(mode="after")
    def _validate_ci(self) -> StatisticalResult:
        if self.ci[0] > self.ci[1]:
            raise ValueError("confidence interval lower bound cannot exceed upper bound")
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class Tier2Result(SchemaRecord):
    RECORD_KIND = "tier2_result"

    effect: NonEmptyStr | None
    decision_diff: dict[str, Any]

    @model_validator(mode="after")
    def _validate_diff(self) -> Tier2Result:
        canonical_bytes(self.decision_diff)
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class Tier3Result(SchemaRecord):
    RECORD_KIND = "tier3_result"

    stat: StatisticalResult | None


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class TierResults(SchemaRecord):
    RECORD_KIND = "tier_results"

    tier0: NonEmptyStr
    tier1: NonEmptyStr
    tier2: Tier2Result
    tier3: Tier3Result


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class BugCandidateDetail(SchemaRecord):
    RECORD_KIND = "bug_candidate_detail"

    violated_contracts: tuple[NonEmptyStr, ...]
    supporting_signals: tuple[NonEmptyStr, ...]
    contradicting_signals: tuple[NonEmptyStr, ...]
    confidence: float = Field(ge=0, le=1)
    required_followup: tuple[NonEmptyStr, ...]

    @model_validator(mode="after")
    def _validate_contracts(self) -> BugCandidateDetail:
        if not self.violated_contracts:
            raise ValueError("bug candidates require at least one violated contract")
        return self


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class VerdictInterpretation(SchemaRecord):
    RECORD_KIND = "verdict_interpretation"

    benign_change: NonEmptyStr | None = None


@dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class VerdictSpec(SchemaRecord):
    RECORD_KIND = "verdict"

    schema_version: Literal["verdict_v1"] = "verdict_v1"
    verdict_id: VerdictId
    check_id: CheckId
    oracle_id: OracleId
    obs_refs: tuple[ObservationRef, ...]
    flakiness_noted: bool
    tier_results: TierResults
    outcome: VerdictOutcome
    bug_candidate_detail: BugCandidateDetail | None
    interpretation: VerdictInterpretation
    arbitration_ref: NonEmptyStr | None
    checker_version: NonEmptyStr
    created_at: NonEmptyStr

    @classmethod
    def create(cls, **fields: Any) -> VerdictSpec:
        payload = cls._payload(**fields)
        return cls(verdict_id=content_id("ver", payload), **fields)

    @staticmethod
    def _payload(**fields: Any) -> dict[str, Any]:
        return {
            "check_id": fields["check_id"],
            "oracle_id": fields["oracle_id"],
            "obs_refs": list(fields["obs_refs"]),
            "flakiness_noted": fields["flakiness_noted"],
            "tier_results": dump_value(fields["tier_results"]),
            "outcome": (
                fields["outcome"].value
                if isinstance(fields["outcome"], VerdictOutcome)
                else fields["outcome"]
            ),
            "bug_candidate_detail": (
                dump_value(fields["bug_candidate_detail"])
                if fields["bug_candidate_detail"] is not None
                else None
            ),
            "interpretation": dump_value(fields["interpretation"]),
            "arbitration_ref": fields["arbitration_ref"],
            "checker_version": fields["checker_version"],
            "created_at": fields["created_at"],
        }

    def to_hash_payload(self) -> dict[str, Any]:
        return self._payload(
            check_id=self.check_id,
            oracle_id=self.oracle_id,
            obs_refs=self.obs_refs,
            flakiness_noted=self.flakiness_noted,
            tier_results=self.tier_results,
            outcome=self.outcome,
            bug_candidate_detail=self.bug_candidate_detail,
            interpretation=self.interpretation,
            arbitration_ref=self.arbitration_ref,
            checker_version=self.checker_version,
            created_at=self.created_at,
        )

    @model_validator(mode="after")
    def _validate_verdict(self) -> VerdictSpec:
        is_bug = self.outcome is VerdictOutcome.BUG_CANDIDATE
        if is_bug != (self.bug_candidate_detail is not None):
            raise ValueError(
                "bug_candidate_detail must be present exactly for bug_candidate outcomes"
            )
        expected = content_id("ver", self.to_hash_payload())
        if self.verdict_id != expected:
            raise ValueError("verdict_id does not match verdict payload")
        return self
