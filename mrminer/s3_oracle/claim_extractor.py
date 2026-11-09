"""Per-PR structured ClaimCandidate extraction with evidence sanitization."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Annotated, Any, Literal, get_args

from pydantic import Field, StringConstraints, model_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass

from mrminer.common.canonical import sha256_digest
from mrminer.common.schema import (
    EvidenceSpec,
    EvidenceStrength,
    GuardDimension,
    GuardDomain,
    GuardMode,
    LlmAudit,
    OracleAspect,
    ShapeConstraints,
)
from mrminer.common.schema.base import SCHEMA_CONFIG, SchemaRecord
from mrminer.llm import StructuredLlmProvider, invoke_structured
from mrminer.s2_graph import (
    ClaimCandidate,
    ClaimedStatement,
    ClaimPredicate,
)
from mrminer.s3_oracle.evidence_bundle import EvidenceSource, PrEvidenceBundle

NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]
EntityId = Annotated[str, StringConstraints(pattern=r"^[a-z]+:[^\s/]+$")]

_EXTRACTOR_VERSION = "s2.claim_extractor/v2"
SupportedEvidenceField = Literal[
    "target.binding",
    "aspect",
    "claimed.kind",
    "claimed.text_normalized",
    "claimed.predicate",
    "domain.dtype",
    "domain.device",
    "domain.overload",
    "domain.layout",
    "domain.shape_mode",
    "domain.phase",
    "domain.backend",
    "domain.shape_constraints",
]
_SUPPORTED_FIELDS = frozenset(get_args(SupportedEvidenceField))
_SYSTEM_PROMPT = f"""\
Extract zero or more compiler-behavior claims from exactly one PR evidence bundle.
Return only the supplied JSON schema. Cite only registered source keys and absolute
character spans. Every claim must cite claimed.predicate. Do not output evidence
strength; the sanitizer derives it from source_kind. Output supports_field must
be one of: {", ".join(sorted(_SUPPORTED_FIELDS))}. Never copy supports_field values
from input graph_evidence; those label graph edges, not claim fields. Set every domain
dimension to unknown with empty values by default. Use SET or ANY only when a separate
evidence span supports that exact domain field; ANY requires an explicit universal
statement. claimed.predicate must cite prose or test evidence that states the behavior.
For each target ATen operator, cite target.binding with an exact span that names the
operator; target.binding may come from a different prose, test, or patch source. A
claimed.predicate span that itself names the operator also satisfies target.binding.
entity_id and graph_evidence are retrieval hints, not claim evidence. A patch mention
alone is only target-binding or supporting metadata, never behavior evidence.

Emit a SEMANTIC claim ONLY when it asserts the operator's computed OUTPUT (values,
dtype, shape, or whether it raises an error) is a specific way or CHANGES under a
specific condition: a correctness fix, a feature, an edge case, an error condition,
or a decomposition producing specific or different output values. A codegen or
compilation crash fix qualifies (pre-fix the operator errored or produced wrong
output; post-fix it produces the correct output). Do NOT emit a SEMANTIC claim that
merely defines what the operator computes or returns ("reverses the elements",
"returns a tuple", "is a mutation operation"), states support / existence /
availability ("is supported by X", "available on Y"), describes usage in tests or
tracing, or reports tags, registration, dispatch routing, call signature, formatting,
or internal implementation details (which function / pass / namespace is used) — those
are not metamorphic oracle claims, so omit them entirely. When the property is about
an error being raised, a control-flow decision, or performance rather than output
values, use aspect effect / decision / performance instead of semantic.
"""
_DIMENSION_FIELDS = (
    "dtype",
    "device",
    "overload",
    "layout",
    "shape_mode",
    "phase",
    "backend",
)
_SOURCE_KEY_WITH_LOCATOR = re.compile(
    r"^(?P<prefix>.+)@chars (?P<start>\d+)-(?P<end>\d+)$"
)

# A SEMANTIC claim is coerced to the positive `semantic_close eq true` predicate
# (MR-C1). If its own text asserts the operator is NOT equivalent, that coercion
# would emit an oracle claiming the opposite of the evidence, so we reject it
# fail-closed rather than invert its polarity. Deliberately narrow: only phrases
# that directly negate equivalence as the claim's assertion (not incidental
# "differs"/"may fail" bug descriptions, which are the correct framing for an
# equality MR the buggy version violates).
_EQUIVALENCE_NEGATION_MARKERS = (
    "not bitwise equivalent",
    "not equivalent",
    "not equal",
    "not identical",
    "not the same",
    "not preserved",
    "no longer equivalent",
    "no longer equal",
    "does not match",
    "do not match",
    "is not consistent with",
)

# A SEMANTIC value-equivalence oracle needs the claim to assert a *behavioural*
# property (a fix / change in what the operator computes). Support/existence/usage/
# identity statements ("is supported by inductor", "is a mutation operation", "is
# used in symbolic tracing", "supports MPS backend") assert no differential output
# property, yet can still mint a meaningless MR when their PR happens to bind a
# program and an axis keyword appears incidentally in evidence (round 3 set_ #130584,
# round 4 nll #95853). Reject them fail-closed. Deliberately narrow: only
# support/existence/usage/identity phrasings — behavioural claims ("now properly
# ignores", "no longer assumes", "uses only target[0] to match") carry no marker.
_SUPPORT_EXISTENCE_MARKERS = (
    "is supported by",
    " supports ",
    "is used in",
    "is used to",
    "is implemented",
    "is available",
    "is listed",
    "is registered",
)
_IDENTITY_DEFINITION_RE = re.compile(
    r"\bis an? [a-z]+ (operation|op|kernel|function|view op)\b"
)


class ClaimExtractionError(ValueError):
    pass


@pydantic_dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class GuardDimensionDraft(SchemaRecord):
    RECORD_KIND = "claim_guard_dimension_draft"

    mode: GuardMode
    values: tuple[NonEmptyStr, ...] = ()

    @model_validator(mode="after")
    def _validate_values(self) -> GuardDimensionDraft:
        if self.mode is GuardMode.SET and not self.values:
            raise ValueError("SET dimension requires values")
        if self.mode is not GuardMode.SET and self.values:
            raise ValueError("only SET dimension may define values")
        return self


@pydantic_dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class GuardDomainDraft(SchemaRecord):
    RECORD_KIND = "claim_guard_domain_draft"

    dtype: GuardDimensionDraft
    device: GuardDimensionDraft
    overload: GuardDimensionDraft
    layout: GuardDimensionDraft
    shape_mode: GuardDimensionDraft
    phase: GuardDimensionDraft
    backend: GuardDimensionDraft
    shape_constraints: ShapeConstraints | None = None


@pydantic_dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class EvidenceSpanDraft(SchemaRecord):
    RECORD_KIND = "claim_evidence_span_draft"

    source_key: NonEmptyStr
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    supports_field: SupportedEvidenceField

    @model_validator(mode="after")
    def _validate_span(self) -> EvidenceSpanDraft:
        if self.char_end <= self.char_start:
            raise ValueError("evidence span end must exceed start")
        return self


@pydantic_dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class ExtractedClaimDraft(SchemaRecord):
    RECORD_KIND = "extracted_claim_draft"

    target_entities: tuple[EntityId, ...]
    aspect: OracleAspect
    claimed_kind: NonEmptyStr
    text_normalized: NonEmptyStr
    predicate: ClaimPredicate
    domain: GuardDomainDraft
    evidence_spans: tuple[EvidenceSpanDraft, ...]
    confidence: float = Field(ge=0, le=1)


@pydantic_dataclass(frozen=True, kw_only=True, config=SCHEMA_CONFIG)
class ClaimExtractionEnvelope(SchemaRecord):
    RECORD_KIND = "claim_extraction_envelope"

    schema_version: Literal["claim_extraction_v2"] = "claim_extraction_v2"
    claims: tuple[ExtractedClaimDraft, ...]


@dataclass(frozen=True, kw_only=True)
class RejectedClaim:
    index: int
    reason: str
    draft: ExtractedClaimDraft

    def to_record(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "reason": self.reason,
            "draft": self.draft.to_record(),
        }


@dataclass(frozen=True, kw_only=True)
class ClaimNormalization:
    index: int
    claim_id: str
    changes: tuple[str, ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "claim_id": self.claim_id,
            "changes": list(self.changes),
        }


@dataclass(frozen=True, kw_only=True)
class ClaimExtractionReport:
    pr_number: int
    claims: tuple[ClaimCandidate, ...]
    rejected: tuple[RejectedClaim, ...]
    normalizations: tuple[ClaimNormalization, ...]
    llm_audit: LlmAudit
    provider_request_id: str | None

    def to_record(self) -> dict[str, Any]:
        return {
            "pr_number": self.pr_number,
            "claims": [claim.to_record() for claim in self.claims],
            "rejected": [item.to_record() for item in self.rejected],
            "normalizations": [
                item.to_record() for item in self.normalizations
            ],
            "llm_audit": self.llm_audit.to_record(),
            "provider_request_id": self.provider_request_id,
        }


class ClaimExtractor:
    def __init__(
        self,
        provider: StructuredLlmProvider | None,
        *,
        extractor_version: str = _EXTRACTOR_VERSION,
    ):
        if not extractor_version:
            raise ValueError("extractor_version must be non-empty")
        self.provider = provider
        self.extractor_version = extractor_version

    def extract_pr(
        self,
        bundle: PrEvidenceBundle,
        *,
        effective_artifact: str,
        allowed_target_entities: tuple[str, ...] | None = None,
    ) -> ClaimExtractionReport:
        if not effective_artifact.startswith("art:"):
            raise ValueError("effective_artifact must be an art: identifier")
        allowed = frozenset(allowed_target_entities or (bundle.entity_id,))
        if bundle.entity_id not in allowed:
            raise ValueError("bundle entity must be an allowed target")
        result = invoke_structured(
            self.provider,
            task="operator_claim_extraction",
            system_prompt=_SYSTEM_PROMPT,
            input_payload=bundle.to_prompt_payload(),
            output_schema=ClaimExtractionEnvelope.json_schema(),
            request_version="operator_claim_extraction_request_v2",
        )
        try:
            envelope = ClaimExtractionEnvelope.from_record(result.payload)
        except Exception as error:
            raise ClaimExtractionError(
                f"LLM claim envelope failed schema validation: {error}"
            ) from error

        claims: list[ClaimCandidate] = []
        rejected: list[RejectedClaim] = []
        normalizations: list[ClaimNormalization] = []
        for index, draft in enumerate(envelope.claims):
            try:
                claim, changes = self._sanitize_claim(
                    draft,
                    bundle=bundle,
                    effective_artifact=effective_artifact,
                    allowed_target_entities=allowed,
                )
                claims.append(claim)
                if changes:
                    normalizations.append(
                        ClaimNormalization(
                            index=index,
                            claim_id=claim.claim_id,
                            changes=changes,
                        )
                    )
            except (ClaimExtractionError, ValueError) as error:
                rejected.append(
                    RejectedClaim(index=index, reason=str(error), draft=draft)
                )
        return ClaimExtractionReport(
            pr_number=bundle.pr_number,
            claims=tuple(claims),
            rejected=tuple(rejected),
            normalizations=tuple(normalizations),
            llm_audit=LlmAudit(model=result.model, prompt_sha=result.prompt_sha),
            provider_request_id=result.request_id,
        )

    def _sanitize_claim(
        self,
        draft: ExtractedClaimDraft,
        *,
        bundle: PrEvidenceBundle,
        effective_artifact: str,
        allowed_target_entities: frozenset[str],
    ) -> tuple[ClaimCandidate, tuple[str, ...]]:
        targets = tuple(sorted(set(draft.target_entities)))
        if not targets or not set(targets) <= allowed_target_entities:
            raise ClaimExtractionError("claim contains an unknown target entity")
        if draft.predicate.subject not in targets:
            raise ClaimExtractionError("predicate subject is not a claim target")
        sources = bundle.source_map()
        normalizations: list[str] = []
        operator_tokens = {
            target.rsplit("::", 1)[-1].lower()
            for target in targets
            if target.startswith("op:aten::")
        }
        retained_spans: list[EvidenceSpanDraft] = []
        for span in draft.evidence_spans:
            if span.supports_field == "target.binding" and operator_tokens:
                quote = self._span_text(span, sources).lower()
                if not any(token in quote for token in operator_tokens):
                    normalizations.append(
                        "evidence.target.binding:dropped:operator_not_in_span"
                    )
                    continue
            retained_spans.append(span)
        evidence = tuple(
            sorted(
                (
                    self._sanitize_span(span, sources)
                    for span in retained_spans
                ),
                key=lambda item: (
                    item.artifact_ref,
                    item.locator,
                    item.supports_field,
                    item.strength.value,
                ),
            )
        )
        unique_evidence = {
            (
                item.artifact_ref,
                item.locator,
                item.supports_field,
                item.strength.value,
            ): item
            for item in evidence
        }
        evidence = tuple(unique_evidence[key] for key in sorted(unique_evidence))
        if not any(item.supports_field == "claimed.predicate" for item in evidence):
            raise ClaimExtractionError("claim predicate has no cited evidence")
        binding_spans = tuple(
            span
            for span in retained_spans
            if span.supports_field in {"claimed.predicate", "target.binding"}
        )
        unbound_operators = {
            token
            for token in operator_tokens
            if not any(
                token in self._span_text(span, sources).lower()
                for span in binding_spans
            )
        }
        if unbound_operators:
            raise ClaimExtractionError(
                "claim predicate evidence does not name a target operator"
            )
        if draft.aspect in {OracleAspect.EFFECT, OracleAspect.DECISION} and not any(
            item.supports_field == "claimed.predicate"
            and item.strength is EvidenceStrength.EXPLICIT_CLAIM
            for item in evidence
        ):
            raise ClaimExtractionError(
                "effect/decision predicate requires explicit prose evidence"
            )
        if not any(
            item.supports_field == "claimed.predicate"
            and item.strength
            in {
                EvidenceStrength.EXPLICIT_CLAIM,
                EvidenceStrength.TEST_ASSERTION,
            }
            for item in evidence
        ):
            raise ClaimExtractionError(
                "claim predicate requires explicit prose or test assertion evidence"
            )

        dimensions: dict[str, GuardDimension] = {}
        for name in _DIMENSION_FIELDS:
            dimension = getattr(draft.domain, name)
            refs = tuple(
                sorted(
                    {
                        item.quote_hash
                        for item in evidence
                        if item.supports_field == f"domain.{name}"
                    }
                )
            )
            mode = dimension.mode
            values = tuple(sorted(set(dimension.values)))
            if mode in {GuardMode.SET, GuardMode.ANY} and not refs:
                normalizations.append(
                    f"domain.{name}:{mode.value}->unknown:no_evidence"
                )
                mode = GuardMode.UNKNOWN
                values = ()
            dimensions[name] = GuardDimension(
                mode=mode,
                values=values,
                evidence_refs=refs,
            )
        shape_constraints = draft.domain.shape_constraints
        if shape_constraints is not None and not any(
            item.supports_field == "domain.shape_constraints" for item in evidence
        ):
            normalizations.append(
                "domain.shape_constraints:set->none:no_evidence"
            )
            shape_constraints = None
        domain = GuardDomain(
            **dimensions,
            shape_constraints=shape_constraints,
        )
        predicate = draft.predicate
        if draft.aspect is OracleAspect.SEMANTIC:
            lowered_text = draft.text_normalized.lower()
            negation = next(
                (m for m in _EQUIVALENCE_NEGATION_MARKERS if m in lowered_text),
                None,
            )
            if negation is not None:
                raise ClaimExtractionError(
                    "semantic-equivalence claim contradicted by non-equivalence "
                    f"language in its text ({negation!r}); refusing to coerce to "
                    "semantic_close eq true"
                )
            support_marker = next(
                (m for m in _SUPPORT_EXISTENCE_MARKERS if m in lowered_text),
                None,
            )
            if support_marker is None and _IDENTITY_DEFINITION_RE.search(
                lowered_text
            ):
                support_marker = "is a <kind> operation"
            if support_marker is not None:
                raise ClaimExtractionError(
                    "semantic claim states support/existence/identity, not a "
                    f"differential behaviour ({support_marker!r}); no value-MR to "
                    "witness"
                )
            semantic_predicate = ClaimPredicate(
                subject=draft.predicate.subject,
                property="semantic_close",
                operator="eq",
                expected=True,
                probe_phase="post_execution",
                preferred_probe="semantic.paired_capture/v1",
            )
            if predicate.to_record() != semantic_predicate.to_record():
                normalizations.append(
                    "claimed.predicate:semantic->semantic_close:eq:true"
                )
            predicate = semantic_predicate
        claim = ClaimCandidate.create(
            source_pr=bundle.pr_number,
            effective_artifact=effective_artifact,
            target_entities=targets,
            pattern_fallback=None,
            aspect=draft.aspect,
            claimed=ClaimedStatement(
                kind=draft.claimed_kind,
                text_normalized=draft.text_normalized,
                predicate=predicate,
            ),
            domain=domain,
            evidence=evidence,
            confidence=draft.confidence,
            extractor_version=self.extractor_version,
        )
        return claim, tuple(normalizations)

    def _sanitize_span(
        self,
        span: EvidenceSpanDraft,
        sources: dict[str, EvidenceSource],
    ) -> EvidenceSpec:
        if span.supports_field not in _SUPPORTED_FIELDS:
            raise ClaimExtractionError(
                f"unsupported evidence field {span.supports_field!r}"
            )
        source = self._resolve_source(span, sources)
        if not (
            source.char_start
            <= span.char_start
            < span.char_end
            <= source.char_end
        ):
            raise ClaimExtractionError("evidence span lies outside the retained source")
        quote = self._span_text(span, sources)
        if not quote:
            raise ClaimExtractionError("evidence quote cannot be empty")
        strength = {
            "prose": EvidenceStrength.EXPLICIT_CLAIM,
            "test": EvidenceStrength.TEST_ASSERTION,
            "patch": EvidenceStrength.INFERRED,
        }[source.source_kind]
        return EvidenceSpec(
            artifact_ref=source.artifact_ref,
            locator=(
                f"{source.source_id}@chars "
                f"{span.char_start}-{span.char_end}"
            ),
            quote=quote,
            quote_hash=sha256_digest(quote),
            strength=strength,
            supports_field=span.supports_field,
            extractor=self.extractor_version,
        )

    @staticmethod
    def _span_text(
        span: EvidenceSpanDraft,
        sources: dict[str, EvidenceSource],
    ) -> str:
        source = ClaimExtractor._resolve_source(span, sources)
        if not (
            source.char_start
            <= span.char_start
            < span.char_end
            <= source.char_end
        ):
            raise ClaimExtractionError("evidence span lies outside the retained source")
        relative_start = span.char_start - source.char_start
        relative_end = span.char_end - source.char_start
        return source.text[relative_start:relative_end]

    @staticmethod
    def _resolve_source(
        span: EvidenceSpanDraft,
        sources: dict[str, EvidenceSource],
    ) -> EvidenceSource:
        direct = sources.get(span.source_key)
        if direct is not None:
            return direct
        match = _SOURCE_KEY_WITH_LOCATOR.fullmatch(span.source_key)
        if match is None or (
            int(match.group("start")),
            int(match.group("end")),
        ) != (span.char_start, span.char_end):
            raise ClaimExtractionError(f"unknown evidence source {span.source_key!r}")
        prefix = f"{match.group('prefix')}@"
        candidates = [
            source
            for key, source in sources.items()
            if key.startswith(prefix)
            and source.char_start <= span.char_start
            and span.char_end <= source.char_end
        ]
        if len(candidates) != 1:
            raise ClaimExtractionError(f"unknown evidence source {span.source_key!r}")
        return candidates[0]
