"""Conservative ClaimCandidate-to-OracleSpec induction."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from mrminer.common.canonical import canonical_json
from mrminer.common.schema import (
    DynamicGuard,
    EnvironmentGuard,
    EvidencePolicy,
    EvidenceStrength,
    LineageSpec,
    LlmAudit,
    OracleAspect,
    OracleBinding,
    OracleClaim,
    OracleGuard,
    OracleProvenance,
    OracleSource,
    OracleSpec,
    OracleVariation,
    ProvenanceOrigin,
    RelationSpec,
    RequiredObservation,
    SourceSelector,
    SpecClass,
    StaticGuard,
    TemplateId,
    TriggerSpec,
    ValiditySpec,
)
from mrminer.common.validate import ContractContext, ContractValidator
from mrminer.llm import StructuredLlmProvider, invoke_structured
from mrminer.s2_graph import ClaimCandidate
from mrminer.s3_oracle.actions import (
    OracleAction,
    OracleActionKind,
    OracleActionStatus,
)
from mrminer.s3_oracle.domain_cluster import (
    ClaimCluster,
    DomainRelation,
    cluster_claims,
    domain_relation,
)


class LineageDecision(str, Enum):
    ADD_EVIDENCE = "add_evidence"
    SUPERSEDE = "supersede"
    REBASELINE = "rebaseline"
    NO_OP = "no_op"


@dataclass(frozen=True, kw_only=True)
class LineageJudgement:
    decision: LineageDecision
    reason: str
    llm_audit: LlmAudit


class LlmLineageJudge:
    """The only LLM entry point allowed to decide same-domain semantic conflict."""

    def __init__(self, provider: StructuredLlmProvider | None):
        self.provider = provider

    def judge(
        self,
        *,
        existing: OracleSpec,
        old_claims: tuple[ClaimCandidate, ...],
        new_claims: tuple[ClaimCandidate, ...],
    ) -> LineageJudgement:
        result = invoke_structured(
            self.provider,
            task="oracle_lineage_judge",
            system_prompt=(
                "Judge only the supplied exact-domain semantic conflict. "
                "Return add_evidence, supersede, rebaseline, or no_op with a reason."
            ),
            input_payload={
                "existing_oracle": existing.to_record(),
                "old_claims": [claim.to_record() for claim in old_claims],
                "new_claims": [claim.to_record() for claim in new_claims],
            },
            output_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["decision", "reason"],
                "properties": {
                    "decision": {
                        "enum": [item.value for item in LineageDecision]
                    },
                    "reason": {"type": "string", "minLength": 1},
                },
            },
            request_version="oracle_lineage_judge_request_v1",
        )
        try:
            decision = LineageDecision(result.payload["decision"])
            reason = result.payload["reason"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid lineage judgement: {error}") from error
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("lineage judgement reason must be non-empty")
        return LineageJudgement(
            decision=decision,
            reason=reason,
            llm_audit=LlmAudit(model=result.model, prompt_sha=result.prompt_sha),
        )


@dataclass(frozen=True, kw_only=True)
class InductionContext:
    validity_artifact: str
    program_ids: tuple[str, ...]
    created_at: str
    pipeline_version: str = "s3.inducer/v1"
    source_origins: tuple[ProvenanceOrigin, ...] = (
        ProvenanceOrigin.PR_REGRESSION_TEST,
    )
    variation_template_id: str | None = None
    param_grid: dict[str, tuple[bool | int | float | str, ...]] = field(
        default_factory=dict
    )
    required_capabilities: tuple[str, ...] = ()
    dynamic_trigger: TriggerSpec | None = None
    relation_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    llm_audit: LlmAudit | None = None

    def __post_init__(self) -> None:
        if not self.validity_artifact.startswith("art:"):
            raise ValueError("validity_artifact must be an art: identifier")
        if not self.created_at or not self.pipeline_version:
            raise ValueError("created_at and pipeline_version must be non-empty")
        if self.program_ids != tuple(sorted(set(self.program_ids))):
            raise ValueError("program_ids must be sorted and unique")
        if self.source_origins != tuple(sorted(set(self.source_origins))):
            raise ValueError("source_origins must be sorted and unique")
        if self.required_capabilities != tuple(
            sorted(set(self.required_capabilities))
        ):
            raise ValueError("required_capabilities must be sorted and unique")


def _predicate_key(claim: ClaimCandidate) -> tuple[str, str, str, str]:
    predicate = claim.claimed.predicate
    return (
        predicate.subject,
        predicate.property,
        predicate.probe_phase,
        predicate.preferred_probe,
    )


def _predicate_value(claim: ClaimCandidate) -> tuple[str, str]:
    predicate = claim.claimed.predicate
    return predicate.operator, canonical_json(predicate.expected)


def _claims_conflict(
    old_claims: tuple[ClaimCandidate, ...],
    new_claims: tuple[ClaimCandidate, ...],
) -> bool:
    old = {_predicate_key(claim): _predicate_value(claim) for claim in old_claims}
    return any(
        key in old and old[key] != _predicate_value(claim)
        for claim in new_claims
        for key in (_predicate_key(claim),)
    )


class OracleInducer:
    def __init__(
        self,
        *,
        lineage_judge: LlmLineageJudge | None = None,
    ):
        self.lineage_judge = lineage_judge

    def induce(
        self,
        *,
        entity_id: str,
        claims: list[ClaimCandidate] | tuple[ClaimCandidate, ...],
        existing_oracles: list[OracleSpec] | tuple[OracleSpec, ...],
        context: InductionContext,
    ) -> list[OracleAction]:
        clustered = cluster_claims(claims, entity_id=entity_id)
        all_claims = {claim.claim_id: claim for claim in claims}
        actions: list[OracleAction] = []
        for cluster in clustered.clusters:
            same_aspect = [
                oracle
                for oracle in existing_oracles
                if oracle.aspect.value == cluster.aspect
                and any(
                    binding.entity_id == entity_id and binding.role == "primary"
                    for binding in oracle.binding
                )
            ]
            exact = [
                oracle
                for oracle in same_aspect
                if domain_relation(oracle.guard.static.domain, cluster.domain)
                is DomainRelation.EXACT
                and oracle.validity.from_artifact == context.validity_artifact
            ]
            if len(exact) > 1:
                actions.append(
                    self._pending(
                        cluster,
                        OracleActionKind.SUSPEND,
                        "multiple existing exact-domain oracles require arbitration",
                        predecessor_ids=tuple(item.oracle_id for item in exact),
                    )
                )
                continue
            if exact:
                actions.append(
                    self._update_exact(
                        cluster,
                        existing=exact[0],
                        all_claims=all_claims,
                        context=context,
                    )
                )
                continue
            blockers = [
                (oracle, domain_relation(oracle.guard.static.domain, cluster.domain))
                for oracle in same_aspect
                if domain_relation(oracle.guard.static.domain, cluster.domain)
                is not DomainRelation.DISJOINT
            ]
            if blockers:
                relation_set = {relation for _, relation in blockers}
                if relation_set <= {
                    DomainRelation.CONTAINS,
                    DomainRelation.CONTAINED_BY,
                }:
                    kind = OracleActionKind.REFINE_GUARD
                    reason = "containment requires an explicit guard refinement"
                elif DomainRelation.PARTIAL in relation_set:
                    kind = OracleActionKind.SPLIT
                    reason = "partial overlap requires domain atomization"
                else:
                    kind = OracleActionKind.SUSPEND
                    reason = "unknown domain overlap requires arbitration"
                actions.append(
                    self._pending(
                        cluster,
                        kind,
                        reason,
                        predecessor_ids=tuple(
                            oracle.oracle_id for oracle, _ in blockers
                        ),
                    )
                )
                continue
            actions.append(self._create(cluster, context=context))
        return actions

    def _create(
        self,
        cluster: ClaimCluster,
        *,
        context: InductionContext,
    ) -> OracleAction:
        try:
            spec = self._fill_spec(
                cluster,
                context=context,
                predecessors=(),
                generation=1,
            )
        except ValueError as error:
            return OracleAction.create(
                kind=OracleActionKind.NO_OP,
                status=OracleActionStatus.NO_OP,
                entity_id=cluster.entity_id,
                aspect=cluster.aspect,
                source_claim_ids=tuple(
                    claim.claim_id for claim in cluster.claims
                ),
                reason=str(error),
            )
        return OracleAction.create(
            kind=OracleActionKind.CREATE,
            status=OracleActionStatus.READY,
            entity_id=cluster.entity_id,
            aspect=cluster.aspect,
            source_claim_ids=tuple(claim.claim_id for claim in cluster.claims),
            new_specs=(spec,),
            reason="new exact-domain claim cluster",
        )

    def _update_exact(
        self,
        cluster: ClaimCluster,
        *,
        existing: OracleSpec,
        all_claims: dict[str, ClaimCandidate],
        context: InductionContext,
    ) -> OracleAction:
        new_claims = tuple(
            claim
            for claim in cluster.claims
            if claim.claim_id not in existing.claim.claim_refs
        )
        if not new_claims:
            return OracleAction.create(
                kind=OracleActionKind.NO_OP,
                status=OracleActionStatus.NO_OP,
                entity_id=cluster.entity_id,
                aspect=cluster.aspect,
                source_claim_ids=tuple(
                    claim.claim_id for claim in cluster.claims
                ),
                predecessor_ids=(existing.oracle_id,),
                reason="all exact-domain claims are already referenced",
            )
        unresolved = set(existing.claim.claim_refs) - set(all_claims)
        if unresolved:
            return self._pending(
                cluster,
                OracleActionKind.SUSPEND,
                f"existing claim refs are unresolved: {sorted(unresolved)}",
                predecessor_ids=(existing.oracle_id,),
            )
        old_claims = tuple(all_claims[item] for item in existing.claim.claim_refs)
        action_kind = OracleActionKind.ADD_EVIDENCE
        audit = context.llm_audit
        reason = "compatible exact-domain claims add evidence"
        if _claims_conflict(old_claims, new_claims):
            if self.lineage_judge is None:
                return self._pending(
                    cluster,
                    OracleActionKind.SUSPEND,
                    "same-domain predicate conflict requires lineage judge",
                    predecessor_ids=(existing.oracle_id,),
                )
            judgement = self.lineage_judge.judge(
                existing=existing,
                old_claims=old_claims,
                new_claims=new_claims,
            )
            reason = judgement.reason
            audit = judgement.llm_audit
            if judgement.decision is LineageDecision.NO_OP:
                return OracleAction.create(
                    kind=OracleActionKind.NO_OP,
                    status=OracleActionStatus.NO_OP,
                    entity_id=cluster.entity_id,
                    aspect=cluster.aspect,
                    source_claim_ids=tuple(
                        claim.claim_id for claim in cluster.claims
                    ),
                    predecessor_ids=(existing.oracle_id,),
                    reason=reason,
                )
            action_kind = OracleActionKind(judgement.decision.value)
        combined = tuple(
            sorted(
                {
                    claim.claim_id: claim
                    for claim in (*old_claims, *cluster.claims)
                }.values(),
                key=lambda item: (item.source_pr, item.claim_id),
            )
        )
        successor_cluster = ClaimCluster(
            cluster_id=cluster.cluster_id,
            entity_id=cluster.entity_id,
            aspect=cluster.aspect,
            domain=cluster.domain,
            claims=combined,
        )
        successor_context = InductionContext(
            validity_artifact=context.validity_artifact,
            program_ids=context.program_ids,
            created_at=context.created_at,
            pipeline_version=context.pipeline_version,
            source_origins=context.source_origins,
            variation_template_id=context.variation_template_id,
            param_grid=context.param_grid,
            required_capabilities=context.required_capabilities,
            dynamic_trigger=context.dynamic_trigger,
            relation_params=context.relation_params,
            llm_audit=audit,
        )
        try:
            successor = self._fill_spec(
                successor_cluster,
                context=successor_context,
                predecessors=(existing.oracle_id,),
                generation=existing.lineage.generation + 1,
            )
        except ValueError as error:
            return OracleAction.create(
                kind=OracleActionKind.NO_OP,
                status=OracleActionStatus.NO_OP,
                entity_id=cluster.entity_id,
                aspect=cluster.aspect,
                source_claim_ids=tuple(
                    claim.claim_id for claim in combined
                ),
                predecessor_ids=(existing.oracle_id,),
                reason=str(error),
            )
        return OracleAction.create(
            kind=action_kind,
            status=OracleActionStatus.READY,
            entity_id=cluster.entity_id,
            aspect=cluster.aspect,
            source_claim_ids=tuple(claim.claim_id for claim in combined),
            predecessor_ids=(existing.oracle_id,),
            new_specs=(successor,),
            reason=reason,
        )

    def _fill_spec(
        self,
        cluster: ClaimCluster,
        *,
        context: InductionContext,
        predecessors: tuple[str, ...],
        generation: int,
    ) -> OracleSpec:
        if not context.program_ids:
            raise ValueError("no executable ProgramManifest IDs were supplied")
        aspect = OracleAspect(cluster.aspect)
        variation_required = aspect in {
            OracleAspect.SEMANTIC,
            OracleAspect.DECISION,
            OracleAspect.PERFORMANCE,
        }
        if variation_required and context.variation_template_id is None:
            raise ValueError(f"{aspect.value} oracle requires a variation template")
        if context.variation_template_id is not None and not context.param_grid:
            raise ValueError("variation template requires a finite param_grid")
        if aspect in {OracleAspect.EFFECT, OracleAspect.DECISION}:
            if not self._claims_have_explicit_predicates(cluster.claims):
                raise ValueError(
                    f"{aspect.value} oracle requires explicit predicate evidence"
                )
        if aspect in {
            OracleAspect.EFFECT,
            OracleAspect.DECISION,
        } and context.dynamic_trigger is None:
            raise ValueError(
                f"{aspect.value} oracle requires an independent pre-trigger"
            )

        relation, template_id, minimum_strength = self._relation(
            aspect,
            cluster,
            context,
        )
        capabilities = set(context.required_capabilities)
        capabilities.update(
            probe
            for observation in relation.required_observations
            for probe in observation.probes
        )
        if context.dynamic_trigger is not None:
            capabilities.add(context.dynamic_trigger.probe)
        members = sorted(
            {
                target
                for claim in cluster.claims
                for target in claim.target_entities
                if target != cluster.entity_id
            }
        )
        bindings = (
            OracleBinding(entity_id=cluster.entity_id, role="primary"),
            *(
                OracleBinding(entity_id=member, role="member")
                for member in members
            ),
        )
        variation = None
        if context.variation_template_id is not None:
            grid = {
                name: tuple(
                    sorted(set(values), key=canonical_json)
                )
                for name, values in sorted(context.param_grid.items())
            }
            variation = OracleVariation(
                variation_template_id=context.variation_template_id,
                param_grid=grid,
            )
        summaries = []
        for claim in cluster.claims:
            text = f"PR #{claim.source_pr}: {claim.claimed.text_normalized}"
            if text not in summaries:
                summaries.append(text)
        spec = OracleSpec.create(
            spec_class=SpecClass.ORACLE,
            aspect=aspect,
            template_id=template_id,
            priority=None,
            binding=tuple(bindings),
            guard=OracleGuard(
                static=StaticGuard(domain=cluster.domain),
                dynamic=DynamicGuard(trigger=context.dynamic_trigger),
                environment=EnvironmentGuard(
                    required_capabilities=tuple(sorted(capabilities))
                ),
            ),
            claim=OracleClaim(
                claim_refs=tuple(
                    sorted(claim.claim_id for claim in cluster.claims)
                ),
                summary="; ".join(summaries),
            ),
            evidence_policy=EvidencePolicy(min_strength=minimum_strength),
            source=OracleSource(
                selector=SourceSelector(
                    origin=context.source_origins,
                    program_ids=context.program_ids,
                )
            ),
            variation=variation,
            relation=relation,
            missing_probe_outcome="unsupported",
            validity=ValiditySpec(from_artifact=context.validity_artifact),
            lineage=LineageSpec(
                generation=generation,
                predecessors=tuple(sorted(predecessors)),
            ),
            provenance=OracleProvenance(
                pipeline_version=context.pipeline_version,
                llm=context.llm_audit,
                created_at=context.created_at,
            ),
        )
        issues = ContractValidator().validate(ContractContext(oracle=spec))
        if issues:
            raise ValueError(
                "generated oracle contract is invalid: "
                + "; ".join(issue.code for issue in issues)
            )
        return spec

    def _relation(
        self,
        aspect: OracleAspect,
        cluster: ClaimCluster,
        context: InductionContext,
    ) -> tuple[RelationSpec, TemplateId, EvidenceStrength]:
        params = dict(context.relation_params.get(aspect.value, {}))
        if aspect is OracleAspect.SEMANTIC:
            params = {"rtol": 1e-4, "atol": 1e-5, **params}
            observations = (
                RequiredObservation(
                    role="source_pair",
                    run_kind="paired_semantic",
                    probes=("semantic.paired_capture/v1",),
                ),
                RequiredObservation(
                    role="followup_pair",
                    run_kind="paired_semantic",
                    probes=("semantic.paired_capture/v1",),
                ),
            )
            return (
                RelationSpec(
                    checker_id="checkers.semantic.paired_close",
                    checker_version="1",
                    params=params,
                    required_observations=observations,
                ),
                TemplateId.MR_C1,
                EvidenceStrength.INFERRED,
            )
        predicates = [
            claim.claimed.predicate.to_record() for claim in cluster.claims
        ]
        if aspect is OracleAspect.EFFECT:
            trigger = context.dynamic_trigger
            assert trigger is not None
            probes = tuple(
                sorted(
                    {
                        trigger.probe,
                        "semantic.paired_capture/v1",
                        *(
                            claim.claimed.predicate.preferred_probe
                            for claim in cluster.claims
                        ),
                    }
                )
            )
            params = {"predicates": predicates, **params}
            return (
                RelationSpec(
                    checker_id="checkers.effect.predicate_holds",
                    checker_version="1",
                    params=params,
                    required_observations=(
                        RequiredObservation(
                            role="source_pair",
                            run_kind="paired_semantic",
                            probes=probes,
                        ),
                    ),
                ),
                TemplateId.MR_C3,
                EvidenceStrength.EXPLICIT_CLAIM,
            )
        if aspect is OracleAspect.DECISION:
            trigger = context.dynamic_trigger
            assert trigger is not None
            probe_tuple = tuple(
                sorted(
                    {
                        trigger.probe,
                        *(
                            claim.claimed.predicate.preferred_probe
                            for claim in cluster.claims
                        ),
                    }
                )
            )
            params = {"predicates": predicates, **params}
            return (
                RelationSpec(
                    checker_id="checkers.decision.predicate_relation",
                    checker_version="1",
                    params=params,
                    required_observations=(
                        RequiredObservation(
                            role="source",
                            run_kind="single",
                            probes=probe_tuple,
                        ),
                        RequiredObservation(
                            role="followup",
                            run_kind="single",
                            probes=probe_tuple,
                        ),
                    ),
                ),
                TemplateId.MR_C2,
                EvidenceStrength.EXPLICIT_CLAIM,
            )
        if not self._claims_have_explicit_predicates(cluster.claims):
            raise ValueError("performance oracle requires explicit predicate evidence")
        params = dict(context.relation_params.get("performance", {}))
        return (
            RelationSpec(
                checker_id="checkers.performance.non_inferiority",
                checker_version="1",
                params=params,
                required_observations=(
                    RequiredObservation(
                        role="baseline",
                        run_kind="performance",
                        probes=("performance.samples/v1",),
                    ),
                    RequiredObservation(
                        role="candidate",
                        run_kind="performance",
                        probes=("performance.samples/v1",),
                    ),
                ),
            ),
            TemplateId.MR_P1,
            EvidenceStrength.EXPLICIT_CLAIM,
        )

    @staticmethod
    def _claims_have_explicit_predicates(
        claims: tuple[ClaimCandidate, ...],
    ) -> bool:
        return all(
            any(
                evidence.supports_field == "claimed.predicate"
                and evidence.strength is EvidenceStrength.EXPLICIT_CLAIM
                for evidence in claim.evidence
            )
            for claim in claims
        )

    @staticmethod
    def _pending(
        cluster: ClaimCluster,
        kind: OracleActionKind,
        reason: str,
        *,
        predecessor_ids: tuple[str, ...],
    ) -> OracleAction:
        return OracleAction.create(
            kind=kind,
            status=OracleActionStatus.PENDING_ARBITRATION,
            entity_id=cluster.entity_id,
            aspect=cluster.aspect,
            source_claim_ids=tuple(claim.claim_id for claim in cluster.claims),
            predecessor_ids=predecessor_ids,
            reason=reason,
        )
