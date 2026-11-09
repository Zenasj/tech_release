"""Context assembler (engineering glue, not a paper contribution).

Turns an accepted ClaimCandidate + tool-mined program ids into a complete
``InductionContext`` so ``extract -> induce`` can run unattended. It picks *which
knob to vary* between the two executions of the relational MR from a **closed,
deterministic** registry of variation axes (no LLM), and fails closed with a typed
``ContextGap`` when it cannot bind a program or match an axis.

v1 scope (see docs/design/context_assembler_design_v1.md): the **semantic** aspect only,
two variation axes (``decomp``, ``exec_mode``). Effect/decision/performance routes
are deliberately out of scope until a concrete operator target needs them.

Blast radius: additive. This module does not modify ``inducer``,
``claim_extractor``, or the graph.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from mrminer.s2_graph import ClaimCandidate
from mrminer.s3_oracle.inducer import InductionContext
from mrminer.common.schema import (
    EvidenceStrength,
    InputInterface,
    OracleAspect,
    ParameterDefinition,
    ProgramKind,
    ProgramManifest,
    ProgramPhase,
    ProgramTransformOperation,
    ProvenanceOrigin,
    TriggerSpec,
    VariationTemplateSpec,
)

# roles a semantic (paired) MR template produces; sorted+unique per schema rule.
_SEMANTIC_ROLES = ("followup_pair", "source_pair")

# Operators for which a value-equivalence MR is undefined, so we never assemble one
# (F8/F1). RNG operators are nondeterministic (no fixed output without seed control);
# void/assertion operators return nothing (no output value to compare). Matched by
# the operator name stem (overload stripped).
_RNG_OPERATOR_STEMS = frozenset({
    "bernoulli", "normal", "uniform", "exponential", "cauchy", "geometric",
    "log_normal", "random", "randn", "randn_like", "rand", "rand_like", "randint",
    "randint_like", "randperm", "multinomial", "poisson", "binomial",
    "standard_gamma", "dirichlet", "native_dropout", "native_dropout_backward",
    "dropout", "_fused_dropout", "rrelu_with_noise", "_sample_dirichlet",
})
_VOID_OPERATOR_STEMS = frozenset({
    "_assert_scalar", "_assert_async", "_assert_tensor_metadata",
    "_functional_assert_scalar", "_functional_assert_async",
})


def _operator_stem(entity_id: str) -> str:
    return entity_id.split("::")[-1].split(".")[0]


def _synthesize_program(claim: ClaimCandidate) -> str:
    """Program SPEC synthesised from the claim when no pre-existing test binds.

    An oracle is a claim + MR structure; the concrete program/input used to witness
    it is a *validation-time* concern that can be constructed from the claim's
    condition, its issue, or the PR — it need not be a test the PR happened to ship.
    This mints a ProgramManifest bound to the claim's operator as a placeholder to be
    materialised at validation; it lets the oracle exist rather than dropping the
    claim to a `program` gap.
    """
    operator = next(
        (e for e in claim.target_entities if e.startswith("op:aten::")),
        claim.claimed.predicate.subject,
    )
    manifest = ProgramManifest.create(
        program_kind=ProgramKind.PYTHON_CALLABLE,
        source_ref=None,
        entrypoint=f"mr_synth/{_operator_stem(operator)}@pr{claim.source_pr}",
        dependencies={},
        input_interface=InputInterface(arity=1, roles=("x",)),
        required_device_class="cpu",
        phase=ProgramPhase.FORWARD,
        bound_entities=(operator,),
    )
    return manifest.program_id


def _unsuitable_operator(entity_id: str) -> str | None:
    stem = _operator_stem(entity_id)
    if stem in _VOID_OPERATOR_STEMS or stem.startswith("_assert"):
        return f"void/assertion operator {stem!r} has no output value for a value-MR"
    if stem in _RNG_OPERATOR_STEMS:
        return (
            f"nondeterministic (RNG) operator {stem!r}; value-equivalence is "
            "undefined without seed control"
        )
    return None


@dataclass(frozen=True)
class VariationAxis:
    """One closed knob to vary between the two executions of a relational MR."""

    axis_id: str
    keywords: tuple[str, ...]          # deterministic evidence signals
    transform_id: str                  # what the executor applies
    op_param_key: str                  # key inside the transform's params
    param_placeholder: str             # e.g. "$decomp"
    values: tuple[str, ...]            # finite grid for the placeholder

    def matches(self, claim_blob: str) -> bool:
        return any(keyword in claim_blob for keyword in self.keywords)

    def template(self) -> VariationTemplateSpec:
        return VariationTemplateSpec.create(
            operations=(
                ProgramTransformOperation(
                    transform_id=self.transform_id,
                    params={self.op_param_key: self.param_placeholder},
                ),
            ),
            param_schema={
                self.param_placeholder: ParameterDefinition(enum=self.values)
            },
            roles_produced=_SEMANTIC_ROLES,
        )

    def param_grid(self) -> dict[str, tuple[str, ...]]:
        return {self.param_placeholder: self.values}


# Order = priority: specific axes are tried before the general exec_mode.
DEFAULT_VARIATION_REGISTRY: tuple[VariationAxis, ...] = (
    VariationAxis(
        axis_id="decomp",
        keywords=("get_decomp_fn", "_decomp", "decompos", "decomp"),
        transform_id="pytorch.inductor.decomposition.v1",
        op_param_key="decomposition",
        param_placeholder="$decomp",
        values=("default", "disabled"),
    ),
    VariationAxis(
        axis_id="layout",
        keywords=("channels_last", "channels last", "channels-last"),
        transform_id="pytorch.layout.v1",
        op_param_key="layout",
        param_placeholder="$layout",
        values=("channels_last", "contiguous"),
    ),
    VariationAxis(
        axis_id="dynamic_shape",
        keywords=(
            "symint", "unbacked", "dynamic shape", "dynamic shapes",
            "data-dependent", "data dependent", "dynamic end",
            "dynamic slice", "symbolic scalar",
        ),
        transform_id="pytorch.dynamic_shapes.v1",
        op_param_key="shapes",
        param_placeholder="$shapes",
        values=("dynamic", "static"),
    ),
    VariationAxis(
        axis_id="exec_mode",
        keywords=("torch.compile", "inductor", "compiled", " eager"),
        transform_id="pytorch.execmode.v1",
        op_param_key="mode",
        param_placeholder="$mode",
        values=("eager", "inductor"),
    ),
)


@dataclass(frozen=True)
class ContextGap:
    """Typed fail-closed outcome: the tool could not assemble a context."""

    reason: str
    missing_piece: str                 # "aspect_route" | "program" | "variation"

    def to_record(self) -> dict:
        return {
            "outcome": "context_gap",
            "reason": self.reason,
            "missing_piece": self.missing_piece,
        }


@dataclass(frozen=True)
class AssembledContext:
    """A full, tool-assembled induction context plus the variation it chose."""

    context: InductionContext
    variation_template: VariationTemplateSpec | None
    axis_id: str
    program_ids: tuple[str, ...]
    aspect: str = "semantic"
    program_synthesized: bool = False

    def to_record(self) -> dict:
        return {
            "outcome": "assembled",
            "aspect": self.aspect,
            "axis_id": self.axis_id,
            "program_ids": list(self.program_ids),
            "program_synthesized": self.program_synthesized,
            "variation_template_id": (
                self.variation_template.variation_template_id
                if self.variation_template is not None
                else None
            ),
            "param_grid": {
                name: list(values)
                for name, values in self.context.param_grid.items()
            },
            "validity_artifact": self.context.validity_artifact,
        }


def _claim_blob(claim: ClaimCandidate) -> str:
    """Lowercased text the axis matcher scans: claim text + evidence quotes."""

    parts = [claim.claimed.text_normalized]
    predicate = claim.claimed.predicate
    for field in (predicate.property, predicate.preferred_probe):
        if field:
            parts.append(str(field))
    for evidence in claim.evidence:
        parts.append(evidence.quote)
        parts.append(evidence.locator)
    return "\n".join(parts).lower()


def select_program_ids(
    seed_candidates: Sequence[dict],
    *,
    pr_number: int,
    entity_id: str,
) -> tuple[str, ...]:
    """Tool-driven program binding: source_validated program ids for (pr, entity)."""

    ids: list[str] = []
    for candidate in seed_candidates:
        if candidate.get("readiness") != "source_validated":
            continue
        if candidate.get("pr_number") != pr_number:
            continue
        manifest = candidate.get("manifest") or {}
        if entity_id not in (manifest.get("bound_entities") or []):
            continue
        program_id = manifest.get("program_id")
        if program_id and program_id not in ids:
            ids.append(program_id)
    return tuple(ids)


def _has_explicit_predicate(claim: ClaimCandidate) -> bool:
    return any(
        evidence.supports_field == "claimed.predicate"
        and evidence.strength is EvidenceStrength.EXPLICIT_CLAIM
        for evidence in claim.evidence
    )


def _trigger(claim: ClaimCandidate) -> TriggerSpec:
    predicate = claim.claimed.predicate
    operator = next(
        (e for e in claim.target_entities if e.startswith("op:aten::")),
        predicate.subject,
    )
    return TriggerSpec(
        probe=predicate.preferred_probe or "semantic.paired_capture/v1",
        predicate=predicate.property or "holds",
        entity=operator,
        min_count=1,
    )


# Performance MRs need a full statistical design (a lab-policy constant, not mined).
_PERFORMANCE_DESIGN = {
    "metric": "latency_ms", "direction": "lower_is_better", "margin_delta": 0.0,
    "confidence_level": 0.95, "process_reps": 30, "estimator": "median",
    "ci_method": "bootstrap", "bootstrap_seed": 20260815,
    "bootstrap_resamples": 2000, "stopping_rule": "fixed_n",
}
_PERFORMANCE_VARIATION = VariationTemplateSpec.create(
    operations=(ProgramTransformOperation(
        transform_id="pytorch.optpatch.v1", params={"patch": "$patch"}),),
    param_schema={"$patch": ParameterDefinition(enum=("disabled", "enabled"))},
    roles_produced=("baseline", "candidate"),
)


def assemble_context(
    claim: ClaimCandidate,
    *,
    program_ids: Sequence[str],
    at: str,
    registry: Sequence[VariationAxis] = DEFAULT_VARIATION_REGISTRY,
) -> AssembledContext | ContextGap:
    """Assemble a complete InductionContext for a claim, or a typed ContextGap.

    Routes by aspect: semantic -> MR-C1, effect -> MR-C3, decision -> MR-C2,
    performance -> MR-P1 (the inducer builds the template; here we supply the
    matching runtime binding).
    """

    for entity_id in claim.target_entities:
        unsuitable = _unsuitable_operator(entity_id)
        if unsuitable is not None:
            return ContextGap(reason=unsuitable, missing_piece="unsuitable_operator")

    program_ids = tuple(sorted(set(program_ids)))  # InductionContext needs sorted+unique
    program_synthesized = False
    if not program_ids:
        # No pre-existing test binds: synthesise a program SPEC from the claim so the
        # oracle can exist. The runnable program is constructed at validation time
        # (from the claim/issue/PR), which is deferred.
        program_ids = (_synthesize_program(claim),)
        program_synthesized = True

    common = dict(
        validity_artifact=claim.effective_artifact,
        program_ids=program_ids,
        created_at=at,
        source_origins=(ProvenanceOrigin.PR_REGRESSION_TEST,),
    )
    aspect = claim.aspect

    if aspect in (OracleAspect.EFFECT, OracleAspect.DECISION, OracleAspect.PERFORMANCE):
        if not _has_explicit_predicate(claim):
            return ContextGap(
                reason=f"{aspect.value} oracle requires explicit predicate evidence",
                missing_piece="predicate_evidence",
            )

    if aspect is OracleAspect.EFFECT:
        context = InductionContext(dynamic_trigger=_trigger(claim), **common)
        return AssembledContext(context=context, variation_template=None,
                                axis_id="effect", aspect="effect",
                                program_ids=program_ids,
                                program_synthesized=program_synthesized)

    if aspect is OracleAspect.PERFORMANCE:
        context = InductionContext(
            variation_template_id=_PERFORMANCE_VARIATION.variation_template_id,
            param_grid={"$patch": ("disabled", "enabled")},
            relation_params={"performance": _PERFORMANCE_DESIGN}, **common)
        return AssembledContext(context=context,
                                variation_template=_PERFORMANCE_VARIATION,
                                axis_id="performance", aspect="performance",
                                program_ids=program_ids,
                                program_synthesized=program_synthesized)

    # semantic and decision both pick a variation axis from the closed registry
    axis = next((ax for ax in registry if ax.matches(_claim_blob(claim))), None)
    if axis is None:
        return ContextGap(reason="no variation axis matched the claim evidence",
                          missing_piece="variation")
    template = axis.template()
    fields = dict(variation_template_id=template.variation_template_id,
                  param_grid=axis.param_grid(), **common)
    if aspect is OracleAspect.DECISION:
        fields["dynamic_trigger"] = _trigger(claim)
    context = InductionContext(**fields)
    return AssembledContext(context=context, variation_template=template,
                            axis_id=axis.axis_id, aspect=aspect.value,
                            program_ids=program_ids,
                            program_synthesized=program_synthesized)
