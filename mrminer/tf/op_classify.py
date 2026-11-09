"""TF op categorization for XLA differential fuzzing.

Replaces the old blanket prefix/substring seed exclusion (TensorArray*/Resource*/
Stateful*/Dataset* -> dropped) with a per-op category grounded in **authoritative
op_def attributes** (from `op_def_registry`). Instead of deleting whole families we
route each op to the harness + comparator that actually fits it, so genuinely
XLA-supported members (e.g. `TensorArrayV3`, `ResourceScatterUpdate`) get tested
rather than thrown away.

Five categories (see `Category`):

  supported              plain compute op -> full eager-vs-XLA value differential.
  needs_wrapper          resource/variable/array-handle op -> wrap state (tf.Variable
                         / TensorArray harness), return read-back tensors, then diff.
  nondeterministic       stateful RNG family -> seed it; compare shape/dtype (not value).
  eager_only             cannot be meaningfully jit_compiled (TF1 ref control-flow,
                         tf.data pipelines/iterators) -> excluded from XLA differential.
  unsupported            has no XLA kernel on this build. NOT dropped: the oracle is
                         "XLA must reject cleanly"; a *silent wrong result* under
                         jit_compile is a bug (this is where #119375-class silent
                         miscompiles live). Assigned at RUNTIME from the runner's
                         xla_unsupported markers (build-exact), with an optional small
                         static hint set below; static classification returns `supported`
                         for plain-compute ops and lets the runtime confirm.

Design notes:
- op_def is the ground truth for structure (ref-ness, DT_RESOURCE/DT_VARIANT dtypes,
  is_stateful). The XlaOpRegistry (which TF op has an XLA kernel) has no Python binding
  in the installed wheel, so supported/unsupported for plain ops is a runtime signal,
  which is more build-accurate than any static list anyway.
- Pure function of the op name; only touches the lightweight op_def_registry (no TF
  session / device init needed).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# protobuf DataType enum values (stable)
DT_STRING = 7
DT_RESOURCE = 20
DT_VARIANT = 21

Category = str  # one of the CATEGORIES below
CATEGORIES = (
    "supported",
    "needs_wrapper",
    "nondeterministic",
    "eager_only",
    "unsupported",
)

# tf.data / iterator infra: variant-based pipelines, cannot be jit_compiled as pure
# tensor compute. Name-based is reliable here (all carry Dataset/Iterator/Optional).
_DATASET_SUBSTR = ("Dataset", "Iterator")
_OPTIONAL_OPS = {"OptionalFromValue", "OptionalGetValue", "OptionalNone",
                 "OptionalHasValue"}

# legacy array-handle infra (produce a resource/variant handle used as a mutable
# container). Testable, but only via a write/read/stack harness -> needs_wrapper.
_ARRAY_PREFIX = ("TensorArray", "TensorList")
_LEGACY_STACK = {"Stack", "StackV2", "StackPush", "StackPushV2", "StackPop",
                 "StackPopV2", "StackClose", "StackCloseV2"}

# stateful RNG family (nondeterministic unless Stateless*/seeded). Gated by is_stateful.
_RNG_RE = re.compile(
    r"(Random|Multinomial|RngReadAndSkip|RngSkip|ParameterizedTruncatedNormal"
    r"|TruncatedNormal|NonDeterministicInts|StatefulUniform|StatefulStandardNormal"
    r"|StatefulTruncatedNormal|Sample)"
)

# optional build-agnostic hints for ops known to lack an XLA kernel even though they are
# plain-compute by structure. Kept intentionally tiny; runtime markers are authoritative.
_STATIC_UNSUPPORTED_HINT: set[str] = set()


@dataclass(frozen=True)
class OpCategory:
    op: str          # bare op stem (no "tf::" prefix)
    category: Category
    reason: str      # short human-readable justification (from op_def evidence)


def _bare(op: str) -> str:
    return op.split("::")[-1]


def _get_opdef(stem: str):
    """Return the OpDef for `stem`, or None if unknown to this build."""
    try:
        from tensorflow.python.framework import op_def_registry as odr
        return odr.get(stem)
    except Exception:
        return None


def classify(op: str) -> OpCategory:
    """Classify a TF op (accepts 'tf::Name' or 'Name') into one Category.

    Uses authoritative op_def attributes; falls back to name heuristics only for
    dataset/array infra whose structure is already implied by the name.
    """
    stem = _bare(op)

    # tf.data pipeline / iterator infra -> eager-only (name-reliable).
    if any(s in stem for s in _DATASET_SUBSTR) or stem in _OPTIONAL_OPS:
        return OpCategory(stem, "eager_only", "tf.data pipeline/iterator (variant), not jit-compilable")

    d = _get_opdef(stem)

    # TF1 ref control-flow / ref-typed ops -> eager-only (authoritative: is_ref).
    if d is not None and (any(a.is_ref for a in d.input_arg)
                          or any(a.is_ref for a in d.output_arg)):
        return OpCategory(stem, "eager_only", "TF1 ref-typed op (is_ref), legacy control-flow")

    # array-handle infra (TensorArray*/TensorList*/legacy Stack) -> needs_wrapper.
    if any(stem.startswith(p) for p in _ARRAY_PREFIX) or stem in _LEGACY_STACK:
        return OpCategory(stem, "needs_wrapper", "array-handle container; test via write/read/stack harness")

    # resource-consuming ops (variables etc.) -> needs_wrapper (bind tf.Variable).
    if d is not None and any((a.type == DT_RESOURCE) for a in d.input_arg):
        return OpCategory(stem, "needs_wrapper", "consumes DT_RESOURCE handle; wrap with tf.Variable, read back")
    if d is not None and any((a.type == DT_RESOURCE) for a in d.output_arg):
        return OpCategory(stem, "needs_wrapper", "produces DT_RESOURCE handle; test via a stateful harness")

    # stateful RNG -> nondeterministic (seed + compare shape/dtype, not values).
    if d is not None and d.is_stateful and _RNG_RE.search(stem) and not stem.startswith("Stateless"):
        return OpCategory(stem, "nondeterministic", "stateful RNG; seed and compare shape/dtype not values")

    # explicit static hint (rare) that a plain-compute op lacks an XLA kernel.
    if stem in _STATIC_UNSUPPORTED_HINT:
        return OpCategory(stem, "unsupported", "known to lack XLA kernel (static hint)")

    # default: plain compute op. supported/unsupported is confirmed at runtime by the
    # runner's xla_unsupported markers (build-exact), which reclassify to `unsupported`.
    return OpCategory(stem, "supported", "plain compute; full eager-vs-XLA differential")


# --- oracle routing table -------------------------------------------------------
# Each category maps to how a seed of that op should be tested.
#   xla_diff   : run under jit_compile and compare to eager
#   comparator : 'value' (exact/rel), 'shape_dtype' (non-deterministic), or None
#   harness    : None | 'variable' | 'array' (how build() should wrap state)
#   silent_wrong_is_bug : for `unsupported`, a successful compile that changes output
#                         is a bug (do not skip); a clean rejection is expected.
ROUTING = {
    "supported":        dict(xla_diff=True,  comparator="value",       harness=None,       silent_wrong_is_bug=False),
    "needs_wrapper":    dict(xla_diff=True,  comparator="value",       harness="auto",     silent_wrong_is_bug=False),
    "nondeterministic": dict(xla_diff=True,  comparator="shape_dtype", harness=None,       silent_wrong_is_bug=False),
    "eager_only":       dict(xla_diff=False, comparator=None,          harness=None,       silent_wrong_is_bug=False),
    "unsupported":      dict(xla_diff=True,  comparator="value",       harness=None,       silent_wrong_is_bug=True),
}


def routing_for(op: str) -> dict:
    return ROUTING[classify(op).category]


# --- per-category generation guidance (injected into the seed note -> LLM prompt) ---
# needs_wrapper splits by structure: array-handle ops can be built PURELY INSIDE fn
# (traceable, no persistent state -> the runner's build-once/reuse-args value diff works
# directly, and this is exactly the #119375 pattern); resource/variable ops cannot create
# the Variable inside a jit_compiled fn (not XLA-traceable), so the Variable is built in
# build() and read back -- best-effort, the runner safely skips ones whose repeated-call
# state would accumulate.
_HINT_ARRAY = (
    "HARNESS REQUIRED (array-handle op): build the `tf.TensorArray(dtype, size=..., "
    "dynamic_size=...)` / TensorList INSIDE fn, write to it (optionally in a tf.while_loop), "
    "and RETURN `.stack()` (or the read tensor). Keep fn pure -- do NOT use tf.Variable and "
    "do NOT return the raw handle. Vary size / dynamic_size / write pattern; note that "
    "dynamic_size=True with size=0 grown in a tf.while_loop is a known XLA fault mode."
)
_HINT_VARIABLE = (
    "HARNESS REQUIRED (resource/variable op): create the `tf.Variable(<init>)` in build() "
    "(NOT inside fn -- a Variable created inside a jit_compiled fn is not XLA-traceable), "
    "capture it; `fn` applies this op to it and RETURNS the read-back tensor (`v.value()` or "
    "the op's tensor output), never the raw resource handle. Prefer inits/ops where the "
    "returned value is a function of THIS call's inputs (e.g. assign-then-read) so the result "
    "is stable across repeated calls. Vary shapes / dtypes / indices."
)
_HINT_RNG = (
    "HARNESS NOTE (stateful RNG): make the draw reproducible -- pass explicit seeds or use the "
    "stateless_* variant so a rerun is identical. The harness compares only SHAPE and DTYPE, "
    "not values; focus variants on shapes / dtypes / params, not exact outputs."
)
_HINT_UNSUPPORTED = (
    "NOTE: this op may have no XLA kernel. The oracle expects XLA to REJECT it cleanly; a "
    "successful jit_compile that changes the result (vs eager) is the bug -- aim variants at "
    "exposing a silent wrong result rather than a clean error."
)


def harness_hint(op: str) -> str:
    """Generation guidance for the op's category (goes into the seed note)."""
    c = classify(op)
    if c.category == "needs_wrapper":
        return _HINT_ARRAY if "array-handle" in c.reason else _HINT_VARIABLE
    if c.category == "nondeterministic":
        return _HINT_RNG
    if c.category == "unsupported":
        return _HINT_UNSUPPORTED
    return ""
