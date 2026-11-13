"""Generate tests from a source seed, target property, and execution feedback."""
from __future__ import annotations

import json
import re
from pathlib import Path

from mrminer.common.canonical import sha256_digest
from mrminer.fuzz.types import GeneratedTest, OracleRef, Seed
from mrminer.llm.provider import StructuredLlmProvider, invoke_structured

TASK = "fuzz.generate_tests"

SYSTEM_PROMPT = (
    "Generate a PyTorch test for the target operator property. Use the seed's "
    "source code as the starting point. First satisfy the property's applicability "
    "conditions and return the observation it constrains. Then explore corner "
    "cases along those same dimensions, including zero and one sizes, unusual "
    "strides or layouts, extreme values, and uncommon dtypes when applicable. "
    "Read the feedback history: after a skipped test, address the stated reason; "
    "after an activating test, try a distinct context. Do not change the checker. "
    "Every code sample must import torch and define build() returning (fn, args), "
    "where fn is callable and args is a tuple. Inside fn, wrap the one source "
    "operator call as mark(source_operator_id, callable, *call_args, **call_kwargs) "
    "and return its result. The harness supplies mark; do not define it. The "
    "source operator ID is in target_oracle.op. Do not call torch.compile inside "
    "the test. Return only the requested JSON object."
)

# Optional generation controls used by comparison experiments.
# NOP_SYSTEM_PROMPT (V3 No-P): keep the relation/aspect INTENT the oracle checks,
# but remove every instruction that targets the property's precondition `P` / its
# applicable input domain and edge. The generator is told WHAT relation to try to
# break, but not WHERE (which dtype/shape/flag/boundary) the property applies, so it
# cannot concentrate inputs at the contract boundary. Used to measure whether `P`
# guidance raises valid-test rate, precondition-satisfaction rate, and violation hit
# rate.
NOP_SYSTEM_PROMPT = (
    "You are a compiler-fuzzing test generator for PyTorch's torch.compile/inductor. "
    "Given a seed (the aten operators an IR fragment exercises) and ONE target "
    "MR-oracle (a relation a real PR established for one operator), write a few small, "
    "self-contained test programs that exercise that operator and try to make the "
    "relation fail. You are given the oracle's `aspect` and `claim` so you know WHAT "
    "relation must hold, but you are NOT told the specific precondition domain (dtype, "
    "shape, stride/memory_format, rank, flags, or boundary values) where it applies: "
    "choose inputs yourself without concentrating on any particular contract edge. "
    "Tailor only to the oracle's `aspect`: for `effect`, try inputs the operator might "
    "reject and see whether the compiled path silently accepts them; for `semantic`, "
    "exercise value-sensitive inputs; for `decision`, drive the predicate. "
    "Each test's `code` MUST define `def build(): return fn, args` where `fn` is a "
    "callable and `args` a tuple of positional arguments; import torch inside. "
    "Wrap the designated call inside fn with mark(target_oracle.op, callable, *args). "
    "Do not call torch.compile yourself -- the harness runs eager vs compiled. "
    "If `mutate_from` is present, REWRITE it into the required `def build(): return "
    "fn, args` form (no `self`/class scaffolding/asserts; construct tensors directly, "
    "honor any device in the seed note). "
    "Output only build()-form tests. Reason briefly, then output the JSON object "
    "promptly (do not spend the whole budget thinking)."
)

# GENERIC_SYSTEM_PROMPT (V4 No-Guidance / Generic-Gen): the generator sees ONLY the
# operator(s), never the property (claim/spec/precondition/relation). It does plain
# generic generation for the op; the mined property-specific checker is still applied
# downstream by the runner. Used to measure whether property-guided generation makes
# the corresponding contract boundary easier to reach.
GENERIC_SYSTEM_PROMPT = (
    "You are a compiler-fuzzing test generator for PyTorch's torch.compile/inductor. "
    "Given a seed naming one or more aten operators, write a few small, self-contained "
    "test programs that exercise the named operator(s) on varied inputs (different "
    "shapes, dtypes, strides, and edge cases) to stress torch.compile/inductor. "
    "Each test's `code` MUST define `def build(): return fn, args` where `fn` is a "
    "callable and `args` a tuple of positional arguments; import torch inside. "
    "Wrap the designated call inside fn with mark(target_oracle.op, callable, *args). "
    "Do not call torch.compile yourself -- the harness runs eager vs compiled. "
    "If `mutate_from` is present, REWRITE it into the required `def build(): return "
    "fn, args` form (no `self`/class scaffolding/asserts; construct tensors directly, "
    "honor any device in the seed note). "
    "Output only build()-form tests. Reason briefly, then output the JSON object "
    "promptly (do not spend the whole budget thinking)."
)

# PRPROMPT_SYSTEM_PROMPT (baseline: LLM-from-raw-PR): the generator is given the real
# upstream PR text (title / description / code diff) that changed an operator's behavior,
# but NONE of the structured, recovered property (no P / T / aspect / relation / checker).
# It must infer from the PR what to test and write a differential test. Judged by the
# GENERIC all-observable oracle (judge_mode="generic"), never by a mined checker -- so this
# isolates the value of the whole mining + structured-property pipeline against simply
# prompting a capable LLM with the raw PR.
PRPROMPT_SYSTEM_PROMPT = (
    "You are a compiler-fuzzing test generator for PyTorch's torch.compile/inductor. "
    "You are given a real upstream pull request (its title, description, and code diff) "
    "that changed the behavior of an operator. Without any further structured hints, write "
    "a few small, self-contained test programs that could catch a regression of the behavior "
    "this PR changed: read the PR to infer what input conditions and behavior it concerns, "
    "and construct inputs that exercise the affected operator at those conditions. "
    "Each test's `code` MUST define `def build(): return fn, args` where `fn` is a callable "
    "and `args` a tuple of positional arguments; import torch inside. "
    "Do not call torch.compile yourself -- the harness runs eager vs compiled and compares. "
    "Output only build()-form tests. Reason briefly, then output the JSON object promptly."
)

# guidance level -> the system prompt used when the caller gives no explicit override.
_GUIDANCE_PROMPTS = {
    "full": SYSTEM_PROMPT,
    "no_p": NOP_SYSTEM_PROMPT,
    "none": GENERIC_SYSTEM_PROMPT,
    "prprompt": PRPROMPT_SYSTEM_PROMPT,
}

# raw PR corpus (for the LLM-from-PR baseline). PR number is parsed from the oracle's
# claim summary ("PR #NNNNN: ..."); the full PR record lives in the mining corpus.
_PR_CORPUS = Path("data/pr_corpus_2023_now/pytorch__pytorch")


def _pr_field(x) -> str:
    """Coerce a PR JSON field (str / {text|patch|summary:...} / list) to plain text."""
    if x is None:
        return ""
    if isinstance(x, dict):
        x = x.get("text") or x.get("patch") or x.get("summary") or ""
    if isinstance(x, (list, tuple)):
        return "\n".join(_pr_field(i) for i in x)
    return str(x)


def _load_pr_text(claim_summary: str, max_chars: int = 6000) -> str:
    """Raw PR text (title + description + truncated diff) for the LLM-from-PR baseline.

    Parses `#NNNNN` out of the oracle's claim summary and reads the collected PR record.
    Falls back to the claim summary itself when the PR predates the corpus / is missing."""
    m = re.search(r"#(\d+)", claim_summary or "")
    if not m:
        return claim_summary or ""
    p = _PR_CORPUS / f"{m.group(1)}.json"
    if not p.exists():
        return claim_summary or ""
    try:
        d = json.loads(p.read_text())
    except Exception:
        return claim_summary or ""
    parts = [f"PR #{m.group(1)}: {_pr_field(d.get('title'))}".strip()]
    body = _pr_field(d.get("body")).strip()
    if body:
        parts.append("Description:\n" + body)
    diff = _pr_field(d.get("touched_files_patch")).strip()
    if diff:
        parts.append("Code diff (truncated):\n" + diff[:3000])
    return "\n\n".join(parts)[:max_chars]

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "tests": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["code"],
            },
        }
    },
    "required": ["tests"],
}


def _input_payload(seed: Seed, oracle: OracleRef, feedback: str,
                   guidance: str = "full") -> dict:
    # guidance controls how much of the mined property the generator sees:
    #   "full" -- op + aspect + variation + claim + spec (precondition)      [reference]
    #   "no_p" -- op + aspect + variation + claim, but NO spec/precondition  [V3 No-P]
    #   "none" -- op only; the property is hidden from generation            [V4 No-Guidance]
    # The checker applied downstream by the runner is UNCHANGED in every case; only
    # what the generator is told changes, so these isolate generation-side guidance.
    if guidance == "prprompt":
        # LLM-from-raw-PR baseline: op + the real PR text, NO structured property.
        target = {"op": oracle.op, "pr": _load_pr_text(oracle.claim)}
    elif guidance == "none":
        target = {"op": oracle.op}
    elif guidance == "no_p":
        target = {
            "op": oracle.op,
            "core_operator": oracle.core_operator or oracle.op,
            "aspect": oracle.aspect,
            "variation": oracle.variation,
            "claim": oracle.claim,
            # spec (machine-readable precondition P) deliberately omitted
        }
    else:  # "full"
        target = {
            "op": oracle.op,
            "core_operator": oracle.core_operator or oracle.op,
            "aspect": oracle.aspect,
            "variation": oracle.variation,   # e.g. {"$decomp": ["default","disabled"]}
            "claim": oracle.claim,           # the PR-grounded intent (guidance)
            "applicability": oracle.applicability,
            "observations": list(oracle.observations),
            "description": oracle.property_record.get("description", ""),
            "transform": oracle.property_record.get("T", ""),
            "spec": getattr(oracle, "params", {}) or {},  # relation.params: the checker + any
                                             # machine-readable precondition (decision predicate,
                                             # perf config) -- the specific property to break
        }
    payload = {
        "seed": {"ops": list(seed.ops), "note": seed.note,
                 "code": seed.parent_code},
        "target_oracle": target,
        "contract": (
            "each tests[i].code defines build() -> (fn, args); fn calls "
            "mark(target_oracle.op, callable, *args) around the designated source "
            "call and returns its result; import framework; no compiler call"
        ),
    }
    if feedback:
        # Feedback contains prior tests, verdicts, and skip reasons.
        payload["feedback"] = feedback
    if seed.parent_code:
        # Preserve the supplied source test as a concrete starting point.
        payload["mutate_from"] = seed.parent_code
    return payload


def generate(
    seed: Seed,
    oracle: OracleRef,
    provider: StructuredLlmProvider,
    *,
    feedback: str = "",
    system_prompt: str | None = None,
    guidance: str = "full",
) -> list[GeneratedTest]:
    """Ask the provider for tests targeting `oracle`, seeded by `seed`.

    `system_prompt` overrides the default PyTorch framing (used by the TF line to
    swap in a tf.function(jit_compile=True) framing); the input payload / output
    schema / build()-form contract are framework-agnostic and stay shared.

    `guidance` trims how much of the mined property the generator sees and, when
    `system_prompt` is not overridden, selects the matching system prompt. The runner's
    checker is unaffected -- only generation guidance changes."""
    eff_prompt = system_prompt or _GUIDANCE_PROMPTS.get(guidance, SYSTEM_PROMPT)
    result = invoke_structured(
        provider,
        task=TASK,
        system_prompt=eff_prompt,
        input_payload=_input_payload(seed, oracle, feedback, guidance),
        output_schema=OUTPUT_SCHEMA,
    )
    tests: list[GeneratedTest] = []
    for i, item in enumerate(result.payload["tests"]):
        code = item["code"]
        for module in ("torch", "tf"):
            if re.search(rf"\b{module}\.", code) and not re.search(
                rf"(?m)^\s*(?:import\s+{module}\b|from\s+{module}\b)", code
            ):
                import_line = "import tensorflow as tf" if module == "tf" else "import torch"
                code = import_line + "\n" + code
        tid = "test:" + sha256_digest({"o": oracle.oracle_id, "i": i, "c": code})[:16]
        tests.append(
            GeneratedTest(
                test_id=tid,
                seed_id=seed.seed_id,
                target_oracle=oracle.oracle_id,
                code=code,
                rationale=item.get("rationale", ""),
            )
        )
    return tests
