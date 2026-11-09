"""LLM generator: (seed + target oracle) -> a batch of tests.

This is the one genuinely new core piece (design doc §3). It reuses the existing
structured-LLM boundary (`mrminer.llm.provider.invoke_structured`), so a
FixtureLlmProvider can stand in for the model in M1 and a real DeepSeekProvider
drops in later with no interface change.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from mrminer.common.canonical import sha256_digest
from mrminer.fuzz.types import GeneratedTest, OracleRef, Seed
from mrminer.llm.provider import StructuredLlmProvider, invoke_structured

TASK = "fuzz.generate_tests"

SYSTEM_PROMPT = (
    "You are a compiler-fuzzing test generator for PyTorch's torch.compile/inductor. "
    "Given a seed (the aten operators an IR fragment exercises) and ONE target "
    "MR-oracle (an invariant a real PR established for one operator), write a few "
    "small, self-contained test programs that exercise that operator in the exact "
    "domain the oracle targets, trying to make the invariant fail. "
    "Treat the oracle's `claim` (and `spec`) as the SPECIFIC property to break: read it "
    "closely and identify the exact precondition and edge it is about -- the dtype, shape, "
    "stride / memory_format, rank, flags, or boundary VALUES that make the property "
    "relevant -- then concentrate your inputs at that edge instead of running the operator "
    "on generic inputs. When the claim names a condition (e.g. a specific dtype like int64 "
    "or bf16, a size-1 or non-contiguous dimension, preserve_format, an out-of-range / "
    "subnormal / NaN value, a bool/uint8 index, a decomposition or lowering path), "
    "construct inputs that ENTER that condition and probe just inside and just outside it; "
    "prefer a handful of sharply-targeted variants over many generic ones. "
    "Tailor inputs to the oracle's `aspect`: for `effect` (an eager-raises-implies-"
    "compiled-raises invariant), deliberately construct inputs the operator SHOULD reject "
    "in that property's domain -- wrong dtype (e.g. int64/bool where a float is expected), "
    "out-of-range or negative index, invalid shape / rank / size mismatch -- and see whether "
    "the compiled path silently ACCEPTS them; for `semantic`, push value-sensitive edges "
    "(precision boundaries, subnormals, non-contiguous / permuted / size-1 dims, special "
    "values); for `decision`, drive the exact condition the predicate names (e.g. a bool or "
    "uint8 index). "
    "The oracle was established by a real fix PR, so the textbook cases are likely already "
    "handled: aim at the RESIDUAL corners that fix may miss -- rare dtype combinations, "
    "size-1 / empty / non-contiguous tensors, composition with an adjacent op, and "
    "just-past-boundary values -- rather than the exact case the PR already covers. "
    "Each test's `code` MUST define `def build(): return fn, args` where `fn` is a "
    "callable and `args` a tuple of positional arguments; import torch inside. "
    "Do not call torch.compile yourself -- the harness runs eager vs compiled. "
    "If `mutate_from` is present, take ITS fn and its inputs as a starting point but "
    "REWRITE each variant into the required `def build(): return fn, args` form: do "
    "NOT keep `self`, `self.common`, `self.device`, class methods, or asserts -- "
    "construct the input tensors directly (honor any device in the seed note, e.g. "
    "device='cuda'), and vary shapes / dtypes / strides / edge cases to stress the "
    "oracle. "
    "If the seed note contains a '[FUSION seed' directive, follow it: build the whole "
    "described op chain inside a SINGLE fn (one compiled region, not separate calls) so "
    "inductor fuses it, keep the named target op as the one under test, and use the "
    "requested dtype where the ops allow it. "
    "Output only build()-form tests. Reason briefly, then output the JSON "
    "object promptly (do not spend the whole budget thinking)."
)

# --- ablation system prompts (paper §11 variants V3/V4) ------------------------
# NOP_SYSTEM_PROMPT (V3 No-P): keep the relation/aspect INTENT the oracle checks,
# but remove every instruction that targets the property's precondition `P` / its
# applicable input domain and edge. The generator is told WHAT relation to try to
# break, but not WHERE (which dtype/shape/flag/boundary) the property applies, so it
# cannot concentrate inputs at the contract boundary. Used to measure whether `P`
# guidance raises valid-test rate, precondition-satisfaction rate, and violation hit
# rate (RQ3, §11.0.4).
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
# the corresponding contract boundary easier to reach (RQ3, §11.0.4).
GENERIC_SYSTEM_PROMPT = (
    "You are a compiler-fuzzing test generator for PyTorch's torch.compile/inductor. "
    "Given a seed naming one or more aten operators, write a few small, self-contained "
    "test programs that exercise the named operator(s) on varied inputs (different "
    "shapes, dtypes, strides, and edge cases) to stress torch.compile/inductor. "
    "Each test's `code` MUST define `def build(): return fn, args` where `fn` is a "
    "callable and `args` a tuple of positional arguments; import torch inside. "
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
            "aspect": oracle.aspect,
            "variation": oracle.variation,
            "claim": oracle.claim,
            # spec (machine-readable precondition P) deliberately omitted
        }
    else:  # "full"
        target = {
            "op": oracle.op,
            "aspect": oracle.aspect,
            "variation": oracle.variation,   # e.g. {"$decomp": ["default","disabled"]}
            "claim": oracle.claim,           # the PR-grounded intent (guidance)
            "spec": getattr(oracle, "params", {}) or {},  # relation.params: the checker + any
                                             # machine-readable precondition (decision predicate,
                                             # perf config) -- the specific property to break
        }
    payload = {
        "seed": {"ops": list(seed.ops), "note": seed.note},
        "target_oracle": target,
        "contract": (
            "each tests[i].code defines build() -> (fn, args); self-contained; "
            "import torch inside; no torch.compile call"
        ),
    }
    if feedback:
        # M3 feedback hook: prior outcomes for this target, so a real model can
        # steer (e.g. "these didn't trigger -- vary the input domain"). The
        # fixture ignores it; it becomes meaningful with a real provider.
        payload["feedback"] = feedback
    if seed.parent_code:
        # M4 mutation hook: a promoted seed carries a prior interesting test; ask
        # the model to mutate it rather than generate cold.
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

    `guidance` ∈ {"full","no_p","none"} is the paper's generation-side ablation knob
    (§11 V3/V4): it trims how much of the mined property the generator sees and, when
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
