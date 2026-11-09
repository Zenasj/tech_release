"""Select TensorFlow operator/PR pairs with changed registration evidence.

The linker reads changed diff lines, then this stage keeps functional operators
with direct schema, kernel, XLA, or gradient evidence. Broad MLIR-only matches
and structural plumbing operators are excluded. By default only merged PRs are
considered; users may include an unmerged PR after verifying that its change
landed through another route.

Example: python scripts/tf_triage.py --corpus runs/tensorflow/pr_corpus/tensorflow__tensorflow
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from mrminer.tf.linker_tf import link_pr, op_id

# High-precision registration signals (a real behavior/impl touch of the op).
_PRECISE = {"schema", "kernel", "xla", "grad"}

# Structural / plumbing ops: appear incidentally in many PRs, not "op X changed".
_STRUCT = {
    "Const", "Identity", "Identity_1", "PartitionedCall", "StatefulPartitionedCall",
    "NoOp", "ReadVariableOp", "Placeholder", "PlaceholderWithDefault", "VarHandleOp",
    "AssignVariableOp", "BatchFunction", "Merge", "Switch", "Enter", "Exit",
    "NextIteration", "Cast", "Shape", "Reshape", "Pack", "Unpack", "StridedSlice",
}

# Non-MR PRs: mechanical / infra / docs / build -- establish no metamorphic relation.
_NON_MR = (
    "automated code change", "internal change", "copybara", "rollback of", "roll back",
    "rollforward", "merge pull request", "bump", "version", "pin ", "requirements",
    "third_party", "bazel", "build fix", "fix build", "golden", "benchmark",
    "microbenchmark", "refactor", "cleanup", "clean up", "typo", "flaky", "[ci]",
    " ci ", "disable", "skip test", "unskip", "docs", "documentation", "rename",
    "comment", "logging", "test only", "add test", "dead code", "nfc", "lint",
    "format", "typing", "type hint", "update to", "deprecat",
)

# Prose signals -> sets aspect (bonus, not required for TF).
_AXIS = ("xla", "jit_compile", "jit compile", "lowering", "lower ", "fuse", "fusion",
         "layout", "dynamic shape", "dynamic shapes", "compile", "constant fold",
         "canonicaliz", "legaliz", "decompos")
_EFFECT = ("raise", "raises", "error", "assert", "mismatch", "overflow", "nan",
           "incorrect", "wrong", "not supported", "unsupported", "reject", "must be",
           "nonfinite", "out of bounds", "invalid", "crash")


def _body_text(d: dict) -> str:
    b = d.get("body")
    if isinstance(b, dict):
        return str(b.get("text") or "")
    return str(b or "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="runs/tensorflow/pr_corpus/tensorflow__tensorflow")
    ap.add_argument("--out", default="runs/tensorflow/mining_plan.json")
    ap.add_argument("--cap", type=int, default=5, help="max PRs per op in the plan")
    ap.add_argument("--include-unmerged", action="store_true",
                    help="include unmerged PRs only when independently verified as landed")
    ns = ap.parse_args()

    registry_path = Path(os.environ.get("MRMINER_TF_OP_UNIVERSE", "runs/tensorflow/op_universe.json"))
    functional = {name for name, item in json.loads(registry_path.read_text())["ops"].items()
                  if item.get("functional")}

    files = sorted(p for p in Path(ns.corpus).glob("*.json") if p.stem.isdigit())
    n_total = 0
    n_unmerged = 0
    n_nonmr = 0
    n_linked = 0
    # op -> list of (pr_number, signals, aspect, tests_added, has_prose)
    op_edges: dict[str, list] = defaultdict(list)
    sig_hist: Counter = Counter()

    for f in files:
        try:
            d = json.loads(f.read_bytes())
        except Exception:
            continue
        n_total += 1
        if not ns.include_unmerged and not d.get("merged"):
            n_unmerged += 1
            continue
        title = str(d.get("title") or "")
        text = (title + " " + _body_text(d)).lower()
        if any(k in text for k in _NON_MR):
            n_nonmr += 1
            continue
        ops = link_pr(d)                      # {op_name: {signals}}, universe-validated
        if not ops:
            continue
        # prose aspect (bonus)
        aspect = None
        if any(k in text for k in _AXIS):
            aspect = "semantic"
        elif any(k in text for k in _EFFECT):
            aspect = "effect"
        tests_added = bool(d.get("tests_added"))
        pr_num = d.get("pr_number")
        linked_here = False
        for name, signals in ops.items():
            if name not in functional:
                continue
            if name in _STRUCT:
                continue
            if not (signals & _PRECISE):      # mlir-only / no precise signal -> skip
                continue
            linked_here = True
            for s in signals & _PRECISE:
                sig_hist[s] += 1
            op_edges[op_id(name)].append(
                {"pr": pr_num, "signals": sorted(signals & _PRECISE),
                 "aspect": aspect, "tests": tests_added, "prose": aspect is not None}
            )
        if linked_here:
            n_linked += 1

    # Mineable = every candidate edge (precise signal, non-structural, non-mlir-only,
    # non-MR PR). "strict" additionally requires a prose axis/effect signal.
    all_edges = sum(len(v) for v in op_edges.values())
    strict_edges = sum(1 for v in op_edges.values() for e in v if e["prose"])
    strict_ops = sum(1 for v in op_edges.values() if any(e["prose"] for e in v))

    # Build the plan: for each mineable op, keep up to `cap` best PRs
    # (prefer ones with prose + tests, then more precise signals).
    def score(e):
        return (e["prose"], e["tests"], len(e["signals"]))

    plan = []
    for op, edges in sorted(op_edges.items(), key=lambda kv: -len(kv[1])):
        best = sorted(edges, key=score, reverse=True)[: ns.cap]
        plan.append({"op": op, "n_prs_linked": len(edges),
                     "plan_prs": [e["pr"] for e in best],
                     "has_prose": any(e["prose"] for e in edges)})
    plan_calls = sum(len(p["plan_prs"]) for p in plan)

    out = {
        "corpus": ns.corpus,
        "prs_total": n_total,
        "prs_unmerged_excluded": n_unmerged,
        "prs_non_mr_rejected": n_nonmr,
        "prs_mineable": n_linked,
        "candidate_ops": len(op_edges),
        "candidate_edges": all_edges,
        "precise_signal_hist": dict(sig_hist),
        "strict_ops_with_prose": strict_ops,
        "strict_edges_with_prose": strict_edges,
        "plan_cap_per_op": ns.cap,
        "plan_ops": len(plan),
        "plan_llm_calls": plan_calls,
        "plan": plan,
    }
    Path(ns.out).parent.mkdir(parents=True, exist_ok=True)
    Path(ns.out).write_text(json.dumps(out, indent=2))

    print(f"PRs scanned            : {n_total}")
    print(f"  unmerged excluded    : {n_unmerged}")
    print(f"  non-MR rejected      : {n_nonmr}  ({100*n_nonmr/max(n_total,1):.1f}%)")
    print(f"  mineable (>=1 op)    : {n_linked}")
    print(f"candidate ops          : {len(op_edges)}   (structural + mlir-only excluded)")
    print(f"candidate (op,PR) edges: {all_edges}")
    print(f"  precise-signal hist  : {dict(sig_hist)}")
    print(f"strict (with prose)    : {strict_ops} ops / {strict_edges} edges")
    print(f"--- PLAN (cap {ns.cap} PRs/op) ---")
    print(f"  ops to mine          : {len(plan)}")
    print(f"  LLM extract calls    : {plan_calls}")
    top = plan[:12]
    print("  top ops:", [(p["op"].split('::')[-1], p["n_prs_linked"]) for p in top])
    print(f"-> {ns.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
