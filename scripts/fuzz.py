#!/usr/bin/env python3
"""Run an appendable, property-guided PyTorch or TensorFlow campaign."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from functools import partial
from pathlib import Path

from mrminer.fuzz.ledger import Ledger
from mrminer.fuzz.llm import LocalOpenAIProvider
from mrminer.fuzz.loop import fuzz_campaign
from mrminer.fuzz.seed_trace import trace_isolated
from mrminer.fuzz.runner import run_test as torch_run_test
from mrminer.fuzz.types import Seed
from mrminer.tf.runner_tf import run_test as tf_run_test


TF_PROMPT = (
    "Generate a TensorFlow test for the target operator property, starting from "
    "the seed's source code. First satisfy its applicability condition and produce "
    "the required observation. Then explore corner cases along the relevant "
    "shape, dtype, layout, value and attribute dimensions. Use feedback history "
    "to fix skipped attempts or vary an already activating context. Each code "
    "sample must import tensorflow as tf and define "
    "def build(): return fn, args, where args is a tuple of tensors and fn uses "
    "only graph-traceable TensorFlow operations. Wrap the designated source "
    "call with mark(target_oracle.op, callable, *call_args, **call_kwargs) and "
    "return its result; the harness supplies mark. Do not call tf.function in the "
    "test: the harness compares eager execution with jit_compile=True. Vary "
    "inputs within the property's applicability domain, especially the stated "
    "shape, dtype, value, and parameter boundaries. Output the requested JSON."
)


def load_index(path: Path) -> dict[str, list[dict]]:
    payload = json.loads(path.read_text())
    index = payload.get("index")
    if not isinstance(index, dict) or not index:
        raise ValueError(f"{path} has no indexed properties")
    selected = {key: entries for key, entries in index.items() if entries}
    missing = [item.get("oracle_id", "<unknown>") for entries in selected.values()
               for item in entries if not item.get("property")]
    if missing:
        raise ValueError(f"{path} contains {len(missing)} records without P,T,pi,rho; "
                         "run synthesize_properties.py and screen_properties.py first")
    return selected


def load_seeds(path: Path) -> list[Seed]:
    """Load traced framework tests/examples and screening witnesses as code seeds."""
    seeds = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row.get("ops", []), list) or not isinstance(row.get("code"), str):
            raise ValueError(f"{path}:{number}: expected optional ops list and code")
        seeds.append(Seed(seed_id=row.get("seed_id") or f"seed:{number}",
                          ops=tuple(row.get("ops") or ()), note=row.get("note", ""),
                          parent_code=row["code"], origin=row.get("origin", "framework")))
    if not seeds:
        raise ValueError(f"{path} contains no seeds")
    return seeds


def load_screening_witnesses(path: Path, index: dict[str, list[dict]]) -> list[Seed]:
    seeds = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        witness = row.get("witness")
        op = row.get("operator")
        if not witness or op not in index:
            continue
        source = next((item.get("guard", {}).get("static", {}).get("source_entry", {}).get("operator")
                       for item in index[op] if item.get("oracle_id") == row.get("oracle_id")), None)
        ops = tuple(dict.fromkeys([op] + ([source] if source else [])))
        generated = witness.get("origin") == "generated_from_pr"
        seeds.append(Seed(seed_id=f"witness:{row['oracle_id']}", ops=ops,
                          note=f"PR {witness['pr_number']} screening input" if generated
                          else f"PR {witness['pr_number']} regression test",
                          parent_code=witness["code"],
                          origin="pr_generated" if generated else "regression"))
    return seeds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=("pytorch", "tensorflow"), required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--seeds", type=Path, required=True,
                        help="JSONL of framework tests, usage examples and screening witnesses")
    parser.add_argument("--screen-audit", type=Path,
                        help="append admitted regression-test witnesses as seeds")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-base", default=os.environ.get("MRMINER_API_BASE", ""))
    parser.add_argument("--model", default=os.environ.get("MRMINER_MODEL", ""))
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--max-tests", type=int,
                        help="stop after exactly this many generated tests")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--judge-mode", choices=("oracle", "no_condition"),
                        default="oracle", help="only no_condition bypasses P during checking")
    parser.add_argument("--trace-timeout", type=int, default=180)
    args = parser.parse_args()
    if not args.model or not args.api_base:
        parser.error("set MRMINER_MODEL and MRMINER_API_BASE or pass --model and --api-base")
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.max_tests is not None and args.max_tests < 1:
        parser.error("--max-tests must be positive")

    index = load_index(args.index)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {"round": 0}
    seeds = load_seeds(args.seeds)
    if args.screen_audit:
        seeds.extend(load_screening_witnesses(args.screen_audit, index))
    traced_seeds = []
    with (args.output_dir / "seed_trace.jsonl").open("a", encoding="utf-8") as trace_file:
        for seed in seeds:
            if seed.parent_code.strip():
                traced = trace_isolated(seed.parent_code, args.framework, args.index,
                                        timeout=args.trace_timeout)
                ops = tuple(traced.get("ops") or ())
            else:
                # Full's bootstrap seeds are operator directives, not source programs.
                traced = {"errors": [], "marked_calls": 0}
                reached = list(seed.ops)
                for core_op, entries in index.items():
                    if core_op in reached:
                        continue
                    if any((((item.get("guard") or {}).get("static") or {})
                            .get("source_entry") or {}).get("operator") in seed.ops
                           for item in entries):
                        reached.append(core_op)
                ops = tuple(reached)
            included = bool(set(ops) & index.keys())
            trace_file.write(json.dumps({"round": checkpoint["round"],
                                         "seed_id": seed.seed_id, "declared_ops": seed.ops,
                                         "traced_ops": ops, "included": included,
                                         "errors": traced.get("errors", []),
                                         "marked_calls": traced.get("marked_calls", 0)},
                                        ensure_ascii=False) + "\n")
            trace_file.flush()
            if included:
                traced_seeds.append(replace(seed, ops=ops))
    seeds = traced_seeds
    if not seeds:
        raise ValueError("no seed reached an indexed operator; inspect seed_trace.jsonl")
    provider = LocalOpenAIProvider(base_url=args.api_base, model=args.model)
    base_runner = torch_run_test if args.framework == "pytorch" else tf_run_test
    runner = partial(base_runner, judge_mode=args.judge_mode)
    prompt = None if args.framework == "pytorch" else TF_PROMPT
    ledger = Ledger(args.output_dir / "findings.jsonl", known_bugs=[])
    tests_path = args.output_dir / "tests.jsonl"
    results_path = args.output_dir / "results.jsonl"
    source_by_test: dict[str, str] = {}

    with tests_path.open("a", encoding="utf-8") as tests, \
            results_path.open("a", encoding="utf-8") as results:
        def on_test(test, oracle) -> None:
            source_by_test[test.test_id] = test.code
            tests.write(json.dumps({"round": checkpoint["round"], "test_id": test.test_id,
                                    "oracle_id": oracle.oracle_id, "operator": oracle.op,
                                    "code": test.code, "judge_mode": args.judge_mode},
                                   ensure_ascii=False) + "\n")
            tests.flush()

        def on_result(test, oracle, finding, wall_ms) -> None:
            results.write(json.dumps({"round": checkpoint["round"],
                                      "test_id": test.test_id, "oracle_id": oracle.oracle_id,
                                      "operator": oracle.op, "verdict": finding.verdict,
                                      "triggered": finding.triggered,
                                      "judge_mode": args.judge_mode,
                                      "detail": finding.detail,
                                      "contexts": list(finding.contexts),
                                      "wall_ms": round(wall_ms, 1)}, ensure_ascii=False) + "\n")
            results.flush()

        campaign = fuzz_campaign(
            seeds, provider, index, iterations=args.iterations,
            max_tests=args.max_tests,
            random_seed=args.seed + checkpoint["round"], run_test=runner,
            system_prompt=prompt, minimize_triggers=True,
            on_test=on_test, on_result=on_result,
        )

    minimized = {item.test_id: item.minimized_code
                 for item in campaign.minimized if item.confirmed}
    confirmed_ids = {item.test_id for item in campaign.minimized if item.confirmed}
    with (args.output_dir / "crashes.jsonl").open("a", encoding="utf-8") as crashes:
        for finding in campaign.crashes:
            crashes.write(json.dumps({"round": checkpoint["round"],
                                      "test_id": finding.test_id,
                                      "oracle_id": finding.oracle_id,
                                      "operator": finding.op,
                                      "verdict": finding.verdict,
                                      "detail": finding.detail,
                                      "code": source_by_test.get(finding.test_id, "")},
                                     ensure_ascii=False) + "\n")
    new_count = 0
    for finding in campaign.triggers:
        if finding.test_id not in confirmed_ids:
            continue
        verdict = ledger.record(
            finding, code=source_by_test.get(finding.test_id, ""),
            minimized=minimized.get(finding.test_id, ""),
        )
        new_count += verdict.status == "NEW"
    checkpoint["round"] += 1
    checkpoint_path.write_text(json.dumps(checkpoint, indent=2) + "\n")
    print(json.dumps({"round": checkpoint["round"], "judge_mode": args.judge_mode,
                      "seeds": len(seeds),
                      "tests": len(campaign.findings), "activated_pairs": campaign.coverage_size,
                      "violations": len(campaign.triggers),
                      "confirmed_violations": len(confirmed_ids),
                      "crashes": len(campaign.crashes), "new_candidates": new_count}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
