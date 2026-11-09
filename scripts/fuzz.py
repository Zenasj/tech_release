#!/usr/bin/env python3
"""Run an appendable, property-guided PyTorch or TensorFlow campaign."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from mrminer.fuzz.ledger import Ledger
from mrminer.fuzz.llm import LocalOpenAIProvider
from mrminer.fuzz.loop import fuzz_campaign
from mrminer.fuzz.runner import run_test as torch_run_test
from mrminer.fuzz.types import Seed
from mrminer.tf.runner_tf import run_test as tf_run_test


TF_PROMPT = (
    "Generate small, self-contained TensorFlow tests for the supplied operator "
    "and property. Each code sample must import tensorflow as tf and define "
    "def build(): return fn, args, where args is a tuple of tensors and fn uses "
    "only graph-traceable TensorFlow operations. Do not call tf.function in the "
    "test: the harness compares eager execution with jit_compile=True. Vary "
    "inputs within the property's applicability domain, especially the stated "
    "shape, dtype, value, and parameter boundaries. Output the requested JSON."
)


def load_index(path: Path) -> dict[str, list[dict]]:
    payload = json.loads(path.read_text())
    index = payload.get("index")
    if not isinstance(index, dict) or not index:
        raise ValueError(f"{path} has no indexed properties")
    return {key: entries for key, entries in index.items() if entries}


def make_seeds(index: dict[str, list[dict]]) -> list[Seed]:
    seeds = []
    for op, entries in sorted(index.items()):
        source_ops = {
            item.get("guard", {}).get("static", {}).get("source_entry", {}).get("operator")
            for item in entries
        }
        source_ops.discard(None)
        seed_ops = tuple(dict.fromkeys([op, *sorted(source_ops)]))
        seeds.append(Seed(seed_id=f"bootstrap:{op}", ops=seed_ops,
                          note=f"Target {op}; source calls: {', '.join(sorted(source_ops)) or op}",
                          origin="op-bootstrap"))
    return seeds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=("pytorch", "tensorflow"), required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-base", default=os.environ.get("MRMINER_API_BASE", ""))
    parser.add_argument("--model", default=os.environ.get("MRMINER_MODEL", ""))
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--max-oracles", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not args.model or not args.api_base:
        parser.error("set MRMINER_MODEL and MRMINER_API_BASE or pass --model and --api-base")
    if args.iterations < 1 or args.max_oracles < 1:
        parser.error("--iterations and --max-oracles must be positive")

    index = load_index(args.index)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {"round": 0}
    seeds = make_seeds(index)
    random.Random(args.seed + checkpoint["round"]).shuffle(seeds)
    provider = LocalOpenAIProvider(base_url=args.api_base, model=args.model)
    runner = torch_run_test if args.framework == "pytorch" else tf_run_test
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
                                    "code": test.code}, ensure_ascii=False) + "\n")
            tests.flush()

        def on_result(test, oracle, finding, wall_ms) -> None:
            results.write(json.dumps({"round": checkpoint["round"],
                                      "test_id": test.test_id, "oracle_id": oracle.oracle_id,
                                      "operator": oracle.op, "verdict": finding.verdict,
                                      "triggered": finding.triggered,
                                      "wall_ms": round(wall_ms, 1)}, ensure_ascii=False) + "\n")
            results.flush()

        campaign = fuzz_campaign(
            seeds, provider, index, iterations=args.iterations,
            max_oracles=args.max_oracles, run_test=runner,
            system_prompt=prompt, minimize_triggers=True,
            on_test=on_test, on_result=on_result,
        )

    minimized = {item.test_id: item.minimized_code
                 for item in campaign.minimized}
    new_count = 0
    for finding in campaign.triggers:
        verdict = ledger.record(
            finding, code=source_by_test.get(finding.test_id, ""),
            minimized=minimized.get(finding.test_id, ""),
        )
        new_count += verdict.status == "NEW"
    checkpoint["round"] += 1
    checkpoint_path.write_text(json.dumps(checkpoint, indent=2) + "\n")
    print(json.dumps({"round": checkpoint["round"], "seeds": len(seeds),
                      "tests": len(campaign.findings), "activated_pairs": campaign.coverage_size,
                      "violations": len(campaign.triggers), "new_candidates": new_count}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
