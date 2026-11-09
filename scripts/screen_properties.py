#!/usr/bin/env python3
"""Keep properties with at least one executable, activated test witness."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from mrminer.fuzz.generator import generate
from mrminer.fuzz.llm import LocalOpenAIProvider
from mrminer.fuzz.oracle_index import oracles_for
from mrminer.fuzz.runner import run_test as torch_run_test
from mrminer.tf.runner_tf import run_test as tf_run_test

from fuzz import TF_PROMPT, make_seeds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=("pytorch", "tensorflow"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--api-base", default=os.environ.get("MRMINER_API_BASE", ""))
    parser.add_argument("--model", default=os.environ.get("MRMINER_MODEL", ""))
    parser.add_argument("--max-tests-per-property", type=int, default=3)
    parser.add_argument("--retry-errors", action="store_true",
                        help="retry previously failed model or runner calls")
    args = parser.parse_args()
    if not args.api_base or not args.model:
        parser.error("set MRMINER_API_BASE and MRMINER_MODEL or pass --api-base and --model")
    if args.max_tests_per_property < 1:
        parser.error("--max-tests-per-property must be positive")
    index = json.loads(args.input.read_text())["index"]
    provider = LocalOpenAIProvider(base_url=args.api_base, model=args.model)
    runner = torch_run_test if args.framework == "pytorch" else tf_run_test
    prompt = None if args.framework == "pytorch" else TF_PROMPT
    screened: dict[str, list[dict]] = defaultdict(list)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    completed = {}
    if args.audit.exists():
        for line in args.audit.read_text().splitlines():
            try:
                previous = json.loads(line)
                completed[previous["oracle_id"]] = previous
            except (json.JSONDecodeError, KeyError):
                continue
    with args.audit.open("a", encoding="utf-8") as audit_file:
        for seed in make_seeds(index):
            op = seed.ops[0]
            refs = {item.oracle_id: item for item in oracles_for(seed.ops, index)}
            for item in index[op]:
                oracle_id = item["oracle_id"]
                if oracle_id in completed and not (
                    args.retry_errors and completed[oracle_id]["status"] == "error"
                ):
                    if completed[oracle_id]["status"] == "activated":
                        screened[op].append(item)
                    continue
                row = {"operator": op, "oracle_id": oracle_id, "status": "inactive",
                       "attempts": [], "model": args.model}
                ref = refs.get(oracle_id)
                if ref is None:
                    row["status"] = "unretrievable"
                else:
                    try:
                        for test in generate(seed, ref, provider, system_prompt=prompt):
                            if len(row["attempts"]) >= args.max_tests_per_property:
                                break
                            finding = runner(test, ref, isolate=True)
                            row["attempts"].append({"test_id": test.test_id,
                                                    "code": test.code, "verdict": finding.verdict,
                                                    "detail": finding.detail})
                            if not finding.verdict.startswith("skip:"):
                                row["status"] = "activated"
                                screened[op].append(item)
                                break
                    except Exception as error:
                        row["status"] = "error"
                        row["error"] = f"{type(error).__name__}: {error}"
                audit_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                audit_file.flush()
                print(json.dumps({"oracle_id": oracle_id, "status": row["status"]}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"index": screened,
                                       "operator_count": len(screened),
                                       "oracle_count": sum(map(len, screened.values()))},
                                      indent=2, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
