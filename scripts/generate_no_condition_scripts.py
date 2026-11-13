#!/usr/bin/env python3
"""Generate an independent script corpus without waiting for checker synthesis.

Run from tech_release, for example:
  MRMINER_API_BASE=http://127.0.0.1:8008/v1 MRMINER_MODEL=qwq-32b \
    python3 scripts/generate_no_condition_scripts.py --framework pytorch \
    --seeds runs/no_condition/full_seeds_pytorch.jsonl \
    --output runs/no_condition/generated_pytorch/tests.jsonl --target 2500

Inputs: the Full seed JSONL and the existing Full oracle indexes under ../reports/.
Output: append-only JSONL scripts at --output; no test verdicts are inferred here.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tech_release"))
sys.path.insert(0, str(ROOT / "scripts"))

from mrminer.fuzz.generator import generate
from mrminer.fuzz.llm import LocalOpenAIProvider
from mrminer.fuzz.oracle_index import oracles_for
from mrminer.fuzz.types import Seed


def load_full_index(framework: str) -> dict:
    if framework == "pytorch":
        from scripts.fuzz_campaign_continuous import load_merged_index

        return load_merged_index(
            "reports/oracle_validation_v2/operator_oracle_index_full_propagated.json"
        )
    from scripts.fuzz_tf_run import load_tf_index

    index, _ = load_tf_index()
    translated = ROOT / "reports/transfer/tf_translated_from_pt.json"
    if translated.exists():
        for op, entries in json.loads(translated.read_text())["index"].items():
            index.setdefault(op, []).extend(entries)
    return index


def load_seeds(path: Path) -> list[Seed]:
    seeds = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        seeds.append(Seed(seed_id=row["seed_id"], ops=tuple(row["ops"]),
                          note=row.get("note", ""), parent_code=row.get("code", ""),
                          origin=row.get("origin", "manual")))
    return seeds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", choices=("pytorch", "tensorflow"), required=True)
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=int, default=2500)
    parser.add_argument("--max-attempts", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--api-base", default=os.environ.get("MRMINER_API_BASE", ""))
    parser.add_argument("--model", default=os.environ.get("MRMINER_MODEL", ""))
    args = parser.parse_args()
    if not args.api_base or not args.model:
        parser.error("set MRMINER_API_BASE and MRMINER_MODEL")
    if args.target < 1:
        parser.error("--target must be positive")

    os.chdir(ROOT)
    index = load_full_index(args.framework)
    seeds = load_seeds(args.seeds if args.seeds.is_absolute() else ROOT / "tech_release" / args.seeds)
    eligible = [(seed, oracles_for(seed.ops, index)) for seed in seeds]
    eligible = [(seed, oracles) for seed, oracles in eligible if oracles]
    if not eligible:
        raise ValueError("no Full seed retrieves a property")
    provider = LocalOpenAIProvider(base_url=args.api_base, model=args.model)
    if args.framework == "tensorflow":
        from scripts.fuzz import TF_PROMPT
        prompt = TF_PROMPT
    else:
        prompt = None

    output = args.output if args.output.is_absolute() else ROOT / "tech_release" / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    errors = output.with_name("generation_errors.jsonl")
    existing = [json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
    seen = {(row["oracle_id"], row["code"]) for row in existing}
    count = len(existing)
    counts = Counter(row["oracle_id"] for row in existing)
    rng = random.Random(args.seed)
    rng.shuffle(eligible)
    print(json.dumps({"framework": args.framework, "seeds": len(seeds),
                      "eligible_seeds": len(eligible), "existing": count,
                      "target": args.target, "output": str(output)}), flush=True)
    attempt = 0
    with output.open("a", encoding="utf-8") as out, errors.open("a", encoding="utf-8") as err:
        while count < args.target and attempt < args.max_attempts:
            seed, oracles = eligible[attempt % len(eligible)]
            if attempt and attempt % len(eligible) == 0:
                rng.shuffle(eligible)
            min_count = min(counts[o.oracle_id] for o in oracles)
            choices = [o for o in oracles if counts[o.oracle_id] == min_count]
            oracle = rng.choice(choices)
            attempt += 1
            try:
                tests = generate(seed, oracle, provider, system_prompt=prompt,
                                 guidance="full")
            except Exception as exc:
                err.write(json.dumps({"attempt": attempt, "seed_id": seed.seed_id,
                                      "oracle_id": oracle.oracle_id,
                                      "error": f"{type(exc).__name__}: {exc}"}) + "\n")
                err.flush()
                continue
            for test in tests:
                key = (oracle.oracle_id, test.code)
                if key in seen:
                    continue
                row = {"test_id": test.test_id, "seed_id": seed.seed_id,
                       "oracle_id": oracle.oracle_id, "operator": oracle.op,
                       "code": test.code, "rationale": test.rationale,
                       "framework": args.framework, "generation_only": True}
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                seen.add(key)
                counts[oracle.oracle_id] += 1
                count += 1
                if count % 25 == 0 or count == args.target:
                    print(f"[{args.framework}] generated={count}/{args.target} attempts={attempt}",
                          flush=True)
                if count >= args.target:
                    break
    return 0 if count == args.target else 1


if __name__ == "__main__":
    raise SystemExit(main())
