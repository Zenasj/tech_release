#!/usr/bin/env python3
"""Export the seed corpus used by the RQ4 Full campaigns as JSONL.

Run from the repository root, for example:
  python3 tech_release/scripts/export_full_seeds.py --framework pytorch \
    --output tech_release/runs/no_condition/full_seeds_pytorch.jsonl
Input: the RQ4 Full oracle indexes under reports/.
Output: one JSONL seed manifest at --output.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def full_seeds(framework: str):
    if framework == "pytorch":
        from scripts.fuzz_campaign_continuous import build_corpus, load_merged_index

        index = load_merged_index(
            "reports/oracle_validation_v2/operator_oracle_index_full_propagated.json"
        )
        seeds = build_corpus(index, "cuda")
    else:
        from scripts.fuzz_tf_run import load_tf_index, tf_controlflow_seeds, tf_seeds

        index, _ = load_tf_index()
        translated = ROOT / "reports/transfer/tf_translated_from_pt.json"
        if translated.exists():
            for op, entries in json.loads(translated.read_text())["index"].items():
                index.setdefault(op, []).extend(entries)
        seeds = tf_seeds(index, "cpu") + tf_controlflow_seeds(index, k=40, device="cpu")
    random.Random(0).shuffle(seeds)
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", choices=("pytorch", "tensorflow"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seeds = full_seeds(args.framework)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for seed in seeds:
            file.write(json.dumps({"seed_id": seed.seed_id, "ops": seed.ops,
                                   "note": seed.note, "code": seed.parent_code,
                                   "origin": seed.origin}, ensure_ascii=False) + "\n")
    print(json.dumps({"framework": args.framework, "seeds": len(seeds),
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
