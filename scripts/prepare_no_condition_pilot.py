#!/usr/bin/env python3
"""Prepare PR-backed property candidates for a 500-test NoCondition pilot.

Example, run from tech_release:
  python3 scripts/prepare_no_condition_pilot.py \
    --framework pytorch --limit 100 \
    --output runs/no_condition_pilot_500/pytorch/source_index.json

Input: the existing project-level operator index and local PR corpus.
Output: a new index with explicit PR provenance; existing files are untouched.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "pytorch": (PROJECT_ROOT / "reports/oracle_validation_v2/operator_oracle_index_full_core_complete.json",
                PROJECT_ROOT / "data/pr_corpus_full/pytorch__pytorch"),
    "tensorflow": (PROJECT_ROOT / "reports/tf/operator_oracle_index_tf_stablehlo_complete.json",
                   PROJECT_ROOT / "data/pr_corpus_tf/tensorflow__tensorflow"),
}


def candidates(framework: str) -> tuple[list[tuple[str, dict]], Path]:
    source, corpus = SOURCES[framework]
    index = json.loads(source.read_text())["index"]
    result = []
    for operator, entries in index.items():
        for entry in entries:
            if framework == "tensorflow":
                provenance = entry.get("provenance") or {}
                if provenance.get("kind") == "translated":
                    continue
                pr = provenance.get("pr")
            else:
                match = re.search(r"\bPR\s*#(\d+)\b",
                                  (entry.get("claim") or {}).get("summary", ""))
                pr = int(match.group(1)) if match else None
            if pr is None or not (corpus / f"{pr}.json").is_file():
                continue
            copy = dict(entry)
            copy["release_provenance"] = {"pr_number": int(pr),
                                          "source_index": str(source)}
            result.append((operator, copy))
    return result, corpus


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=SOURCES, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    eligible, corpus = candidates(args.framework)
    if len(eligible) < args.limit:
        parser.error(f"only {len(eligible)} PR-backed records are available")
    random.Random(args.seed).shuffle(eligible)
    selected = eligible[:args.limit]
    index: dict[str, list[dict]] = defaultdict(list)
    for operator, item in selected:
        index[operator].append(item)
    payload = {"index": dict(sorted(index.items())),
               "pilot": {"framework": args.framework, "selection_seed": args.seed,
                         "eligible": len(eligible), "selected": len(selected),
                         "pr_corpus": str(corpus)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "eligible": len(eligible),
                      "selected": len(selected), "operators": len(index),
                      "corpus": str(corpus)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
