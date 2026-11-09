#!/usr/bin/env python3
"""Mine executable properties from an attributed PyTorch PR corpus."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from mrminer.fuzz.llm import LocalOpenAIProvider
from mrminer.s2_graph import EntityGraph
from mrminer.s3_oracle.claim_extractor import ClaimExtractor
from mrminer.s3_oracle.context_assembler import AssembledContext, assemble_context
from mrminer.s3_oracle.evidence_bundle import EvidenceBudget, EvidenceRetriever
from mrminer.s3_oracle.inducer import OracleInducer


def operators(args: argparse.Namespace) -> list[str]:
    selected = list(args.operator)
    if args.operators_file:
        selected.extend(line.strip() for line in args.operators_file.read_text().splitlines()
                        if line.strip() and not line.lstrip().startswith("#"))
    if not selected:
        raise ValueError("provide --operator or --operators-file")
    return list(dict.fromkeys(selected))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--operator", action="append", default=[],
                        help="Attributed operator ID, e.g. op:aten::clamp")
    parser.add_argument("--pr-number", type=int, action="append", default=[],
                        help="Optional PR number filter; may be repeated")
    parser.add_argument("--operators-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("MRMINER_MODEL", ""))
    parser.add_argument("--api-base", default=os.environ.get("MRMINER_API_BASE", ""))
    parser.add_argument("--max-prs", type=int, default=20)
    parser.add_argument("--artifact-id", default="art:target-build")
    args = parser.parse_args()
    if not args.model or not args.api_base:
        parser.error("set MRMINER_MODEL and MRMINER_API_BASE or pass --model and --api-base")
    if args.max_prs < 1:
        parser.error("--max-prs must be positive")
    selected = operators(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    graph = EntityGraph(args.graph_root)
    retriever = EvidenceRetriever(
        graph, corpus_root=args.corpus_root, budget=EvidenceBudget(max_prs=args.max_prs)
    )
    provider = LocalOpenAIProvider(base_url=args.api_base, model=args.model)
    extractor = ClaimExtractor(provider)
    inducer = OracleInducer()
    at = datetime.now(timezone.utc).isoformat()
    previous = json.loads(args.output.read_text()) if args.output.exists() else {}
    if previous and previous.get("model") != args.model:
        raise ValueError("existing output uses a different model; choose a new output path")
    index: dict[str, list[dict]] = defaultdict(list, previous.get("index", {}))
    seen = {item["oracle_id"] for entries in index.values() for item in entries}
    records: list[dict] = list(previous.get("records", []))
    counts: Counter = Counter(previous.get("counts", {}))
    completed = {(row["operator"], row["pr_number"]) for row in records}

    def persist() -> None:
        payload = {"index": dict(sorted(index.items())), "operator_count": len(index),
                   "oracle_count": len(seen), "model": args.model,
                   "generated_at": at, "counts": dict(counts), "records": records}
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(args.output)

    for op in selected:
        bundle = retriever.retrieve(op, pr_numbers=set(args.pr_number) or None)
        if args.pr_number and not bundle.prs:
            raise ValueError(f"none of the requested PRs are attributed to {op}")
        for pr in bundle.prs:
            if (op, pr.pr_number) in completed:
                continue
            row = {"operator": op, "pr_number": pr.pr_number,
                   "status": "pending", "claim_ids": [], "property_ids": []}
            try:
                report = extractor.extract_pr(pr, effective_artifact=args.artifact_id)
                row["claim_ids"] = [claim.claim_id for claim in report.claims]
                row["rejected_claims"] = len(report.rejected)
                for claim in report.claims:
                    outcome = assemble_context(claim, program_ids=(), at=at)
                    if not isinstance(outcome, AssembledContext):
                        counts["context_gap"] += 1
                        continue
                    for action in inducer.induce(
                        entity_id=op, claims=(claim,), existing_oracles=(),
                        context=outcome.context,
                    ):
                        if action.status.value != "ready":
                            counts["not_ready"] += 1
                            continue
                        for spec in action.new_specs:
                            if spec.oracle_id in seen:
                                continue
                            seen.add(spec.oracle_id)
                            item = spec.to_record()
                            item["release_provenance"] = {"pr_number": pr.pr_number,
                                                          "source_operator": op}
                            index[op].append(item)
                            row["property_ids"].append(spec.oracle_id)
                            counts["properties"] += 1
                row["status"] = "ok"
                counts["processed_prs"] += 1
            except Exception as error:
                row["status"] = "error"
                row["error"] = f"{type(error).__name__}: {error}"
                counts["errors"] += 1
            records.append(row)
            completed.add((op, pr.pr_number))
            persist()
            print(json.dumps({"operator": op, "pr": pr.pr_number,
                              "status": row["status"], "properties": len(row["property_ids"])},
                             ensure_ascii=False), flush=True)

    persist()
    print(json.dumps({"output": str(args.output), "counts": dict(counts)}))
    return 2 if counts["errors"] and not counts["processed_prs"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
