#!/usr/bin/env python3
"""Screen candidate properties against executable PR-grounded inputs."""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from mrminer.fuzz.llm import LocalOpenAIProvider
from mrminer.fuzz.oracle_index import oracles_for
from mrminer.fuzz.regression_adaptation import FALLBACK_PROMPT, SCHEMA, validate_code
from mrminer.fuzz.runner import run_test as torch_run_test
from mrminer.fuzz.types import GeneratedTest
from mrminer.llm.provider import invoke_structured
from mrminer.tf.runner_tf import run_test as tf_run_test


def supporting_prs(item: dict) -> set[int]:
    release = item.get("release_provenance") or {}
    provenance = item.get("provenance") or {}
    values = [release.get("pr_number"), provenance.get("pr")]
    values.extend(item.get("supporting_prs") or [])
    return {int(value) for value in values if value is not None}


def load_witnesses(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row.get("pr_number"), int) or not isinstance(row.get("code"), str):
            raise ValueError(f"{path}:{number}: expected integer pr_number and code")
        rows.append(row)
    return rows


def generate_from_pr(operator: str, item: dict, corpus: Path, provider) -> dict | None:
    source = (((item.get("guard") or {}).get("static") or {})
              .get("source_entry") or {}).get("operator") or operator
    for pr_number in sorted(supporting_prs(item)):
        path = corpus / f"{pr_number}.json"
        if not path.is_file():
            continue
        pr = json.loads(path.read_text(encoding="utf-8"))
        evidence = {
            "framework": "tensorflow" if operator.startswith("op:stablehlo::") else "pytorch",
            "operator": operator, "source_operator": source,
            "pr_number": pr_number, "pr_title": pr.get("title", ""),
            "pr_body": str(pr.get("body") or "")[:4000],
            "patches": [{"path": patch.get("path") or patch.get("filename"),
                         "patch": str(patch.get("patch") or "")[:3000]}
                        for patch in (pr.get("touched_files_patch") or [])[:8]],
            "tests_added": [{"test_name": test.get("test_name"),
                             "added_code": str(test.get("added_code") or "")[:3000]}
                            for test in (pr.get("tests_added") or [])[:4]],
            "property": item["property"],
        }
        response = invoke_structured(provider, task="regression.screen_input",
                                     system_prompt=FALLBACK_PROMPT,
                                     input_payload=evidence, output_schema=SCHEMA)
        code = response.payload["code"]
        validate_code(code, source)
        return {"pr_number": pr_number, "oracle_id": item["oracle_id"],
                "test_name": "", "code": code, "origin": "generated_from_pr",
                "prompt_sha": response.prompt_sha, "model": response.model,
                "request_id": response.request_id}
    return None


def screen(index: dict[str, list[dict]], witnesses: list[dict], runner,
           audit_path: Path, fallback=None) -> dict[str, list[dict]]:
    by_pr: dict[int, list[dict]] = defaultdict(list)
    for witness in witnesses:
        by_pr[witness["pr_number"]].append(witness)
    admitted: dict[str, list[dict]] = defaultdict(list)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("w", encoding="utf-8") as audit:
        for op, items in index.items():
            for item in items:
                oid = item["oracle_id"]
                source = (item.get("guard") or {}).get("static", {}).get("source_entry") or {}
                ops = [op]
                if source.get("operator"):
                    ops.append(source["operator"])
                ref = next((ref for ref in oracles_for(ops, {op: [item]})
                            if ref.oracle_id == oid), None)
                row = {"operator": op, "oracle_id": oid, "status": "no_witness",
                       "attempts": []}
                if not item.get("property"):
                    row["status"] = "missing_checker"
                    audit.write(json.dumps(row, ensure_ascii=False) + "\n")
                    continue
                eligible = [w for pr in supporting_prs(item) for w in by_pr[pr]
                            if not w.get("oracle_id") or w["oracle_id"] == oid]
                if eligible and ref is None:
                    row["status"] = "unretrievable"
                for i, witness in enumerate(eligible if ref else []):
                    test = GeneratedTest(test_id=f"regression:{oid}:{i}",
                                         seed_id=f"pr:{witness['pr_number']}",
                                         target_oracle=oid, code=witness["code"])
                    try:
                        finding = runner(test, ref, isolate=True)
                        verdict, detail = finding.verdict, finding.detail
                        traced = list(finding.contexts)
                        activated = verdict == "pass" or (
                            finding.triggered and verdict not in
                            {"compiled_crash", "worker_died", "worker_timeout"})
                    except Exception as error:
                        verdict = "skip:runner_error"
                        detail = f"{type(error).__name__}: {error}"
                        activated = False
                        traced = []
                    attempt = {"pr_number": witness["pr_number"],
                               "test_name": witness.get("test_name", ""),
                               "origin": witness.get("origin", "adapted_regression"),
                               "code": witness["code"],
                               "verdict": verdict, "detail": detail,
                               "contexts": traced}
                    row["attempts"].append(attempt)
                    if activated:
                        row["status"] = "activated"
                        row["witness"] = attempt
                        admitted[op].append(item)
                        break
                if ref and row["status"] != "activated" and fallback is not None:
                    try:
                        candidate = fallback(op, item)
                        if candidate is not None:
                            test = GeneratedTest(test_id=f"screen-input:{oid}",
                                                 seed_id=f"pr:{candidate['pr_number']}",
                                                 target_oracle=oid, code=candidate["code"])
                            finding = runner(test, ref, isolate=True)
                            attempt = {**candidate, "verdict": finding.verdict,
                                       "detail": finding.detail,
                                       "contexts": list(finding.contexts)}
                            row["attempts"].append(attempt)
                            if finding.verdict == "pass" or (
                                finding.triggered and finding.verdict not in
                                {"compiled_crash", "worker_died", "worker_timeout"}):
                                row["status"] = "activated"
                                row["witness"] = attempt
                                admitted[op].append(item)
                    except Exception as error:
                        row["fallback_error"] = f"{type(error).__name__}: {error}"
                if eligible and row["status"] == "no_witness":
                    row["status"] = "inactive"
                if row["attempts"] and row["status"] == "no_witness":
                    row["status"] = "inactive"
                audit.write(json.dumps(row, ensure_ascii=False) + "\n")
    return admitted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=("pytorch", "tensorflow"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--witnesses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--api-base", default=os.environ.get("MRMINER_API_BASE", ""))
    parser.add_argument("--model", default=os.environ.get("MRMINER_MODEL", ""))
    args = parser.parse_args()
    index = json.loads(args.input.read_text())["index"]
    runner = torch_run_test if args.framework == "pytorch" else tf_run_test
    fallback = None
    if args.corpus:
        if not args.api_base or not args.model:
            parser.error("set MRMINER_API_BASE and MRMINER_MODEL")
        provider = LocalOpenAIProvider(base_url=args.api_base, model=args.model)
        fallback = lambda operator, item: generate_from_pr(
            operator, item, args.corpus, provider)
    admitted = screen(index, load_witnesses(args.witnesses), runner, args.audit,
                      fallback=fallback)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"index": admitted,
                                       "operator_count": len(admitted),
                                       "oracle_count": sum(map(len, admitted.values()))},
                                      indent=2, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
