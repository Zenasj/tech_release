#!/usr/bin/env python3
"""Adapt collected PR regression-test additions into screening witnesses."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from mrminer.fuzz.llm import LocalOpenAIProvider
from mrminer.fuzz.regression_adaptation import ADAPT_PROMPT, SCHEMA, validate_code
from mrminer.llm.provider import invoke_structured


def supporting_prs(item: dict) -> list[int]:
    release = item.get("release_provenance") or {}
    provenance = item.get("provenance") or {}
    values = [release.get("pr_number"), provenance.get("pr")]
    values.extend(item.get("supporting_prs") or [])
    return sorted({int(value) for value in values if value is not None})


def source_operator(item: dict, index_operator: str) -> str:
    entry = ((item.get("guard") or {}).get("static") or {}).get("source_entry") or {}
    return entry.get("operator") or index_operator


def adapt(index: dict[str, list[dict]], corpus: Path, provider, *,
          max_tests_per_property: int | None = None, max_items: int | None = None
          ) -> tuple[list[dict], list[dict]]:
    witnesses: list[dict] = []
    audit: list[dict] = []
    attempted = 0
    for operator, items in sorted(index.items()):
        for item in items:
            if not item.get("property"):
                continue
            oracle_id = item["oracle_id"]
            source = source_operator(item, operator)
            for pr_number in supporting_prs(item):
                path = corpus / f"{pr_number}.json"
                if not path.is_file():
                    audit.append({"operator": operator, "oracle_id": oracle_id,
                                  "pr_number": pr_number, "status": "missing_pr_record"})
                    continue
                pr = json.loads(path.read_text(encoding="utf-8"))
                patches = {record.get("path") or record.get("filename"): record.get("patch", "")
                           for record in pr.get("touched_files_patch") or []}
                tests = [test for test in pr.get("tests_added") or []
                         if (test.get("added_code") or "").strip()]
                selected = tests[:max_tests_per_property] if max_tests_per_property else tests
                for test in selected:
                    if max_items is not None and attempted >= max_items:
                        return witnesses, audit
                    attempted += 1
                    test_path = test.get("path") or ""
                    test_name = test.get("test_name") or ""
                    added_code = test["added_code"]
                    key = {"operator": operator, "oracle_id": oracle_id,
                           "pr_number": pr_number, "test_name": test_name,
                           "test_node_id": test.get("test_node_id", ""),
                           "test_path": test_path,
                           "source_sha256": hashlib.sha256(added_code.encode()).hexdigest()}
                    payload = {**key, "framework": "tensorflow" if source.startswith("op:tf::")
                               or operator.startswith("op:stablehlo::") else "pytorch",
                               "source_operator": source,
                               "pr_title": pr.get("title", ""),
                               "added_code": added_code,
                               "test_patch": str(patches.get(test_path) or "")[:10000],
                               "property": item["property"]}
                    try:
                        response = invoke_structured(
                            provider, task="regression.adapt", system_prompt=ADAPT_PROMPT,
                            input_payload=payload, output_schema=SCHEMA)
                        code = response.payload["code"]
                        validate_code(code, source)
                        witnesses.append({"pr_number": pr_number, "oracle_id": oracle_id,
                                          "test_name": test_name, "test_node_id": key["test_node_id"],
                                          "test_path": test_path,
                                          "source_sha256": key["source_sha256"], "code": code})
                        audit.append({**key, "status": "adapted",
                                      "prompt_sha": response.prompt_sha,
                                      "model": response.model,
                                      "request_id": response.request_id,
                                      "source_added_code": added_code, "code": code})
                    except Exception as error:
                        audit.append({**key, "status": "adaptation_error",
                                      "error": f"{type(error).__name__}: {error}"})
                if not tests:
                    audit.append({"operator": operator, "oracle_id": oracle_id,
                                  "pr_number": pr_number, "status": "no_added_test_code"})
    return witnesses, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--api-base", default=os.environ.get("MRMINER_API_BASE", ""))
    parser.add_argument("--model", default=os.environ.get("MRMINER_MODEL", ""))
    parser.add_argument("--max-tests-per-property", type=int)
    parser.add_argument("--max-items", type=int)
    args = parser.parse_args()
    if not args.api_base or not args.model:
        parser.error("set MRMINER_API_BASE and MRMINER_MODEL")
    if args.max_tests_per_property is not None and args.max_tests_per_property < 1:
        parser.error("--max-tests-per-property must be positive")
    if args.max_items is not None and args.max_items < 1:
        parser.error("--max-items must be positive")
    index = json.loads(args.input.read_text(encoding="utf-8"))["index"]
    provider = LocalOpenAIProvider(base_url=args.api_base, model=args.model)
    witnesses, audit = adapt(index, args.corpus, provider,
                             max_tests_per_property=args.max_tests_per_property,
                             max_items=args.max_items)
    for path in (args.output, args.audit):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                   for row in witnesses), encoding="utf-8")
    args.audit.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                  for row in audit), encoding="utf-8")
    print(json.dumps({"adapted": len(witnesses),
                      "adaptation_errors": sum(row["status"] == "adaptation_error"
                                               for row in audit),
                      "audit": str(args.audit)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
