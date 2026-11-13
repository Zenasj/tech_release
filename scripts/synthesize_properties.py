#!/usr/bin/env python3
"""Synthesize executable P, T, pi, and rho fields from attributed PR evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path

from mrminer.fuzz.llm import LocalOpenAIProvider
from mrminer.fuzz.property_model import PropertyChecker
from mrminer.llm.provider import invoke_structured


SCHEMA = {
    "type": "object",
    "properties": {
        "P": {"type": "string"}, "T": {"type": "string"},
        "pi": {"type": "string"}, "rho": {"type": "string"},
        "origin_condition": {"type": "string"},
        "description": {"type": "string"},
    },
    "required": ["P", "T", "pi", "rho", "origin_condition", "description"],
}


PROMPT = """\
Construct ONE executable operator property supported by the supplied PR evidence.
Return only the JSON schema. P, pi, rho, and origin_condition are Python
expressions, not statements. P receives c (source operator, marked call args
in c.args/c.kwargs, designated site, inferred flag, and PyTorch FX site operands
in c.site_args/c.site_kwargs; TensorFlow StableHLO instruction metadata is
in c.site_metadata). pi receives c and r, where r.kind is
'return', 'raise', or 'abort', r.exception is the exception class, and
r.output_at(c.output_path) reads the marked call's returned value. Return
undefined() when the requested observation does not exist. rho receives the two
projected values as a and b and must yield a Python bool. close(a,b) compares
values with shape and dtype checks. np, math, and the selected framework module
(torch or tf) are available. T must be eager_to_inductor, eager_to_xla,
identity_eager, or identity_compiled, as appropriate to the framework. Use an
identity transform for a single-execution requirement. P must accept the PR's
regression examples and must not generalize beyond its supported intent. When
the PR supports a mapping from a rejected source call to the indexed core op,
give its argument condition as origin_condition; otherwise use an empty string.
Never infer a property from an operator name alone. For a value property, project
the relevant values; for metadata, project the exact metadata; for validation,
project the execution outcome or exception category. Do not output checker code
outside these four expressions.
"""


def _pr_number(item: dict) -> int | None:
    release = item.get("release_provenance") or {}
    provenance = item.get("provenance") or {}
    value = release.get("pr_number") or provenance.get("pr")
    return int(value) if value is not None else None


def _text(value) -> str:
    if isinstance(value, dict):
        return str(value.get("text") or "")
    return str(value or "")


def _evidence(corpus: Path, pr_number: int | None) -> dict:
    if pr_number is None:
        return {}
    path = corpus / f"{pr_number}.json"
    if not path.exists():
        return {}
    pr = json.loads(path.read_text())
    patches = [f"{f.get('path') or f.get('filename')}:\n{str(f.get('patch') or '')[:4000]}"
               for f in (pr.get("touched_files_patch") or []) if f.get("patch")]
    tests = [f"{t.get('test_name')}:\n{str(t.get('added_code') or t.get('code') or '')[:3000]}"
             for t in (pr.get("tests_added") or [])]
    comments = [_text(c) for c in (pr.get("comments") or [])]
    return {"pr_number": pr_number, "title": pr.get("title", ""),
            "body": _text(pr.get("body"))[:5000],
            "patches": patches[:12], "regression_tests": tests[:12],
            "discussion": comments[:12]}


def synthesize(index: dict[str, list[dict]], corpus: Path, provider,
               framework: str, max_items: int | None = None) -> tuple[dict, list[dict]]:
    result = deepcopy(index)
    audit = []
    attempted = 0
    for operator, items in result.items():
        for item in items:
            if item.get("property"):
                continue
            if max_items is not None and attempted >= max_items:
                return result, audit
            attempted += 1
            pr_number = _pr_number(item)
            evidence = _evidence(corpus, pr_number)
            row = {"oracle_id": item["oracle_id"], "operator": operator,
                   "pr_number": pr_number, "status": "pending"}
            if not evidence:
                row["status"] = "missing_pr_evidence"
                audit.append(row)
                continue
            source = ((item.get("guard") or {}).get("static") or {}).get("source_entry") or {}
            payload = {"framework": framework, "core_operator": operator,
                       "source_operator": source.get("operator", operator),
                       "candidate": {"claim": item.get("claim"), "guard": item.get("guard"),
                                     "relation": item.get("relation")},
                       "attribution": {"home_tier": item.get("home_tier"),
                                       "lowering_motif": ((item.get("guard") or {}).get("static") or {}).get("lowering_motif")},
                       "evidence": evidence}
            row["evidence_sha256"] = hashlib.sha256(
                json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            try:
                response = invoke_structured(provider, task="property.synthesize",
                                             system_prompt=PROMPT, input_payload=payload,
                                             output_schema=SCHEMA)
                record = {key: response.payload[key] for key in SCHEMA["required"]}
                PropertyChecker(record, framework=framework)
                item["property"] = record
                row["status"] = "synthesized"
            except Exception as error:
                row["status"] = "error"
                row["error"] = f"{type(error).__name__}: {error}"
            audit.append(row)
    return result, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=("pytorch", "tensorflow"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True,
                        help="directory containing <PR number>.json records")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--api-base", default=os.environ.get("MRMINER_API_BASE", ""))
    parser.add_argument("--model", default=os.environ.get("MRMINER_MODEL", ""))
    parser.add_argument("--max-items", type=int)
    args = parser.parse_args()
    if not args.api_base or not args.model:
        parser.error("set MRMINER_API_BASE and MRMINER_MODEL")
    document = json.loads(args.input.read_text())
    provider = LocalOpenAIProvider(base_url=args.api_base, model=args.model)
    index, audit = synthesize(document["index"], args.corpus, provider,
                              args.framework, args.max_items)
    document["index"] = index
    document["property_synthesis"] = {"model": args.model, "framework": args.framework,
                                      "temperature": 0.6, "top_p": 0.95,
                                      "corpus": str(args.corpus)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(args.output)
    args.audit.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in audit))
    print(json.dumps({"synthesized": sum(r["status"] == "synthesized" for r in audit),
                      "errors": sum(r["status"] == "error" for r in audit),
                      "missing_evidence": sum(r["status"] == "missing_pr_evidence" for r in audit)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
