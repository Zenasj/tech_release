#!/usr/bin/env python3
"""Build executable seed programs from official framework operator tests."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path

from mrminer.fuzz.llm import LocalOpenAIProvider
from mrminer.fuzz.regression_adaptation import SCHEMA, validate_code
from mrminer.fuzz.seed_trace import trace_isolated
from mrminer.llm.provider import invoke_structured


PROMPT = """\
Adapt this official framework operator test into one executable seed. Preserve
the test's operator call, input construction, argument values, and relevant
program structure. Do not invent a different test. Return JSON with Python
code defining build() -> (fn, args). The runner supplies mark. Wrap exactly
the designated source call with mark(source_operator_id, callable, *args,
**kwargs), using the supplied source operator ID verbatim, and return its
result from fn. Do not call torch.compile or tf.function in the seed; the
runner selects eager and compiled modes. Import required public framework
modules at file scope.
"""


def source_operators(index: dict[str, list[dict]]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for core, items in index.items():
        for item in items:
            source = (((item.get("guard") or {}).get("static") or {})
                      .get("source_entry") or {}).get("operator") or core
            result[source].add(core)
    return dict(result)


def _snake(name: str) -> str:
    name = re.sub(r"V[0-9]+$", "", name)
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def aliases(source: str, framework: str) -> set[str]:
    name = source.rsplit("::", 1)[-1].split(".")[0]
    if framework == "pytorch":
        return {f"torch.{name}", f"torch.nn.functional.{name}",
                f"F.{name}", f"torch.ops.aten.{name}"}
    stem = _snake(name)
    flat = stem.replace("_", "")
    internal = {f"{module}.{variant}"
                for module in ("array_ops", "gen_array_ops", "math_ops",
                               "gen_math_ops", "linalg_ops")
                for variant in (stem, flat, name[0].lower() + name[1:])}
    return {f"tf.raw_ops.{name}", f"tf.{stem}", f"tf.{flat}",
            f"tf.math.{stem}", f"tf.math.{flat}",
            f"tf.linalg.{stem}", f"tf.linalg.{flat}",
            f"tensorflow.raw_ops.{name}", *internal}


def call_path(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{call_path(node.value)}.{node.attr}"
    return ""


def matches_call(node: ast.Call, names: set[str]) -> bool:
    path = call_path(node.func)
    return path in names or any(path.startswith(name + ".") for name in names)


def test_files(source_root: Path, framework: str) -> list[Path]:
    if framework == "pytorch":
        roots = [source_root / "test/inductor", source_root / "test"]
        paths = list(roots[0].glob("test_*.py"))
        paths.extend(roots[1].glob("test_*ops*.py"))
        return sorted(set(paths))
    roots = [source_root / "tensorflow/python/kernel_tests",
             source_root / "tensorflow/compiler/tests"]
    return sorted({path for root in roots if root.is_dir()
                   for path in root.rglob("*_test.py")})


def collect_candidates(source_root: Path, framework: str,
                       source_ops: set[str]) -> dict[str, list[dict]]:
    names = {op: aliases(op, framework) for op in source_ops}
    tokens = {name.rsplit(".", 1)[-1].lower()
              for group in names.values() for name in group}
    candidates: dict[str, list[dict]] = defaultdict(list)
    for path in test_files(source_root, framework):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            if not any(token in text.lower() for token in tokens):
                continue
            tree = ast.parse(text, filename=str(path))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test"):
                continue
            code = ast.get_source_segment(text, node)
            if not code:
                continue
            rel = str(path.relative_to(source_root))
            for source, group in names.items():
                if not any(isinstance(call, ast.Call) and matches_call(call, group)
                           for call in ast.walk(node)):
                    continue
                candidates[source].append({
                    "source_operator": source, "source_file": rel,
                    "test_name": node.name, "line": node.lineno,
                    "source_code": code,
                    "source_sha256": hashlib.sha256(code.encode()).hexdigest(),
                })
    def score(row: dict, op: str):
        name = row["test_name"].lower()
        difficult = any(word in name for word in (
            "bad", "error", "invalid", "unknown", "grad", "variable",
            "string", "resource", "empty"))
        stem = _snake(op.rsplit("::", 1)[-1])
        if framework == "pytorch":
            file_rank = 0 if "test/inductor/" in row["source_file"] else 1
        elif "tensorflow/compiler/tests/" in row["source_file"]:
            file_rank = 0
        elif stem in Path(row["source_file"]).stem.lower():
            file_rank = 1
        else:
            file_rank = 2
        return (file_rank, 1 if difficult else 0,
                len(row["source_code"]), row["source_file"], row["line"])
    return {op: sorted(rows, key=lambda row: score(row, op))
            for op, rows in candidates.items()}


class MarkCall(ast.NodeTransformer):
    def __init__(self, source: str, names: set[str]):
        self.source = source
        self.names = names
        self.count = 0

    def visit_Call(self, node: ast.Call):
        if matches_call(node, self.names):
            self.count += 1
            return ast.copy_location(ast.Call(
                func=ast.Name(id="mark", ctx=ast.Load()),
                args=[ast.Constant(self.source), node.func,
                      *[self.visit(arg) for arg in node.args]],
                keywords=[self.visit(kw) for kw in node.keywords]), node)
        return self.generic_visit(node)


def direct_pytorch_seed(row: dict, source: str) -> str | None:
    try:
        method = ast.parse(row["source_code"]).body[0]
    except SyntaxError:
        return None
    if not isinstance(method, ast.FunctionDef):
        return None
    harness = None
    for offset, statement in enumerate(method.body):
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            continue
        if call_path(statement.value.func) in {
            "self.common", "self.check_model", "self.check_model_gpu",
            "self.check_model_cuda",
        } and len(statement.value.args) >= 2:
            harness = (offset, statement.value)
            break
    if harness is None:
        return None
    offset, call = harness
    body = method.body[:offset]
    if any(isinstance(node, ast.Name) and node.id == "self"
           for statement in body for node in ast.walk(statement)):
        return None
    marker = MarkCall(source, aliases(source, "pytorch"))
    body = [marker.visit(statement) for statement in body]
    if marker.count != 1:
        return None
    body.append(ast.Return(value=ast.Tuple(
        elts=[call.args[0], call.args[1]], ctx=ast.Load())))
    build = ast.FunctionDef(name="build", args=ast.arguments(
        posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[],
        kwarg=None, defaults=[]), body=body, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[build], type_ignores=[]))
    code = ("import torch\nimport torch.nn.functional as F\n"
            "from torch import nn\nimport math\n\n" + ast.unparse(module) + "\n")
    validate_code(code, source)
    return code


def normalize_mark_source(code: str, source: str, framework: str) -> str:
    tree = ast.parse(code)
    changed = False
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call) or call_path(call.func) != "mark":
            continue
        if len(call.args) < 2:
            continue
        if not isinstance(call.args[0], ast.Constant) or not isinstance(
                call.args[0].value, str):
            continue
        if call.args[0].value == source:
            continue
        if call_path(call.args[1]) not in aliases(source, framework):
            continue
        call.args[0] = ast.Constant(source)
        changed = True
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n" if changed else code


def adapt_with_model(row: dict, source: str, framework: str, provider) -> tuple[str, dict]:
    payload = {**row, "source_code": row["source_code"][:14000],
               "source_operator": source, "framework": framework}
    response = invoke_structured(provider, task="official.seed_adapt",
                                 system_prompt=PROMPT, input_payload=payload,
                                 output_schema=SCHEMA)
    code = normalize_mark_source(response.payload["code"], source, framework)
    return code, {"model": response.model, "prompt_sha": response.prompt_sha,
                  "request_id": response.request_id}


def prepare(index_path: Path, source_root: Path, framework: str, provider=None, *,
            max_per_operator: int = 2, max_attempts_per_operator: int = 8,
            trace_timeout: int = 180) -> tuple[list[dict], list[dict]]:
    index = json.loads(index_path.read_text(encoding="utf-8"))["index"]
    mapping = source_operators(index)
    candidates = collect_candidates(source_root, framework, set(mapping))
    seeds, audit = [], []
    for source, cores in sorted(mapping.items()):
        accepted = 0
        for row in candidates.get(source, [])[:max_attempts_per_operator]:
            if accepted >= max_per_operator:
                break
            record = dict(row)
            record["core_operators"] = sorted(cores)
            code = None
            try:
                code = direct_pytorch_seed(row, source) if framework == "pytorch" else None
                provenance = {"adapter": "ast"} if code else {}
                if code is None:
                    if provider is None:
                        raise ValueError("model endpoint required for this test")
                    code, provenance = adapt_with_model(row, source, framework, provider)
                    provenance["adapter"] = "model"
                validate_code(code, source)
                traced = trace_isolated(code, framework, index_path, timeout=trace_timeout)
                reached = sorted(set(traced.get("ops") or ()) & cores)
                record.update(provenance)
                record["traced_operators"] = traced.get("ops", [])
                record["trace_errors"] = traced.get("errors", [])
                if not reached:
                    record["status"] = "not_retrievable"
                    audit.append(record)
                    continue
                seed_id = f"official:{framework}:{row['source_file']}:{row['test_name']}:{row['line']}:{source}"
                seeds.append({"seed_id": seed_id, "ops": reached, "code": code,
                              "note": f"{row['source_file']}:{row['line']}",
                              "origin": "framework",
                              "source_operator": source,
                              "source_sha256": row["source_sha256"]})
                record["status"] = "accepted"
                record["seed_id"] = seed_id
                accepted += 1
            except Exception as error:
                record["status"] = "adaptation_error"
                record["error"] = f"{type(error).__name__}: {error}"
                if code is not None:
                    record["proposed_code"] = code
            audit.append(record)
        if not candidates.get(source):
            audit.append({"source_operator": source, "core_operators": sorted(cores),
                          "status": "no_official_test"})
    return seeds, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=("pytorch", "tensorflow"), required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--max-per-operator", type=int, default=2)
    parser.add_argument("--max-attempts-per-operator", type=int, default=8)
    parser.add_argument("--trace-timeout", type=int, default=180)
    parser.add_argument("--api-base", default=os.environ.get("MRMINER_API_BASE", ""))
    parser.add_argument("--model", default=os.environ.get("MRMINER_MODEL", ""))
    args = parser.parse_args()
    if args.max_per_operator < 1 or args.max_attempts_per_operator < 1:
        parser.error("per-operator limits must be positive")
    provider = (LocalOpenAIProvider(base_url=args.api_base, model=args.model)
                if args.api_base and args.model else None)
    seeds, audit = prepare(args.index, args.source_root, args.framework, provider,
                           max_per_operator=args.max_per_operator,
                           max_attempts_per_operator=args.max_attempts_per_operator,
                           trace_timeout=args.trace_timeout)
    for path in (args.output, args.audit):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                   for row in seeds), encoding="utf-8")
    args.audit.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                  for row in audit), encoding="utf-8")
    print(json.dumps({"seeds": len(seeds), "accepted": sum(
        row["status"] == "accepted" for row in audit),
        "uncovered_sources": sum(row["status"] == "no_official_test" for row in audit)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
