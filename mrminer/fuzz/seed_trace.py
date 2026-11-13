"""Trace a seed's marked calls to its actual Core ATen or StableHLO operators."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from mrminer.fuzz.property_model import Context, PropertyChecker
from mrminer.fuzz.property_runtime import _clone_inputs
from mrminer.fuzz.provenance import TraceRecorder

_MARK = "__SEED_TRACE__"
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def trace_code(code: str, framework: str, index: dict[str, list[dict]]) -> dict:
    namespace = {"mark": lambda _op, fn, *a, **kw: fn(*a, **kw)}
    try:
        exec(code, namespace)  # noqa: S102 -- isolated worker
        fn, args = namespace["build"]()
        args = _clone_inputs(tuple(args), framework)
    except Exception as error:
        return {"ops": [], "errors": [f"build: {type(error).__name__}: {error}"]}
    recorder = TraceRecorder(lambda call: _clone_inputs(call, framework))
    namespace["mark"] = recorder.mark
    try:
        fn(*args)
    except Exception:
        pass  # a rejected marked call may still yield an inferred context
    ops: set[str] = set()
    errors: list[str] = []
    for call in recorder.calls:
        if not call.raised:
            ops.add(call.source_operator)
        try:
            if framework == "pytorch":
                from torch._decomp import core_aten_decompositions
                from torch.fx.experimental.proxy_tensor import make_fx
                graph = make_fx(lambda *a: call.fn(*a, **call.kwargs),
                                decomposition_table=core_aten_decompositions(),
                                tracing_mode="fake")(*call.args)
                traced = {f"op:{node.target._schema.name}" for node in graph.graph.nodes
                          if node.op == "call_function" and hasattr(node.target, "_schema")}
            else:
                from mrminer.tf.stablehlo_ops import program_stablehlo_sites
                traced = {site["operator"] for site in program_stablehlo_sites(
                    lambda *a: call.fn(*a, **call.kwargs), call.args)}
            ops.update(traced)
        except Exception as error:
            errors.append(f"call {call.call_id}: {type(error).__name__}: {error}")
            if not call.raised:
                continue
            for core, entries in index.items():
                for item in entries:
                    source = ((item.get("guard") or {}).get("static") or {}).get("source_entry") or {}
                    if source.get("operator", core) != call.source_operator or not item.get("property"):
                        continue
                    try:
                        checker = PropertyChecker(item["property"], framework=framework)
                        context = Context(core, call.source_operator, "inferred:seed",
                                          call.call_id, call.args, call.kwargs, None, True)
                        if checker.origin_matches(context):
                            ops.add(core)
                    except Exception:
                        continue
    return {"ops": sorted(ops), "errors": errors, "marked_calls": len(recorder.calls)}


def trace_isolated(code: str, framework: str, index_path: Path,
                   timeout: int = 180) -> dict:
    fd, job_path = tempfile.mkstemp(suffix=".seed-trace.json")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump({"code": code, "framework": framework,
                       "index_path": str(index_path.resolve())}, handle)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(_PACKAGE_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        from mrminer.fuzz.gpu import place_env
        place_env(env)
        process = subprocess.run([sys.executable, "-m", "mrminer.fuzz.seed_trace",
                                  "--job", job_path], capture_output=True,
                                 text=True, timeout=timeout, env=env)
        for line in reversed(process.stdout.splitlines()):
            if line.startswith(_MARK):
                return json.loads(line[len(_MARK):])
        return {"ops": [], "errors": [f"worker exited {process.returncode}: "
                                     f"{process.stderr[-300:]}"]}
    except subprocess.TimeoutExpired:
        return {"ops": [], "errors": [f"trace timed out after {timeout}s"]}
    finally:
        os.unlink(job_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    index = json.loads(Path(job["index_path"]).read_text())["index"]
    result = trace_code(job["code"], job["framework"], index)
    print(_MARK + json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
