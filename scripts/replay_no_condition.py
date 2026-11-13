#!/usr/bin/env python3
"""Replay Full tests against every reached property, with and without P.

Example (run from tech_release):
  python3 scripts/replay_no_condition.py --framework pytorch \
    --index runs/pt_full/operator_oracle_index_screened.json \
    --tests runs/pt_full/tests.jsonl --output-dir runs/pt_no_condition

Inputs are a screened property index and Full's tests.jsonl. Outputs are
paired_results.jsonl and summary.json in a new output directory. No tests are
generated, and the property library is never modified.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import traceback
from collections import Counter
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from mrminer.fuzz.oracle_index import oracles_for  # noqa: E402
from mrminer.fuzz.seed_trace import trace_isolated  # noqa: E402

MARK = "__NO_CONDITION_RESULT__"


def load_index(path: Path) -> dict[str, list[dict]]:
    payload = json.loads(path.read_text())
    index = payload.get("index")
    if not isinstance(index, dict) or not index:
        raise ValueError(f"{path}: expected a nonempty property index")
    for op, entries in index.items():
        for item in entries:
            record = item.get("property") or {}
            missing = {"P", "T", "pi", "rho"} - record.keys()
            if missing:
                raise ValueError(f"{path}: {op}/{item.get('oracle_id')} lacks {sorted(missing)}")
    return index


def read_tests(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("code"), str) or not row["code"].strip():
                raise ValueError(f"{path}:{line_number}: missing test code")
            rows.append({"script_index": len(rows), "source_line": line_number,
                         "test_id": row.get("test_id"),
                         "target_oracle_id": row.get("oracle_id"),
                         "code": row["code"]})
            if len(rows) == limit:
                break
    if len(rows) != limit:
        raise ValueError(f"{path}: requested {limit} tests, found {len(rows)}")
    return rows


def worker(job_path: Path) -> int:
    from mrminer.fuzz.property_runtime import check_program_pair

    job = json.loads(job_path.read_text())
    try:
        if job["framework"] == "pytorch":
            import torch
            version = torch.__version__
            device = (torch.cuda.get_device_name(torch.cuda.current_device())
                      if torch.cuda.is_available() else "cpu")
        else:
            import tensorflow as tf
            version = tf.__version__
            device = [item.name for item in tf.config.list_physical_devices()]
        result = check_program_pair(job["code"], job["property"],
                                    operator=job["core_operator"],
                                    source_operator=job["source_operator"],
                                    framework=job["framework"])
        payload = {"result": result, "environment": {"framework_version": version,
                   "device": device, "CUDA_VISIBLE_DEVICES":
                   os.environ.get("CUDA_VISIBLE_DEVICES", ""), "random_seed": 0}}
    except BaseException as error:
        payload = {"error": f"{type(error).__name__}: {error}",
                   "traceback": traceback.format_exc()}
    print(MARK + json.dumps(payload, default=str, ensure_ascii=False))
    return 0


def replay_pair(job: dict, timeout: int) -> dict:
    fd, name = tempfile.mkstemp(suffix=".no-condition.json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(job, stream)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(PACKAGE_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        from mrminer.fuzz.gpu import place_env
        place_env(env)
        process = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                  "--worker", name], capture_output=True,
                                 text=True, timeout=timeout, env=env)
        for line in reversed(process.stdout.splitlines()):
            if line.startswith(MARK):
                return json.loads(line[len(MARK):])
        return {"error": f"worker exited with signal/exit code {process.returncode}",
                "stderr": process.stderr[-4000:]}
    except subprocess.TimeoutExpired as error:
        return {"error": f"worker timed out after {timeout}s",
                "stderr": (error.stderr or b"").decode(errors="replace")[-4000:]}
    finally:
        os.unlink(name)


def blocked_fail_sites(result: dict) -> list[str]:
    conditional = result["conditional"].get("contexts", [])
    no_condition = result["no_condition"].get("contexts", [])
    return [f"{before['call_id']}:{before['site']}"
            for before, after in zip(conditional, no_condition)
            if before["verdict"] == "skip:P" and after["verdict"] == "fail"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=("pytorch", "tensorflow"))
    parser.add_argument("--index", type=Path)
    parser.add_argument("--tests", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int, default=2500)
    parser.add_argument("--trace-timeout", type=int, default=180)
    parser.add_argument("--check-timeout", type=int, default=180)
    parser.add_argument("--prepare-only", action="store_true",
                        help="validate input files without executing tests")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker)
    if not all((args.framework, args.index, args.tests)):
        parser.error("--framework, --index, and --tests are required")
    if not args.prepare_only and args.output_dir is None:
        parser.error("--output-dir is required for replay")
    if args.limit < 1 or args.trace_timeout < 1 or args.check_timeout < 1:
        parser.error("limit and timeouts must be positive")

    index = load_index(args.index)
    tests = read_tests(args.tests, args.limit)
    if args.prepare_only:
        print(json.dumps({"scripts": len(tests), "indexed_properties":
                          sum(map(len, index.values()))}))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "paired_results.jsonl"
    counts: Counter = Counter()
    with output.open("x", encoding="utf-8") as stream:
        for test in tests:
            trace = trace_isolated(test["code"], args.framework, args.index,
                                   timeout=args.trace_timeout)
            counts["trace_errors"] += bool(trace.get("errors"))
            refs = oracles_for(trace.get("ops", []), index)
            seen = set()
            checks = []
            for ref in refs:
                key = (ref.oracle_id, ref.core_operator, ref.op)
                if key in seen:
                    continue
                seen.add(key)
                payload = replay_pair({"code": test["code"],
                                       "property": ref.property_record,
                                       "core_operator": ref.core_operator,
                                       "source_operator": ref.op,
                                       "framework": args.framework},
                                      args.check_timeout)
                check = {"property_id": ref.oracle_id,
                         "core_operator": ref.core_operator,
                         "source_operator": ref.op, **payload}
                if "result" in payload:
                    pair = payload["result"]
                    sites = blocked_fail_sites(pair)
                    check["blocked_fail_sites"] = sites
                    check["new_alarm"] = bool(sites and
                        pair["conditional"]["verdict"] != "fail" and
                        pair["no_condition"]["verdict"] == "fail")
                    counts[f"conditional:{pair['conditional']['verdict']}"] += 1
                    counts[f"no_condition:{pair['no_condition']['verdict']}"] += 1
                    counts["new_alarms"] += check["new_alarm"]
                else:
                    counts["worker_errors"] += 1
                checks.append(check)
            counts["scripts"] += 1
            counts["property_pairs"] += len(checks)
            row = dict(test)
            row.update({"trace": trace, "checks": checks})
            stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            stream.flush()

    summary = {"framework": args.framework, "tests": str(args.tests.resolve()),
               "index": str(args.index.resolve()), "seed": 0,
               "counts": dict(counts)}
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
