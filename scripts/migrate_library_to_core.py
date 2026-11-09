#!/usr/bin/env python3
"""Transform the existing PyTorch property library onto Core ATen subjects.

Reads the operator-indexed property library, re-homes each property onto a Core
ATen operator (per scripts/core_homing.py), records the entry operator in the
applicability predicate P (`guard.static.source_entry`), flags `core_homed`, and
re-indexes by the new subject. Results (and the verdict `set_`-style residuals)
follow the chosen policy: residuals keep their original subject, still carry the
entry in P, and are flagged `core_homed=False`. No property is dropped.

The original library is never modified: output goes to a NEW file plus a CSV
audit report.

Homing per subject operator is computed once and applies to all its oracles.
Tier-E tracing synthesises generic fake inputs from each operator's schema; ops
whose shape constraints the generic inputs do not satisfy fail to trace and are
recorded as residual (not homed) rather than mis-homed.

Pass --input, --output, and --report to select the generated release artifacts.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import torch
from torch._ops import OpOverload, OpOverloadPacket
from torch.fx.experimental.proxy_tensor import make_fx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core_homing as ch  # noqa: E402

IN_PATH = Path("runs/pytorch/operator_oracle_index.json")
OUT_PATH = Path("runs/pytorch/operator_oracle_index_core.json")
REPORT_PATH = Path("runs/pytorch/core_homing_report.csv")
CORE_ONLY = False

_SHAPE_TEMPLATES = [(4, 4), (2, 3, 4), (2, 3, 4, 4), (6,), (4, 3)]


def _pick_overload(pkt: OpOverloadPacket) -> OpOverload | None:
    names = list(pkt.overloads())
    for pref in ("default", *names):
        op = getattr(pkt, pref, None)
        if isinstance(op, OpOverload) and not op._schema.name.endswith("_out"):
            return op
    return None


def _synth_arg(arg, shape):
    """Best-effort fake/real value for one schema argument; None means 'skip (use default)'."""
    t = str(arg.type)
    if arg.has_default_value():
        # let the op use its default for anything that has one, except the first tensors
        return "DEFAULT"
    if "Tensor?" in t or t.startswith("Optional"):
        return "DEFAULT"
    if t.startswith("Tensor"):
        if "[]" in t:
            return [torch.randn(shape), torch.randn(shape)]
        return torch.randn(shape)
    if "SymInt[]" in t or "int[]" in t:
        return list(shape)
    if "SymInt" in t or t.startswith("int"):
        return 1
    if t.startswith("float") or t.startswith("Scalar"):
        return 1.0
    if t.startswith("bool"):
        return False
    if "ScalarType" in t:
        return torch.float32
    return "DEFAULT"


def _try_trace_home(name: str) -> ch.HomeResult:
    """Tier-E: synthesise inputs from schema and trace the Core ATen decomposition."""
    pkt = getattr(torch.ops.aten, name, None)
    if not isinstance(pkt, OpOverloadPacket):
        return ch.HomeResult(name, name, False, "E-noinput", "no packet")
    op = _pick_overload(pkt)
    if op is None:
        return ch.HomeResult(name, name, False, "E-noinput", "no usable overload")
    args_schema = op._schema.arguments
    last_err = ""
    for shape in _SHAPE_TEMPLATES:
        tensor_inputs = []
        call_args = []
        ok = True
        for a in args_schema:
            v = _synth_arg(a, shape)
            if v == "DEFAULT":
                if a.has_default_value():
                    continue  # positional-with-default: rely on kwargs skip -> use default
                else:
                    v = None
            if isinstance(v, torch.Tensor):
                idx = len(tensor_inputs)
                tensor_inputs.append(v)
                call_args.append(("T", idx))
            elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
                idxs = []
                for tt in v:
                    idxs.append(len(tensor_inputs))
                    tensor_inputs.append(tt)
                call_args.append(("TL", idxs))
            else:
                call_args.append(("C", v))
        if not ok:
            continue

        def fn(*ts):
            built = []
            for kind, payload in call_args:
                if kind == "T":
                    built.append(ts[payload])
                elif kind == "TL":
                    built.append([ts[i] for i in payload])
                else:
                    built.append(payload)
            return op(*built)

        try:
            gm = make_fx(fn, decomposition_table=ch.core_aten_decompositions(),
                         tracing_mode="fake")(*tensor_inputs)
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {str(e)[:60]}"
            continue
        producer = ch._output_producer(gm)
        if producer is None:
            return ch.HomeResult(name, name, False, "E-multiout", "tuple/getitem output")
        out_pkt = ch.packet_name(producer)
        if ch.is_core(out_pkt):
            return ch.HomeResult(out_pkt, name, True, "E-traced", f"output={producer}")
        return ch.HomeResult(name, name, False, "E-noncore-out", f"output={producer}")
    return ch.HomeResult(name, name, False, "E-tracefail", last_err or "all shape templates failed")


def resolve_home(name: str) -> ch.HomeResult:
    r = ch.home_subject(name)           # settles A / B / (E-noinput for non-core)
    if r.tier in ("A", "B"):
        return r
    return _try_trace_home(name)        # Tier E: attempt trace


def migrate(limit: int | None, report_only: bool) -> int:
    doc = json.loads(IN_PATH.read_text())
    index = doc["index"]
    op_keys = list(index.keys())
    if limit:
        op_keys = op_keys[:limit]

    homing: dict[str, ch.HomeResult] = {}
    for k in op_keys:
        name = ch.packet_name(k)
        homing[k] = resolve_home(name)

    # --- audit report ---
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    tier_oracles: dict[str, int] = defaultdict(int)
    homed_oracles = total_oracles = 0
    for k in op_keys:
        r = homing[k]
        n = len(index[k])
        total_oracles += n
        tier_oracles[r.tier] += n
        if r.core_homed:
            homed_oracles += n
        rows.append((k, n, r.new_subject, r.entry, r.core_homed, r.tier, r.note))

    with REPORT_PATH.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["old_op_key", "oracles", "new_subject", "entry", "core_homed", "tier", "note"])
        for row in rows:
            w.writerow(row)

    print(f"input ops: {len(op_keys)}  oracles: {total_oracles}")
    fraction = 100 * homed_oracles / total_oracles if total_oracles else 0
    print(f"core-homed oracles: {homed_oracles}/{total_oracles} ({fraction:.0f}%)")
    print("by tier (oracles):")
    for t, c in sorted(tier_oracles.items(), key=lambda x: -x[1]):
        print(f"   {t:16s} {c}")
    print(f"report: {REPORT_PATH}")

    if report_only:
        return 0

    # --- build re-homed, re-indexed library ---
    new_index: dict[str, list] = defaultdict(list)
    for k in op_keys:
        r = homing[k]
        if CORE_ONLY and not r.core_homed:
            continue
        new_key = f"op:aten::{r.new_subject}"
        entry_key = f"op:aten::{r.entry}"
        for oracle in index[k]:
            o = deepcopy(oracle)
            for b in o.get("binding", []):
                if b.get("role") == "primary":
                    b["entity_id"] = new_key
            guard = o.setdefault("guard", {}).setdefault("static", {})
            guard["source_entry"] = {
                "operator": entry_key,
                "relation": {"A": "identity", "B": "in_place_functional"}.get(r.tier, "decomposition_output"),
                "homed": r.core_homed,
            }
            o["core_homed"] = r.core_homed
            o["home_tier"] = r.tier
            new_index[new_key].append(o)

    out_doc = dict(doc)
    out_doc["index"] = {k: new_index[k] for k in sorted(new_index)}
    out_doc["operator_count"] = len(new_index)
    out_doc["oracle_count"] = sum(len(v) for v in new_index.values())
    out_doc["core_homing"] = {
        "source": str(IN_PATH),
        "core_homed_oracles": homed_oracles,
        "total_oracles": total_oracles,
        "core_only": CORE_ONLY,
        "deferred_oracles": total_oracles - homed_oracles,
        "policy": "residual keeps original subject, entry in P, core_homed=False",
    }
    OUT_PATH.write_text(json.dumps(out_doc, ensure_ascii=False, indent=2))
    print(f"wrote: {OUT_PATH}  (ops {out_doc['operator_count']}, oracles {out_doc['oracle_count']})")
    return 0


def main() -> int:
    global IN_PATH, OUT_PATH, REPORT_PATH, CORE_ONLY
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=IN_PATH)
    ap.add_argument("--output", type=Path, default=OUT_PATH)
    ap.add_argument("--report", type=Path, default=REPORT_PATH)
    ap.add_argument("--core-only", action="store_true",
                    help="keep only verified Core ATen homes in the output index")
    ap.add_argument("--limit", type=int, default=None, help="only process the first N op keys")
    ap.add_argument("--report-only", action="store_true", help="print audit + CSV, do not write the new library")
    args = ap.parse_args()
    IN_PATH, OUT_PATH, REPORT_PATH, CORE_ONLY = args.input, args.output, args.report, args.core_only
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    return migrate(args.limit, args.report_only)


if __name__ == "__main__":
    raise SystemExit(main())
