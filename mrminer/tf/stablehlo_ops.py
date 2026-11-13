"""Enumerate StableHLO instructions from TensorFlow's textual compiler IR.

The bundled operator inventory lives in configs/stablehlo_spec_ops.json.
Each instruction retains its SSA name, ordinal, and source location reference.
"""
from __future__ import annotations

import json
import re
from pathlib import Path


DEFAULT_INDEX = Path(__file__).resolve().parents[2] / "configs/stablehlo_spec_ops.json"


def attach_source_contextual_ops(
    lowered_ops: list[str], source_ops: list[str], index: dict[str, list[dict]],
) -> list[str]:
    """Add the opaque boundary key when a deferred TF source op is present.

    This is an index attachment. It does not assert that a custom_call was
    emitted by the compiler; each attached property carries that distinction.
    """
    fallback_key = "op:stablehlo::custom_call"
    fallback_sources = {
        (item.get("guard") or {}).get("static", {}).get("source_entry", {}).get("operator")
        for item in index.get(fallback_key, [])
        if item.get("home_tier") == "STABLEHLO_SOURCE_CONTEXTUAL"
    }
    result = list(dict.fromkeys(lowered_ops))
    if set(source_ops) & fallback_sources:
        result = [fallback_key] + [op for op in result if op != fallback_key]
    return result


_OP = re.compile(r"\bstablehlo\.([a-z][a-z0-9_]*)\b")
_LOCATION = re.compile(r"\bloc\(([^)]+)\)")


def program_stablehlo_sites(fn, args, index_path: Path = DEFAULT_INDEX) -> list[dict]:
    """Enumerate individual StableHLO instructions in the marked call's IR."""
    import tensorflow as tf

    args = tuple(args) if isinstance(args, (list, tuple)) else (args,)
    compiled = tf.function(fn, jit_compile=True)
    ir = compiled.experimental_get_compiler_ir(*args)(stage="stablehlo")
    if not isinstance(ir, str) or not ir:
        raise ValueError("TensorFlow did not return textual StableHLO IR")
    inventory = json.loads(index_path.read_text(encoding="utf-8"))["ops"]
    if isinstance(inventory, list):
        inventory = {f"op:stablehlo::{name}" for name in inventory}
    else:
        inventory = set(inventory)
    return parse_stablehlo_sites(ir, inventory)


def parse_stablehlo_sites(ir: str, inventory: set[str]) -> list[dict]:
    """Preserve one record per textual instruction, including repeated op names."""
    sites = []
    counts: dict[str, int] = {}
    for line in ir.splitlines():
        stripped = line.strip()
        if not (stripped.startswith("%") or stripped.startswith("stablehlo.")
                or stripped.startswith('"stablehlo.')):
            continue
        match = _OP.search(line)
        if not match:
            continue
        op_id = f"op:stablehlo::{match.group(1)}"
        if op_id not in inventory or op_id == "op:stablehlo::return":
            continue
        ordinal = counts.get(op_id, 0)
        counts[op_id] = ordinal + 1
        ssa = stripped.split("=", 1)[0].strip() if stripped.startswith("%") else ""
        location = _LOCATION.search(line)
        sites.append({"operator": op_id, "ordinal": ordinal, "ssa": ssa,
                      "location": location.group(1) if location else "",
                      "ir_line": stripped})
    return sites


def program_stablehlo_ops(fn, args, index_path: Path = DEFAULT_INDEX) -> list[str]:
    return list(dict.fromkeys(site["operator"] for site in
                              program_stablehlo_sites(fn, args, index_path)))
