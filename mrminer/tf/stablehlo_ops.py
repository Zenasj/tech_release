"""Extract StableHLO operator IDs from a TensorFlow function's lowering.

The pinned dialect inventory lives in reports/tf/stablehlo_operator_index.json.
The TensorFlow bridge emits MLIR bytecode, whose operation names are stored as
NUL-delimited strings. Exact symbol matching avoids TF-op-name heuristics.
"""
from __future__ import annotations

import json
from pathlib import Path


DEFAULT_INDEX = Path("reports/tf/stablehlo_operator_index.json")


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


def program_stablehlo_ops(fn, args, index_path: Path = DEFAULT_INDEX) -> list[str]:
    import tensorflow as tf
    from tensorflow.compiler.mlir.tensorflow_to_stablehlo.python import (
        pywrap_tensorflow_to_stablehlo as bridge,
    )

    args = tuple(args) if isinstance(args, (list, tuple)) else (args,)
    concrete = tf.function(fn).get_concrete_function(*args)
    module = tf.mlir.experimental.convert_function(concrete)
    bytecode = bridge.tensorflow_module_to_stablehlo(module)
    if not bytecode:
        raise ValueError("TF-to-StableHLO conversion returned no module")
    inventory = json.loads(index_path.read_text(encoding="utf-8"))["ops"]
    found = []
    for op_id in inventory:
        if op_id == "op:stablehlo::return":
            continue
        name = op_id.rsplit("::", 1)[-1].encode("ascii")
        if b"\x00" + name + b"\x00" in bytecode:
            found.append(op_id)
    return found
