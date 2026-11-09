"""The TF op universe -- analog of aten's native_functions.yaml registry.

Every registered TF op is exposed on `tf.raw_ops`; that closed, enumerable set is
our operator universe (oracles are keyed to `op:tf::<Name>`, mirroring
`op:aten::<name>` on the PyTorch side). Optionally enrich each op with its OpDef
(inputs/outputs/attrs) from the C++ op registry.

    PYTHONPATH=. python -m mrminer.tf.op_registry --out reports/tf/op_universe.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def op_id(name: str) -> str:
    return f"op:tf::{name}"


def tf_op_universe() -> list[str]:
    """All registered TF op names (the closed operator set)."""
    import tensorflow as tf

    return sorted(n for n in dir(tf.raw_ops) if not n.startswith("_"))


def op_def(name: str):
    """OpDef (schema) for one op, or None. API differs across TF versions, so try
    the stable paths and fall back gracefully."""
    try:
        from tensorflow.python.framework import op_def_registry

        return op_def_registry.get(name)
    except Exception:
        return None


def dump(out: Path) -> dict:
    from tensorflow.core.framework import types_pb2

    ops = tf_op_universe()
    excluded = {types_pb2.DT_RESOURCE, types_pb2.DT_VARIANT, types_pb2.DT_STRING}
    index = {}
    for name in ops:
        d = op_def(name)
        entry = {"op": op_id(name)}
        if d is not None:
            try:
                entry["n_inputs"] = len(d.input_arg)
                entry["n_outputs"] = len(d.output_arg)
                entry["attrs"] = [a.name for a in d.attr]
                entry["functional"] = bool(
                    not d.is_stateful and d.output_arg
                    and (d.input_arg or name == "Const")
                    and all(not arg.is_ref and arg.type not in excluded
                            for arg in (*d.input_arg, *d.output_arg))
                    and all(attr.type not in {"func", "list(func)"} for attr in d.attr)
                )
            except Exception:
                pass
        index[name] = entry
    out.parent.mkdir(parents=True, exist_ok=True)
    eligible = sum(bool(item.get("functional")) for item in index.values())
    out.write_text(json.dumps({"op_count": len(ops), "functional_count": eligible,
                               "ops": index}, indent=2))
    return {"op_count": len(ops), "functional_count": eligible, "out": str(out)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/tensorflow/op_universe.json")
    ns = ap.parse_args()
    res = dump(Path(ns.out))
    print(f"TF op universe: {res['op_count']} ops -> {res['out']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
