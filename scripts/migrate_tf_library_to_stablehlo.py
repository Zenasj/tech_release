#!/usr/bin/env python3
"""Re-index mined TensorFlow properties by observed StableHLO lowering ops."""
from __future__ import annotations

import argparse
import csv
import json
import inspect
import re
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

NATIVE = Path("runs/tensorflow/operator_oracle_index_tf.json")
TRANSFER = None
UNIVERSE = Path("runs/tensorflow/stablehlo_operator_index.json")
TF_PATTERNS = None
STABLEHLO_TD = None
OUT = Path("runs/tensorflow/operator_oracle_index_tf_stablehlo.json")
OUT_COMPLETE = Path("runs/tensorflow/operator_oracle_index_tf_stablehlo_complete.json")
DEFERRED = Path("runs/tensorflow/operator_oracle_index_tf_stablehlo_deferred.json")
AUDIT = Path("runs/tensorflow/stablehlo_homing_report.csv")
ATTACHMENT_AUDIT = Path("runs/tensorflow/stablehlo_attachment_report.csv")

# Preferred retrieval anchors; every candidate is checked against an actual
# representative TF->StableHLO lowering. Composite paths retain their motif.
PROBE_TARGETS = {
    "AddV2": "add", "Sub": "subtract", "Mul": "multiply",
    "RealDiv": "divide", "Mod": "remainder", "Pow": "power",
    "Maximum": "maximum", "Minimum": "minimum", "Atan2": "atan2",
    "Complex": "complex", "LeftShift": "shift_left",
    "BitwiseAnd": "and", "BitwiseOr": "or", "BitwiseXor": "xor",
    "LogicalAnd": "and", "LogicalOr": "or",
    "All": "reduce", "Any": "reduce", "Max": "reduce", "Mean": "reduce",
    "Min": "reduce", "Sum": "reduce", "MatMul": "dot_general",
    "BatchMatMulV2": "dot_general", "Conv2D": "convolution",
    "AvgPool": "reduce_window", "MaxPool": "reduce_window",
    "Cast": "convert", "Round": "round_nearest_even", "Sign": "sign",
    "Reshape": "reshape", "ExpandDims": "reshape", "Squeeze": "reshape",
    "Transpose": "transpose", "ReverseV2": "reverse",
    "ConcatV2": "concatenate", "SelectV2": "select",
    "Relu": "maximum", "FFT": "fft", "RFFT": "fft", "IRFFT": "fft",
    "GatherV2": "gather", "GatherNd": "gather", "Slice": "slice",
    "PadV2": "pad", "MatrixTriangularSolve": "triangular_solve",
    "Reciprocal": "divide",
    "TensorScatterAdd": "scatter", "TensorScatterSub": "scatter",
    "TensorScatterUpdate": "scatter", "TensorScatterMax": "scatter",
    "TensorScatterMin": "scatter", "ScatterNdNonAliasingAdd": "scatter",
    "UniformQuantize": "uniform_quantize", "UniformDequantize": "uniform_dequantize",
    "Equal": "compare", "NotEqual": "compare", "Less": "compare",
    "LessEqual": "compare", "Greater": "compare", "GreaterEqual": "compare",
    "Cumsum": "reduce_window", "Cumprod": "reduce_window",
    "CumulativeLogsumexp": "reduce_window", "ArgMin": "reduce",
    "ArgMax": "reduce", "Softmax": "divide", "LogSoftmax": "subtract",
    "Round": "select", "Range": "dynamic_iota", "StatelessIf": "if",
    "Qr": "custom_call",
}
IGNORED_ANCHORS = {"constant", "convert", "reshape", "broadcast_in_dim", "iota"}


def pattern_mapping() -> dict[str, tuple[str, str]]:
    if TF_PATTERNS is None or STABLEHLO_TD is None:
        return {}
    td = STABLEHLO_TD.read_text()
    classes = dict(re.findall(
        r'^def StableHLO_(\w+Op)\s*:\s*StableHLO_\w+<\s*"([a-z][a-z0-9_]*)"',
        td, re.MULTILINE))
    patterns = TF_PATTERNS.read_text()
    mapping = {}
    for tf_name, dialect, op_class in re.findall(
        r'\[TF_(\w+Op),\s*(MHLO|CHLO)_(\w+Op)\]', patterns):
        target = classes.get(op_class.replace("Broadcast", ""))
        if target:
            name = tf_name.removesuffix("Op")
            mapping[name] = (target, "tablegen-mhlo" if dialect == "MHLO" else "tablegen-chlo-broadcast")
    return mapping


def probe(name: str) -> tuple[set[str], str]:
    """Compile one representative raw TF op to StableHLO; inspect exact bytecode symbols."""
    import tensorflow as tf
    from tensorflow.compiler.mlir.tensorflow_to_stablehlo.python import pywrap_tensorflow_to_stablehlo as bridge

    x = tf.TensorSpec([2, 2], tf.float32)
    i = tf.TensorSpec([2, 2], tf.int32)
    raw = getattr(tf.raw_ops, name)
    try:
        if name in {"All", "Any"}:
            spec = tf.TensorSpec([2, 2], tf.bool)
            fn = lambda v: raw(input=v, axis=tf.constant([1], tf.int32), keep_dims=False)
            specs = (spec,)
        elif name in {"Cumsum", "Cumprod", "CumulativeLogsumexp"}:
            fn, specs = lambda v: raw(x=v, axis=tf.constant(1, tf.int32)), (x,)
        elif name == "Range":
            fn, specs = lambda start: raw(start=start, limit=tf.constant(6), delta=tf.constant(1)), (tf.TensorSpec([], tf.int32),)
        elif name == "Fill":
            fn, specs = lambda v: raw(dims=tf.constant([2, 2]), value=v), (tf.TensorSpec([], tf.float32),)
        elif name.startswith("FusedBatchNorm") and "Grad" not in name:
            z = tf.TensorSpec([1, 2, 2, 2], tf.float32)
            c = tf.TensorSpec([2], tf.float32)
            fn, specs = lambda a,b,d,e,f: raw(x=a, scale=b, offset=d, mean=e, variance=f, is_training=False), (z,c,c,c,c)
        elif name == "MaxPoolWithArgmax":
            z = tf.TensorSpec([1, 4, 4, 1], tf.float32)
            fn, specs = lambda v: raw(input=v, ksize=[1,2,2,1], strides=[1,1,1,1], padding="VALID"), (z,)
        elif name == "MirrorPad":
            fn, specs = lambda v: raw(input=v, paddings=tf.constant([[1,1],[1,1]]), mode="REFLECT"), (x,)
        elif name == "TopKV2":
            fn, specs = lambda v: raw(input=v, k=tf.constant(1)), (x,)
        elif name in {"ResizeBicubic", "ResizeBilinear", "ResizeNearestNeighbor"}:
            z = tf.TensorSpec([1, 4, 4, 1], tf.float32)
            fn, specs = lambda v: raw(images=v, size=tf.constant([2, 2])), (z,)
        elif name in {"Softmax", "LogSoftmax"}:
            fn, specs = lambda v: raw(logits=v), (x,)
        elif name == "SplitV":
            fn, specs = lambda v: raw(value=v, size_splits=tf.constant([1,1]), axis=tf.constant(0), num_split=2), (x,)
        elif name in {"ArgMin", "ArgMax"}:
            fn, specs = lambda v: raw(input=v, dimension=tf.constant(1)), (x,)
        elif name == "InTopK":
            fn, specs = lambda v: raw(predictions=v, targets=tf.constant([0,1]), k=1), (x,)
        elif name == "Qr":
            fn, specs = lambda v: raw(input=v), (x,)
        elif name == "QuantizeV2":
            fn, specs = lambda v: raw(input=v, min_range=tf.constant(-1.0), max_range=tf.constant(1.0), T=tf.qint8), (x,)
        elif name == "Tile":
            fn, specs = lambda v: raw(input=v, multiples=tf.constant([2, 2])), (x,)
        elif name == "Unpack":
            fn, specs = lambda v: raw(value=v, num=2), (x,)
        elif name == "MatrixBandPart":
            fn, specs = lambda v: raw(input=v, num_lower=tf.constant(0), num_upper=tf.constant(1)), (x,)
        elif name in {"QuantizeAndDequantizeV3", "QuantizeAndDequantizeV4"}:
            fn, specs = lambda v: raw(input=v, input_min=tf.constant(-1.0), input_max=tf.constant(1.0), num_bits=8), (x,)
        elif name == "AvgPool3D":
            z = tf.TensorSpec([1, 3, 3, 3, 1], tf.float32)
            fn, specs = lambda v: raw(input=v, ksize=[1,2,2,2,1], strides=[1,1,1,1,1], padding="VALID"), (z,)
        elif name == "Concat":
            fn, specs = lambda a,b: raw(concat_dim=tf.constant(1), values=[a,b]), (x,x)
        elif name in {"Angle", "Conj"}:
            fn, specs = lambda v: raw(input=v), (tf.TensorSpec([2,2], tf.complex64),)
        elif name == "Gather":
            fn, specs = lambda v: raw(params=v, indices=tf.constant([0,1])), (x,)
        elif name == "LinSpace":
            fn, specs = lambda: raw(start=tf.constant(0.0), stop=tf.constant(1.0), num=tf.constant(4)), ()
        elif name == "LowerBound":
            fn, specs = lambda a,b: raw(sorted_inputs=a, values=b), (x,x)
        elif name == "OneHot":
            fn, specs = lambda: raw(indices=tf.constant([0,1]), depth=tf.constant(3), on_value=tf.constant(1.0), off_value=tf.constant(0.0)), ()
        elif name == "Pack":
            fn, specs = lambda a,b: raw(values=[a,b]), (x,x)
        elif name == "RightShift":
            fn, specs = lambda a,b: raw(x=a,y=b), (i,i)
        elif name == "Roll":
            fn, specs = lambda v: raw(input=v, shift=tf.constant(1), axis=tf.constant(0)), (x,)
        elif name == "SegmentMean":
            fn, specs = lambda v: raw(data=v, segment_ids=tf.constant([0,1])), (x,)
        elif name == "Split":
            fn, specs = lambda v: raw(axis=tf.constant(0), value=v, num_split=2), (x,)
        elif name == "TanhGrad":
            fn, specs = lambda a,b: raw(y=a,dy=b), (x,x)
        elif name == "Bincount":
            fn, specs = lambda v: raw(arr=v, size=tf.constant(4), weights=tf.constant([], tf.int32)), (tf.TensorSpec([4], tf.int32),)
        elif name == "Dequantize":
            fn, specs = lambda v: raw(input=v, min_range=tf.constant(-1.0), max_range=tf.constant(1.0)), (tf.TensorSpec([2,2], tf.qint8),)
        elif name == "Eig":
            fn, specs = lambda v: raw(input=v, Tout=tf.complex64), (x,)
        elif name == "FakeQuantWithMinMaxArgs":
            fn, specs = lambda v: raw(inputs=v), (x,)
        elif name.startswith("FusedBatchNormGrad"):
            z = tf.TensorSpec([1, 2, 2, 2], tf.float32)
            c = tf.TensorSpec([2], tf.float32)
            if name.endswith("V3"):
                fn, specs = lambda a,b,d,e,f,g: raw(y_backprop=a,x=b,scale=d,reserve_space_1=e,reserve_space_2=f,reserve_space_3=g), (z,z,c,c,c,c)
            else:
                fn, specs = lambda a,b,d,e,f: raw(y_backprop=a,x=b,scale=d,reserve_space_1=e,reserve_space_2=f), (z,z,c,c,c)
        elif name == "FusedPadConv2D":
            z = tf.TensorSpec([1, 4, 4, 1], tf.float32)
            k = tf.TensorSpec([2, 2, 1, 1], tf.float32)
            fn, specs = lambda a,b: raw(input=a,paddings=tf.constant([[0,0],[1,1],[1,1],[0,0]]),filter=b,mode="REFLECT",strides=[1,1,1,1],padding="VALID"), (z,k)
        elif name == "ImageProjectiveTransformV2":
            z = tf.TensorSpec([1, 4, 4, 1], tf.float32)
            fn, specs = lambda v: raw(images=v,transforms=tf.constant([[1.,0.,0.,0.,1.,0.,0.,0.]]),output_shape=tf.constant([4,4]),interpolation="NEAREST"), (z,)
        elif name == "MatrixSolveLs":
            fn, specs = lambda a,b: raw(matrix=a,rhs=b,l2_regularizer=tf.constant(0.0, tf.float64)), (x,x)
        elif name == "NonMaxSuppressionV3":
            boxes = tf.TensorSpec([3,4], tf.float32)
            scores = tf.TensorSpec([3], tf.float32)
            fn, specs = lambda a,b: raw(boxes=a,scores=b,max_output_size=tf.constant(2),iou_threshold=tf.constant(0.5),score_threshold=tf.constant(0.0)), (boxes,scores)
        elif name == "OptionalFromValue":
            fn, specs = lambda v: raw(components=[v]), (x,)
        elif name == "QuantizeAndDequantizeV2":
            fn, specs = lambda v: raw(input=v,input_min=tf.constant(-1.0),input_max=tf.constant(1.0)), (x,)
        elif name == "QuantizeAndDequantizeV4Grad":
            fn, specs = lambda a,b: raw(gradients=a,input=b,input_min=tf.constant(-1.0),input_max=tf.constant(1.0)), (x,x)
        elif name == "ResizeNearestNeighborGrad":
            z = tf.TensorSpec([1, 4, 4, 1], tf.float32)
            fn, specs = lambda v: raw(grads=v,size=tf.constant([2,2])), (z,)
        elif name == "ScaleAndTranslate":
            z = tf.TensorSpec([1, 4, 4, 1], tf.float32)
            fn, specs = lambda v: raw(images=v,size=tf.constant([2,2]),scale=tf.constant([1.,1.]),translation=tf.constant([0.,0.])), (z,)
        elif name == "SegmentProdV2":
            fn, specs = lambda v: raw(data=v,segment_ids=tf.constant([0,0]),num_segments=tf.constant(2)), (x,)
        elif name == "Where":
            fn, specs = lambda v: raw(condition=v), (tf.TensorSpec([2,2], tf.bool),)
        elif name == "CTCLoss":
            z = tf.TensorSpec([3, 1, 2], tf.float32)
            fn, specs = lambda v: raw(inputs=v, labels_indices=tf.constant([[0, 0]], tf.int64),
                                       labels_values=tf.constant([1], tf.int32),
                                       sequence_length=tf.constant([3], tf.int32)), (z,)
        elif name == "StatelessIf":
            fn, specs = (lambda v: tf.cond(tf.reduce_sum(v) > 0,
                                           lambda: v + tf.constant(1.0),
                                           lambda: v - tf.constant(1.0))), (x,)
        elif name in {"UniformQuantize", "UniformDequantize"}:
            spec = tf.TensorSpec([2,2],tf.float32 if name == "UniformQuantize" else tf.qint8)
            dtype = tf.qint8 if name == "UniformQuantize" else tf.float32
            fn, specs = lambda v: raw(input=v,scales=tf.constant(0.1),zero_points=tf.constant(0),Tout=dtype,quantization_min_val=-128,quantization_max_val=127), (spec,)
        elif name in {"Max", "Mean", "Min", "Sum"}:
            fn = lambda v: raw(input=v, axis=tf.constant([1], tf.int32), keep_dims=False)
            specs = (x,)
        elif name in {"MatMul", "BatchMatMulV2"}:
            fn = (lambda a, b: raw(a=a, b=b)) if name == "MatMul" else (lambda a, b: raw(x=a, y=b))
            specs = (x, x)
        elif name == "Conv2D":
            z = tf.TensorSpec([1, 4, 4, 1], tf.float32)
            k = tf.TensorSpec([2, 2, 1, 1], tf.float32)
            fn = lambda a, b: raw(input=a, filter=b, strides=[1, 1, 1, 1], padding="VALID")
            specs = (z, k)
        elif name in {"AvgPool", "MaxPool"}:
            z = tf.TensorSpec([1, 4, 4, 1], tf.float32)
            fn = (lambda a: raw(value=a, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="VALID")) if name == "AvgPool" else (lambda a: raw(input=a, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="VALID"))
            specs = (z,)
        elif name == "Cast":
            fn, specs = lambda v: raw(x=v, DstT=tf.int32), (x,)
        elif name == "Reshape":
            fn, specs = lambda v: raw(tensor=v, shape=tf.constant([4], tf.int32)), (x,)
        elif name == "ExpandDims":
            fn, specs = lambda v: raw(input=v, axis=tf.constant(0, tf.int32)), (x,)
        elif name == "Squeeze":
            fn, specs = lambda v: raw(input=v, axis=[0]), (tf.TensorSpec([1, 2], tf.float32),)
        elif name == "Transpose":
            fn, specs = lambda v: raw(x=v, perm=tf.constant([1, 0], tf.int32)), (x,)
        elif name == "ReverseV2":
            fn, specs = lambda v: raw(tensor=v, axis=tf.constant([1], tf.int32)), (x,)
        elif name == "ConcatV2":
            fn, specs = lambda a, b: raw(values=[a, b], axis=tf.constant(1, tf.int32)), (x, x)
        elif name == "SelectV2":
            fn, specs = lambda c, a, b: raw(condition=c, t=a, e=b), (tf.TensorSpec([2, 2], tf.bool), x, x)
        elif name == "Relu":
            fn, specs = lambda v: raw(features=v), (x,)
        elif name in {"FFT", "RFFT", "IRFFT"}:
            spec = tf.TensorSpec([2, 4], tf.complex64 if name != "RFFT" else tf.float32)
            if name == "FFT":
                fn = lambda v: raw(input=v)
            else:
                fn = lambda v: raw(input=v, fft_length=tf.constant([4], tf.int32))
            specs = (spec,)
        elif name == "GatherV2":
            fn, specs = lambda v, ix: raw(params=v, indices=ix, axis=tf.constant(0, tf.int32)), (x, i)
        elif name == "GatherNd":
            fn, specs = lambda v, ix: raw(params=v, indices=ix), (x, tf.TensorSpec([1, 1], tf.int32))
        elif name == "Slice":
            fn, specs = lambda v: raw(input=v, begin=tf.constant([0, 0], tf.int32), size=tf.constant([1, 2], tf.int32)), (x,)
        elif name == "PadV2":
            fn, specs = lambda v: raw(input=v, paddings=tf.constant([[1, 1], [0, 0]], tf.int32), constant_values=tf.constant(0.0)), (x,)
        elif name == "MatrixTriangularSolve":
            fn, specs = lambda a, b: raw(matrix=a, rhs=b), (x, x)
        elif name.startswith("TensorScatter"):
            ix = tf.constant([[0], [1]], tf.int32)
            fn, specs = lambda v, u: raw(tensor=v, indices=ix, updates=u), (x, x)
        elif name.startswith("ResourceScatterNd"):
            variable = tf.Variable(tf.zeros([2, 2], tf.float32))
            fn, specs = (lambda u: (raw(ref=variable.handle, indices=tf.constant([[0], [1]]), updates=u),
                                     variable.read_value())[1]), (x,)
        elif name == "ResourceScatterUpdate":
            variable = tf.Variable(tf.zeros([2, 2], tf.float32))
            fn, specs = (lambda u: (raw(resource=variable.handle, indices=tf.constant([0, 1]), updates=u),
                                     variable.read_value())[1]), (x,)
        elif name == "ScatterNdNonAliasingAdd":
            ix = tf.constant([[0], [1]], tf.int32)
            fn, specs = lambda v, u: raw(input=v, indices=ix, updates=u), (x, x)
        elif name in {"Equal", "NotEqual", "Less", "LessEqual", "Greater", "GreaterEqual"}:
            fn, specs = lambda a, b: raw(x=a, y=b), (x, x)
        elif name == "Complex":
            fn, specs = lambda a, b: raw(real=a, imag=b), (x, x)
        elif name in {"AddV2", "Sub", "Mul", "RealDiv", "Mod", "Pow", "Maximum", "Minimum", "Atan2", "Complex", "LeftShift", "BitwiseAnd", "BitwiseOr", "BitwiseXor", "LogicalAnd", "LogicalOr"}:
            spec = (tf.TensorSpec([2, 2], tf.bool) if name.startswith("Logical") else
                    i if name in {"LeftShift", "BitwiseAnd", "BitwiseOr", "BitwiseXor"} else x)
            fn, specs = lambda a, b: raw(x=a, y=b), (spec, spec)
        else:
            signature = inspect.signature(raw)
            required = [p.name for p in signature.parameters.values()
                        if p.default is inspect.Parameter.empty and p.name != "name"]
            if required == ["x", "y"]:
                fn, specs = lambda a,b: raw(x=a,y=b), (x,x)
            elif len(required) == 1 and required[0] in {"x", "input", "features", "tensor", "value", "logits"}:
                arg = required[0]
                spec = (tf.TensorSpec([2, 2], tf.bool) if name == "LogicalNot" else
                        i if name == "Invert" else x)
                fn, specs = lambda v: raw(**{arg:v}), (spec,)
            else:
                return set(), "probe-recipe-unavailable"
        concrete = tf.function(fn).get_concrete_function(*specs)
        module = tf.mlir.experimental.convert_function(concrete)
        bytecode = bridge.tensorflow_module_to_stablehlo(module)
        if not bytecode:
            return set(), "lowering-returned-empty"
        # The bridge can leave an unsupported TF operation in the module while
        # emitting StableHLO constants for its attributes. Its original TF op
        # name then remains in the bytecode symbol table. Such constants do
        # not establish a StableHLO implementation of the source operation.
        universe = json.loads(UNIVERSE.read_text())["ops"]
        found = {op.split("::")[-1] for op in universe
                 if b"\x00" + op.split("::")[-1].encode() + b"\x00" in bytecode}
        if b"\x00" + name.encode() + b"\x00" in bytecode:
            # Qr is also the custom_call target name. Its appearance in the
            # symbol table is therefore not evidence of an unresolved TF op;
            # the independent XLA HLO probe confirms custom_call_target="Qr".
            if name == "Qr" and "custom_call" in found:
                return found - {"return"}, "probe-ok-qr-custom-call-target"
            return set(), "source-tf-op-remains-after-conversion"
        return found - {"return"}, "probe-ok"
    except Exception as exc:  # best-effort probe; preserve exact reason for audit
        lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
        return set(), f"{type(exc).__name__}: {(lines[-1] if lines else str(exc))[:140]}"


def main() -> None:
    global NATIVE, TRANSFER, UNIVERSE, TF_PATTERNS, STABLEHLO_TD
    global OUT, OUT_COMPLETE, DEFERRED, AUDIT, ATTACHMENT_AUDIT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-probes", action="store_true", help="audit without dynamic homing")
    parser.add_argument("--native-index", type=Path, default=NATIVE)
    parser.add_argument("--transfer-index", type=Path)
    parser.add_argument("--universe", type=Path, default=UNIVERSE)
    parser.add_argument("--tf-patterns", type=Path)
    parser.add_argument("--stablehlo-td", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/tensorflow"))
    args = parser.parse_args()
    NATIVE, TRANSFER, UNIVERSE = args.native_index, args.transfer_index, args.universe
    TF_PATTERNS, STABLEHLO_TD = args.tf_patterns, args.stablehlo_td
    OUT = args.out_dir / OUT.name
    OUT_COMPLETE = args.out_dir / OUT_COMPLETE.name
    DEFERRED = args.out_dir / DEFERRED.name
    AUDIT = args.out_dir / AUDIT.name
    ATTACHMENT_AUDIT = args.out_dir / ATTACHMENT_AUDIT.name
    import tensorflow as tf
    universe = set(json.loads(UNIVERSE.read_text())["ops"])
    source_map = pattern_mapping()
    merged = defaultdict(list)
    sources = [("native", NATIVE)]
    if TRANSFER is not None:
        sources.append(("transfer", TRANSFER))
    for label, path in sources:
        for key, items in json.loads(path.read_text())["index"].items():
            for item in items:
                merged[key].append((label, item))

    homed, deferred = defaultdict(list), defaultdict(list)
    audit = []
    counts = Counter()
    for key in sorted(merged):
        name = key.split("::")[-1]
        pattern_target, pattern_evidence = source_map.get(name, (None, ""))
        observed, probe_note = (set(), "not-run") if args.no_probes else probe(name)
        # A stateful scatter can leave only its constant indices in the bridge
        # output. That constant is not an implementation of the scatter.
        if name.startswith("ResourceScatter") and observed == {"constant"}:
            observed, probe_note = set(), "stateful-op-not-lowered; constant indices only"
        eligible = sorted(op for op in observed if f"op:stablehlo::{op}" in universe)
        preferred = [PROBE_TARGETS.get(name), pattern_target]
        target = next((op for op in preferred if op in eligible), None)
        if target is None:
            distinctive = [op for op in eligible if op not in IGNORED_ANCHORS]
            target = (distinctive or eligible or [None])[0]
        evidence = ("tf-to-stablehlo-probe" if target else
                    "no-confirmed-spec-operation")
        if target and len(eligible) > 1:
            evidence = "tf-to-stablehlo-composite-probe"
        new_key = f"op:stablehlo::{target}" if target else key
        audit.append({"tf_operator": key, "stablehlo_operator": new_key if target else "",
                      "properties": len(merged[key]), "status": "HOMED" if target else "DEFER",
                      "evidence": evidence,
                      "probe_ops": ",".join(sorted(observed)), "probe_note": probe_note})
        for label, source in merged[key]:
            item = deepcopy(source)
            guard = item.setdefault("guard", {}).setdefault("static", {})
            guard["source_entry"] = {"operator": key, "origin": label,
                                     "lowering_evidence": evidence}
            if target:
                guard["lowering_motif"] = [f"op:stablehlo::{op}" for op in eligible]
            item["_lib_source"] = label
            item["core_homed"] = bool(target)
            item["home_tier"] = "STABLEHLO" if target else "DEFER"
            if target:
                bindings = item.setdefault("binding", [])
                for binding in bindings:
                    if binding.get("role") == "primary":
                        binding["entity_id"] = new_key
                if not any(binding.get("role") == "primary" for binding in bindings):
                    bindings.append({"role": "primary", "entity_id": new_key})
                homed[new_key].append(item)
                counts[f"homed_{label}"] += 1
            else:
                item["deferred_class"] = evidence
                deferred[key].append(item)
                counts[f"deferred_{label}"] += 1

    expected = sum(len(items) for items in merged.values())
    if sum(counts.values()) != expected:
        raise ValueError(f"property conservation failed: {sum(counts.values())} != {expected}")
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit[0].keys())
        writer.writeheader()
        writer.writerows(audit)
    metadata = {"source_libraries": [str(path) for _, path in sources],
                "stablehlo_universe": str(UNIVERSE), "total_properties": expected,
                "tensorflow_probe_runtime": tf.__version__,
                "tensorflow_pattern_source_version": None,
                "counts": dict(counts), "mapping_policy":
                "one probe-confirmed StableHLO retrieval anchor per source operator; full observed lowering motif and TF source retained in applicability"}
    OUT.write_text(json.dumps({"index": dict(sorted(homed.items())),
                               "operator_count": len(homed),
                               "oracle_count": sum(map(len, homed.values())),
                               "stablehlo_homing": metadata}, ensure_ascii=False, indent=2) + "\n")
    DEFERRED.write_text(json.dumps({"index": dict(sorted(deferred.items())),
                                    "operator_count": len(deferred),
                                    "oracle_count": sum(map(len, deferred.values())),
                                    "stablehlo_homing": metadata}, ensure_ascii=False, indent=2) + "\n")
    # Source-contextual attachment: StableHLO custom_call is the explicit
    # opaque boundary, not an assertion that the present TF bridge emitted a
    # custom_call. These properties stay marked as unlowered in their metadata.
    complete = defaultdict(list, {key: list(items) for key, items in homed.items()})
    fallback_key = "op:stablehlo::custom_call"
    if fallback_key not in universe:
        raise ValueError("StableHLO spec inventory lacks custom_call")
    for source_key, items in deferred.items():
        for source in items:
            item = deepcopy(source)
            guard = item.setdefault("guard", {}).setdefault("static", {})
            guard["opaque_boundary"] = {
                "operator": fallback_key,
                "call_target_name": f"mrminer.tf::{source_key.rsplit('::', 1)[-1]}",
                "materialized_in_compiler": False,
            }
            item["home_tier"] = "STABLEHLO_SOURCE_CONTEXTUAL"
            item["index_attached"] = True
            item["core_homed"] = False
            bindings = item.setdefault("binding", [])
            for binding in bindings:
                if binding.get("role") == "primary":
                    binding["entity_id"] = fallback_key
            if not any(binding.get("role") == "primary" for binding in bindings):
                bindings.append({"role": "primary", "entity_id": fallback_key})
            complete[fallback_key].append(item)
    total = sum(map(len, complete.values()))
    if total != expected or len({p["oracle_id"] for entries in complete.values() for p in entries}) != expected:
        raise ValueError("complete StableHLO index did not conserve unique properties")
    complete_meta = dict(metadata)
    complete_meta["attachment_counts"] = {
        "observed_lowering": sum(map(len, homed.values())),
        "source_contextual_opaque": sum(map(len, deferred.values())),
    }
    complete_meta["fallback_policy"] = (
        "custom_call is a source-contextual index boundary only; no claim that "
        "the current compiler emitted this operation or accepts the source types")
    OUT_COMPLETE.write_text(json.dumps({
        "index": dict(sorted(complete.items())), "operator_count": len(complete),
        "oracle_count": total, "stablehlo_homing": complete_meta,
    }, ensure_ascii=False, indent=2) + "\n")
    with ATTACHMENT_AUDIT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "tf_operator", "properties", "attachment_operator", "attachment_tier",
            "observed_lowering", "probe_note"])
        writer.writeheader()
        for row in audit:
            lowered = row["status"] == "HOMED"
            writer.writerow({
                "tf_operator": row["tf_operator"], "properties": row["properties"],
                "attachment_operator": row["stablehlo_operator"] if lowered else fallback_key,
                "attachment_tier": "OBSERVED_LOWERING" if lowered else "SOURCE_CONTEXTUAL_OPAQUE",
                "observed_lowering": lowered, "probe_note": row["probe_note"],
            })
    print(json.dumps({"stablehlo_ops": len(homed), "homed": sum(map(len, homed.values())),
                      "deferred": sum(map(len, deferred.values())), "complete": total,
                      "complete_index": str(OUT_COMPLETE), "counts": dict(counts)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
