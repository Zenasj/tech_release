"""TF/XLA execution runner: eager vs tf.function(jit_compile=True) differential.

Mirrors mrminer/fuzz/runner.py (same Finding contract, same subprocess isolation,
same self-consistency guard) but for TensorFlow: the "compiled" side is XLA via
`tf.function(fn, jit_compile=True)`. A generated test's `build()` returns
`(fn, args)` where fn is a callable over tf tensors and args a tuple of tf tensors.

M-TF scope: semantic (eager vs XLA value) + effect (eager raises, XLA accepts);
decision/performance stubbed. XLA frequently cannot compile an op -> that raises,
which we record as compiled_crash (triage separates "legitimately not XLA-
compilable" from real bugs).

Worker:  python -m mrminer.tf.runner_tf --job <job.json>
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from mrminer.fuzz.types import Finding, GeneratedTest, OracleRef
from mrminer.tf.op_registry import op_id

REL = 1e-3
_MARK = "__FUZZ_RESULT__"
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# XLA legitimately does not compile every TF op (documented support boundary): it
# raises a clean "unsupported operation" rather than miscompiling. That is NOT a bug
# (mirror of PyTorch's make_fallback), so treat it as skip, not compiled_crash.
_XLA_UNSUPPORTED = (
    "Detected unsupported operations",
    "unsupported op",
    "not compilable",
    "could not be legalized",
    "requires compilation but",
    "Support for TensorList",
    "TensorList is not initialized",
    "only supported in eager mode",   # test used an eager-only API (e.g. tf.data as_numpy_iterator)
    "NotImplementedError",            # the framework rejects this op by design, not a compiler bug
    "not currently supported",
    "not yet supported",
    "is not supported",
    "Invalid type for triangular solve",   # XLA lacks fp16/low-precision matrix solve/inverse
)


def _is_xla_unsupported(msg: str) -> bool:
    m = msg.lower()
    return any(s.lower() in m for s in _XLA_UNSUPPORTED)


def _is_oom(msg: str) -> bool:
    m = msg.lower()
    # transient resource/environment failures (GPU memory, cuDNN/DNN init) -> not a bug
    return ("resourceexhausted" in m or "out of memory" in m or "oom" in m
            or "dnn library" in m or "failedprecondition" in m and "dnn" in m)


# The generated test itself is not graph-traceable (an eager-only Python construct),
# so it fails at tf.function TRACING, before XLA -- a test-quality issue, not a bug.
_NOT_TRACEABLE = (
    "OperatorNotAllowedInGraphError",
    "Iterating over a symbolic",
    "only supports singleton tf.Variables",
    "tf.Variable is only created once",
    "not allowed in Graph execution",
    "Cannot convert",
    "using a `tf.Tensor` as a Python `bool`",
    "must return zero or more Tensors",   # fn returned a non-Tensor (resource/op/None-mix)
    "To be compatible with tf.function",
)


def _is_not_traceable(msg: str) -> bool:
    return any(s.lower() in msg.lower() for s in _NOT_TRACEABLE)


# Python-level test-construction errors (the generated code is malformed), NOT compiler
# bugs. Kept deliberately narrow so a real compiler crash is never swallowed:
#  - "module '<pkg>' has no attribute" = the test called a non-existent API (e.g. tf.erfinv
#     instead of tf.math.erfinv). This does NOT match "'<obj>' object has no attribute ..."
#     (which is how genuine internal compiler crashes surface, e.g. the Inductor
#     ExternKernelSchedulerNode.group crash), so real bugs are unaffected.
#  - "unexpected keyword argument" = the op was called with a wrong kwarg name; a pure
#     Python signature TypeError raised at the call site, never a miscompile.
_BAD_TEST = (
    "module 'tensorflow' has no attribute",
    "module 'tf.",            # tf.math / tf.linalg / ... has no attribute <x>
    "unexpected keyword argument",
    # All three below are raised at graph/function *construction* (identically on the
    # eager-traced and compiled paths, before the op under test ever compiles), so
    # classifying them as a bad test cannot mask a compiler miscompile:
    "the same nested structure",   # malformed tf.while_loop / tf.cond (loop_vars / branch structure mismatch)
    "has no attribute",            # calling a removed/nonexistent API, e.g. tf.image.resize_bilinear
    "that does not match type",    # mixed-dtype args violating an op's type constraint (eager rejects too)
    "cannot access local variable",  # Python UnboundLocalError in the generated harness closure
    "is not compatible with",      # TypeSpec/shape-invariant incompatibility at while_loop/tf.function construction
)


def _is_bad_test(msg: str) -> bool:
    return any(s.lower() in msg.lower() for s in _BAD_TEST)


def _np(x):
    """tf tensor / nested -> numpy (or leave scalars)."""
    try:
        import numpy as np
    except Exception:
        np = None
    if hasattr(x, "numpy"):
        return x.numpy()
    if isinstance(x, (list, tuple)):
        return [_np(v) for v in x]
    return x


def _err(a, b) -> float:
    import numpy as np

    if isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            return float("inf")
        return max((_err(x, y) for x, y in zip(a, b)), default=0.0)
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return float("inf")
    if a.dtype.kind not in "fc":
        return 0.0 if np.array_equal(a, b) else float("inf")
    return float(np.max(np.abs(a.astype("float64") - b.astype("float64")))) if a.size else 0.0


def _scale(a) -> float:
    import numpy as np

    if isinstance(a, list):
        return max((_scale(x) for x in a), default=1.0)
    a = np.asarray(a)
    return max(float(np.max(np.abs(a.astype("float64")))), 1.0) if (a.size and a.dtype.kind in "fc") else 1.0


# dtype-aware relative tolerance (mirrors the PyTorch runner): a low-precision (bf16/fp16)
# XLA kernel rounds differently from eager, so a fixed fp32-grade REL flags pure rounding
# as "diverged" (the Softplus fp16 FP). Loosen only for low precision; keep REL otherwise.
# Not looser than the rounding band, so a real low-precision miscompile still diverges grossly.
_REL_BY_DTYPE = {"bfloat16": 1.5e-2, "float16": 4e-3}


def _rel_tol(a) -> float:
    """Loosest per-dtype tolerance over all arrays in the reference output."""
    import numpy as np

    if isinstance(a, list):
        return max((_rel_tol(x) for x in a), default=REL)
    dt = getattr(getattr(np.asarray(a), "dtype", None), "name", "")
    return _REL_BY_DTYPE.get(dt, REL)


def _r(triggered: bool, verdict: str, detail: str = "") -> dict:
    return {"triggered": triggered, "verdict": verdict, "detail": detail}


def program_ops(fn, args, dialect: str = "tf") -> list[str]:
    """Operator IDs from the TF graph or its StableHLO lowering."""
    import tensorflow as tf

    try:
        if dialect == "stablehlo":
            from mrminer.tf.stablehlo_ops import program_stablehlo_ops
            return program_stablehlo_ops(fn, args)
        if dialect != "tf":
            raise ValueError(f"unknown operator dialect: {dialect}")
        cf = tf.function(fn).get_concrete_function(*args)
        skip = {"Placeholder", "Const", "Identity", "NoOp", "ReadVariableOp"}
        return sorted({op_id(n.op) for n in cf.graph.as_graph_def().node if n.op not in skip})
    except Exception:
        return []


def _shape_dtype(a) -> object:
    """A shape/dtype signature for value-agnostic comparison (nondeterministic ops)."""
    import numpy as np

    if isinstance(a, list):
        return [_shape_dtype(x) for x in a]
    a = np.asarray(a)
    return (tuple(a.shape), str(a.dtype))


def _judge(code: str, aspect: str, category: str = "supported") -> dict:
    import tensorflow as tf

    # Compare eager vs XLA under the SAME precision policy. On Ampere+ GPUs eager
    # conv/matmul default to TF32 (~10-bit mantissa) while XLA uses full fp32, which
    # diverges at ~1e-3 on normal-magnitude values and floods false positives (the
    # Conv2D 1x1x128 rel_err=0.0039 was purely this). Disabling TF32 makes the compare
    # strictly full-fp32 -- it never loosens tolerance, so a real conv miscompile still
    # diverges. (Mirrors TF's own low-tolerance numerical tests.)
    try:
        tf.config.experimental.enable_tensor_float_32_execution(False)
    except Exception:
        pass

    # Generated snippets may use the harness-provided TensorFlow alias without
    # repeating an import. Explicit imports in the snippet can still override it.
    ns: dict = {"tf": tf}
    try:
        exec(code, ns)  # noqa: S102 -- isolated worker process
        if "build" not in ns:
            return _r(False, "skip:bad_test", "no build()")
        fn, args = ns["build"]()
    except Exception as e:
        return _r(False, "skip:bad_test", f"{type(e).__name__}: {e}")

    if aspect == "semantic":
        try:
            ref = _np(fn(*args))
        except Exception:
            return _r(False, "skip:eager_error", "eager errored -> no reference")
        if ref is None:
            return _r(False, "skip:bad_test", "fn returned None (no tensor to compare)")
        # nondeterministic ops: compare shape/dtype only (values legitimately vary).
        if category == "nondeterministic":
            try:
                out = _np(tf.function(fn, jit_compile=True)(*args))
            except Exception as e:
                full = f"{type(e).__name__}: {e}"
                if _is_oom(full):
                    return _r(False, "skip:oom", full.splitlines()[0][:160])
                if _is_xla_unsupported(full):
                    return _r(False, "skip:xla_unsupported", full.splitlines()[0][:160])
                if _is_not_traceable(full):
                    return _r(False, "skip:not_traceable", full.splitlines()[0][:160])
                if _is_bad_test(full):
                    return _r(False, "skip:bad_test", full.splitlines()[0][:160])
                return _r(True, "compiled_crash", full.splitlines()[0][:160])
            same = _shape_dtype(out) == _shape_dtype(ref)
            return _r(not same, "pass" if same else "diverged",
                      f"shape/dtype {'match' if same else 'MISMATCH'} (rng: values not compared)")
        try:
            ref2 = _np(fn(*args))
            if _err(ref2, ref) / _scale(ref) > _rel_tol(ref):
                return _r(False, "skip:nondeterministic", "eager differs across two runs")
        except Exception:
            return _r(False, "skip:nondeterministic", "second eager run errored")
        try:
            out = _np(tf.function(fn, jit_compile=True)(*args))
        except Exception as e:
            full = f"{type(e).__name__}: {e}"
            if _is_oom(full):
                return _r(False, "skip:oom", full.splitlines()[0][:160])
            if _is_xla_unsupported(full):
                return _r(False, "skip:xla_unsupported", full.splitlines()[0][:160])
            if _is_not_traceable(full):
                return _r(False, "skip:not_traceable", full.splitlines()[0][:160])
            if _is_bad_test(full):
                return _r(False, "skip:bad_test", full.splitlines()[0][:160])
            return _r(True, "compiled_crash", full.splitlines()[0][:160])
        err = _err(out, ref) / _scale(ref)
        tol = _rel_tol(ref)
        return _r(err > tol, "diverged" if err > tol else "pass", f"rel_err={err:.4f} tol={tol:g}")

    if aspect == "effect":
        try:
            fn(*args)
            eager_raised = False
        except Exception:
            eager_raised = True
        if not eager_raised:
            return _r(False, "skip:eager_ok", "eager did not raise")
        try:
            tf.function(fn, jit_compile=True)(*args)
            return _r(True, "compiled_accepts_invalid", "eager raised, XLA did not")
        except Exception:
            return _r(False, "pass", "both raised")

    return _r(False, "skip:aspect_todo", f"{aspect} judging not implemented")


# ---------- generic all-observable baseline (paper §11 GenericOracle, TF side) ----------
def _obs_mismatch_tf(a, b) -> str | None:
    """First differing observable between the eager reference and the XLA output, or None.
    Blind all-observable comparison: structure, shape, dtype, then value. (TF exposes no
    stride/layout observable at this boundary, unlike the PyTorch runner, so it is omitted;
    both sides are already numpy via _np, so a type mismatch surfaces as shape/dtype.)"""
    import numpy as np

    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if not (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple))) or len(a) != len(b):
            return "structure"
        for x, y in zip(a, b):
            m = _obs_mismatch_tf(x, y)
            if m:
                return m
        return None
    aa, bb = np.asarray(a), np.asarray(b)
    if tuple(aa.shape) != tuple(bb.shape):
        return "shape"
    if str(aa.dtype) != str(bb.dtype):
        return "dtype"
    if _err(a, b) / _scale(a) > _rel_tol(a):
        return "value"
    return None


def _judge_generic_tf(code: str, category: str = "supported") -> dict:
    """GenericOracle / Generic-All-Observable for TF: ignore the mined property; run eager
    vs XLA on the same input and flag ANY difference across easily-observable channels
    (exception raised-or-not, structure/shape/dtype/value) plus crashes. This is the
    oracle-level control for RQ2/RQ4: it has no per-op PR semantic knowledge, so it exposes
    the TP/FP trade-off of blind checking. Test-VALIDITY gates (bad test, nondeterminism,
    OOM, XLA-unsupported, not-traceable) are kept IDENTICAL to the mined-oracle path
    (_judge) so that only the ORACLE differs across conditions on the same fixed test."""
    import tensorflow as tf

    try:
        tf.config.experimental.enable_tensor_float_32_execution(False)  # same precision policy as _judge
    except Exception:
        pass

    ns: dict = {"tf": tf}
    try:
        exec(code, ns)  # noqa: S102 -- isolated worker process
        if "build" not in ns:
            return _r(False, "skip:bad_test", "no build()")
        fn, args = ns["build"]()
    except Exception as e:
        return _r(False, "skip:bad_test", f"{type(e).__name__}: {e}")

    # eager reference, with exception-raised as an observable
    eager_raised = False
    ref = None
    try:
        ref = _np(fn(*args))
    except Exception:
        eager_raised = True
    if not eager_raised:
        if ref is None:
            return _r(False, "skip:bad_test", "fn returned None (no tensor to compare)")
        if category != "nondeterministic":  # rng/nondet ops compare shape/dtype only (below)
            try:
                ref2 = _np(fn(*args))
                if _err(ref2, ref) / _scale(ref) > _rel_tol(ref):
                    return _r(False, "skip:nondeterministic", "eager differs across two runs")
            except Exception:
                return _r(False, "skip:nondeterministic", "second eager run errored")

    # XLA compiled side, same skip gates as the mined-oracle path
    comp_raised = False
    out = None
    comp_detail = ""
    try:
        out = _np(tf.function(fn, jit_compile=True)(*args))
    except Exception as e:
        full = f"{type(e).__name__}: {e}"
        if _is_oom(full):
            return _r(False, "skip:oom", full.splitlines()[0][:160])
        if _is_xla_unsupported(full):
            return _r(False, "skip:xla_unsupported", full.splitlines()[0][:160])
        if _is_not_traceable(full):
            return _r(False, "skip:not_traceable", full.splitlines()[0][:160])
        if _is_bad_test(full):
            return _r(False, "skip:bad_test", full.splitlines()[0][:160])
        comp_raised = True
        comp_detail = full.splitlines()[0][:160]

    # exception observable
    if eager_raised != comp_raised:
        if eager_raised and not comp_raised:
            return _r(True, "generic:compiled_accepts_invalid", "eager raised, XLA did not")
        return _r(True, "generic:compiled_crash", comp_detail)
    if eager_raised and comp_raised:
        return _r(False, "pass", "both raised")

    # both succeeded: all-observable compare
    if category == "nondeterministic":
        same = _shape_dtype(out) == _shape_dtype(ref)
        return _r(not same, "pass" if same else "generic:shape_diff",
                  f"shape/dtype {'match' if same else 'MISMATCH'} (rng: values not compared)")
    m = _obs_mismatch_tf(ref, out)
    if m:
        extra = f" rel_err={_err(ref, out) / _scale(ref):.4f} tol={_rel_tol(ref):g}" if m == "value" else ""
        return _r(True, f"generic:{m}_diff", f"eager vs XLA {m} differ{extra}")
    return _r(False, "pass", "all observables agree")


def _parse_result(proc: subprocess.CompletedProcess) -> dict:
    for line in reversed((proc.stdout or "").splitlines()):
        if line.startswith(_MARK):
            return json.loads(line[len(_MARK):])
    tail = ((proc.stderr or "") or "")[-200:].replace("\n", " ")
    return _r(False, "worker_died", f"rc={proc.returncode} {tail}")


def run_test(test: GeneratedTest, oracle: OracleRef, *, isolate: bool = True, timeout: int = 180,
             judge_mode: str = "oracle") -> Finding:
    """judge_mode="oracle" (default) uses the mined property-specific checker selected by
    aspect; judge_mode="generic" applies the all-observable eager-vs-XLA baseline
    (_judge_generic_tf) -- the GenericOracle control, which holds the test set fixed and
    swaps only the oracle."""
    try:
        from mrminer.tf.op_classify import classify
        category = classify(oracle.op).category
    except Exception:
        category = "supported"
    if not isolate:
        if oracle.property_record and judge_mode in {"oracle", "no_condition"}:
            from mrminer.fuzz.property_runtime import check_program
            res = check_program(test.code, oracle.property_record,
                                operator=oracle.core_operator or oracle.op,
                                source_operator=oracle.op, framework="tensorflow",
                                ignore_condition=judge_mode == "no_condition")
        else:
            res = (_judge_generic_tf(test.code, category) if judge_mode == "generic"
                   else _judge(test.code, oracle.aspect, category))
    else:
        job = {"code": test.code, "aspect": oracle.aspect, "category": category,
               "judge_mode": judge_mode, "op": oracle.op,
               "core_operator": oracle.core_operator or oracle.op,
               "property": oracle.property_record}
        fd, jp = tempfile.mkstemp(suffix=".tffuzzjob.json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(job, f)
            env = dict(os.environ)
            env["PYTHONPATH"] = str(_REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
            from mrminer.fuzz.gpu import place_env
            place_env(env)                               # route this test to the freest pooled GPU
            try:
                proc = subprocess.run(
                    [sys.executable, "-m", "mrminer.tf.runner_tf", "--job", jp],
                    capture_output=True, text=True, timeout=timeout, env=env,
                )
                res = _parse_result(proc)
            except subprocess.TimeoutExpired:
                res = _r(False, "worker_timeout", f">{timeout}s")
        finally:
            os.unlink(jp)
    return Finding(
        test_id=test.test_id, oracle_id=oracle.oracle_id, op=oracle.op,
        aspect=oracle.aspect, triggered=res["triggered"],
        verdict=res["verdict"], detail=res["detail"],
        contexts=tuple(res.get("contexts") or ()),
    )


def _worker_main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ns = ap.parse_args(argv)
    job = json.loads(Path(ns.job).read_text())
    try:
        if job.get("judge_mode", "oracle") == "generic":
            res = _judge_generic_tf(job["code"], job.get("category", "supported"))
        elif job.get("property"):
            from mrminer.fuzz.property_runtime import check_program
            res = check_program(job["code"], job["property"],
                                operator=job["core_operator"], source_operator=job["op"],
                                framework="tensorflow",
                                ignore_condition=job.get("judge_mode") == "no_condition")
        else:
            res = _judge(job["code"], job["aspect"], job.get("category", "supported"))
    except Exception as e:
        res = _r(False, "skip:worker_error", f"{type(e).__name__}: {e}")
    print(_MARK + json.dumps(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(_worker_main())
