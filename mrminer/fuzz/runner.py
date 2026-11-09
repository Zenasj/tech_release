"""Execute a generated test and judge it against its target oracle.

Semi-black-box: treat the compiler as a black box, run eager vs inductor on the
same input, decide whether the oracle was triggered (violated). The
eager-vs-inductor / eager-raises differential is the pattern proven in
`scripts/targeted_spec_scan.py`; it is re-implemented here (self-contained) so the
fuzz package does not edit the original script.

M2:
  * **subprocess isolation** (default): each test runs in its own process, so an
    untrusted / crashing test (CUDA IMA, device-side assert, segfault) cannot
    poison the driver. Required before running real-LLM output.
  * aspects judged: `semantic` (eager vs inductor) and `effect` (eager raises but
    compiled silently accepts = compiled dropped a validation).
  * `performance` (non-inferiority): exec_mode latency proxy -- median-of-N eager vs
    inductor. OBSERVE-ONLY: in-loop timing on a shared GPU is noise-prone (micro-ops show
    fixed-overhead ratios that don't reproduce under controlled benchmarking), so it never
    raises a trigger (`observed:slower` / `pass`); real non-inferiority needs an isolated
    benchmark (future work). The mined $patch on/off toggle also can't be actuated here.
  * `decision` (predicate_relation): probe registry keyed on the mined predicate's
    `preferred_probe`. The cudagraph-unsafe probe captures the INDUCTOR post-grad graph
    (via post_grad_custom_post_pass) -- the stage where is_cudagraph_unsafe_fx_node is
    reliable -- and triggers `decision_mismatch` when the marking disagrees with the
    PR-established expectation; unmechanized probes honor missing_probe_outcome ->
    skip:probe_unsupported.

Run as a worker:  python -m mrminer.fuzz.runner --job <job.json>
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

REL = 1e-3
_RESULT_MARK = "__FUZZ_RESULT__"
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Ops whose output is UNINITIALISED memory: eager vs compiled read different
# garbage, so a semantic eager-vs-compiled comparison is meaningless (any
# "divergence" is noise, any "pass" is luck). Not testable for the semantic MR.
_UNINIT_OPS = {
    "op:aten::empty", "op:aten::empty_strided", "op:aten::empty_like",
    "op:aten::empty_permuted", "op:aten::new_empty", "op:aten::new_empty_strided",
}


# ---------- comparison helpers (self-contained) ----------
def _err(a, b) -> float:
    import torch

    if isinstance(a, (tuple, list)):
        if not isinstance(b, (tuple, list)) or len(a) != len(b):
            return float("inf")
        return max((_err(x, y) for x, y in zip(a, b)), default=0.0)
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or a.shape != b.shape:
            return float("inf")
        if not a.is_floating_point():
            return 0.0
        return (a.float() - b.float()).abs().max().item() if a.numel() else 0.0
    try:
        return abs(float(a) - float(b))
    except Exception:
        return 0.0 if a == b else float("inf")


def _scale(o) -> float:
    import torch

    if isinstance(o, (tuple, list)):
        return max((_scale(x) for x in o), default=1.0)
    if isinstance(o, torch.Tensor) and o.numel():
        return max(o.float().abs().max().item(), 1.0)
    return 1.0


# dtype-aware relative tolerance: low-precision (bf16/fp16) fused kernels reorder ops and
# round differently from eager, so a fixed fp32-grade REL flags pure rounding as
# "diverged". Loosen the bar for low-precision outputs; keep REL for fp32/fp64/other.
# bf16: 1 ulp ~= 2^-8 ~= 0.004; allow ~4 ulp for a fused chain -> 1.5e-2.
# fp16: 1 ulp ~= 2^-10 ~= 0.001; ~4 ulp -> 4e-3. (Deliberately NOT looser than the
# rounding band, so a real low-precision miscompile still diverges grossly.)
_REL_BY_DTYPE = {"torch.bfloat16": 1.5e-2, "torch.float16": 4e-3}


def _rel_tol(o) -> float:
    """The loosest per-dtype tolerance over all tensors in the reference output."""
    import torch

    if isinstance(o, (tuple, list)):
        return max((_rel_tol(x) for x in o), default=REL)
    if isinstance(o, torch.Tensor):
        return _REL_BY_DTYPE.get(str(o.dtype), REL)
    return REL


def _ftz_subnormal_explains(ns: dict, out) -> bool:
    """True iff a semantic divergence is explained by flush-to-zero of *nonzero
    subnormal inputs*. Rebuild fresh inputs, flush their subnormals to 0, re-run
    eager, and check it reproduces the COMPILED output. Conservative: fires only
    when (a) the input actually carries nonzero subnormals and (b) flushing them
    makes eager match the compiled result — so a normal-range real miscompile is
    never muted. Collapses the known default-FTZ class (cf. #185480 / #199671)
    into one verdict instead of re-alerting on each affected op."""
    import torch

    try:
        fn, args = ns["build"]()
    except Exception:
        return False
    had_sub = False

    def _flush(o):
        nonlocal had_sub
        if torch.is_tensor(o) and o.is_floating_point() and o.numel():
            tiny = torch.finfo(o.dtype).tiny  # smallest positive *normal*
            m = (o != 0) & (o.abs() < tiny)
            if bool(m.any()):
                had_sub = True
                return torch.where(m, torch.zeros_like(o), o)
        return o

    try:
        fargs = [_flush(a) for a in args]
    except Exception:
        return False
    if not had_sub:
        return False  # no subnormal input -> not an FTZ case, never mute
    try:
        with torch.no_grad():
            ref_ftz = fn(*fargs)
        return _err(ref_ftz, out) / _scale(out) <= _rel_tol(out)
    except Exception:
        return False


def _poison(ref) -> None:
    """Defeat the caching allocator's block reuse so uninitialised output (from
    empty / empty_strided / empty_like &c.) differs across two eager runs and is
    caught as nondeterministic. Without this, two consecutive empty() calls reuse
    one freed block and return identical garbage -> a false "deterministic" pass,
    then the compiled run allocates elsewhere -> a false "diverged". Allocate +
    free NaN decoys sized like each float output, on that output's own device.
    (Ported from scripts/sample_and_test_model.py.)"""
    import torch

    per_device: dict = {}

    def collect(o):
        if isinstance(o, (tuple, list)):
            for x in o:
                collect(x)
        elif isinstance(o, torch.Tensor) and o.is_floating_point() and o.numel():
            per_device.setdefault(o.device, []).append(o.numel())

    collect(ref)
    decoys = [torch.full((n,), float("nan"), device=dev)
              for dev, ns in per_device.items() for n in ns]
    del decoys


# ---------- generic all-observable baseline (paper §11 V2 / §11.0.1) ----------
def _obs_mismatch(a, b) -> str | None:
    """First differing observable between a reference and a compiled output, or None.
    Blind all-observable comparison: structure, type, shape, dtype, stride (layout),
    then value. (Aliasing is not observable in this harness and is omitted.)"""
    import torch

    if isinstance(a, (tuple, list)) or isinstance(b, (tuple, list)):
        if not (isinstance(a, (tuple, list)) and isinstance(b, (tuple, list))) or len(a) != len(b):
            return "structure"
        for x, y in zip(a, b):
            m = _obs_mismatch(x, y)
            if m:
                return m
        return None
    at, bt = isinstance(a, torch.Tensor), isinstance(b, torch.Tensor)
    if at != bt:
        return "type"
    if at:
        if tuple(a.shape) != tuple(b.shape):
            return "shape"
        if a.dtype != b.dtype:
            return "dtype"
        if a.stride() != b.stride():
            return "stride"
        if _err(a, b) / _scale(a) > _rel_tol(a):
            return "value"
        return None
    try:
        return "value" if _err(a, b) > REL else None
    except Exception:
        return None


def _judge_generic(code: str, op: str = "") -> dict:
    """V2 GenericOracle / Generic-All-Observable: ignore the mined property; run eager
    vs inductor on the same input and flag ANY difference across easily-observable
    channels (exception raised-or-not, structure/type/shape/dtype/stride/value) plus
    crashes. This is the strong oracle-level control for RQ2 (§11.0.3): it has no per-op
    PR semantic knowledge, so it exposes the TP/FP trade-off of blind checking.

    Test-VALIDITY gates (bad test, RNG, uninitialised memory, nondeterminism, OOM) are
    kept identical to the MRMiner oracle path so that only the ORACLE differs across
    baselines on the same test -- an invalid test is skipped under both."""
    import torch
    import torch._dynamo

    if op in _UNINIT_OPS:
        return _r(False, "skip:uninitialized_op", f"{op} returns uninitialised memory")

    ns: dict = {}
    try:
        exec(code, ns)  # noqa: S102 -- isolated worker process
        if "build" not in ns:
            return _r(False, "skip:bad_test", "no build()")
        fn, args = ns["build"]()
    except Exception as e:
        return _r(False, "skip:bad_test", f"{type(e).__name__}: {e}")

    from torch.utils._python_dispatch import TorchDispatchMode

    class _Trace(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.names: set[str] = set()

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            nm = getattr(getattr(func, "_schema", None), "name", None)
            if nm:
                self.names.add(nm)
            return func(*args, **(kwargs or {}))

    tr = _Trace()
    eager_raised = False
    ref = None
    try:
        with torch.no_grad(), tr:
            ref = fn(*args)
    except Exception:
        eager_raised = True
    if not eager_raised:
        _RNG = ("rand", "dropout", "bernoulli", "multinomial", "normal", "poisson", "rrelu")
        if any(k in nm for nm in tr.names for k in _RNG):
            return _r(False, "skip:uses_rng", "program uses RNG -> compile RNG differs by design")
        try:
            _poison(ref)
            with torch.no_grad():
                ref2 = fn(*args)
            if _err(ref2, ref) / _scale(ref) > _rel_tol(ref):
                return _r(False, "skip:nondeterministic", "eager differs across two runs")
        except Exception:
            return _r(False, "skip:nondeterministic", "second eager run errored")

    torch._dynamo.reset()
    comp_raised = False
    out = None
    try:
        with torch.no_grad():
            out = torch.compile(fn, backend="inductor")(*args)
    except Exception as e:
        full = f"{type(e).__name__}: {e}".splitlines()[0]
        if "out of memory" in full.lower() or "OutOfMemoryError" in type(e).__name__:
            return _r(False, "skip:oom", full[:160])
        comp_raised = True
        comp_detail = full[:160]

    # exception observable
    if eager_raised != comp_raised:
        if eager_raised and not comp_raised:
            return _r(True, "generic:compiled_accepts_invalid", "eager raised, compiled did not")
        return _r(True, "generic:compiled_crash", comp_detail)
    if eager_raised and comp_raised:
        return _r(False, "pass", "both raised")

    # both succeeded: compare all observables
    m = _obs_mismatch(ref, out)
    if m:
        extra = ""
        if m == "value":
            extra = f" rel_err={_err(ref, out) / _scale(ref):.4f} tol={_rel_tol(ref):g}"
        return _r(True, f"generic:{m}_diff", f"eager vs inductor {m} differ{extra}")
    return _r(False, "pass", "all observables agree")


# ---------- in-process judging (runs inside the worker) ----------
def _judge(code: str, aspect: str, op: str = "", params: dict | None = None,
           judge_mode: str = "oracle") -> dict:
    """Return {'triggered': bool, 'verdict': str, 'detail': str}.

    judge_mode="oracle" (default) uses the mined property-specific checker selected by
    `aspect`. judge_mode="generic" ignores the property and applies the all-observable
    baseline (_judge_generic) -- the §11 V2 control."""
    import torch
    import torch._dynamo

    if judge_mode == "generic":
        return _judge_generic(code, op)

    if aspect == "semantic" and op in _UNINIT_OPS:
        return _r(False, "skip:uninitialized_op", f"{op} returns uninitialised memory")

    ns: dict = {}
    try:
        exec(code, ns)  # noqa: S102 -- isolated worker process
        if "build" not in ns:
            return _r(False, "skip:bad_test", "no build()")
        fn, args = ns["build"]()
    except Exception as e:
        return _r(False, "skip:bad_test", f"{type(e).__name__}: {e}")

    if aspect == "semantic":
        # Trace the aten ops the program dispatches to during the reference run, so
        # we can skip programs whose output is RNG-dependent: torch.compile's RNG
        # differs from eager BY DESIGN, so an eager!=compiled difference there is
        # expected, not a bug (also catches random-shape programs, e.g. a size drawn
        # from torch.randint). Mirrors scripts/sample_and_test_model.py's guard.
        from torch.utils._python_dispatch import TorchDispatchMode

        class _Trace(TorchDispatchMode):
            def __init__(self):
                super().__init__()
                self.names: set[str] = set()

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                nm = getattr(getattr(func, "_schema", None), "name", None)
                if nm:
                    self.names.add(nm)
                return func(*args, **(kwargs or {}))

        tr = _Trace()
        try:
            with torch.no_grad(), tr:
                ref = fn(*args)
        except Exception:
            return _r(False, "skip:eager_error", "eager errored -> no reference")
        _RNG = ("rand", "dropout", "bernoulli", "multinomial", "normal", "poisson", "rrelu")
        if any(k in nm for nm in tr.names for k in _RNG):
            return _r(False, "skip:uses_rng", "program uses RNG -> compile RNG differs by design")
        # self-consistency guard: a second eager run must match, else the program
        # is nondeterministic / stateful (in-place buffer mutation) and there is no
        # fixed reference to compare the compiled run against.
        try:
            _poison(ref)  # make uninitialised-memory outputs differ on the 2nd run
            with torch.no_grad():
                ref2 = fn(*args)
            if _err(ref2, ref) / _scale(ref) > _rel_tol(ref):
                return _r(False, "skip:nondeterministic", "eager differs across two runs")
        except Exception:
            return _r(False, "skip:nondeterministic", "second eager run errored")
        torch._dynamo.reset()
        try:
            with torch.no_grad():
                out = torch.compile(fn, backend="inductor")(*args)
        except Exception as e:
            full = f"{type(e).__name__}: {e}".splitlines()[0]
            if "out of memory" in full.lower() or "OutOfMemoryError" in type(e).__name__:
                return _r(False, "skip:oom", full[:160])   # resource pressure, not a bug
            return _r(True, "compiled_crash", full[:160])
        err = _err(out, ref) / _scale(ref)
        tol = _rel_tol(ref)
        if err > tol and _ftz_subnormal_explains(ns, out):
            # Divergence is the known default-FTZ-on-subnormal class (cf. #185480 / #199671),
            # reproduced by flushing subnormal inputs to zero. Collapse it instead of re-alerting.
            return _r(False, "skip:ftz_subnormal",
                      f"subnormal-input FTZ reproduces compiled output (known FTZ class); rel_err={err:.4f}")
        return _r(err > tol, "diverged" if err > tol else "pass", f"rel_err={err:.4f} tol={tol:g}")

    if aspect == "effect":
        # oracle says the op SHOULD raise on this (invalid) input. Bug = eager
        # raises but compiled silently accepts (dropped the validation).
        try:
            with torch.no_grad():
                fn(*args)
            eager_raised = False
        except Exception:
            eager_raised = True
        if not eager_raised:
            return _r(False, "skip:eager_ok", "eager did not raise -> domain not exercised")
        torch._dynamo.reset()
        try:
            with torch.no_grad():
                torch.compile(fn, backend="inductor")(*args)
            return _r(True, "compiled_accepts_invalid", "eager raised, compiled did not")
        except Exception:
            return _r(False, "pass", "both raised")

    if aspect == "performance":
        return _judge_performance(fn, args)

    if aspect == "decision":
        return _judge_decision(fn, args, op, params or {})

    return _r(False, "skip:aspect_todo", f"{aspect} judging not implemented")


def _iter_tensors(obj):
    import torch
    if isinstance(obj, (tuple, list)):
        for x in obj:
            yield from _iter_tensors(x)
    elif isinstance(obj, torch.Tensor):
        yield obj


def _judge_performance(fn, args) -> dict:
    """R_O = non-inferiority: compilation must not make the op meaningfully slower.

    The mined config toggles a specific optimization patch we cannot actuate on an
    arbitrary build, so we use exec_mode (eager vs inductor) latency as the observable
    channel. Timing in the shared multi-lane GPU is noisy, so we only flag a *large*
    regression (median-of-N, conservative margin); precise judging needs an isolated
    benchmark (future work)."""
    import time
    import torch
    import torch._dynamo

    cuda = any(t.is_cuda for t in _iter_tensors(args))

    def sync():
        if cuda:
            torch.cuda.synchronize()

    try:
        with torch.no_grad():
            fn(*args)
        sync()
    except Exception as e:
        return _r(False, "skip:eager_error", f"{type(e).__name__}")

    torch._dynamo.reset()
    try:
        comp = torch.compile(fn, backend="inductor")
        with torch.no_grad():
            comp(*args)            # warmup: pays the one-time compile cost here
        sync()
    except Exception as e:
        return _r(False, "skip:compile_error", f"{type(e).__name__}")  # crashes are covered by semantic/effect

    reps = max(5, int(os.environ.get("MRMINER_PERF_REPS", "30")))

    def median_time(call, reps=reps, warmup=5):
        with torch.no_grad():
            for _ in range(warmup):
                call()
        sync()
        ts = []
        with torch.no_grad():
            for _ in range(reps):
                sync()
                t0 = time.perf_counter()
                call()
                sync()
                ts.append(time.perf_counter() - t0)
        ts.sort()
        return ts[len(ts) // 2]

    try:
        e = median_time(lambda: fn(*args))
        c = median_time(lambda: comp(*args))
    except Exception as ex:
        return _r(False, "skip:timing_error", f"{type(ex).__name__}")

    ratio = c / e if e > 0 else float("inf")
    MARGIN, FLOOR = 3.0, 5e-5      # need >3x AND >50us absolute, to clear loop noise
    # In-loop timing on a shared GPU is noise-prone: micro-ops show large fixed-overhead
    # ratios that do NOT reproduce under controlled benchmarking (e.g. a constant ~2.3ms
    # compiled floor across unrelated tiny ops). So we OBSERVE but never raise a trigger
    # from it -- otherwise every such noise spike becomes a spurious NEW signature.
    # Real non-inferiority needs an isolated benchmark (warmup/repeat/isolation); future work.
    slower = c > e * MARGIN and (c - e) > FLOOR
    return _r(False, "observed:slower" if slower else "pass",
              f"compiled/eager={ratio:.2f}x (e={e*1e3:.3f}ms c={c*1e3:.3f}ms)")


def _capture_post_grad_graph(fn, args):
    """Compile fn under inductor with a post-grad hook that captures the post-grad FX
    graph -- the stage where cudagraph-unsafe meta is reliable (unlike the dynamo graph).
    Returns a torch.fx.Graph, None if nothing was captured, or a short error string.
    fx_graph_cache is disabled during the call so the post-grad pass actually runs."""
    import torch
    import torch._dynamo
    from torch._inductor import config as icfg

    cap: dict = {}

    def _plain(graph):
        cap.setdefault("graph", graph)
        return None

    try:
        from torch._inductor.custom_graph_pass import CustomGraphPass

        class _Cap(CustomGraphPass):
            def __call__(self, graph):
                cap.setdefault("graph", graph)

            def uuid(self):
                return "mrminer-decision-capture-v1"

        hook = _Cap()
    except Exception:
        hook = _plain

    old_pass = icfg.post_grad_custom_post_pass
    old_cache = getattr(icfg, "fx_graph_cache", None)
    icfg.post_grad_custom_post_pass = hook
    try:
        icfg.fx_graph_cache = False
    except Exception:
        pass
    try:
        torch._dynamo.reset()
        with torch.no_grad():
            torch.compile(fn, backend="inductor")(*args)
    except Exception as ex:
        return f"{type(ex).__name__}: {str(ex)[:60]}"
    finally:
        icfg.post_grad_custom_post_pass = old_pass
        if old_cache is not None:
            try:
                icfg.fx_graph_cache = old_cache
            except Exception:
                pass
    return cap.get("graph")


def _judge_decision(fn, args, op: str, params: dict) -> dict:
    """R_O = the compiler makes the decision the source PR established. Each predicate
    names a probe; we mechanize the ones we can and honor missing_probe_outcome by
    skipping the rest (skip:probe_unsupported)."""
    import torch
    import torch._dynamo

    preds = (params or {}).get("predicates") or []
    if not preds:
        return _r(False, "skip:probe_unsupported", "no predicate params")

    target_short = op.split("::")[-1]
    for p in preds:
        prop = (p.get("property") or "").lower()
        probe = (p.get("preferred_probe") or "").lower()
        expected = str(p.get("expected", "")).lower()

        # --- cudagraph-unsafe family: probe the marking on the INDUCTOR post-grad graph ---
        if "cudagraph" in prop or "cudagraph_unsafe_fx_node" in probe:
            try:
                from torch._inductor.utils import is_cudagraph_unsafe_fx_node
            except Exception:
                return _r(False, "skip:probe_unsupported", "is_cudagraph_unsafe_fx_node unavailable")
            graph = _capture_post_grad_graph(fn, args)
            if isinstance(graph, str):
                return _r(False, "skip:probe_error", graph)
            if graph is None:
                return _r(False, "skip:probe_error", "no post-grad graph captured")
            base = target_short.lstrip("_")
            nodes = [n for n in graph.nodes
                     if n.op == "call_function" and (target_short in str(n.target) or base in str(n.target))]
            if not nodes:
                return _r(False, "skip:eager_ok", f"{target_short} not in post-grad graph")
            try:
                unsafe = any(bool(is_cudagraph_unsafe_fx_node(n)) for n in nodes)
            except Exception as ex:
                return _r(False, "skip:probe_error", f"{type(ex).__name__}: {str(ex)[:60]}")
            exp_unsafe = expected in ("true", "unsafe")
            if unsafe != exp_unsafe:
                return _r(True, "decision_mismatch",
                          f"{target_short}.cudagraph_unsafe={unsafe} expected={exp_unsafe} (post-grad)")
            return _r(False, "pass", f"cudagraph_unsafe={unsafe} matches expected (post-grad)")

        # --- fallback / kernel-count: observe generated_kernel_count (expected is prose) ---
        if "generated_kernel_count" in probe:
            from torch._inductor import metrics
            torch._dynamo.reset()
            metrics.reset()
            try:
                with torch.no_grad():
                    torch.compile(fn, backend="inductor")(*args)
            except Exception as ex:
                return _r(False, "skip:probe_error", f"{type(ex).__name__}")
            return _r(False, "pass", f"generated_kernel_count={metrics.generated_kernel_count}")

    return _r(False, "skip:probe_unsupported", f"probe not mechanized: {probe[:40]}")


def _r(triggered: bool, verdict: str, detail: str = "") -> dict:
    return {"triggered": triggered, "verdict": verdict, "detail": detail}


# ---------- subprocess driver (runs in the parent) ----------
def _parse_result(proc: subprocess.CompletedProcess) -> dict:
    for line in reversed((proc.stdout or "").splitlines()):
        if line.startswith(_RESULT_MARK):
            return json.loads(line[len(_RESULT_MARK):])
    # No result line: the worker died hard (segfault / CUDA IMA / OOM / timeout).
    # Conservatively flag for triage rather than auto-calling it a bug.
    tail = ((proc.stderr or "") or "")[-200:].replace("\n", " ")
    return _r(False, "worker_died", f"rc={proc.returncode} {tail}")


def run_test(
    test: GeneratedTest,
    oracle: OracleRef,
    *,
    isolate: bool = True,
    timeout: int = 180,
    judge_mode: str = "oracle",
) -> Finding:
    if not isolate:
        res = _judge(test.code, oracle.aspect, oracle.op, getattr(oracle, "params", {}),
                     judge_mode=judge_mode)
    else:
        job = {"code": test.code, "aspect": oracle.aspect, "op": oracle.op,
               "params": getattr(oracle, "params", {}), "judge_mode": judge_mode}
        fd, jp = tempfile.mkstemp(suffix=".fuzzjob.json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(job, f)
            env = dict(os.environ)
            env["PYTHONPATH"] = str(_REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
            from mrminer.fuzz.gpu import place_env
            place_env(env)                               # route this test to the freest pooled GPU
            try:
                proc = subprocess.run(
                    [sys.executable, "-m", "mrminer.fuzz.runner", "--job", jp],
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
    )


def _worker_main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ns = ap.parse_args(argv)
    job = json.loads(Path(ns.job).read_text())
    try:
        res = _judge(job["code"], job["aspect"], job.get("op", ""), job.get("params") or {},
                     judge_mode=job.get("judge_mode", "oracle"))
    except Exception as e:  # never let the worker die silently on a Python error
        res = _r(False, "skip:worker_error", f"{type(e).__name__}: {e}")
    print(_RESULT_MARK + json.dumps(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(_worker_main())
