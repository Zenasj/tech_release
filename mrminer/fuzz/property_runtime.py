"""Run one explicit operator property on provenance-bearing test contexts."""
from __future__ import annotations

import random
import traceback
from typing import Any

from mrminer.fuzz.property_model import ExecutionResult, PropertyChecker
from mrminer.fuzz.provenance import TraceRecorder, contexts_for


def _clone_inputs(value: Any, framework: str, storages: dict | None = None) -> Any:
    storages = storages if storages is not None else {}
    if framework == "pytorch":
        import torch
        if isinstance(value, torch.Tensor):
            key = (value.device, value.untyped_storage()._cdata)
            if key not in storages:
                raw = torch.empty(0, dtype=torch.uint8, device=value.device)
                raw.set_(value.untyped_storage(), 0,
                         (value.untyped_storage().nbytes(),), (1,))
                storages[key] = raw.clone()
            base = storages[key].view(value.dtype)
            cloned = base.as_strided(value.shape, value.stride(), value.storage_offset())
            if value.requires_grad:
                cloned.requires_grad_(True)
            return cloned
    else:
        import tensorflow as tf
        if isinstance(value, tf.Variable):
            raise ValueError("mutable TensorFlow variables are not supported as paired inputs")
        if tf.is_tensor(value):
            return tf.identity(value)
    if isinstance(value, tuple):
        return tuple(_clone_inputs(item, framework, storages) for item in value)
    if isinstance(value, list):
        return [_clone_inputs(item, framework, storages) for item in value]
    if isinstance(value, dict):
        return {key: _clone_inputs(item, framework, storages) for key, item in value.items()}
    return value


def _seed(framework: str, seed: int) -> None:
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    if framework == "pytorch":
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    else:
        import tensorflow as tf
        tf.random.set_seed(seed)


def _execute(fn, args: tuple, mode: str, framework: str) -> ExecutionResult:
    try:
        if framework == "pytorch":
            import torch
            if mode == "inductor":
                torch._dynamo.reset()
                value = torch.compile(fn, backend="inductor")(*args)
            else:
                value = fn(*args)
        else:
            import tensorflow as tf
            value = (tf.function(fn, jit_compile=True)(*args) if mode == "xla"
                     else fn(*args))
        return ExecutionResult("return", output=value, post_state=args)
    except Exception as error:
        return ExecutionResult("raise", exception=type(error).__name__, post_state=args,
                               exception_traceback=traceback.format_exc())


def _modes(transform: str, framework: str) -> tuple[str, str]:
    if transform == "eager_to_inductor":
        return "eager", "inductor"
    if transform == "eager_to_xla":
        return "eager", "xla"
    if transform == "identity_eager":
        return "eager", "eager"
    if transform == "identity_compiled":
        return ("inductor", "inductor") if framework == "pytorch" else ("xla", "xla")
    raise ValueError(f"unsupported transform: {transform}")


def check_program(code: str, record: dict, *, operator: str,
                  source_operator: str, framework: str,
                  ignore_condition: bool = False) -> dict:
    return _check_program(code, record, operator=operator,
                          source_operator=source_operator, framework=framework,
                          compare_condition=False,
                          ignore_condition=ignore_condition)


def check_program_pair(code: str, record: dict, *, operator: str,
                       source_operator: str, framework: str) -> dict:
    """Check both variants on the same contexts and execution results."""
    return _check_program(code, record, operator=operator,
                          source_operator=source_operator, framework=framework,
                          compare_condition=True, ignore_condition=False)


def _check_program(code: str, record: dict, *, operator: str,
                   source_operator: str, framework: str,
                   compare_condition: bool, ignore_condition: bool) -> dict:
    def finish(result: dict) -> dict:
        return {"conditional": result, "no_condition": result.copy()} \
            if compare_condition else result

    try:
        checker = PropertyChecker(record, framework=framework)
    except Exception as error:
        return finish({"triggered": False, "verdict": "skip:checker_error",
                       "detail": f"{type(error).__name__}: {error}"})
    namespace: dict = {"mark": lambda _op, fn, *a, **kw: fn(*a, **kw)}
    try:
        exec(code, namespace)  # noqa: S102 -- isolated worker
        fn, raw_args = namespace["build"]()
        raw_args = tuple(raw_args)
        if framework == "pytorch":
            import torch
            if any(isinstance(cell.cell_contents, torch.Tensor)
                   for cell in (getattr(fn, "__closure__", None) or ())):
                raise ValueError("tensor captured by function closure")
        trace_args = _clone_inputs(raw_args, framework)
        first_args = _clone_inputs(raw_args, framework)
        second_args = _clone_inputs(raw_args, framework)
    except Exception as error:
        return finish({"triggered": False, "verdict": "skip:bad_test",
                       "detail": f"{type(error).__name__}: {error}"})

    recorder = TraceRecorder(lambda call: _clone_inputs(call, framework))
    namespace["mark"] = recorder.mark
    _seed(framework, 0)
    trace_result = _execute(fn, trace_args, "eager", framework)
    namespace["mark"] = lambda _op, target, *a, **kw: target(*a, **kw)
    contexts, errors = contexts_for(trace_result, recorder.calls,
                                    operator=operator, source_operator=source_operator,
                                    checker=checker, framework=framework)
    if not contexts:
        return finish({"triggered": False, "verdict": "skip:no_context",
                       "detail": "; ".join(errors) or "marked source call did not reach target"})
    baseline_mode, followup_mode = _modes(record["T"], framework)
    _seed(framework, 0)
    baseline = _execute(fn, first_args, baseline_mode, framework)
    _seed(framework, 0)
    followup = _execute(fn, second_args, followup_mode, framework)
    def summarize(ignore_condition: bool) -> dict:
        ranks = {"skip": 0, "pass": 1, "fail": 2}
        verdicts = [checker.check(context, baseline, followup,
                                  ignore_condition=ignore_condition)
                    for context in contexts]
        verdict, detail = max(verdicts,
                              key=lambda pair: ranks[pair[0].split(":", 1)[0]])
        context_records = [{"operator": c.operator, "source_operator": c.source_operator,
                            "site": c.site, "call_id": c.call_id,
                            "inferred": c.inferred, "output_path": c.output_path,
                            "site_metadata": c.site_metadata,
                            "verdict": pair[0], "reason": pair[1]}
                           for c, pair in zip(contexts, verdicts)]
        return {"triggered": verdict == "fail", "verdict": verdict,
                "detail": f"{detail}; contexts={len(contexts)}; "
                          f"sites={','.join(c.site for c in contexts)}",
                "contexts": context_records}

    conditional = summarize(ignore_condition)
    if compare_condition:
        executions = [{"kind": r.kind, "exception": r.exception,
                       "exception_traceback": r.exception_traceback}
                      for r in (baseline, followup)]
        return {"conditional": conditional, "no_condition": summarize(True),
                "executions": executions}
    return conditional
