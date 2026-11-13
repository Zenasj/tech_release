"""Associate a marked source call with its Core ATen or StableHLO lowering.

Each test marks the call it intends to exercise using
``mark(source_operator, callable, *args, **kwargs)``. The recorder captures the
concrete call, then traces that call in isolation. This prevents an unrelated
operator elsewhere in the test from satisfying an operator-index lookup.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from mrminer.fuzz.property_model import Context, ExecutionResult, PropertyChecker


@dataclass
class MarkedCall:
    call_id: int
    source_operator: str
    fn: Callable
    args: tuple
    kwargs: dict
    output: Any = None
    raised: bool = False


class TraceRecorder:
    def __init__(self, snapshot: Callable | None = None):
        self.calls: list[MarkedCall] = []
        self.snapshot = snapshot

    def mark(self, source_operator: str, fn: Callable, *args, **kwargs):
        if not isinstance(source_operator, str) or not source_operator.startswith("op:"):
            raise ValueError("mark requires a source operator ID such as op:aten::ones_like")
        saved_args, saved_kwargs = ((tuple(args), dict(kwargs)) if self.snapshot is None
                                    else self.snapshot((tuple(args), dict(kwargs))))
        call = MarkedCall(len(self.calls), source_operator, fn, saved_args, saved_kwargs)
        self.calls.append(call)  # retain even if validation raises before lowering
        try:
            call.output = fn(*args, **kwargs)
        except Exception:
            call.raised = True
            raise
        return call.output


def output_path(container: Any, marked: Any) -> tuple[str | int, ...] | None:
    """Find the marked call's result in the returned value by object identity."""
    if container is marked:
        return ()
    if isinstance(container, (tuple, list)):
        for index, value in enumerate(container):
            path = output_path(value, marked)
            if path is not None:
                return (index, *path)
    if isinstance(container, dict):
        for key, value in container.items():
            path = output_path(value, marked)
            if path is not None:
                return (key, *path)
    return None


def _torch_sites(call: MarkedCall, operator: str) -> list[tuple[str, tuple, dict, dict]]:
    from torch._decomp import core_aten_decompositions
    from torch.fx.node import map_arg
    from torch.fx.experimental.proxy_tensor import make_fx

    def marked_fn(*args):
        return call.fn(*args, **call.kwargs)

    graph = make_fx(marked_fn, decomposition_table=core_aten_decompositions(),
                    tracing_mode="fake")(*call.args)
    sites = []
    for node in graph.graph.nodes:
        if node.op != "call_function":
            continue
        schema = getattr(node.target, "_schema", None)
        name = getattr(schema, "name", "")
        if name and f"op:{name}" == operator:
            site_args = map_arg(node.args, lambda n: n.meta.get("val"))
            site_kwargs = map_arg(node.kwargs, lambda n: n.meta.get("val"))
            sites.append((f"fx:{node.name}", site_args, site_kwargs,
                          {"target": str(node.target)}))
    return sites


def _tf_sites(call: MarkedCall, operator: str) -> list[tuple[str, tuple, dict, dict]]:
    from mrminer.tf.stablehlo_ops import program_stablehlo_sites

    def marked_fn(*args):
        return call.fn(*args, **call.kwargs)

    observed = program_stablehlo_sites(marked_fn, call.args)
    return [(f"stablehlo:{call.call_id}:{item['ordinal']}:{item['ssa']}", (), {}, item)
            for item in observed if item["operator"] == operator]


def contexts_for(result: ExecutionResult, calls: list[MarkedCall], *,
                 operator: str, source_operator: str, checker: PropertyChecker,
                 framework: str) -> tuple[list[Context], list[str]]:
    contexts: list[Context] = []
    errors: list[str] = []
    for call in calls:
        if call.source_operator != source_operator:
            continue
        path = output_path(result.output, call.output) if result.kind == "return" else None
        try:
            sites = (_torch_sites(call, operator) if framework == "pytorch"
                     else _tf_sites(call, operator)) if result.kind == "return" else []
        except Exception as error:
            sites = []
            errors.append(f"trace {call.call_id}: {type(error).__name__}: {error}")
        for site, site_args, site_kwargs, site_metadata in sites:
            contexts.append(Context(operator, source_operator, site, call.call_id,
                                    call.args, call.kwargs, path,
                                    site_args=tuple(site_args), site_kwargs=site_kwargs,
                                    site_metadata=site_metadata))
        if result.kind == "raise" and call.raised and not sites:
            inferred = Context(operator, source_operator, f"inferred:{call.call_id}",
                               call.call_id, call.args, call.kwargs, None, True)
            try:
                if checker.origin_matches(inferred):
                    contexts.append(inferred)
            except Exception as error:
                errors.append(f"origin {call.call_id}: {type(error).__name__}: {error}")
    return contexts, errors
