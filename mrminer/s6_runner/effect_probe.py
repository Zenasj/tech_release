"""PyTorch FX pre-trigger and compiler-counter effect observation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import re
from typing import Any, Protocol

from mrminer.common.canonical import sha256_digest
from mrminer.common.schema import DecisionObservation, PatternSpec
from mrminer.s6_runner.pattern_matcher import (
    RestrictedPatternMatcher,
    RuntimeGraph,
    RuntimeGraphNode,
)

PATTERN_PRESENCE_PROBE = "graph.pattern_presence/pre_optimization/v1"
COUNTER_PROBE = "decision.counter/post_execution/v1"


class DecisionProbe(Protocol):
    def before_compile(self, program: Any, materialized_input: Any) -> None: ...

    def after_runtime(
        self,
        program: Any,
        materialized_input: Any,
        output: Any,
    ) -> DecisionObservation: ...


def _input_values(materialized_input: Any) -> tuple[Any, ...]:
    if isinstance(materialized_input, tuple):
        return materialized_input
    if isinstance(materialized_input, list):
        return tuple(materialized_input)
    if isinstance(materialized_input, Mapping):
        return tuple(materialized_input[key] for key in sorted(materialized_input))
    return (materialized_input,)


def _value_metadata(value: Any) -> tuple[str, str | None, str | None]:
    import torch

    if isinstance(value, torch.Tensor):
        return (
            "tensor",
            str(value.dtype).removeprefix("torch."),
            value.device.type,
        )
    if isinstance(value, (bool, int, float, complex, str)):
        return "scalar", None, None
    return "unknown", None, None


def _qualified_target(target: Any) -> str:
    module = getattr(target, "__module__", None)
    name = getattr(target, "__name__", None)
    if isinstance(name, str) and name:
        return f"{module}.{name}" if isinstance(module, str) and module else name
    return str(target)


def _operator_identity(
    *,
    node: Any,
    graph_module: Any,
) -> tuple[str, str]:
    target = node.target
    schema = getattr(target, "_schema", None)
    schema_name = getattr(schema, "name", None)
    if isinstance(schema_name, str) and "::" in schema_name:
        namespace, name = schema_name.split("::", 1)
        overload = getattr(target, "_overloadname", None) or "default"
        return f"op:{namespace}::{name}", str(overload)

    if node.op == "call_module":
        submodule = graph_module.get_submodule(str(target))
        target_name = type(submodule).__name__.lower()
    elif node.op == "call_method":
        target_name = str(target).lower()
    else:
        target_name = str(getattr(target, "__name__", target)).lower()
    aliases = {
        "addmm": "op:aten::addmm",
        "linear": "op:aten::linear",
        "relu": "op:aten::relu",
    }
    if target_name in aliases:
        return aliases[target_name], "default"

    qualified = _qualified_target(target)
    stable = re.sub(r"[^A-Za-z0-9_.:-]+", "_", qualified).strip("_")
    if not stable:
        stable = sha256_digest(
            {"node_op": node.op, "target_type": type(target).__qualname__}
        )
    return f"op:runtime::{stable}", "default"


def _dependencies(value: Any, node_ids: Mapping[Any, int]) -> tuple[int, ...]:
    values: set[int] = set()

    def visit(item: Any) -> None:
        if _hashable(item) and item in node_ids:
            values.add(node_ids[item])
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        elif isinstance(item, Mapping):
            for key in sorted(item, key=str):
                visit(item[key])

    visit(value)
    return tuple(sorted(values))


def _hashable(value: Any) -> bool:
    try:
        hash(value)
    except TypeError:
        return False
    return True


def torch_fx_runtime_graph(
    program: Any,
    materialized_input: Any,
) -> RuntimeGraph:
    """Trace a program without invoking the compiler or reading compiler counters."""

    import torch

    graph_module = torch.fx.symbolic_trace(program)
    supported = [
        node
        for node in graph_module.graph.nodes
        if node.op
        in {
            "placeholder",
            "get_attr",
            "call_function",
            "call_method",
            "call_module",
        }
    ]
    node_ids = {node: index for index, node in enumerate(supported)}
    inputs = iter(_input_values(materialized_input))
    runtime_nodes: list[RuntimeGraphNode] = []
    for node in supported:
        node_id = node_ids[node]
        if node.op == "placeholder":
            if (
                isinstance(materialized_input, Mapping)
                and str(node.target) in materialized_input
            ):
                value = materialized_input[str(node.target)]
            else:
                try:
                    value = next(inputs)
                except StopIteration:
                    value = None
            value_kind, dtype, device = _value_metadata(value)
            runtime_nodes.append(
                RuntimeGraphNode(
                    node_id=node_id,
                    kind="placeholder",
                    value_kind=value_kind,
                    dtype=dtype,
                    device=device,
                )
            )
            continue
        if node.op == "get_attr":
            value = graph_module
            for part in str(node.target).split("."):
                value = getattr(value, part)
            value_kind, dtype, device = _value_metadata(value)
            runtime_nodes.append(
                RuntimeGraphNode(
                    node_id=node_id,
                    kind="get_attr",
                    value_kind=value_kind,
                    dtype=dtype,
                    device=device,
                )
            )
            continue
        op, overload = _operator_identity(node=node, graph_module=graph_module)
        runtime_nodes.append(
            RuntimeGraphNode(
                node_id=node_id,
                kind="op",
                value_kind="tensor",
                op=op,
                overload=overload,
                arg_inputs=tuple(
                    _dependencies(argument, node_ids) for argument in node.args
                ),
            )
        )
    return RuntimeGraph(nodes=tuple(runtime_nodes))


@dataclass
class TorchFxEffectProbe:
    pattern: PatternSpec
    counter_namespace: str
    counter_name: str
    matcher: RestrictedPatternMatcher = field(
        default_factory=RestrictedPatternMatcher
    )
    _pattern_count: int | None = None

    def __post_init__(self) -> None:
        if not self.counter_namespace or not self.counter_name:
            raise ValueError("effect counter namespace and name must be non-empty")

    @property
    def pattern_entity_id(self) -> str:
        return f"pat:{self.pattern.canonical_hash}"

    def before_compile(self, program: Any, materialized_input: Any) -> None:
        from torch._dynamo.utils import counters

        graph = torch_fx_runtime_graph(program, materialized_input)
        self._pattern_count = self.matcher.count(self.pattern, graph)
        counters.clear()

    def after_runtime(
        self,
        program: Any,
        materialized_input: Any,
        output: Any,
    ) -> DecisionObservation:
        del program, materialized_input, output
        from torch._dynamo.utils import counters

        if self._pattern_count is None:
            raise RuntimeError("effect probe was not prepared before runtime")
        counter = counters[self.counter_namespace][self.counter_name]
        if isinstance(counter, bool) or not isinstance(counter, int) or counter < 0:
            raise TypeError("effect counter must be a non-negative integer")
        return DecisionObservation(
            pre_optimization={
                "pattern_presence": {
                    self.pattern_entity_id: self._pattern_count
                }
            },
            post_pass={"pattern_application": {}},
            post_codegen={},
            graph_breaks=(),
            fallbacks=(),
            counters={self.counter_name: counter},
            probe_versions={
                PATTERN_PRESENCE_PROBE: "v1",
                COUNTER_PROBE: "v1",
            },
        )
