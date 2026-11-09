"""M7.3: generate minimal reconstructible programs from restricted PatternSpecs.

A restricted PatternSpec is a small DAG of typed wildcard inputs feeding op
nodes (e.g. ``relu(linear(input, weight, bias))``). This module turns such a
pattern into a runnable program factory plus the tensor input specs its
wildcards require, so the graph's existing pattern entities become a scalable
source of validated (program, input) pairs (M7.1 mints the input, M7.2 validates
the pair). Ops outside the recipe registry, or structures the registry cannot
shape, fail closed with a typed reason rather than guessing.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum

from mrminer.s2_graph.api_resolution import resolve_aten_to_api
from mrminer.common.schema import (
    InputInterface,
    PatternSpec,
    ProgramKind,
    ProgramManifest,
    ProgramPhase,
)
from mrminer.s3_oracle.input_builder import TensorInputSpec

Shape = tuple[int, ...]


# --- op recipes (torch imported lazily so generation stays torch-free) --------


def _linear(x, w, b):  # input @ weight.T + bias
    import torch.nn.functional as F

    return F.linear(x, w, b)


def _relu(x):
    import torch

    return torch.relu(x)


def _sigmoid(x):
    import torch

    return torch.sigmoid(x)


def _tanh(x):
    import torch

    return torch.tanh(x)


def _gelu(x):
    import torch.nn.functional as F

    return F.gelu(x)


@dataclass(frozen=True, kw_only=True)
class OpRecipe:
    torch_fn: Callable[..., object]
    arity: int
    wildcard_shapes: dict[int, Shape]
    output_shape: Callable[[list[Shape]], Shape]


def _unary(fn: Callable[..., object]) -> OpRecipe:
    return OpRecipe(
        torch_fn=fn,
        arity=1,
        wildcard_shapes={0: (64, 16)},
        output_shape=lambda shapes: shapes[0],
    )


_REGISTRY: dict[str, OpRecipe] = {
    "op:aten::linear": OpRecipe(
        torch_fn=_linear,
        arity=3,
        wildcard_shapes={0: (64, 32), 1: (16, 32), 2: (16,)},
        # F.linear(input(M,K), weight(N,K), bias(N,)) -> (M, N)
        output_shape=lambda shapes: (shapes[0][0], shapes[1][0]),
    ),
    "op:aten::relu": _unary(_relu),
    "op:aten::sigmoid": _unary(_sigmoid),
    "op:aten::tanh": _unary(_tanh),
    "op:aten::gelu": _unary(_gelu),
}


def supported_ops() -> frozenset[str]:
    return frozenset(_REGISTRY)


class GenerationStatus(str, Enum):
    GENERATED = "generated"
    UNSUPPORTED_OP = "unsupported_op"
    UNSUPPORTED_STRUCTURE = "unsupported_structure"


@dataclass(frozen=True, kw_only=True)
class GeneratedProgram:
    status: GenerationStatus
    reason: str | None = None
    program_factory: Callable[[], Callable[..., object]] | None = None
    input_specs: tuple[TensorInputSpec, ...] = ()
    manifest: ProgramManifest | None = None


def _topological_order(pattern: PatternSpec) -> list[int] | None:
    incoming: dict[int, set[int]] = defaultdict(set)
    outgoing: dict[int, set[int]] = defaultdict(set)
    ids = {node.idx for node in pattern.nodes}
    for edge in pattern.edges:
        incoming[edge.to_idx].add(edge.from_idx)
        outgoing[edge.from_idx].add(edge.to_idx)
    ready = sorted(idx for idx in ids if not incoming[idx])
    order: list[int] = []
    seen: set[int] = set()
    while ready:
        current = ready.pop(0)
        order.append(current)
        seen.add(current)
        for successor in sorted(outgoing[current]):
            if incoming[successor] <= seen and successor not in seen:
                if successor not in ready:
                    ready.append(successor)
        ready.sort()
    return order if len(order) == len(ids) else None


def generate_program_from_pattern(
    pattern: PatternSpec,
    *,
    dtype: str = "float32",
    device_class: str = "cpu",
    api_ids: Iterable[str] | None = None,
) -> GeneratedProgram:
    nodes = {node.idx: node for node in pattern.nodes}
    for node in pattern.nodes:
        if node.kind == "op" and node.op not in _REGISTRY:
            return GeneratedProgram(
                status=GenerationStatus.UNSUPPORTED_OP,
                reason=f"op {node.op!r} is not in the recipe registry",
            )

    order = _topological_order(pattern)
    if order is None:
        return GeneratedProgram(
            status=GenerationStatus.UNSUPPORTED_STRUCTURE,
            reason="pattern graph is not a DAG",
        )

    # arg_pos -> source node idx, per op node
    args_by_node: dict[int, dict[int, int]] = defaultdict(dict)
    for edge in pattern.edges:
        args_by_node[edge.to_idx][edge.arg_pos] = edge.from_idx

    shapes: dict[int, Shape] = {}
    for idx in order:
        node = nodes[idx]
        if node.kind != "op":
            continue
        recipe = _REGISTRY[node.op]
        args = args_by_node[idx]
        if sorted(args) != list(range(recipe.arity)):
            return GeneratedProgram(
                status=GenerationStatus.UNSUPPORTED_STRUCTURE,
                reason=f"node {idx} arg positions do not match arity {recipe.arity}",
            )
        arg_shapes: list[Shape] = []
        for pos in range(recipe.arity):
            source = args[pos]
            if nodes[source].kind == "wildcard":
                wanted = recipe.wildcard_shapes.get(pos)
                if wanted is None:
                    return GeneratedProgram(
                        status=GenerationStatus.UNSUPPORTED_STRUCTURE,
                        reason=f"no wildcard shape recipe for {node.op} arg {pos}",
                    )
                existing = shapes.get(source)
                if existing is not None and existing != wanted:
                    return GeneratedProgram(
                        status=GenerationStatus.UNSUPPORTED_STRUCTURE,
                        reason=f"wildcard {source} has conflicting shapes",
                    )
                shapes[source] = wanted
                arg_shapes.append(wanted)
            else:
                arg_shapes.append(shapes[source])
        shapes[idx] = recipe.output_shape(arg_shapes)

    input_specs: list[TensorInputSpec] = []
    for wc_idx in pattern.boundary.inputs:
        node = nodes[wc_idx]
        if wc_idx not in shapes:
            return GeneratedProgram(
                status=GenerationStatus.UNSUPPORTED_STRUCTURE,
                reason=f"wildcard input {wc_idx} is not consumed by any op",
            )
        input_specs.append(
            TensorInputSpec(
                name=node.wc or f"in{wc_idx}",
                shape=shapes[wc_idx],
                dtype=dtype,
                device_class=device_class,
            )
        )

    def program_factory() -> Callable[..., object]:
        def program(*inputs: object) -> object:
            value: dict[int, object] = {}
            for wc_idx, tensor in zip(pattern.boundary.inputs, inputs):
                value[wc_idx] = tensor
            for idx in order:
                node = nodes[idx]
                if node.kind != "op":
                    continue
                recipe = _REGISTRY[node.op]
                args = args_by_node[idx]
                call_args = [value[args[pos]] for pos in range(recipe.arity)]
                value[idx] = recipe.torch_fn(*call_args)
            return value[pattern.boundary.output]

        return program

    # Pattern matching stays at the aten level (the pre-optimization FX graph is
    # aten; see effect_probe / pattern_matcher). For graph routing we bind the
    # generated program to the public Python API operators the pattern
    # corresponds to (graph_model_v1): each aten op resolves to its namesake API
    # when available, else keeps the aten operator as fallback.
    resolved_ids = frozenset(api_ids or ())
    bound = {
        resolve_aten_to_api(node.op, resolved_ids) or node.op
        for node in pattern.nodes
        if node.kind == "op" and node.op is not None
    }
    manifest = ProgramManifest.create(
        program_kind=ProgramKind.PYTHON_CALLABLE,
        source_ref=None,
        entrypoint=f"pattern_program/v1:{pattern.canonical_hash}",
        dependencies={},
        input_interface=InputInterface(
            arity=len(input_specs),
            roles=tuple(spec.name for spec in input_specs),
        ),
        required_device_class=device_class,
        phase=ProgramPhase.FORWARD,
        bound_entities=tuple(sorted(bound)),
    )

    return GeneratedProgram(
        status=GenerationStatus.GENERATED,
        program_factory=program_factory,
        input_specs=tuple(input_specs),
        manifest=manifest,
    )
