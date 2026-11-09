"""Compiler-independent matching for restricted PatternSpec runtime graphs."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Literal

from mrminer.common.schema import PatternConstraint, PatternNode, PatternSpec

RuntimeNodeKind = Literal["placeholder", "get_attr", "op"]
RuntimeValueKind = Literal["tensor", "scalar", "unknown"]


@dataclass(frozen=True, kw_only=True)
class RuntimeGraphNode:
    node_id: int
    kind: RuntimeNodeKind
    value_kind: RuntimeValueKind
    op: str | None = None
    overload: str | None = None
    arg_inputs: tuple[tuple[int, ...], ...] = ()
    dtype: str | None = None
    device: str | None = None

    def __post_init__(self) -> None:
        if self.node_id < 0:
            raise ValueError("runtime node_id must be non-negative")
        if self.kind == "op":
            if self.op is None or not self.op.startswith("op:"):
                raise ValueError("runtime op nodes require an operator entity")
            if not self.overload:
                raise ValueError("runtime op nodes require an overload")
        elif self.op is not None or self.overload is not None:
            raise ValueError("non-op runtime nodes cannot define op or overload")
        for dependencies in self.arg_inputs:
            if dependencies != tuple(sorted(set(dependencies))):
                raise ValueError(
                    "runtime argument dependencies must be sorted and unique"
                )
            if any(item < 0 for item in dependencies):
                raise ValueError("runtime dependencies must be non-negative")


@dataclass(frozen=True, kw_only=True)
class RuntimeGraph:
    nodes: tuple[RuntimeGraphNode, ...]

    def __post_init__(self) -> None:
        node_ids = tuple(item.node_id for item in self.nodes)
        if not node_ids or node_ids != tuple(sorted(set(node_ids))):
            raise ValueError(
                "runtime graph nodes must be non-empty with sorted unique IDs"
            )
        available = set(node_ids)
        for node in self.nodes:
            referenced = {
                dependency
                for dependencies in node.arg_inputs
                for dependency in dependencies
            }
            missing = referenced - available
            if missing:
                raise ValueError(
                    f"runtime node {node.node_id} references missing nodes: "
                    f"{sorted(missing)}"
                )

    def by_id(self) -> dict[int, RuntimeGraphNode]:
        return {item.node_id: item for item in self.nodes}


@dataclass(frozen=True, kw_only=True)
class PatternMatch:
    bindings: tuple[tuple[int, int], ...]

    def runtime_node(self, pattern_node: int) -> int:
        for pattern_id, runtime_id in self.bindings:
            if pattern_id == pattern_node:
                return runtime_id
        raise KeyError(pattern_node)


class RestrictedPatternMatcher:
    """Deterministic subgraph matcher for the frozen restricted_v1 grammar."""

    @staticmethod
    def _node_compatible(
        pattern: PatternNode,
        runtime: RuntimeGraphNode,
    ) -> bool:
        if pattern.kind == "op":
            return (
                runtime.kind == "op"
                and runtime.op == pattern.op
                and runtime.overload == pattern.overload
            )
        if pattern.type_constraint == "tensor":
            return runtime.value_kind == "tensor"
        if pattern.type_constraint == "scalar":
            return runtime.value_kind == "scalar"
        return False

    @staticmethod
    def _edge_holds(
        *,
        source_runtime: int,
        target_runtime: RuntimeGraphNode,
        arg_pos: int,
    ) -> bool:
        return (
            arg_pos < len(target_runtime.arg_inputs)
            and source_runtime in target_runtime.arg_inputs[arg_pos]
        )

    @staticmethod
    def _constraint_holds(
        constraint: PatternConstraint,
        output: RuntimeGraphNode,
    ) -> bool:
        observed = (
            output.dtype
            if constraint.dimension == "dtype"
            else output.device
        )
        if observed is None:
            return False
        if constraint.operator == "eq":
            return observed == constraint.values[0]
        return observed in constraint.values

    def match(
        self,
        pattern: PatternSpec,
        graph: RuntimeGraph,
    ) -> tuple[PatternMatch, ...]:
        pattern_nodes = {item.idx: item for item in pattern.nodes}
        runtime_nodes = graph.by_id()
        candidates = {
            pattern_id: tuple(
                runtime.node_id
                for runtime in graph.nodes
                if self._node_compatible(pattern_node, runtime)
            )
            for pattern_id, pattern_node in pattern_nodes.items()
        }
        if any(not values for values in candidates.values()):
            return ()

        order = tuple(
            sorted(
                pattern.canonical_node_order,
                key=lambda item: (
                    len(candidates[item]),
                    pattern.canonical_node_order.index(item),
                ),
            )
        )
        bindings: dict[int, int] = {}
        used_runtime: set[int] = set()
        matches: list[PatternMatch] = []

        def partial_edges_hold() -> bool:
            for edge in pattern.edges:
                source = bindings.get(edge.from_idx)
                target = bindings.get(edge.to_idx)
                if source is None or target is None:
                    continue
                if not self._edge_holds(
                    source_runtime=source,
                    target_runtime=runtime_nodes[target],
                    arg_pos=edge.arg_pos,
                ):
                    return False
            return True

        def visit(position: int) -> None:
            if position == len(order):
                output_id = bindings[pattern.boundary.output]
                output = runtime_nodes[output_id]
                if not all(
                    self._constraint_holds(item, output)
                    for item in pattern.constraints
                ):
                    return
                matches.append(
                    PatternMatch(bindings=tuple(sorted(bindings.items())))
                )
                return
            pattern_id = order[position]
            for runtime_id in candidates[pattern_id]:
                if runtime_id in used_runtime:
                    continue
                bindings[pattern_id] = runtime_id
                used_runtime.add(runtime_id)
                if partial_edges_hold():
                    visit(position + 1)
                used_runtime.remove(runtime_id)
                del bindings[pattern_id]

        visit(0)
        return tuple(sorted(set(matches), key=lambda item: item.bindings))

    def count(self, pattern: PatternSpec, graph: RuntimeGraph) -> int:
        return len(self.match(pattern, graph))


def runtime_node_fields() -> tuple[str, ...]:
    """Expose the accepted fact surface for audit tests and adapters."""

    return tuple(item.name for item in fields(RuntimeGraphNode))
