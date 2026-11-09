"""Operator relation graph for evidence propagation during attribution.

Many aten operators share internal implementation: one operator's decomposition /
reference calls another (e.g. `ones_like` -> `full_like` -> `empty_like`). A PR that
changes operator B therefore also constrains B's callers, yet code-location attribution
(mrminer.s2_graph.linker) attributes the change only to the operator named at the edited
site. This module recovers the call/decomposition relation statically from the framework
sources and lets attribution expand a PR's directly-touched operators with their related
(caller) operators as *candidates*; whether a recovered property actually applies to a
related operator is decided downstream by property recovery/screening (Design §4.2).

Edges are `caller_op -> callee_op`: the registered decomposition/reference function of
`caller_op` calls `callee_op` in its body. Propagation uses the REVERSE direction: an
edit to `callee_op` is a candidate for its callers (dependents).
"""
from __future__ import annotations

import ast
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

# Source files whose top-level functions are registered as an operator's
# decomposition / reference / lowering. Relative to the torch package root.
_DECOMP_SOURCES = (
    "_refs/__init__.py",
    "_decomp/decompositions.py",
    "_inductor/decomposition.py",
    "_meta_registrations.py",
)

# Registration / wrapper helpers that appear as decorators or utility calls but
# are not aten operators. They must never enter the operator vocabulary, or they
# pollute both the relation edges and the caller out-degree used for gating.
_NON_OPS = frozenset({
    "register_decomposition", "register_meta", "register_lowering",
    "register_decomposition_for_jvp", "out_wrapper", "pw_cast_for_opmath",
    "pw_cast_for_int_to_real", "elementwise_type_promotion_wrapper",
    "_make_elementwise_unary_reference", "_make_elementwise_binary_reference",
})


def _registered_op(fn: ast.FunctionDef) -> str | None:
    """The operator a function is registered for: @register_decomposition(aten.X) /
    @register_meta(aten.X) / @register_lowering(aten.X), else the function name itself
    (refs/decompositions name the function after the operator)."""
    for dec in fn.decorator_list:
        if not isinstance(dec, ast.Call):
            continue
        name = dec.func.attr if isinstance(dec.func, ast.Attribute) else getattr(dec.func, "id", "")
        if name not in ("register_decomposition", "register_meta", "register_lowering",
                        "register_decomposition_for_jvp"):
            continue
        for arg in dec.args:
            op = _op_from_ref(arg)
            if op and op not in _NON_OPS:
                return op
    return fn.name


def _op_from_ref(node: ast.AST) -> str | None:
    """Resolve `aten.X`, `aten.X.default`, `torch.ops.aten.X`, `prims.X`, or a bare `X`
    to the operator short name `X`."""
    if isinstance(node, ast.Attribute):
        # peel .default / .Tensor overloads, then take the leaf before an aten/prims ns
        parts = []
        cur: ast.AST = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        base = getattr(cur, "id", "")
        parts.append(base)
        parts.reverse()  # e.g. ['aten','full_like','default'] or ['torch','full_like']
        for i, p in enumerate(parts):
            if p in ("aten", "prims") and i + 1 < len(parts):
                return parts[i + 1]
        if parts and parts[0] in ("torch",) and len(parts) >= 2:
            return parts[1]
        return None
    if isinstance(node, ast.Name):
        return node.id
    return None


def _called_ops(fn: ast.FunctionDef, vocab: set[str]) -> set[str]:
    """Operators called in a function body: torch.X(...), aten.X(...), prims.X(...),
    or a bare X(...) when X is a known operator name. Only the function BODY is
    inspected: the decorator list (e.g. @register_decomposition(aten.X)) is skipped,
    otherwise the registration decorator would be miscounted as a call."""
    out: set[str] = set()
    for stmt in fn.body:
        for node in ast.walk(stmt):
            if not isinstance(node, ast.Call):
                continue
            op = _op_from_ref(node.func)
            if op and op in vocab and op != fn.name and op not in _NON_OPS:
                out.add(op)
    return out


@dataclass
class OpRelationGraph:
    """Static operator relation graph recovered from the framework sources.

    `callers` is the reverse adjacency `callee_op -> {caller_ops}` (an edit to a callee
    is a candidate for its callers). `outdeg` is `op -> {ops it calls}`; its size is the
    caller's out-degree, used to keep propagation to pure one-to-one delegations."""
    callers: dict[str, set[str]] = field(default_factory=dict)
    outdeg: dict[str, set[str]] = field(default_factory=dict)


def build_op_relation_graph(torch_root: str | Path) -> OpRelationGraph:
    """Build the operator relation graph from the framework decomposition/reference
    sources: an edge `caller_op -> callee_op` means the registered function of
    `caller_op` calls `callee_op` in its body."""
    torch_root = Path(torch_root)
    fn_ops: list[tuple[str, ast.FunctionDef]] = []
    vocab: set[str] = set()
    for rel in _DECOMP_SOURCES:
        p = torch_root / rel
        if not p.exists():
            continue
        try:
            tree = ast.parse(p.read_text(), filename=str(p))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                op = _registered_op(node)
                if op and op not in _NON_OPS:
                    fn_ops.append((op, node))
                    vocab.add(op)
    callers: dict[str, set[str]] = defaultdict(set)  # callee -> {callers}
    outdeg: dict[str, set[str]] = defaultdict(set)   # caller -> {callees}
    for caller_op, fn in fn_ops:
        for callee in _called_ops(fn, vocab):
            callers[callee].add(caller_op)
            outdeg[caller_op].add(callee)
    return OpRelationGraph(callers=dict(callers), outdeg=dict(outdeg))


# A change to a high-fanout utility operator (e.g. empty, where, called by dozens of
# decompositions) should not be propagated to all of its callers, or it floods the
# candidate set. Operators with more than MAX_FANOUT direct callers are not propagated
# FROM (their own direct attribution is unaffected). This is a secondary backstop; the
# primary gate is the one-to-one delegation rule below.
MAX_FANOUT = 40
# Only propagate B -> A when A calls exactly this many distinct operators. A caller that
# forwards to a single operator (A := wrapper around B, e.g. ones_like := full_like(.,1))
# shares B's semantics, so B's property very likely transfers. A caller that composes
# several operators (one-to-many) merely happens to use B; its property is not implied.
MAX_CALLER_OUTDEG = 1


def related_operators(ops, graph: OpRelationGraph, hops: int = 1,
                      max_fanout: int = MAX_FANOUT,
                      max_caller_outdeg: int = MAX_CALLER_OUTDEG) -> set[str]:
    """Operators related to `ops` by up to `hops` reverse (caller) steps, excluding the
    inputs. A change to operator B is a candidate for caller A only when A is a pure
    one-to-one delegation (A's out-degree <= `max_caller_outdeg`, i.e. A forwards to a
    single operator); one-to-many composers are not followed. Propagation FROM an operator
    with more than `max_fanout` callers is additionally skipped (fanout backstop). These are
    attribution CANDIDATES; applicability is decided downstream by property recovery (§4.2)."""
    callers, outdeg = graph.callers, graph.outdeg
    seen = set(ops)
    frontier = deque((o, 0) for o in ops)
    out: set[str] = set()
    while frontier:
        op, d = frontier.popleft()
        if d >= hops:
            continue
        op_callers = callers.get(op, ())
        if len(op_callers) > max_fanout:          # high-fanout utility op: do not propagate
            continue
        for caller in op_callers:                 # ops that call `op` -> affected by its change
            if caller in seen:
                continue
            if len(outdeg.get(caller, ())) > max_caller_outdeg:  # one-to-many composer: skip
                continue
            seen.add(caller)
            out.add(caller)
            frontier.append((caller, d + 1))
    return out


def op_short_name(op_key: str) -> str:
    """`op:aten::full_like` / `op:aten::full_like.default` -> `full_like`."""
    tail = op_key.rsplit("::", 1)[-1]
    return tail.split(".", 1)[0]


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "torch"
    g = build_op_relation_graph(root)
    print(f"[op-relation] built from {root}: {len(g.callers)} operators have callers")
    for seed in ("full_like", "empty_like", "empty_strided", "fill"):
        rel = related_operators({seed}, g, hops=1)
        print(f"  edit {seed:14s} -> one-to-one delegating callers: "
              f"{sorted(rel)[:12]}{' …' if len(rel) > 12 else ''}")
