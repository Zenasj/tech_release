"""Executable P, T, projection, and relation for an operator property.

Expressions are evaluated only in an isolated test worker. The checker has
three outcomes: Skip if P is false or either projection is undefined, Pass if
the relation holds, and Fail otherwise. A malformed checker is reported as an
error and cannot pass screening.
"""
from __future__ import annotations

import ast
import builtins
import math
from dataclasses import dataclass
from typing import Any


class UndefinedProjection(Exception):
    pass


def undefined() -> None:
    raise UndefinedProjection()


def close(a: Any, b: Any, *, rtol: float = 1e-4, atol: float = 1e-5) -> bool:
    """Compare nested numeric results without hiding shape or dtype differences."""
    import numpy as np

    if isinstance(a, (tuple, list)) or isinstance(b, (tuple, list)):
        return (type(a) is type(b) and len(a) == len(b)
                and all(close(x, y, rtol=rtol, atol=atol) for x, y in zip(a, b)))
    def as_array(value):
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        if hasattr(value, "numpy"):
            return value.numpy()
        return value

    a, b = as_array(a), as_array(b)
    aa, bb = np.asarray(a), np.asarray(b)
    return (aa.shape == bb.shape and aa.dtype == bb.dtype
            and bool(np.allclose(aa, bb, rtol=rtol, atol=atol, equal_nan=True)))


@dataclass(frozen=True)
class Context:
    operator: str
    source_operator: str
    site: str
    call_id: int
    args: tuple
    kwargs: dict
    output_path: tuple[str | int, ...] | None
    inferred: bool = False
    site_args: tuple = ()
    site_kwargs: dict | None = None
    site_metadata: dict | None = None


@dataclass(frozen=True)
class ExecutionResult:
    kind: str  # return | raise | abort
    output: Any = None
    exception: str = ""
    post_state: Any = None
    exception_traceback: str = ""

    def output_at(self, path: tuple[str | int, ...] | None) -> Any:
        if self.kind != "return" or path is None:
            return undefined()
        value = self.output
        for part in path:
            value = value[part]
        return value


_BUILTINS = {name: getattr(builtins, name) for name in (
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "int",
    "isinstance", "len", "list", "max", "min", "range", "round", "set",
    "str", "sum", "tuple", "zip",
)}


def _compile(expr: str, field: str):
    if not isinstance(expr, str) or not expr.strip():
        raise ValueError(f"property {field} must be a nonempty expression")
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError(f"property {field} cannot use private names")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError(f"property {field} cannot use private attributes")
    return compile(tree, f"<property.{field}>", "eval")


class PropertyChecker:
    def __init__(self, record: dict, *, framework: str):
        self.record = record
        self.framework = framework
        self.p = _compile(record["P"], "P")
        self.pi = _compile(record["pi"], "pi")
        self.rho = _compile(record["rho"], "rho")
        self.origin_condition = (_compile(record["origin_condition"], "origin_condition")
                                 if record.get("origin_condition") else None)
        allowed = {"pytorch": {"eager_to_inductor", "identity_eager", "identity_compiled"},
                   "tensorflow": {"eager_to_xla", "identity_eager", "identity_compiled"}}
        if record.get("T") not in allowed[framework]:
            raise ValueError(f"unsupported T for {framework}: {record.get('T')}")

    def _eval(self, code, **values):
        import numpy as np
        namespace = {"__builtins__": _BUILTINS, "math": math,
                     "np": np, "close": close, "undefined": undefined}
        if self.framework == "pytorch":
            import torch
            namespace["torch"] = torch
        else:
            import tensorflow as tf
            namespace["tf"] = tf
        return eval(code, namespace, values)  # noqa: S307 -- isolated worker

    def origin_matches(self, context: Context) -> bool:
        return bool(self.origin_condition and self._eval(self.origin_condition, c=context))

    def check(self, context: Context, baseline: ExecutionResult,
              followup: ExecutionResult, *, ignore_condition: bool = False
              ) -> tuple[str, str]:
        if not ignore_condition:
            try:
                if not bool(self._eval(self.p, c=context)):
                    return "skip:P", "applicability condition is false"
            except Exception as error:
                return "skip:checker_error", f"P: {type(error).__name__}: {error}"
        values = []
        for side, result in (("baseline", baseline), ("followup", followup)):
            try:
                if result.kind == "abort":
                    raise UndefinedProjection()
                values.append(self._eval(self.pi, c=context, r=result))
            except UndefinedProjection:
                return f"skip:pi_{side}", f"projection undefined on {side}"
            except Exception as error:
                return "skip:checker_error", f"pi/{side}: {type(error).__name__}: {error}"
        try:
            holds = self._eval(self.rho, a=values[0], b=values[1])
            import numpy as np
            if not isinstance(holds, (bool, np.bool_)):
                return "skip:checker_error", "rho did not return a Boolean"
            return ("pass", "relation holds") if bool(holds) else ("fail", "relation violated")
        except Exception as error:
            return "skip:checker_error", f"rho: {type(error).__name__}: {error}"
