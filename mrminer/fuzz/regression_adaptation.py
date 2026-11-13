"""Code validation and model prompts for PR-grounded screening inputs."""
from __future__ import annotations

import ast


SCHEMA = {
    "type": "object",
    "properties": {"code": {"type": "string"}},
    "required": ["code"],
    "additionalProperties": False,
}

ADAPT_PROMPT = """\
Adapt the supplied PR regression test to the screening runner. Return one JSON
object containing executable Python code. Preserve the original test's input
construction, operator arguments, and behavior relevant to its assertion. Do
not invent a new test or alter a boundary value to make the property pass.
Define build() returning (fn, args), where fn returns the marked call's result.
The runner supplies mark(source_operator_id, callable, *args, **kwargs).
Use the supplied source operator ID exactly as the first mark argument.
Put framework imports at module level. Do not run unittest or pytest at import
time, and do not compile the call yourself; the runner handles execution modes.
The collected added_code is a diff excerpt, so use its test name, test path,
and patch context to reconstruct only the behavior supported by the PR.
"""

FALLBACK_PROMPT = """\
Construct one executable screening input for the supplied property from its
supporting PR evidence. Use the PR's changed behavior, condition, and concrete
inputs when available. Do not infer a behavior from the operator name alone.
Return a JSON object containing code. Define build() returning (fn, args),
where fn returns the marked source call. The runner supplies
mark(source_operator_id, callable, *args, **kwargs). Use the supplied source
operator ID exactly as the first mark argument. Import the framework at module
level. Do not compile the call yourself; the runner handles execution modes.
"""


def validate_code(code: str, source: str) -> None:
    tree = ast.parse(code)
    builds = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
              and node.name == "build"]
    if len(builds) != 1 or isinstance(builds[0], ast.AsyncFunctionDef):
        raise ValueError("expected one top-level synchronous build()")
    marks = [node for node in ast.walk(builds[0]) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "mark"]
    if not marks:
        raise ValueError("build() has no marked call")
    if any(not node.args or not isinstance(node.args[0], ast.Constant)
           or node.args[0].value != source for node in marks):
        raise ValueError("marked source operator does not match property source")
    if not any(isinstance(node, ast.Return) for node in builds[0].body):
        raise ValueError("build() does not return (fn, args)")
    compile(tree, "<screening-input>", "exec")
