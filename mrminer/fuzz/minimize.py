"""Minimize reproducing tests while preserving remaining calls and inputs."""
from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass

from mrminer.fuzz.runner import run_test as torch_run_test
from mrminer.fuzz.types import GeneratedTest, OracleRef


@dataclass(frozen=True)
class MinimizeResult:
    oracle_id: str
    verdict: str
    before_lines: int
    after_lines: int
    minimized_code: str
    test_id: str = ""
    confirmed: bool = False


def is_repro(
    code: str, oracle: OracleRef, target_verdict: str, *,
    isolate: bool = True, runner: Callable = torch_run_test,
) -> bool:
    """True iff `code` still triggers `oracle` with `target_verdict`."""
    if "build" not in code:
        return False
    test = GeneratedTest(test_id="min", seed_id="min", target_oracle=oracle.oracle_id, code=code)
    f = runner(test, oracle, isolate=isolate)
    return f.triggered and f.verdict == target_verdict


def ddmin_lines(code: str, predicate: Callable[[list[str]], bool]) -> str:
    """Return a line-minimal subset of `code` still satisfying `predicate`."""
    lines = code.splitlines()

    def test(idxs: list[int]) -> bool:
        return bool(idxs) and predicate([lines[i] for i in idxs])

    idxs = list(range(len(lines)))
    if not test(idxs):
        return code  # full input isn't a repro under this predicate -> leave as-is

    n = 2
    while len(idxs) >= 2:
        chunk = max(1, len(idxs) // n)
        subsets = [idxs[i:i + chunk] for i in range(0, len(idxs), chunk)]
        reduced = False
        for s in subsets:
            drop = set(s)
            comp = [i for i in idxs if i not in drop]
            if test(comp):
                idxs = comp
                n = max(n - 1, 2)
                reduced = True
                break
        if not reduced:
            if n >= len(idxs):
                break
            n = min(len(idxs), n * 2)
    return "\n".join(lines[i] for i in idxs)


def minimize(
    code: str, oracle: OracleRef, verdict: str, *,
    isolate: bool = True, runner: Callable = torch_run_test,
    test_id: str = "",
) -> MinimizeResult:
    """Delete whole standalone calls, preserving input construction and arguments."""
    lines = code.splitlines()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        tree = None
    spans: list[set[int]] = []
    if tree is not None:
        for node in ast.walk(tree):
            if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                    and node.lineno == node.value.lineno
                    and not lines[node.lineno - 1][:node.col_offset].strip()):
                spans.append(set(range(node.lineno - 1, node.end_lineno)))
    kept = set(range(len(lines)))
    for span in spans:
        candidate = kept - span
        reduced = "\n".join(lines[i] for i in sorted(candidate))
        if is_repro(reduced, oracle, verdict, isolate=isolate, runner=runner):
            kept = candidate
    mini = "\n".join(lines[i] for i in sorted(kept))
    confirmed = is_repro(mini, oracle, verdict, isolate=isolate, runner=runner)
    return MinimizeResult(
        oracle_id=oracle.oracle_id,
        verdict=verdict,
        before_lines=len(code.splitlines()),
        after_lines=len(mini.splitlines()),
        minimized_code=mini,
        test_id=test_id,
        confirmed=confirmed,
    )
