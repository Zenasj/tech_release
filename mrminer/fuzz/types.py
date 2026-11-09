"""Core data types for the fuzzing agent (M1)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Seed:
    """A starting point for generation: an IR fragment described by the aten ops
    it exercises. May optionally carry the code of a prior interesting test
    (`parent_code`) so the generator can mutate it rather than start cold."""

    seed_id: str
    ops: tuple[str, ...]          # e.g. ("op:aten::__lshift__",)
    note: str = ""
    parent_code: str = ""         # a prior test's code to mutate from (if promoted)
    origin: str = "manual"        # manual | op-bootstrap | promoted


@dataclass(frozen=True)
class OracleRef:
    """The slice of an OracleSpec the agent needs: the target (op), what kind of
    check it is, the variation knob, and the human-readable claim (LLM guidance)."""

    oracle_id: str
    op: str
    aspect: str                   # semantic | effect | decision | performance
    variation: dict               # param_grid, e.g. {"$decomp": ["default", "disabled"]}
    claim: str                    # the PR-grounded intent -> steers generation
    params: dict = field(default_factory=dict)  # relation.params (decision predicates / perf config)


@dataclass(frozen=True)
class GeneratedTest:
    """One test the LLM produced, aimed at triggering `target_oracle`.

    Contract for `code`: a self-contained snippet that defines
        def build():
            return fn, args      # a callable and a tuple of positional args
    importing whatever it needs (torch, etc.)."""

    test_id: str
    seed_id: str
    target_oracle: str
    code: str
    rationale: str = ""


@dataclass(frozen=True)
class Finding:
    """Result of running one test against its target oracle. `triggered=True`
    means the MR-oracle was violated -- a bug candidate."""

    test_id: str
    oracle_id: str
    op: str
    aspect: str
    triggered: bool
    verdict: str                  # e.g. "pass" | "compiled_crash" | "diverged" | "skip:..."
    detail: str = ""
