"""Property-guided generation, execution, and behavioral feedback."""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from mrminer.fuzz.corpus import Corpus
from mrminer.fuzz.coverage import Coverage
from mrminer.fuzz.generator import generate
from mrminer.fuzz.minimize import MinimizeResult, minimize
from mrminer.fuzz.oracle_index import oracles_for
from mrminer.fuzz.runner import run_test as torch_run_test
from mrminer.fuzz.types import Finding, GeneratedTest, Seed
from mrminer.llm.provider import StructuredLlmProvider


def _aspect_round_robin(oracles, max_oracles, picked):
    """Append from `oracles` into `picked` (up to max_oracles) round-robin across aspects,
    so no aspect a seed touches is starved. oracles_for returns index order (semantic,
    effect, performance, decision); a plain [:max_oracles] would never reach decision/
    performance for ops that also carry semantic oracles."""
    buckets: dict[str, list] = {}
    for o in oracles:
        buckets.setdefault(o.aspect, []).append(o)
    while len(picked) < max_oracles and any(buckets.values()):
        for lst in buckets.values():
            if lst and len(picked) < max_oracles:
                picked.append(lst.pop(0))


def _select_oracles(oracles, max_oracles, coverage=None, feedback=True):
    """Prefer uncovered (operator, property) targets while balancing aspects."""
    if max_oracles is None or len(oracles) <= max_oracles:
        return oracles
    picked: list = []
    if feedback and coverage is not None:
        seen = coverage.seen
        uncovered = [o for o in oracles if (o.op, o.oracle_id) not in seen]
        covered = [o for o in oracles if (o.op, o.oracle_id) in seen]
        _aspect_round_robin(uncovered, max_oracles, picked)   # coverage-guided: uncovered first
        _aspect_round_robin(covered, max_oracles, picked)     # fill remainder if budget left
    else:
        _aspect_round_robin(oracles, max_oracles, picked)
    return picked


# A runner is the pluggable execution boundary: (test, oracle) -> Finding. The
# PyTorch line passes mrminer.fuzz.runner.run_test (eager vs inductor); the TF
# line passes mrminer.tf.runner_tf.run_test (eager vs XLA). Both share this loop,
# coverage, corpus, and ddmin -- only the runner and the generator prompt differ.
Runner = Callable[..., Finding]


def fuzz_seed(
    seed: Seed,
    provider: StructuredLlmProvider,
    index: dict[str, list[dict]],
    *,
    max_oracles: int | None = None,
    isolate: bool = True,
    run_test: Runner = torch_run_test,
    system_prompt: str | None = None,
    guidance: str = "full",
) -> list[Finding]:
    """Run one seed end to end and return every test's finding."""
    oracles = _select_oracles(oracles_for(seed.ops, index), max_oracles)
    findings: list[Finding] = []
    for oracle in oracles:
        for test in generate(seed, oracle, provider, system_prompt=system_prompt,
                             guidance=guidance):
            findings.append(run_test(test, oracle, isolate=isolate))
    return findings


@dataclass
class CampaignResult:
    findings: list[Finding] = field(default_factory=list)
    coverage_size: int = 0
    triggers: list[Finding] = field(default_factory=list)
    per_seed: list[dict] = field(default_factory=list)   # seed_id -> stats
    trace: list[dict] = field(default_factory=list)       # per-iteration record
    minimized: list[MinimizeResult] = field(default_factory=list)


def _feedback_for(prev: list[Finding]) -> str:
    """Summarize prior outcomes for a target into a short hint for the generator."""
    if not prev:
        return ""
    verdicts = sorted({f.verdict for f in prev})
    if any(f.triggered for f in prev):
        return ""  # already triggered -- no need to steer
    return (
        "previous attempts on this target did not trigger the oracle "
        f"(verdicts: {', '.join(verdicts)}); try a different input domain / edge case."
    )


def fuzz_campaign(
    seeds,
    provider: StructuredLlmProvider,
    index: dict[str, list[dict]],
    *,
    iterations: int = 10,
    max_oracles: int | None = 3,
    isolate: bool = True,
    grow: bool = True,
    minimize_triggers: bool = False,
    run_test: Runner = torch_run_test,
    system_prompt: str | None = None,
    guidance: str = "full",
    feedback: bool = True,
    on_test: Callable[[GeneratedTest, object], None] | None = None,
    on_result: Callable[[GeneratedTest, object, Finding, float], None] | None = None,
) -> CampaignResult:
    """Run a feedback loop over generated tests and executable properties.

    Each round samples a seed, prioritizes uncovered properties, generates tests,
    executes them, and uses verdicts in subsequent prompts. Tests that activate
    new properties or violate an oracle can become new seeds. ``guidance`` controls
    property detail in prompts; ``feedback`` controls target prioritization, prompt
    hints, and seed promotion. Optional callbacks record generated code and verdicts.
    """
    corpus = Corpus.from_seeds(seeds)
    coverage = Coverage()
    result = CampaignResult()
    history: dict[str, list[Finding]] = {}   # (seed_id|oracle_id) -> prior findings

    for it in range(iterations):
        if not corpus.entries:
            break
        # Sample a seed uniformly and prioritize uncovered properties for that seed.
        entry = corpus.select_uniform()
        oracles = _select_oracles(oracles_for(entry.seed.ops, index), max_oracles,
                                  coverage=coverage, feedback=feedback)

        round_findings: list[Finding] = []
        gained = 0
        promoted = 0
        for oracle in oracles:
            key = f"{entry.seed.seed_id}|{oracle.oracle_id}"
            fb_hint = _feedback_for(history.get(key, [])) if feedback else ""
            for test in generate(entry.seed, oracle, provider, feedback=fb_hint,
                                 system_prompt=system_prompt, guidance=guidance):
                if on_test is not None:
                    on_test(test, oracle)
                _t0 = time.perf_counter()
                f = run_test(test, oracle, isolate=isolate)
                _wall_ms = (time.perf_counter() - _t0) * 1000.0
                if on_result is not None:
                    on_result(test, oracle, f, _wall_ms)
                round_findings.append(f)
                history.setdefault(key, []).append(f)
                is_new = coverage.observe_one(f)
                gained += int(is_new)
                # a triggered test is a bug candidate -> minimize it to a repro
                if minimize_triggers and f.triggered:
                    result.minimized.append(
                        minimize(test.code, oracle, f.verdict, isolate=isolate,
                                 runner=run_test, test_id=test.test_id)
                    )
                # promote interesting tests to new (mutatable) seeds.
                # feedback OFF (A5): corpus stays fixed -- promotion is part of the loop.
                if grow and feedback and (f.triggered or is_new):
                    promo = Seed(
                        seed_id=f"seed:promoted:{test.test_id}",
                        ops=(entry.seed.ops if oracle.op not in index
                             or any(op.startswith("op:stablehlo::") for op in entry.seed.ops)
                             else (oracle.op,)),
                        note=f"promoted from {entry.seed.seed_id}",
                        parent_code=test.code,
                        origin="promoted",
                    )
                    promoted += int(corpus.add(promo))

        trig = [f for f in round_findings if f.triggered]
        corpus.update(entry, gained, len(trig), it)

        result.findings.extend(round_findings)
        result.triggers.extend(trig)
        result.trace.append({
            "iter": it, "seed": entry.seed.seed_id,
            "tests": len(round_findings), "new_cov": gained,
            "triggers": len(trig), "promoted": promoted,
            "cov_size": coverage.size, "corpus": len(corpus.entries),
        })

    result.coverage_size = coverage.size
    result.per_seed = [
        {"seed": e.seed.seed_id, "runs": e.runs, "new_cov": e.new_cov, "triggers": e.triggers}
        for e in corpus.entries
    ]
    return result
