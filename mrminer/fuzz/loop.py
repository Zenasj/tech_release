"""Property-guided generation, execution, and behavioral feedback."""
from __future__ import annotations

import time
import random
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
    crashes: list[Finding] = field(default_factory=list)
    per_seed: list[dict] = field(default_factory=list)   # seed_id -> stats
    trace: list[dict] = field(default_factory=list)       # per-iteration record
    minimized: list[MinimizeResult] = field(default_factory=list)


def _feedback_for(prev: list[tuple[str, Finding]]) -> str:
    """Summarize prior outcomes for a target into a short hint for the generator."""
    if not prev:
        return ""
    return "\n\n".join(
        f"Attempt {i}:\n{code}\nVerdict: {finding.verdict}\nReason: {finding.detail}"
        for i, (code, finding) in enumerate(prev[-5:], 1)
    )


def fuzz_campaign(
    seeds,
    provider: StructuredLlmProvider,
    index: dict[str, list[dict]],
    *,
    iterations: int = 10,
    max_tests: int | None = None,
    isolate: bool = True,
    minimize_triggers: bool = False,
    run_test: Runner = torch_run_test,
    system_prompt: str | None = None,
    guidance: str = "full",
    feedback: bool = True,
    random_seed: int = 0,
    on_test: Callable[[GeneratedTest, object], None] | None = None,
    on_result: Callable[[GeneratedTest, object, Finding, float], None] | None = None,
) -> CampaignResult:
    """Run a feedback loop over generated tests and executable properties.

    Each round takes the next seed in shuffled order, targets its least-activated
    property, generates a test, and feeds its verdict into subsequent prompts.
    Optional callbacks record generated code and verdicts.
    """
    corpus = Corpus.from_seeds(seeds)
    coverage = Coverage()
    result = CampaignResult()
    history: dict[str, list[tuple[str, Finding]]] = {}
    rng = random.Random(random_seed)
    seed_order = list(corpus.entries)
    rng.shuffle(seed_order)

    for it in range(iterations):
        if max_tests is not None and len(result.findings) >= max_tests:
            break
        if not corpus.entries:
            break
        # Cycle through the fixed seed corpus in random order, as in the paper.
        if it and it % len(seed_order) == 0:
            rng.shuffle(seed_order)
        entry = seed_order[it % len(seed_order)]
        available = oracles_for(entry.seed.ops, index)
        rng.shuffle(available)
        available.sort(key=lambda oracle: coverage.counts.get(oracle.oracle_id, 0))
        oracles = available[:1] if available else []

        round_findings: list[Finding] = []
        gained = 0
        for oracle in oracles:
            key = f"{entry.seed.seed_id}|{oracle.oracle_id}"
            fb_hint = _feedback_for(history.get(key, [])) if feedback else ""
            for test in generate(entry.seed, oracle, provider, feedback=fb_hint,
                                 system_prompt=system_prompt, guidance=guidance):
                if max_tests is not None and len(result.findings) + len(round_findings) >= max_tests:
                    break
                if on_test is not None:
                    on_test(test, oracle)
                _t0 = time.perf_counter()
                f = run_test(test, oracle, isolate=isolate)
                _wall_ms = (time.perf_counter() - _t0) * 1000.0
                if on_result is not None:
                    on_result(test, oracle, f, _wall_ms)
                round_findings.append(f)
                history.setdefault(key, []).append((test.code, f))
                is_new = coverage.observe_one(f)
                gained += int(is_new)
                # a triggered test is a bug candidate -> minimize it to a repro
                if minimize_triggers and f.triggered and f.verdict not in {
                    "compiled_crash", "worker_died", "worker_timeout"}:
                    result.minimized.append(
                        minimize(test.code, oracle, f.verdict, isolate=isolate,
                                 runner=run_test, test_id=test.test_id)
                    )

        crashes = [f for f in round_findings if f.verdict in {
            "compiled_crash", "worker_died", "worker_timeout"}]
        trig = [f for f in round_findings if f.triggered and f not in crashes]
        corpus.update(entry, gained, len(trig), it)

        result.findings.extend(round_findings)
        result.triggers.extend(trig)
        result.crashes.extend(crashes)
        result.trace.append({
            "iter": it, "seed": entry.seed.seed_id,
            "tests": len(round_findings), "new_cov": gained,
            "triggers": len(trig), "crashes": len(crashes),
            "cov_size": coverage.size, "corpus": len(corpus.entries),
        })

    result.coverage_size = coverage.size
    result.per_seed = [
        {"seed": e.seed.seed_id, "runs": e.runs, "new_cov": e.new_cov, "triggers": e.triggers}
        for e in corpus.entries
    ]
    return result
