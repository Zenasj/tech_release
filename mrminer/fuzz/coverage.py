"""Oracle coverage signal for the feedback loop (M3).

Coverage = the set of distinct **(operator, oracle)** pairs the campaign has actually
exercised with a *valid* test (i.e. a test that really checked that oracle, not a
skipped one). This is a testing-adequacy metric over our mined MR-oracle library, not
source-code coverage of the compiler: it measures how much of "each operator's oracles"
the campaign has covered, so the fuzzer is rewarded for exercising *all* oracles bound
to an operator. It is interpretable (|covered| / |library|, and per-operator), aligns
with the method's contribution (the operator-bound oracle library), and reads out
directly as a coverage curve.

The oracle's outcome (verdict) is NOT a coverage dimension: it only says whether that
oracle is currently *satisfied* or *violated* (a violation = a finding), and a skipped
verdict means the oracle was not validly exercised, so it does not count as coverage.
Depth within one oracle (exercising its metamorphic variation axes R_I / reaching new
compiler code) is a complementary signal, tracked separately.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from mrminer.fuzz.types import Finding


@dataclass
class Coverage:
    seen: set[tuple[str, str]] = field(default_factory=set)   # (op, oracle_id) validly exercised
    triggers: set[str] = field(default_factory=set)           # oracle_ids that were violated

    def observe_one(self, f: Finding) -> bool:
        """Fold in one finding; return True iff it covered a new (operator, oracle) pair.

        A `skip:*` verdict means the oracle was not validly exercised -> not coverage.
        A triggered finding still records the oracle as violated (for the findings set)."""
        if f.triggered:
            self.triggers.add(f.oracle_id)
        if f.verdict.startswith("skip"):
            return False
        key = (f.op, f.oracle_id)
        new = key not in self.seen
        self.seen.add(key)
        return new

    def observe(self, findings: list[Finding]) -> int:
        """Fold in a batch; return how many *new* (operator, oracle) pairs it covered."""
        return sum(1 for f in findings if self.observe_one(f))

    @property
    def size(self) -> int:
        return len(self.seen)

    def per_operator(self) -> dict[str, int]:
        """op -> number of its oracles covered so far (for per-operator coverage reporting)."""
        by_op: dict[str, int] = {}
        for op, _oid in self.seen:
            by_op[op] = by_op.get(op, 0) + 1
        return by_op
