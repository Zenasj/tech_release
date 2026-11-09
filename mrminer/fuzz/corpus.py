"""Seed corpus + scheduler for the feedback loop (M3).

Greybox-style: each seed accumulates stats (runs, new coverage it produced,
triggers it found); the scheduler favors seeds that recently produced new
coverage or triggers, while still exploring under-run seeds. Interesting tests
can be promoted to new seeds (a hook the campaign uses).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from mrminer.fuzz.types import Seed


@dataclass
class SeedEntry:
    seed: Seed
    runs: int = 0
    new_cov: int = 0          # cumulative new-coverage signals attributed to this seed
    triggers: int = 0
    last_new_cov_run: int = -1

    def energy(self) -> float:
        """Scheduling weight: reward productivity, discount how often already run."""
        base = 1.0 + 3.0 * self.new_cov + 6.0 * self.triggers
        return base / (1.0 + self.runs)


@dataclass
class Corpus:
    entries: list[SeedEntry] = field(default_factory=list)
    _ids: set[str] = field(default_factory=set)
    rng: random.Random = field(default_factory=lambda: random.Random(0))

    @classmethod
    def from_seeds(cls, seeds) -> "Corpus":
        c = cls()
        for s in seeds:
            c.add(s)
        return c

    def add(self, seed: Seed) -> bool:
        if seed.seed_id in self._ids:
            return False
        self.entries.append(SeedEntry(seed=seed))
        self._ids.add(seed.seed_id)
        return True

    def select(self) -> SeedEntry:
        """Weighted-random pick by energy (favors productive + under-run seeds).
        Greybox scheduler, kept for callers that want energy-based seed scheduling."""
        weights = [e.energy() for e in self.entries]
        return self.rng.choices(self.entries, weights=weights, k=1)[0]

    def select_uniform(self) -> SeedEntry:
        """Uniform-random pick. Design B (§4.3): seed selection is NOT coverage-guided --
        the seed only supplies a set of operators to test; the coverage guidance lives in
        which (op, property) target is then chosen (see _select_oracles)."""
        return self.rng.choice(self.entries)

    def update(self, entry: SeedEntry, gained_cov: int, triggers: int, run_idx: int) -> None:
        entry.runs += 1
        entry.new_cov += gained_cov
        entry.triggers += triggers
        if gained_cov > 0:
            entry.last_new_cov_run = run_idx
