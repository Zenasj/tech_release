"""Seed sources for the campaign (M4).

Bootstrap a real initial corpus from the oracle library: every operator that
carries at least one oracle becomes a seed exercising that operator. (Other
sources -- official test programs, the private model zoo, hand-written IR
fragments -- plug in the same way by emitting `Seed`s; index-bootstrap is the
one implemented here.)
"""
from __future__ import annotations

from mrminer.fuzz.types import Seed


def seeds_from_index(index: dict[str, list[dict]], *, limit: int | None = None) -> list[Seed]:
    ops = sorted(index.keys())
    if limit is not None:
        ops = ops[:limit]
    return [
        Seed(seed_id=f"seed:op:{op.split('::')[-1]}", ops=(op,), origin="op-bootstrap")
        for op in ops
    ]
