"""Offline checks for the released generation and indexing boundaries."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from mrminer.fuzz.generator import generate
from mrminer.fuzz.oracle_index import oracles_for
from mrminer.fuzz.types import OracleRef, Seed
from mrminer.llm.provider import FixtureLlmProvider


ROOT = Path(__file__).resolve().parents[1]


class ReleaseSmokeTest(unittest.TestCase):
    def test_missing_framework_import_is_repaired(self) -> None:
        oracle = OracleRef(oracle_id="test:one", op="op:aten::full_like",
                           aspect="semantic", variation={}, claim="Preserve strides")
        seed = Seed(seed_id="seed:one", ops=(oracle.op,))
        provider = FixtureLlmProvider(responses={
            "fuzz.generate_tests": {"tests": [{"code":
                "def build():\n    x = torch.ones(2)\n    return lambda y: torch.full_like(y, 1), (x,)"}]}
        })
        tests = generate(seed, oracle, provider)
        self.assertEqual(len(tests), 1)
        self.assertTrue(tests[0].code.startswith("import torch\n"))
        compile(tests[0].code, "<generated-test>", "exec")

    def test_stablehlo_source_context_is_retrievable(self) -> None:
        index = {"op:stablehlo::gather": [{
            "oracle_id": "test:gather", "aspect": "semantic",
            "guard": {"static": {"source_entry": {"operator": "op:tf::GatherV2"}}},
            "claim": {"summary": "Gathered values agree"},
            "variation": {"param_grid": {"$exec_mode": ["eager", "xla"]}},
        }]}
        self.assertEqual(oracles_for(["op:stablehlo::gather"], index), [])
        refs = oracles_for(["op:stablehlo::gather", "op:tf::GatherV2"], index)
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].op, "op:tf::GatherV2")

    def test_specification_inventory_is_unique(self) -> None:
        names = json.loads((ROOT / "configs/stablehlo_spec_ops.json").read_text())["ops"]
        self.assertEqual(len(names), 111)
        self.assertEqual(len(set(names)), 111)


if __name__ == "__main__":
    unittest.main()
