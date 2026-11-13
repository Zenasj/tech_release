"""Offline checks for the released generation and indexing boundaries."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from importlib.util import spec_from_file_location, module_from_spec

from mrminer.fuzz.generator import generate
from mrminer.fuzz.oracle_index import oracles_for
from mrminer.fuzz.types import OracleRef, Seed
from mrminer.fuzz.types import Finding
from mrminer.fuzz.loop import fuzz_campaign
from mrminer.fuzz.minimize import minimize
from mrminer.llm.provider import FixtureLlmProvider


ROOT = Path(__file__).resolve().parents[1]


def _script(name: str):
    spec = spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
            "property": {"P": "True", "T": "eager_to_xla",
                         "pi": "r.output", "rho": "close(a,b)"},
        }]}
        self.assertEqual(oracles_for(["op:stablehlo::gather"], index), [])
        refs = oracles_for(["op:stablehlo::gather", "op:tf::GatherV2"], index)
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].op, "op:tf::GatherV2")
        self.assertEqual(refs[0].core_operator, "op:stablehlo::gather")
        self.assertEqual(refs[0].property_record["T"], "eager_to_xla")

    def test_specification_inventory_is_unique(self) -> None:
        names = json.loads((ROOT / "configs/stablehlo_spec_ops.json").read_text())["ops"]
        self.assertEqual(len(names), 111)
        self.assertEqual(len(set(names)), 111)

    def test_screening_requires_supporting_pr_regression_test(self) -> None:
        screen = _script("screen_properties")
        item = {"oracle_id": "test:screen", "aspect": "semantic",
                "claim": {"summary": "Value agrees"},
                "release_provenance": {"pr_number": 42},
                "property": {"P": "True", "T": "identity_eager",
                             "pi": "r.output", "rho": "a == b"}}
        index = {"op:aten::abs": [item]}
        def runner(test, oracle, *, isolate):
            return Finding(test.test_id, oracle.oracle_id, oracle.op, oracle.aspect,
                           False, "pass", "")
        with TemporaryDirectory() as tmp:
            audit = Path(tmp) / "audit.jsonl"
            wrong = screen.screen(index, [{"pr_number": 43, "code": "def build(): pass"}],
                                  runner, audit)
            self.assertFalse(wrong)
            right = screen.screen(index, [{"pr_number": 42, "code": "def build(): pass"}],
                                  runner, audit)
            self.assertEqual(len(right["op:aten::abs"]), 1)

    def test_campaign_selects_least_activated_property(self) -> None:
        index = {"op:aten::abs": [
            {"oracle_id": oid, "aspect": "semantic", "claim": {"summary": oid}}
            for oid in ("test:a", "test:b")
        ]}
        seed = Seed(seed_id="seed:abs", ops=("op:aten::abs",), parent_code="def build(): pass")
        provider = FixtureLlmProvider(responses={"fuzz.generate_tests": {
            "tests": [{"code": "def build(): pass"}]}})
        chosen = []
        def runner(test, oracle, *, isolate):
            chosen.append(oracle.oracle_id)
            return Finding(test.test_id, oracle.oracle_id, oracle.op, oracle.aspect,
                           False, "pass", "")
        fuzz_campaign([seed], provider, index, iterations=4, run_test=runner)
        self.assertEqual(len(chosen), 4)
        self.assertEqual(set(chosen[:2]), {"test:a", "test:b"})
        self.assertEqual(set(chosen[2:]), {"test:a", "test:b"})
        self.assertTrue(all("def build(): pass" in str(req.input_payload)
                            for req in provider.requests))
        self.assertTrue(any("Verdict: pass" in str(req.input_payload)
                            for req in provider.requests[2:]))

    def test_named_tensorflow_implementation_file_is_attributed(self) -> None:
        from mrminer.tf import linker_tf
        saved = linker_tf._UNIVERSE
        try:
            linker_tf._UNIVERSE = {"Svd", "GatherV2"}
            pr = {"touched_files_patch": [{"path": "xla/hlo/builder/lib/svd.cc",
                                           "patch": "@@ -1 +1 @@\n-old\n+new"}]}
            self.assertEqual(linker_tf.link_pr(pr), {"Svd": {"implementation_path"}})
        finally:
            linker_tf._UNIVERSE = saved

    def test_online_seed_manifest_keeps_source_program(self) -> None:
        fuzz = _script("fuzz")
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "seeds.jsonl"
            code = "import torch\ndef build():\n    return torch.abs, (torch.ones(1),)"
            path.write_text(json.dumps({"seed_id": "official:test_abs", "ops": ["op:aten::abs"],
                                        "code": code}) + "\n")
            seeds = fuzz.load_seeds(path)
            self.assertEqual(seeds[0].parent_code, code)
            self.assertEqual(seeds[0].ops, ("op:aten::abs",))

    def test_minimization_preserves_input_assignments(self) -> None:
        code = ("import torch\ndef build():\n    x = torch.ones(2)\n"
                "    print('unused')\n    return torch.abs, (x,)")
        oracle = OracleRef("test:min", "op:aten::abs", "semantic", {}, "abs")
        def runner(test, ref, *, isolate):
            return Finding(test.test_id, ref.oracle_id, ref.op, ref.aspect,
                           True, "diverged", "")
        result = minimize(code, oracle, "diverged", runner=runner)
        self.assertTrue(result.confirmed)
        self.assertIn("x = torch.ones(2)", result.minimized_code)
        self.assertNotIn("print('unused')", result.minimized_code)

    def test_property_synthesis_preserves_pr_evidence(self) -> None:
        synth = _script("synthesize_properties")
        index = {"op:aten::abs": [{"oracle_id": "test:pr42", "aspect": "semantic",
                                    "claim": {"summary": "absolute values"},
                                    "release_provenance": {"pr_number": 42}}]}
        provider = FixtureLlmProvider(responses={"property.synthesize": {
            "P": "True", "T": "identity_eager", "pi": "r.output",
            "rho": "close(a,b)", "origin_condition": "", "description": "absolute values"}})
        with TemporaryDirectory() as tmp:
            corpus = Path(tmp)
            (corpus / "42.json").write_text(json.dumps({"title": "Fix abs",
                "body": {"text": "Preserve absolute values"},
                "tests_added": [{"test_name": "test_abs", "added_code": "assert abs(-1)==1"}]}))
            result, audit = synth.synthesize(index, corpus, provider, "pytorch")
        self.assertEqual(audit[0]["status"], "synthesized")
        self.assertEqual(result["op:aten::abs"][0]["property"]["rho"], "close(a,b)")
        self.assertEqual(provider.requests[0].input_payload["evidence"]["pr_number"], 42)


if __name__ == "__main__":
    unittest.main()
