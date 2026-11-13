"""Check the PR-test adaptation boundary without a model or compiler."""
from __future__ import annotations

import json
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from tempfile import TemporaryDirectory

from mrminer.llm.provider import FixtureLlmProvider
from mrminer.fuzz.types import Finding


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_regression_witnesses.py"
SPEC = spec_from_file_location("prepare_regression_witnesses", SCRIPT)
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
SCREEN_SPEC = spec_from_file_location(
    "screen_properties", SCRIPT.parent / "screen_properties.py")
SCREEN = module_from_spec(SCREEN_SPEC)
SCREEN_SPEC.loader.exec_module(SCREEN)


class RegressionWitnessPreparationTest(unittest.TestCase):
    def test_collected_test_is_adapted_with_source_audit(self):
        code = ("import torch\n"
                "def build():\n"
                "    x = torch.ones(2, 3).t()\n"
                "    def fn(a):\n"
                "        return mark('op:aten::full_like', torch.full_like, a, 1)\n"
                "    return fn, (x,)\n")
        provider = FixtureLlmProvider(responses={"regression.adapt": {"code": code}})
        index = {"op:aten::full_like": [{
            "oracle_id": "pr:42:stride",
            "release_provenance": {"pr_number": 42},
            "guard": {"static": {"source_entry": {"operator": "op:aten::full_like"}}},
            "property": {"P": "True", "T": "identity_eager",
                         "pi": "r.output", "rho": "close(a,b)"},
        }]}
        added = "def test_full_like_stride():\n    x = torch.ones(2, 3).t()"
        with TemporaryDirectory() as tmp:
            corpus = Path(tmp)
            (corpus / "42.json").write_text(json.dumps({
                "title": "Preserve full_like strides",
                "tests_added": [{"path": "test/test_ops.py",
                                 "test_name": "test_full_like_stride",
                                 "test_node_id": "test/test_ops.py::test_full_like_stride@L10",
                                 "added_code": added}],
                "touched_files_patch": [{"path": "test/test_ops.py",
                                         "patch": "+def test_full_like_stride():\n+    x = torch.ones(2, 3).t()"}],
            }))
            witnesses, audit = MODULE.adapt(index, corpus, provider)
        self.assertEqual(len(witnesses), 1)
        self.assertEqual(witnesses[0]["pr_number"], 42)
        self.assertEqual(witnesses[0]["code"], code)
        self.assertEqual(audit[0]["status"], "adapted")
        self.assertEqual(audit[0]["source_added_code"], added)
        self.assertEqual(provider.requests[0].input_payload["test_name"],
                         "test_full_like_stride")

    def test_wrong_source_mark_is_rejected(self):
        code = ("def build():\n"
                "    def fn(a): return mark('op:aten::abs', abs, a)\n"
                "    return fn, (1,)\n")
        with self.assertRaisesRegex(ValueError, "source operator"):
            MODULE.validate_code(code, "op:aten::full_like")

    def test_screening_generates_input_when_pr_has_no_added_test(self):
        code = ("import torch\n"
                "def build():\n"
                "    x = torch.ones(2)\n"
                "    def fn(a): return mark('op:aten::abs', torch.abs, a)\n"
                "    return fn, (x,)\n")
        provider = FixtureLlmProvider(
            responses={"regression.screen_input": {"code": code}})
        item = {"oracle_id": "pr:42:abs", "aspect": "semantic",
                "release_provenance": {"pr_number": 42},
                "guard": {"static": {"source_entry": {"operator": "op:aten::abs"}}},
                "property": {"P": "True", "T": "identity_eager",
                             "pi": "r.output", "rho": "close(a,b)"}}
        def runner(test, oracle, *, isolate):
            return Finding(test.test_id, oracle.oracle_id, oracle.op,
                           oracle.aspect, False, "pass", "")
        with TemporaryDirectory() as tmp:
            corpus = Path(tmp)
            (corpus / "42.json").write_text(json.dumps({
                "title": "Fix abs", "body": {"text": "Preserve values"},
                "tests_added": [], "touched_files_patch": []}))
            fallback = lambda op, candidate: SCREEN.generate_from_pr(
                op, candidate, corpus, provider)
            admitted = SCREEN.screen({"op:aten::abs": [item]}, [], runner,
                                     corpus / "audit.jsonl", fallback=fallback)
            audit = json.loads((corpus / "audit.jsonl").read_text())
        self.assertEqual(len(admitted["op:aten::abs"]), 1)
        self.assertEqual(audit["witness"]["origin"], "generated_from_pr")
        self.assertEqual(len(provider.requests), 1)

    def test_screening_retries_after_inactive_adaptation(self):
        item = {"oracle_id": "pr:42:abs", "aspect": "semantic",
                "release_provenance": {"pr_number": 42},
                "property": {"P": "True", "T": "identity_eager",
                             "pi": "r.output", "rho": "close(a,b)"}}
        calls = []
        def runner(test, oracle, *, isolate):
            calls.append(test.code)
            verdict = "skip:inactive" if test.code == "old" else "pass"
            return Finding(test.test_id, oracle.oracle_id, oracle.op,
                           oracle.aspect, False, verdict, "")
        def fallback(operator, candidate):
            return {"pr_number": 42, "code": "new", "origin": "generated_from_pr"}
        with TemporaryDirectory() as tmp:
            audit_path = Path(tmp) / "audit.jsonl"
            admitted = SCREEN.screen(
                {"op:aten::abs": [item]}, [{"pr_number": 42, "code": "old"}],
                runner, audit_path, fallback=fallback)
            audit = json.loads(audit_path.read_text())
        self.assertEqual(calls, ["old", "new"])
        self.assertEqual(len(admitted["op:aten::abs"]), 1)
        self.assertEqual(audit["witness"]["origin"], "generated_from_pr")


if __name__ == "__main__":
    unittest.main()
