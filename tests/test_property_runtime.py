"""Integration checks for explicit property semantics and marked-call tracing."""
from __future__ import annotations

import unittest

from mrminer.fuzz.property_model import Context, ExecutionResult, PropertyChecker
from mrminer.fuzz.property_runtime import check_program, check_program_pair, _clone_inputs
from mrminer.fuzz.runner import run_test
from mrminer.fuzz.types import Finding, GeneratedTest, OracleRef
from mrminer.fuzz.provenance import TraceRecorder
from mrminer.fuzz.seed_trace import trace_code, trace_isolated
from mrminer.fuzz.loop import fuzz_campaign
from mrminer.fuzz.types import Seed
from mrminer.llm.provider import FixtureLlmProvider
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from importlib.util import module_from_spec, spec_from_file_location


TORCH_CODE = """\
import torch
def build():
    x = torch.tensor([-1.0, 2.0])
    def fn(a):
        return mark('op:aten::abs', torch.abs, a)
    return fn, (x,)
"""


class PropertyRuntimeTest(unittest.TestCase):
    def test_partial_projection_does_not_evaluate_relation(self) -> None:
        checker = PropertyChecker({"P": "True", "T": "identity_eager",
                                   "pi": "r.output_at(c.output_path)",
                                   "rho": "False"}, framework="pytorch")
        context = Context("op:aten::abs", "op:aten::abs", "fx:abs", 0, (), {}, None)
        verdict, _ = checker.check(context, ExecutionResult("raise", exception="ValueError"),
                                   ExecutionResult("raise", exception="ValueError"))
        self.assertEqual(verdict, "skip:pi_baseline")

    def test_relation_failure_is_distinct_from_skipping(self) -> None:
        checker = PropertyChecker({"P": "True", "T": "identity_eager",
                                   "pi": "r.output", "rho": "a == b"},
                                  framework="pytorch")
        context = Context("op:aten::abs", "op:aten::abs", "fx:abs", 0, (), {}, ())
        verdict, _ = checker.check(context, ExecutionResult("return", output=1),
                                   ExecutionResult("return", output=2))
        self.assertEqual(verdict, "fail")

    def test_no_condition_bypasses_only_applicability(self) -> None:
        checker = PropertyChecker({"P": "False", "T": "identity_eager",
                                   "pi": "r.output", "rho": "a == b"},
                                  framework="pytorch")
        context = Context("op:aten::abs", "op:aten::abs", "fx:abs", 0, (), {}, ())
        baseline = ExecutionResult("return", output=1)
        followup = ExecutionResult("return", output=2)
        self.assertEqual(checker.check(context, baseline, followup)[0], "skip:P")
        self.assertEqual(checker.check(context, baseline, followup,
                                       ignore_condition=True)[0], "fail")

    def test_paired_replay_shares_contexts_and_observations(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is unavailable")
        record = {"P": "False", "T": "identity_eager",
                  "pi": "tuple(r.output_at(c.output_path).shape)",
                  "rho": "False"}
        pair = check_program_pair(TORCH_CODE, record, operator="op:aten::abs",
                                  source_operator="op:aten::abs", framework="pytorch")
        self.assertEqual(pair["conditional"]["verdict"], "skip:P", pair)
        self.assertEqual(pair["no_condition"]["verdict"], "fail", pair)
        self.assertEqual(pair["conditional"]["contexts"][0]["site"],
                         pair["no_condition"]["contexts"][0]["site"])

    def test_marked_call_snapshots_pre_mutation_inputs(self) -> None:
        import copy
        values = [1]
        recorder = TraceRecorder(copy.deepcopy)
        recorder.mark("op:aten::append", lambda arg: arg.append(2), values)
        self.assertEqual(recorder.calls[0].args[0], [1])
        self.assertEqual(values, [1, 2])

    def test_marked_torch_call_reaches_core_operator(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is unavailable")
        record = {"P": "c.source_operator == 'op:aten::abs'",
                  "T": "identity_eager",
                  "pi": "tuple(r.output_at(c.output_path).shape)",
                  "rho": "a == b"}
        result = check_program(TORCH_CODE, record, operator="op:aten::abs",
                               source_operator="op:aten::abs", framework="pytorch")
        self.assertEqual(result["verdict"], "pass", result)
        self.assertIn("fx:", result["detail"])
        self.assertEqual(result["contexts"][0]["source_operator"], "op:aten::abs")

    def test_unmarked_torch_call_has_no_provenance(self) -> None:
        code = TORCH_CODE.replace("mark('op:aten::abs', torch.abs, a)", "torch.abs(a)")
        record = {"P": "True", "T": "identity_eager", "pi": "r.output",
                  "rho": "close(a, b)"}
        result = check_program(code, record, operator="op:aten::abs",
                               source_operator="op:aten::abs", framework="pytorch")
        self.assertEqual(result["verdict"], "skip:no_context")

    def test_decomposed_source_call_reaches_core_site(self) -> None:
        code = """\
import torch
def build():
    x = torch.empty_strided((5, 1), (1, 5))
    def fn(a):
        return mark('op:aten::ones_like', torch.ones_like, a)
    return fn, (x,)
"""
        record = {"P": "tuple(c.site_args[0].stride()) == (1, 5)",
                  "T": "identity_eager",
                  "pi": "tuple(r.output_at(c.output_path).stride())",
                  "rho": "a == b"}
        result = check_program(code, record, operator="op:aten::full_like",
                               source_operator="op:aten::ones_like", framework="pytorch")
        self.assertEqual(result["verdict"], "pass", result)
        self.assertIn("fx:full_like", result["detail"])

    def test_eager_to_inductor_transform(self) -> None:
        record = {"P": "True", "T": "eager_to_inductor",
                  "pi": "r.output_at(c.output_path) if r.kind == 'return' else undefined()",
                  "rho": "close(a, b)"}
        result = check_program(TORCH_CODE, record, operator="op:aten::abs",
                               source_operator="op:aten::abs", framework="pytorch")
        self.assertEqual(result["verdict"], "pass", result)

    def test_rejected_marked_call_uses_supported_inferred_context(self) -> None:
        code = """\
import torch
def build():
    x = torch.ones(2)
    def fn(a):
        return mark('op:aten::reshape', torch.reshape, a, (3,))
    return fn, (x,)
"""
        record = {"P": "c.inferred and c.args[1] == (3,)",
                  "T": "identity_eager", "origin_condition": "c.args[1] == (3,)",
                  "pi": "r.exception if r.kind == 'raise' else 'return'",
                  "rho": "a == b"}
        result = check_program(code, record, operator="op:aten::view",
                               source_operator="op:aten::reshape", framework="pytorch")
        self.assertEqual(result["verdict"], "pass", result)
        self.assertIn("inferred:", result["detail"])

    def test_input_copies_preserve_stride_offset_and_aliasing(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is unavailable")
        base = torch.arange(20, dtype=torch.float32)
        x = base.as_strided((3, 2), (2, 1), 4)
        y = base.as_strided((2, 2), (3, 1), 5)
        a, b = _clone_inputs((x, y), "pytorch")
        self.assertEqual((a.stride(), a.storage_offset()), (x.stride(), x.storage_offset()))
        self.assertEqual((b.stride(), b.storage_offset()), (y.stride(), y.storage_offset()))
        self.assertEqual(a.untyped_storage()._cdata, b.untyped_storage()._cdata)
        self.assertNotEqual(a.untyped_storage()._cdata, x.untyped_storage()._cdata)
        grad_input = torch.ones(2, requires_grad=True)
        self.assertTrue(_clone_inputs(grad_input, "pytorch").requires_grad)

    def test_isolated_worker_executes_property_record(self) -> None:
        record = {"P": "True", "T": "identity_eager",
                  "pi": "tuple(r.output_at(c.output_path).shape)", "rho": "a == b"}
        oracle = OracleRef("test:worker", "op:aten::abs", "semantic", {}, "abs",
                           core_operator="op:aten::abs", property_record=record)
        test = GeneratedTest("test:worker", "seed:worker", oracle.oracle_id, TORCH_CODE)
        result = run_test(test, oracle, isolate=True, timeout=60)
        self.assertEqual(result.verdict, "pass", result)
        self.assertEqual(result.contexts[0]["operator"], "op:aten::abs")

    def test_seed_retrieval_uses_actual_core_trace(self) -> None:
        traced = trace_code(TORCH_CODE, "pytorch", {})
        self.assertIn("op:aten::abs", traced["ops"])
        self.assertEqual(traced["marked_calls"], 1)

    def test_seed_trace_worker_returns_operator_mapping(self) -> None:
        with TemporaryDirectory() as tmp:
            index = Path(tmp) / "index.json"
            index.write_text(json.dumps({"index": {}}))
            traced = trace_isolated(TORCH_CODE, "pytorch", index, timeout=60)
        self.assertIn("op:aten::abs", traced["ops"])

    def test_screening_admits_executable_pr_witness(self) -> None:
        script = Path(__file__).resolve().parents[1] / "scripts/screen_properties.py"
        spec = spec_from_file_location("release_screen", script)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        item = {"oracle_id": "test:pr42", "aspect": "semantic",
                "release_provenance": {"pr_number": 42},
                "claim": {"summary": "absolute value shape"},
                "property": {"P": "True", "T": "identity_eager",
                             "pi": "tuple(r.output_at(c.output_path).shape)",
                             "rho": "a == b"}}
        with TemporaryDirectory() as tmp:
            audit = Path(tmp) / "audit.jsonl"
            admitted = module.screen({"op:aten::abs": [item]},
                                     [{"pr_number": 42, "code": TORCH_CODE}],
                                     run_test, audit)
            row = json.loads(audit.read_text().splitlines()[0])
        self.assertEqual(row["status"], "activated")
        self.assertEqual(row["witness"]["contexts"][0]["operator"], "op:aten::abs")
        self.assertEqual(admitted["op:aten::abs"][0]["oracle_id"], "test:pr42")

    def test_feedback_campaign_checks_traced_property(self) -> None:
        property_record = {"P": "True", "T": "identity_eager",
                           "pi": "tuple(r.output_at(c.output_path).shape)",
                           "rho": "a == b"}
        index = {"op:aten::abs": [{"oracle_id": "test:campaign", "aspect": "semantic",
                                    "claim": {"summary": "absolute value shape"},
                                    "property": property_record}]}
        ops = tuple(trace_code(TORCH_CODE, "pytorch", index)["ops"])
        seed = Seed("seed:campaign", ops, parent_code=TORCH_CODE)
        provider = FixtureLlmProvider(responses={"fuzz.generate_tests": {
            "tests": [{"code": TORCH_CODE}]}})
        campaign = fuzz_campaign([seed], provider, index, iterations=1,
                                 run_test=run_test, minimize_triggers=False)
        self.assertEqual(campaign.coverage_size, 1)
        self.assertEqual(campaign.findings[0].verdict, "pass")
        self.assertEqual(campaign.findings[0].contexts[0]["operator"], "op:aten::abs")

    def test_campaign_stops_at_exact_generated_test_limit(self) -> None:
        property_record = {"P": "True", "T": "identity_eager",
                           "pi": "r.output", "rho": "a == b"}
        index = {"op:aten::abs": [{"oracle_id": "test:limit", "aspect": "semantic",
                                    "claim": {"summary": "test property"},
                                    "property": property_record}]}
        provider = FixtureLlmProvider(responses={"fuzz.generate_tests": {
            "tests": [{"code": TORCH_CODE}, {"code": TORCH_CODE}]}})
        seed = Seed("seed:limit", ("op:aten::abs",), parent_code=TORCH_CODE)
        def runner(test, oracle, **_kwargs):
            return Finding(test.test_id, oracle.oracle_id, oracle.op,
                           oracle.aspect, False, "pass")
        campaign = fuzz_campaign([seed], provider, index, iterations=3,
                                 max_tests=1, run_test=runner)
        self.assertEqual(len(campaign.findings), 1)

    def test_no_condition_runner_uses_its_own_verdict(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is unavailable")
        record = {"P": "False", "T": "identity_eager",
                  "pi": "tuple(r.output_at(c.output_path).shape)", "rho": "False"}
        oracle = OracleRef("test:no-condition", "op:aten::abs", "semantic", {},
                           "abs", core_operator="op:aten::abs", property_record=record)
        test = GeneratedTest("test:no-condition", "seed:no-condition",
                             oracle.oracle_id, TORCH_CODE)
        self.assertEqual(run_test(test, oracle, isolate=False).verdict, "skip:P")
        self.assertEqual(run_test(test, oracle, isolate=False,
                                  judge_mode="no_condition").verdict, "fail")


if __name__ == "__main__":
    unittest.main()
