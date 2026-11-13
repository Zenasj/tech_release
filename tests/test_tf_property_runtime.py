"""TensorFlow marked-call to StableHLO integration check."""
from __future__ import annotations

import unittest

from mrminer.fuzz.property_runtime import check_program
from mrminer.fuzz.types import GeneratedTest, OracleRef
from mrminer.tf.runner_tf import run_test
from mrminer.tf.stablehlo_ops import parse_stablehlo_sites
from mrminer.fuzz.seed_trace import trace_code


CODE = """\
import tensorflow as tf
def build():
    x = tf.constant([-1.0, 2.0])
    def fn(a):
        return mark('op:tf::Abs', tf.abs, a)
    return fn, (x,)
"""


class TensorFlowPropertyRuntimeTest(unittest.TestCase):
    def test_repeated_stablehlo_instructions_remain_distinct(self) -> None:
        ir = ("    %0 = stablehlo.abs %arg0 : tensor<2xf32> loc(#loc1)\n"
              "    %1 = stablehlo.abs %0 : tensor<2xf32> loc(#loc2)\n")
        sites = parse_stablehlo_sites(ir, {"op:stablehlo::abs"})
        self.assertEqual([site["ordinal"] for site in sites], [0, 1])
        self.assertEqual([site["ssa"] for site in sites], ["%0", "%1"])

    def test_seed_retrieval_uses_actual_stablehlo_trace(self) -> None:
        try:
            import tensorflow  # noqa: F401
        except ImportError:
            self.skipTest("TensorFlow is unavailable")
        traced = trace_code(CODE, "tensorflow", {})
        self.assertIn("op:tf::Abs", traced["ops"])
        self.assertIn("op:stablehlo::abs", traced["ops"])

    def test_marked_tf_call_reaches_stablehlo(self) -> None:
        try:
            import tensorflow  # noqa: F401
        except ImportError:
            self.skipTest("TensorFlow is unavailable")
        record = {"P": "c.source_operator == 'op:tf::Abs'",
                  "T": "identity_eager",
                  "pi": "tuple(r.output_at(c.output_path).shape)",
                  "rho": "a == b"}
        result = check_program(CODE, record, operator="op:stablehlo::abs",
                               source_operator="op:tf::Abs", framework="tensorflow")
        self.assertEqual(result["verdict"], "pass", result)
        self.assertIn("stablehlo:", result["detail"])
        self.assertEqual(result["contexts"][0]["source_operator"], "op:tf::Abs")

    def test_eager_to_xla_transform(self) -> None:
        try:
            import tensorflow  # noqa: F401
        except ImportError:
            self.skipTest("TensorFlow is unavailable")
        record = {"P": "True", "T": "eager_to_xla",
                  "pi": "r.output_at(c.output_path) if r.kind == 'return' else undefined()",
                  "rho": "close(a, b)"}
        result = check_program(CODE, record, operator="op:stablehlo::abs",
                               source_operator="op:tf::Abs", framework="tensorflow")
        self.assertEqual(result["verdict"], "pass", result)

    def test_isolated_tf_worker_executes_property_record(self) -> None:
        try:
            import tensorflow  # noqa: F401
        except ImportError:
            self.skipTest("TensorFlow is unavailable")
        record = {"P": "True", "T": "identity_eager",
                  "pi": "tuple(r.output_at(c.output_path).shape)", "rho": "a == b"}
        oracle = OracleRef("test:tf-worker", "op:tf::Abs", "semantic", {}, "abs",
                           core_operator="op:stablehlo::abs", property_record=record)
        test = GeneratedTest("test:tf-worker", "seed:tf-worker", oracle.oracle_id, CODE)
        result = run_test(test, oracle, isolate=True, timeout=60)
        self.assertEqual(result.verdict, "pass", result)
        self.assertEqual(result.contexts[0]["operator"], "op:stablehlo::abs")


if __name__ == "__main__":
    unittest.main()
