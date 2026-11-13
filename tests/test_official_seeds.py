"""Check extraction of framework test programs into traced seed inputs."""
from __future__ import annotations

import ast
import json
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_official_seeds.py"
SPEC = spec_from_file_location("prepare_official_seeds", SCRIPT)
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class OfficialSeedTest(unittest.TestCase):
    def test_pytorch_common_program_preserves_official_inputs(self):
        source = (
            "def test_abs(self):\n"
            "    def fn(a):\n"
            "        return (a / (torch.abs(a) + 1),)\n"
            "    self.common(fn, (torch.randn(17),))\n"
        )
        row = {"source_code": source}
        code = MODULE.direct_pytorch_seed(row, "op:aten::abs")
        self.assertIsNotNone(code)
        ast.parse(code)
        self.assertIn("torch.randn(17)", code)
        self.assertIn("mark('op:aten::abs', torch.abs, a)", code)

    def test_prepared_seed_retains_source_identity_and_trace(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            test_dir = root / "test/inductor"
            test_dir.mkdir(parents=True)
            (test_dir / "test_demo.py").write_text(
                "import torch\nclass TestOps:\n"
                "    def test_abs(self):\n"
                "        def fn(a): return torch.abs(a)\n"
                "        self.common(fn, (torch.tensor([-2.0]),))\n")
            index_path = root / "index.json"
            index_path.write_text(json.dumps({"index": {"op:aten::abs": [
                {"oracle_id": "one", "guard": {"static": {
                    "source_entry": {"operator": "op:aten::abs"}}}}
            ]}}))
            with patch.object(MODULE, "trace_isolated",
                              return_value={"ops": ["op:aten::abs"], "errors": []}):
                seeds, audit = MODULE.prepare(index_path, root, "pytorch")
        self.assertEqual(len(seeds), 1)
        self.assertEqual(seeds[0]["origin"], "framework")
        self.assertEqual(seeds[0]["ops"], ["op:aten::abs"])
        self.assertEqual(audit[0]["status"], "accepted")
        self.assertEqual(audit[0]["adapter"], "ast")

    def test_tensorflow_source_id_is_normalized_for_matching_call(self):
        code = ("import tensorflow as tf\n"
                "def build():\n"
                "    def fn(a, i):\n"
                "        return mark('op:stablehlo::gather', tf.gather, a, i)\n"
                "    return fn, (tf.constant([1, 2]), tf.constant([0]))\n")
        fixed = MODULE.normalize_mark_source(
            code, "op:tf::GatherV2", "tensorflow")
        self.assertIn("op:tf::GatherV2", fixed)
        MODULE.validate_code(fixed, "op:tf::GatherV2")


if __name__ == "__main__":
    unittest.main()
