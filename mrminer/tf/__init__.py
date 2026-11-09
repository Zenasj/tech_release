"""TensorFlow/XLA supplementary experiment (design: docs/design/tf_replication_plan_v1.md).

TF-specific pieces of the method, mirroring the PyTorch side:
  op_registry  -- the TF OpDef universe (analog of native_functions.yaml / aten ops)
  runner_tf    -- eager vs tf.function(jit_compile=True) differential + op trace + judge

These reuse the framework-generic parts under mrminer/fuzz/ (types, coverage,
corpus, loop, minimize, llm, generator) by import; only the operator layer and the
execution runner are TF-specific. Keep this package separate from the PyTorch code.
"""
