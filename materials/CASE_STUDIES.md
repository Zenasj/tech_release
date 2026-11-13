# Case Studies

Two property-evidence records that document how Tech mines an operator property from a merged PR, screens it against the regression tests that cover the change, and then detects the reported bug during online testing. Each record separates the direct evidence in the source change from the requirement Tech recovers, states the applicability condition and checker used in screening, and distinguishes the screening witness from the later bug-triggering input. The two bugs are among the issues listed in [ISSUE_REPORT.md](ISSUE_REPORT.md).

Notation used below: `P` is the applicability condition, `π` the observation projection, `ρ` the expected relation, and `chk` the executable checker stored with a property record (see the pipeline table in the top-level [README](../README.md)).

## Case 1 — Rejection of incompatible `nll_loss` shapes

PyTorch, operator `aten.div`, reported as PyTorch issue [#200244](https://github.com/pytorch/pytorch/issues/200244).

| Field | Record |
| --- | --- |
| Source change | PyTorch PR [#175956](https://github.com/pytorch/pytorch/pull/175956), landing commit on the `main` branch of the evaluated build. |
| Direct evidence | The change preserves batch-size validation in `nll_loss_nd` under symbolic shapes (`nll_loss_nd_symint`) and in the `nll_loss_forward` decomposition. |
| Inferred extension | The candidate property extends batch-size validation to rejection of incompatible input and target shapes, including spatial-size mismatches. |
| Basis for the extension | Tech reads the PR diff together with the code comments stating that a known mismatch raises an error and that a symbolic mismatch requires a runtime check, and generalizes the preserved batch check into a required shape constraint on the input and target. This generalization follows the stated intent. |
| Property and checker | Origin: `nll_loss`. Applicability condition `P`: the call reaches `aten.div` from the `nll_loss_nd` entry under mean reduction, and its input and target violate the required shape constraint. Observation `π`: rejection versus normal return. Expected relation `ρ`: agreement between eager and compiled outcomes. Checker `chk`: stored with the property record indexed at `aten.div`, projecting each execution to reject or return and testing outcome agreement. |
| Screening witness | Witness source: PyTorch's official regression test `test_nll_loss_mismatched_batch` in `test/test_nn.py`. PR #175956 does not add this test, but it is validated against it. The PR carries the `ciflow/trunk` and `ciflow/inductor` labels and merged green, so its continuous integration exercises the `test_nn.py` suite that holds this test, and the test asserts the same `Expected ... batch_size` rejection that the refactored check in `nll_loss_nd_symint` emits. Executed input: a two-dimensional `nll_loss` call under mean reduction whose input and target disagree on batch size, with input `(10, 3)` and target `(3,)`. The source test asserts a `ValueError` whose message matches `Expected ... batch_size`, which is the message the preserved check emits. Adaptation: the test is reduced to the `nll_loss` call and its inputs, with the assertion replaced by the rejection projection, and no other change. |
| Screening execution | Build and configuration: PyTorch `2.15.0a0+git1a080e5`, CPU device, TorchInductor with default options, seed 0. Condition satisfied: true. Eager observation: rejection (a `ValueError` for batch-size mismatch). Compiled observation: rejection, since the batch-size check that the PR preserves is present on the compiled path. Checker verdict: Pass. Replay: the screening record for this property under `runs/`. |
| Reported violation | Input shape `(2, 3, 2, 2)` and target shape `(2, 2, 1)`, a spatial-size mismatch with matching batch size. Eager and `aot_eager` raise a `RuntimeError` for size mismatch, while Inductor returns the scalar `tensor(-0.7886)`, because the `nll_loss2d_forward` decomposition omits the spatial-size check that eager execution performs. Reproducer and checker version: the minimized script filed with PyTorch issue #200244, checked by the same `chk` as above. |

**How the record holds together.** The direct evidence is a batch-size check, so the screening witness is a batch mismatch, which both executions reject because the compiled path keeps that check, giving a Pass. The property generalizes the batch check to a required shape constraint. The reported violation satisfies that constraint on the batch dimension but violates it on the spatial dimensions: eager rejects, while the compiled `nll_loss2d_forward` decomposition skips the spatial check and returns a scalar. This generalization from batch to spatial is the inferred extension, and the spatial mismatch is the bug rather than the witness.

## Case 2 — SVD reconstruction accuracy

TensorFlow, operator `stablehlo.reshape` in the `Svd` lowering, reported as TensorFlow issue [#127974](https://github.com/tensorflow/tensorflow/issues/127974).

| Field | Record |
| --- | --- |
| Source change | TensorFlow PR [#76518](https://github.com/tensorflow/tensorflow/pull/76518), revision `27840bcfa397f4e36c66b8cdf044d69956896660`. |
| Direct evidence | PR #76518 relocates the XLA SVD implementation and its regression tests into `xla/hlo/builder`. The relocated test file contains `SVDTest.Simple2D`, which reconstructs the input with `ComputeMatmulUDVT` and compares it against the original using `ErrorSpec(1e-3, 1e-3)`. |
| Source location | `third_party/xla/xla/hlo/builder/lib/svd_test.cc`. |
| Derived comparison | The checker evaluates `e(A) = ||R - A||_F / ||A||_F` with `R = U diag(S) Vᵀ`, and requires `e(A) <= ε_svd`. |
| Checker configuration | Applicability condition `P`: the call reaches the `Svd` lowering from `tf.linalg.svd` on a nondegenerate real matrix with `||A||_F > 0`. Threshold: `ε_svd = 1e-3` on the normalized residual, set three orders above the residual of correct float32 reconstructions, and distinct from the elementwise `ErrorSpec(1e-3, 1e-3)` of the source test. Arithmetic precision: `R`, the difference `R - A`, and both Frobenius norms are computed in float64. Zero-input handling: inputs with `||A||_F` below a small floor are skipped to avoid division by zero. |
| Screening witness | Source test: `Simple2D` is the evidence, not the screening input verbatim. Executed input: a nondegenerate, well-conditioned `2 x 2` real matrix at unit scale, with entries of order one. Adaptation from the source test: the elementwise `ErrorSpec` comparison is replaced by the normalized residual `e(A)`. |
| Screening execution | Build and configuration: TensorFlow `2.22.0-dev20260921` with XLA, GPU device (NVIDIA RTX A6000, CUDA 12.6), `jit_compile=True`, seed 0. Condition satisfied: true. Observed residual: `≈ 6e-7`. Threshold: `1e-3`. Checker verdict: Pass. Replay: the screening record for this property under `runs/`. |
| Reported violation | `A = 1e-20 x [[1, 2], [3, 4]]` with condition number about 15, stored as float32. Under `jit_compile=True`, XLA returns the diagonal entries as singular values and a reconstruction residual of `0.658`, while eager execution and float64 stay near `1e-7`. Reproducer and checker version: the minimized script filed with TensorFlow issue #127974, checked by the same `chk` as above. |

**How the record holds together.** `Simple2D` is the source evidence, while the screening witness is a unit-scale nondegenerate matrix on which the normalized residual is far below `ε_svd`, giving a Pass. The threshold `ε_svd` is the checker's own bound on the scale-invariant residual, not the source test's `ErrorSpec`. The residual of `0.658` comes from the later bug-triggering input, not from the screening run.
