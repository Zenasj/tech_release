# Tech

Tech mines operator properties from merged PRs and uses them to guide testing of PyTorch/Inductor and TensorFlow/XLA. This release contains the implementation, collection rules, [the 44 reported issues](materials/ISSUE_REPORT.md), and [two worked case studies](materials/CASE_STUDIES.md). The commands below create PR, property, and testing artifacts under `runs/`.

## Pipeline

| Stage | Implementation | Output |
| --- | --- | --- |
| PR collection | `mrminer.s1_miner.run_s1` | Versioned PR records, diffs, added test excerpts, discussions, linked issues |
| Operator attribution | `mrminer.s2_graph.cli` (PyTorch); `mrminer.tf.linker_tf` and `scripts/tf_triage.py` (TensorFlow) | Operator/PR associations |
| Property mining | `scripts/mine_pytorch.py`; `scripts/tf_mine.py` | Source-backed candidate property index |
| Operator indexing | `scripts/migrate_library_to_core.py`; `scripts/migrate_tf_library_to_stablehlo.py` | Core ATen or StableHLO index and attribution audit |
| Property construction | `scripts/synthesize_properties.py` | Executable $P,T,\pi,\rho$ records grounded in attributed PR evidence |
| Regression-test adaptation | `scripts/prepare_regression_witnesses.py` | Adapted screening inputs and source-to-adaptation audit |
| Candidate screening | `scripts/screen_properties.py` | Index of properties activated by PR-grounded screening inputs |
| Official seed preparation | `scripts/prepare_official_seeds.py` | Traced seed programs and source-test audit |
| Online testing | `scripts/fuzz.py` | Generated tests, verdicts, minimized candidate violations, batch checkpoint |

The pipeline collects and attributes changes, constructs and screens executable properties, and uses the admitted properties to generate and check tests. Each property record specifies its applicability $P$, execution transform $T$, observation projection $\pi$, and expected relation $\rho$. For each marked source call, the runner traces Core ATen or StableHLO sites and evaluates the applicable property checker. Violations are recorded as defect candidates for triage.

## Requirements

- Python 3.10 or newer; a Git checkout of the target PyTorch or TensorFlow version.
- PyTorch installation matching the checkout for the PyTorch line; TensorFlow 2.21 or newer with XLA and textual StableHLO compiler IR for the TensorFlow line.
- GitHub read token for practical collection throughput.
- An OpenAI-compatible chat-completions endpoint that returns JSON in `choices[0].message.content`, such as a local vLLM deployment.

Use separate environments for PyTorch and TensorFlow when their dependency versions conflict. From this directory:

```bash
python -m pip install -e '.[pytorch]'
# Or, in a separate environment:
python -m pip install -e '.[tensorflow]'
```

Copy `.env.example` to `.env`, fill the relevant paths and endpoint settings, then load it:

```bash
set -a
. ./.env
set +a
```

For a local vLLM endpoint, set `MRMINER_API_BASE=http://127.0.0.1:8008/v1`, `MRMINER_MODEL` to the served model ID, and `MRMINER_VLLM_EXTENSIONS=1`. Set `MRMINER_API_KEY` when the endpoint requires bearer authentication.
Set `MRMINER_GPU_POOL` to a comma-separated list of GPU indices only when workers should be placed in that pool; leaving it empty preserves the caller's device visibility.

## PyTorch quickstart

Run from this directory. The paper's mining window is 2023-01-01 through 2026-09-08. For any tested build, exclude PRs merged at or after its commit. `--max-prs 3` is useful for a small collector trial; omit it for full collection.

```bash
python -m mrminer.s1_miner.run_s1 \
  --repo pytorch/pytorch --since 2023-01-01 --until 2026-09-08 \
  --compile-stack-only --config configs/s1_pytorch.json \
  --out-root runs/pytorch/pr_corpus

python -m mrminer.s2_graph.cli \
  --graph-root runs/pytorch/graph build \
  --corpus-root runs/pytorch/pr_corpus \
  --pytorch-source-root "$PYTORCH_SOURCE"

python scripts/mine_pytorch.py \
  --graph-root runs/pytorch/graph \
  --corpus-root runs/pytorch/pr_corpus/pytorch__pytorch \
  --operator op:aten::full_like --pr-number 158898 \
  --output runs/pytorch/operator_oracle_index.json

python scripts/migrate_library_to_core.py \
  --input runs/pytorch/operator_oracle_index.json \
  --output runs/pytorch/operator_oracle_index_core.json \
  --report runs/pytorch/core_homing_report.csv --core-only

python scripts/synthesize_properties.py --framework pytorch \
  --input runs/pytorch/operator_oracle_index_core.json \
  --corpus runs/pytorch/pr_corpus/pytorch__pytorch \
  --output runs/pytorch/operator_properties.json \
  --audit runs/pytorch/property_synthesis.jsonl

python scripts/prepare_regression_witnesses.py \
  --input runs/pytorch/operator_properties.json \
  --corpus runs/pytorch/pr_corpus/pytorch__pytorch \
  --output runs/pytorch/regression_witnesses.jsonl \
  --audit runs/pytorch/regression_witness_adaptations.jsonl

python scripts/screen_properties.py --framework pytorch \
  --input runs/pytorch/operator_properties.json \
  --witnesses runs/pytorch/regression_witnesses.jsonl \
  --corpus runs/pytorch/pr_corpus/pytorch__pytorch \
  --output runs/pytorch/operator_oracle_index_screened.json \
  --audit runs/pytorch/screen_audit.jsonl

python scripts/prepare_official_seeds.py --framework pytorch \
  --index runs/pytorch/operator_oracle_index_screened.json \
  --source-root "$PYTORCH_SOURCE" \
  --output runs/pytorch/seed_manifest.jsonl \
  --audit runs/pytorch/official_seed_audit.jsonl

python scripts/fuzz.py --framework pytorch \
  --index runs/pytorch/operator_oracle_index_screened.json \
  --seeds runs/pytorch/seed_manifest.jsonl \
  --screen-audit runs/pytorch/screen_audit.jsonl \
  --output-dir runs/pytorch/fuzz --iterations 30
```

For a wider mining run, remove `--pr-number`, pass `--operator` repeatedly, or provide one `op:aten::<name>` per line with `--operators-file`. The migration audit records each attribution, and `--core-only` selects records with a Core ATen index.

## TensorFlow quickstart

The TensorFlow collector uses the same PR schema. The attribution step first enumerates the installed raw op registry and then links PR edits to registration or implementation sites. The bundled StableHLO specification inventory defines the operator universe; the pinned `StablehloOps.td` is audited against it. A representative lowering probe establishes each mapped index entry.

```bash
python -m mrminer.s1_miner.run_s1 \
  --repo tensorflow/tensorflow --since 2023-01-01 --until 2026-09-08 \
  --compile-stack-only --config configs/s1_tensorflow.json \
  --out-root runs/tensorflow/pr_corpus

python -m mrminer.tf.op_registry --out runs/tensorflow/op_universe.json
python scripts/tf_triage.py \
  --corpus runs/tensorflow/pr_corpus/tensorflow__tensorflow \
  --out runs/tensorflow/mining_plan.json
python scripts/tf_mine.py \
  --plan runs/tensorflow/mining_plan.json \
  --corpus runs/tensorflow/pr_corpus/tensorflow__tensorflow \
  --records runs/tensorflow/mine_records.jsonl \
  --index runs/tensorflow/operator_oracle_index_tf.json

python scripts/build_stablehlo_operator_index.py \
  --source "$STABLEHLO_TD" \
  --out runs/tensorflow/stablehlo_operator_index.json
python scripts/migrate_tf_library_to_stablehlo.py \
  --native-index runs/tensorflow/operator_oracle_index_tf.json \
  --universe runs/tensorflow/stablehlo_operator_index.json \
  --stablehlo-td "$STABLEHLO_TD" --out-dir runs/tensorflow

python scripts/synthesize_properties.py --framework tensorflow \
  --input runs/tensorflow/operator_oracle_index_tf_stablehlo.json \
  --corpus runs/tensorflow/pr_corpus/tensorflow__tensorflow \
  --output runs/tensorflow/operator_properties.json \
  --audit runs/tensorflow/property_synthesis.jsonl

python scripts/prepare_regression_witnesses.py \
  --input runs/tensorflow/operator_properties.json \
  --corpus runs/tensorflow/pr_corpus/tensorflow__tensorflow \
  --output runs/tensorflow/regression_witnesses.jsonl \
  --audit runs/tensorflow/regression_witness_adaptations.jsonl

python scripts/screen_properties.py --framework tensorflow \
  --input runs/tensorflow/operator_properties.json \
  --witnesses runs/tensorflow/regression_witnesses.jsonl \
  --corpus runs/tensorflow/pr_corpus/tensorflow__tensorflow \
  --output runs/tensorflow/operator_oracle_index_screened.json \
  --audit runs/tensorflow/screen_audit.jsonl

python scripts/prepare_official_seeds.py --framework tensorflow \
  --index runs/tensorflow/operator_oracle_index_screened.json \
  --source-root "$TENSORFLOW_SOURCE" \
  --output runs/tensorflow/seed_manifest.jsonl \
  --audit runs/tensorflow/official_seed_audit.jsonl

python scripts/fuzz.py --framework tensorflow \
  --index runs/tensorflow/operator_oracle_index_screened.json \
  --seeds runs/tensorflow/seed_manifest.jsonl \
  --screen-audit runs/tensorflow/screen_audit.jsonl \
  --output-dir runs/tensorflow/fuzz --iterations 30
```

The TensorFlow migration audit records each representative lowering probe. Screening uses the index of observed StableHLO mappings.

## Regression witnesses and official seeds

`prepare_regression_witnesses.py` reads the collected `tests_added` excerpts and their PR patches, adapts them to the `build() -> (fn, args)` interface through the configured model, and writes `regression_witnesses.jsonl`. Each adapted test marks its source call with `mark(source_operator_id, callable, *args, **kwargs)`; the runner supplies `mark`. The adaptation audit retains the original added code, adapted code, source location, and model request identity.

The screening audit records attempted inputs, their traced operator sites, and their verdicts. Screening admits a property when an input activates its checker. The runner executes each input in an isolated subprocess.

`prepare_official_seeds.py` reads operator tests from the framework source checkout and writes executable, traced programs to `seed_manifest.jsonl`. Each seed retains its official test location and source hash. The runner recomputes the operators reached by the marked call before retrieving properties. The screening audit can be passed separately to add admitted witnesses.

The online loop records the resulting seed mappings in `seed_trace.jsonl`, cycles through the usable seeds in random order, selects the least activated property reachable from each seed, and supplies prior tests and verdicts for that seed/property pair to generation. A seed with no marked call reaching an indexed operator is excluded. Crashes and timeouts do not count as property activations and are saved separately in `crashes.jsonl`.

`P`, `pi`, `rho`, and the optional rejection condition are synthesized as Python expressions from PR evidence and executed in the isolated test worker. PyTorch uses `make_fx` with the Core ATen decomposition table on each marked call and records each matching FX site. TensorFlow obtains textual StableHLO IR for each marked call and records each matching instruction, including its IR location. Both screening and fuzzing save the traced source call, indexed operator, site, and per-context verdict.

## Outputs and repeat runs

Collection and PyTorch and TensorFlow mining keep resume state under `runs/`. The screening JSONL audit records attempted inputs and verdicts for each candidate; screening rewrites its audit and output on each invocation. Fuzzing appends `tests.jsonl` and `results.jsonl`, writes deduplicated candidates to `findings.jsonl`, and numbers each completed invocation in `checkpoint.json`. Re-running the command starts a new feedback loop and appends its results; feedback history is scoped to one invocation.

Generated outputs and credential files are ignored by Git.
