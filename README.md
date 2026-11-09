# Tech (working release name)

This directory packages the research prototype for PR-derived operator properties and property-guided testing of PyTorch/Inductor and TensorFlow/XLA. The name can be changed before publication. It contains source code, collector rules, and [the 44 reported issues](materials/ISSUE_REPORT.md), but no collected PR corpus, mined property library, test campaign, model weights, or API credentials. The issue statuses in that report are a dated snapshot, not live GitHub status.

## Pipeline

| Stage | Implementation | Artifact created by the user |
| --- | --- | --- |
| PR collection | `mrminer.s1_miner.run_s1` | Versioned PR records, diffs, regression tests, discussions, linked issues |
| Operator attribution | `mrminer.s2_graph.cli` (PyTorch); `mrminer.tf.linker_tf` and `scripts/tf_triage.py` (TensorFlow) | Operator/PR associations |
| Property mining | `scripts/mine_pytorch.py`; `scripts/tf_mine.py` | Source-backed candidate property index |
| Operator indexing | `scripts/migrate_library_to_core.py`; `scripts/migrate_tf_library_to_stablehlo.py` | Core ATen or StableHLO index and attribution audit |
| Candidate screening | `scripts/screen_properties.py` | Index of properties with an activated execution witness |
| Online testing | `scripts/fuzz.py` | Generated tests, verdicts, minimized candidate violations, batch checkpoint |

The workflow follows the paper's four design steps: collect and attribute changes, mine and screen properties, generate tests for an operator/property pair, and check executions with activation feedback. A property violation is a defect candidate and requires triage against its source evidence and the target framework's intended behavior.

## Requirements

- Python 3.10 or newer; a Git checkout of the target PyTorch or TensorFlow version.
- PyTorch installation matching the checkout for the PyTorch line; TensorFlow with XLA and its StableHLO bridge for the TensorFlow line.
- GitHub read token for practical collection throughput.
- An OpenAI-compatible chat-completions endpoint that returns JSON in `choices[0].message.content`. A local vLLM deployment works. No model is bundled.

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

For a local vLLM endpoint, set `MRMINER_API_BASE=http://127.0.0.1:8008/v1`, `MRMINER_MODEL` to the served model ID, and `MRMINER_VLLM_EXTENSIONS=1`. For a service requiring bearer authentication, set `MRMINER_API_KEY`. The example file deliberately contains no credentials.

## PyTorch quickstart

Run from this directory. Choose a date window ending before or at the tested commit's history. `--max-prs 3` is useful for a small collector trial; omit it for full collection.

```bash
python -m mrminer.s1_miner.run_s1 \
  --repo pytorch/pytorch --since 2025-07-01 --until 2025-07-31 \
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

python scripts/screen_properties.py --framework pytorch \
  --input runs/pytorch/operator_oracle_index_core.json \
  --output runs/pytorch/operator_oracle_index_screened.json \
  --audit runs/pytorch/screen_audit.jsonl

python scripts/fuzz.py --framework pytorch \
  --index runs/pytorch/operator_oracle_index_screened.json \
  --output-dir runs/pytorch/fuzz --iterations 30
```

For a wider mining run, remove `--pr-number`, pass `--operator` repeatedly, or provide one `op:aten::<name>` per line with `--operators-file`. The migration audit records operators that could not be shown to map to a Core ATen operator; `--core-only` keeps these out of the indexed library, while the source candidate file remains available for inspection.

## TensorFlow quickstart

The TensorFlow collector uses the same PR schema. The attribution step first enumerates the installed raw op registry and then links PR edits to registration or implementation sites. The bundled StableHLO specification inventory defines the operator universe; the pinned `StablehloOps.td` is audited against it. A representative lowering probe establishes each mapped index entry.

```bash
python -m mrminer.s1_miner.run_s1 \
  --repo tensorflow/tensorflow --since 2026-09-01 --until 2026-09-30 \
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

python scripts/screen_properties.py --framework tensorflow \
  --input runs/tensorflow/operator_oracle_index_tf_stablehlo.json \
  --output runs/tensorflow/operator_oracle_index_screened.json \
  --audit runs/tensorflow/screen_audit.jsonl
python scripts/fuzz.py --framework tensorflow \
  --index runs/tensorflow/operator_oracle_index_screened.json \
  --output-dir runs/tensorflow/fuzz --iterations 30
```

If TensorFlow's bridge is unavailable for an operator, the migration audit records it as deferred. The screening command uses the observed-lowering index, not the contextual attachment file, so the published test run does not imply that an unobserved `custom_call` was emitted.

## Outputs and repeat runs

Collection, PyTorch and TensorFlow mining, and property screening keep resume state under `runs/`. The property screening JSONL audit stores the generated witness and verdict for every candidate. Fuzzing appends `tests.jsonl` and `results.jsonl`, writes deduplicated candidates to `findings.jsonl`, and numbers each completed invocation in `checkpoint.json`. Re-running the command starts a new feedback loop and appends its results; feedback history and seed promotion are scoped to one invocation.

Generated outputs and common credential files are ignored by Git. Publish this directory as a new repository or archive if the parent project's history is not intended for release. Git commits record the configured author name and email; choose the public identity before committing. Review the chosen license and staged files before publication.
