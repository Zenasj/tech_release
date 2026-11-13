#!/usr/bin/env bash
# Continue a PR-backed NoCondition pilot after property and seed preparation.
# Example (from tech_release):
#   bash scripts/run_no_condition_pilot.sh pytorch
# Inputs: synthesized properties and the seed corpus exported from RQ4 Full.
# Outputs: runs/no_condition_pilot_2500/<framework>/{witness,screen,no_condition} artifacts.
set -euo pipefail

framework="${1:?usage: run_no_condition_pilot.sh pytorch|tensorflow}"
case "$framework" in
  pytorch)
    short=pt
    corpus="${MRMINER_PR_CORPUS:-runs/pytorch/pr_corpus/pytorch__pytorch}"
    ;;
  tensorflow)
    short=tf
    corpus="${MRMINER_PR_CORPUS:-runs/tensorflow/pr_corpus/tensorflow__tensorflow}"
    ;;
  *) echo "unknown framework: $framework" >&2; exit 2 ;;
esac

cd "$(dirname "$0")/.."
prepared="runs/no_condition_pilot_500/$framework"
root="runs/no_condition_pilot_2500/$framework"
seeds="runs/no_condition/full_seeds_$framework.jsonl"
mkdir -p "$root"
export PYTHONPATH="$PWD"
export NO_PROXY=127.0.0.1,localhost
export MRMINER_API_BASE="${MRMINER_API_BASE:-http://127.0.0.1:8008/v1}"
export MRMINER_MODEL="${MRMINER_MODEL:-qwq-32b}"
export MRMINER_GPU_POOL="${MRMINER_GPU_POOL:-}"

echo "[$framework] waiting for executable properties; using Full seed corpus $seeds"
if [[ ! -s "$seeds" ]]; then
  echo "[$framework] Full seed corpus missing: $seeds" >&2
  exit 1
fi
while [[ ! -s "$prepared/operator_properties.json" ]]; do
  if [[ ! -s "$prepared/operator_properties.json" ]] && ! tmux has-session -t "nc_pilot_$short" 2>/dev/null; then
    echo "[$framework] property synthesis exited without output" >&2
    exit 1
  fi
  sleep 15
done

echo "[$framework] adapting PR regression tests"
python3 -u scripts/prepare_regression_witnesses.py \
  --input "$prepared/operator_properties.json" --corpus "$corpus" \
  --output "$root/regression_witnesses.jsonl" \
  --audit "$root/regression_witness_adaptations.jsonl" \
  --max-tests-per-property 1

# Serialize GPU use so screening and generation do not contend.
exec 9>"runs/no_condition_pilot_2500/gpu.lock"
flock -x 9
echo "[$framework] screening properties on GPU pool $MRMINER_GPU_POOL"
python3 -u scripts/screen_properties.py --framework "$framework" \
  --input "$prepared/operator_properties.json" \
  --witnesses "$root/regression_witnesses.jsonl" --corpus "$corpus" \
  --output "$root/operator_oracle_index_screened.json" \
  --audit "$root/screen_audit.jsonl"

echo "[$framework] generating independent NoCondition tests"
python3 -u scripts/fuzz.py --framework "$framework" --judge-mode no_condition \
  --index "$root/operator_oracle_index_screened.json" \
  --seeds "$seeds" \
  --screen-audit "$root/screen_audit.jsonl" \
  --output-dir "$root/no_condition" \
  --iterations 10000 --max-tests 2500 --seed 0
echo "[$framework] complete"
