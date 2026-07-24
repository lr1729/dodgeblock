#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
demo_dir="${DODGEBLOCK_DEMO_DIR:?set DODGEBLOCK_DEMO_DIR}"
output_dir="${DODGEBLOCK_COUNTERFACTUAL_OUTPUT:?set DODGEBLOCK_COUNTERFACTUAL_OUTPUT}"
workers="${DODGEBLOCK_COUNTERFACTUAL_WORKERS:-12}"
shard_base="${DODGEBLOCK_COUNTERFACTUAL_SHARD_BASE:-4000}"
search_seed_base="${DODGEBLOCK_COUNTERFACTUAL_RNG_BASE:-24000}"
horizon="${DODGEBLOCK_COUNTERFACTUAL_HORIZON:-360}"
futures="${DODGEBLOCK_COUNTERFACTUAL_FUTURES:-2}"
branch_frames="${DODGEBLOCK_COUNTERFACTUAL_BRANCH_FRAMES:-6}"
stride="${DODGEBLOCK_COUNTERFACTUAL_STRIDE:-60}"
max_states="${DODGEBLOCK_COUNTERFACTUAL_MAX_STATES:-128}"
selection_mode="${DODGEBLOCK_COUNTERFACTUAL_SELECTION_MODE:-mixed}"
control_interval="${DODGEBLOCK_COUNTERFACTUAL_CONTROL_INTERVAL:-1}"

mapfile -t demos < <(find "$demo_dir" -type f -name 'demo-*.json.gz' | sort -V)
if (( ${#demos[@]} < workers )); then
  echo "requested $workers workers but found only ${#demos[@]} demos" >&2
  exit 2
fi

mkdir -p "$output_dir/logs"
declare -a children=()
cleanup() {
  if (( ${#children[@]} > 0 )); then
    kill "${children[@]}" 2>/dev/null || true
  fi
}
trap cleanup INT TERM

for ((index = 0; index < workers; index++)); do
  query=$((index + 1))
  shard=$((shard_base + query))
  node "$root/rl/counterfactual-teacher.mjs" \
    --demo "${demos[$index]}" \
    --output-dir "$output_dir" \
    --shard-seed "$shard" \
    --search-seed $((search_seed_base + query)) \
    --horizon "$horizon" \
    --futures "$futures" \
    --branch-frames "$branch_frames" \
    --selection-mode "$selection_mode" \
    --control-interval "$control_interval" \
    --stride "$stride" \
    --max-states "$max_states" \
    > "$output_dir/logs/shard-$shard.log" 2>&1 &
  children+=("$!")
done

failed=0
for child in "${children[@]}"; do
  wait "$child" || failed=1
done
exit "$failed"
