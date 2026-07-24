#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case_dir="${DODGEBLOCK_CASE_DIR:?set DODGEBLOCK_CASE_DIR to a death-case directory}"
output_dir="${DODGEBLOCK_SEARCH_OUTPUT:?set DODGEBLOCK_SEARCH_OUTPUT}"
workers="${DODGEBLOCK_SEARCH_WORKERS:-12}"
seed_base="${DODGEBLOCK_SEARCH_SEED_BASE:-200}"
search_seed_base="${DODGEBLOCK_SEARCH_RNG_BASE:-17000}"
iterations="${DODGEBLOCK_SEARCH_ITERATIONS:-2000000}"
rewind_frames="${DODGEBLOCK_SEARCH_REWIND:-120}"
checkpoint_interval="${DODGEBLOCK_SEARCH_CHECKPOINT_INTERVAL:-50000}"

mapfile -t cases < <(find "$case_dir" -type f -name '*.json.gz' | sort)
if (( ${#cases[@]} < workers )); then
  echo "requested $workers workers but found only ${#cases[@]} cases in $case_dir" >&2
  exit 2
fi

mkdir -p "$output_dir"
declare -a children=()
cleanup() {
  if (( ${#children[@]} > 0 )); then
    kill "${children[@]}" 2>/dev/null || true
  fi
}
trap cleanup INT TERM

for ((index = 0; index < workers; index++)); do
  query=$((index + 1))
  case_path="${cases[$index]}"
  query_dir="$output_dir/query-$query"
  mkdir -p "$query_dir"
  printf '%s\n' "$case_path" > "$query_dir/source-case.txt"

  node "$root/rl/go-explore.mjs" \
    --seed $((seed_base + query)) \
    --search-seed $((search_seed_base + query)) \
    --start-case "$case_path" \
    --start-rewind "$rewind_frames" \
    --iterations "$iterations" \
    --checkpoint-interval "$checkpoint_interval" \
    --output-dir "$query_dir" \
    > "$query_dir/run.log" 2>&1 &
  children+=("$!")
done

failed=0
for child in "${children[@]}"; do
  wait "$child" || failed=1
done
exit "$failed"
