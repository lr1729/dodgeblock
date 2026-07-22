#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
checkpoint_dir="${DODGEBLOCK_CHECKPOINT_DIR:-${HOME}/dodgeblock-r2d2/checkpoints}"
python_bin="${DODGEBLOCK_PYTHON:-python3}"
mkdir -p "$checkpoint_dir"

args=(
  "$root/rl/train_v2.py"
  --workers "${DODGEBLOCK_WORKERS:-8}"
  --envs-per-worker "${DODGEBLOCK_ENVS_PER_WORKER:-64}"
  --total-frames "${DODGEBLOCK_TOTAL_FRAMES:-2000000000}"
  --device "${DODGEBLOCK_DEVICE:-cuda}"
  --checkpoint-dir "$checkpoint_dir"
  --archive-probability "${DODGEBLOCK_ARCHIVE_PROBABILITY:-0.25}"
)

if [[ -e "$checkpoint_dir/latest.pt" ]]; then
  args+=(--resume "$checkpoint_dir/latest.pt")
fi

exec "$python_bin" "${args[@]}"
