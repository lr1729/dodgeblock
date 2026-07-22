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
  --seed "${DODGEBLOCK_SEED:-1}"
  --gamma "${DODGEBLOCK_GAMMA:-0.99999}"
  --n-step "${DODGEBLOCK_N_STEP:-20}"
  --batch-size "${DODGEBLOCK_BATCH_SIZE:-64}"
  --replay-capacity "${DODGEBLOCK_REPLAY_CAPACITY:-32768}"
  --minimum-replay "${DODGEBLOCK_MINIMUM_REPLAY:-1024}"
  --replay-ratio "${DODGEBLOCK_REPLAY_RATIO:-2.0}"
)

if [[ -e "$checkpoint_dir/latest.pt" ]]; then
  args+=(--resume "$checkpoint_dir/latest.pt")
fi

if [[ "${DODGEBLOCK_EVAL_AFTER_TRAIN:-0}" == "1" ]]; then
  "$python_bin" "${args[@]}"
  "$python_bin" "$root/rl/evaluate_v2.py" "$checkpoint_dir/latest.pt" \
    --workers "${DODGEBLOCK_EVAL_WORKERS:-8}" \
    --episodes "${DODGEBLOCK_EVAL_EPISODES:-256}" \
    --device "${DODGEBLOCK_DEVICE:-cuda}"
else
  exec "$python_bin" "${args[@]}"
fi
