#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
checkpoint_dir="${DODGEBLOCK_CHECKPOINT_DIR:-${HOME}/dodgeblock-ppo-v2/checkpoints}"
python_bin="${DODGEBLOCK_PYTHON:-python3}"
mkdir -p "$checkpoint_dir"

args=(
  "$root/rl/ppo_v2.py"
  --workers "${DODGEBLOCK_WORKERS:-8}"
  --envs-per-worker "${DODGEBLOCK_ENVS_PER_WORKER:-64}"
  --total-frames "${DODGEBLOCK_TOTAL_FRAMES:-100000000}"
  --rollout "${DODGEBLOCK_ROLLOUT:-128}"
  --epochs "${DODGEBLOCK_EPOCHS:-3}"
  --minibatch "${DODGEBLOCK_MINIBATCH:-4096}"
  --device "${DODGEBLOCK_DEVICE:-cuda}"
  --checkpoint-dir "$checkpoint_dir"
  --archive-probability "${DODGEBLOCK_ARCHIVE_PROBABILITY:-0.25}"
  --seed "${DODGEBLOCK_SEED:-7}"
  --gamma "${DODGEBLOCK_GAMMA:-0.99999}"
  --gae-lambda "${DODGEBLOCK_GAE_LAMBDA:-0.95}"
  --learning-rate "${DODGEBLOCK_LEARNING_RATE:-0.00025}"
  --weight-decay "${DODGEBLOCK_WEIGHT_DECAY:-0.00001}"
  --clip-coef "${DODGEBLOCK_CLIP_COEF:-0.1}"
  --entropy-coef "${DODGEBLOCK_ENTROPY_COEF:-0.02}"
  --value-coef "${DODGEBLOCK_VALUE_COEF:-0.5}"
  --target-kl "${DODGEBLOCK_TARGET_KL:-0.03}"
  --death-penalty "${DODGEBLOCK_DEATH_PENALTY:-1.0}"
  --alive-reward "${DODGEBLOCK_ALIVE_REWARD:-0.001}"
  --checkpoint-interval "${DODGEBLOCK_CHECKPOINT_INTERVAL:-5000000}"
)

if [[ -e "$checkpoint_dir/latest.pt" ]]; then
  args+=(--resume "$checkpoint_dir/latest.pt")
fi

if [[ "${DODGEBLOCK_EVAL_AFTER_TRAIN:-0}" == "1" ]]; then
  "$python_bin" "${args[@]}"
  "$python_bin" "$root/rl/evaluate_ppo_v2.py" "$checkpoint_dir/latest.pt" \
    --workers "${DODGEBLOCK_EVAL_WORKERS:-8}" \
    --episodes "${DODGEBLOCK_EVAL_EPISODES:-256}" \
    --device "${DODGEBLOCK_DEVICE:-cuda}"
else
  exec "$python_bin" "${args[@]}"
fi
