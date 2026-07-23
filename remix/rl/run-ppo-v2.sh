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
  --rollout "${DODGEBLOCK_ROLLOUT:-256}"
  --epochs "${DODGEBLOCK_EPOCHS:-3}"
  --minibatch "${DODGEBLOCK_MINIBATCH:-4096}"
  --device "${DODGEBLOCK_DEVICE:-cuda}"
  --checkpoint-dir "$checkpoint_dir"
  --archive-probability "${DODGEBLOCK_ARCHIVE_PROBABILITY:-0.25}"
  --seed "${DODGEBLOCK_SEED:-7}"
  --gamma "${DODGEBLOCK_GAMMA:-1.0}"
  --gae-lambda "${DODGEBLOCK_GAE_LAMBDA:-0.995}"
  --learning-rate "${DODGEBLOCK_LEARNING_RATE:-0.00025}"
  --learning-rate-end "${DODGEBLOCK_LEARNING_RATE_END:-0.000025}"
  --weight-decay "${DODGEBLOCK_WEIGHT_DECAY:-0.00001}"
  --clip-coef "${DODGEBLOCK_CLIP_COEF:-0.1}"
  --entropy-coef-start "${DODGEBLOCK_ENTROPY_COEF_START:-0.01}"
  --entropy-coef-end "${DODGEBLOCK_ENTROPY_COEF_END:-0.0001}"
  --value-coef "${DODGEBLOCK_VALUE_COEF:-0.5}"
  --target-kl "${DODGEBLOCK_TARGET_KL:-0.03}"
  --death-penalty "${DODGEBLOCK_DEATH_PENALTY:-1.0}"
  --alive-reward "${DODGEBLOCK_ALIVE_REWARD:-0.001}"
  --target-height "${DODGEBLOCK_TARGET_HEIGHT:-10000}"
  --reward-mode "${DODGEBLOCK_REWARD_MODE:-target}"
  --demonstration-probability "${DODGEBLOCK_DEMONSTRATION_PROBABILITY:-0.8}"
  --demonstration-probability-end "${DODGEBLOCK_DEMONSTRATION_PROBABILITY_END:-0.2}"
  --demonstration-snapshot-capacity "${DODGEBLOCK_DEMONSTRATION_SNAPSHOT_CAPACITY:-256}"
  --reverse-curriculum-initial-frames "${DODGEBLOCK_REVERSE_CURRICULUM_INITIAL_FRAMES:-60}"
  --demonstration-randomize-probability "${DODGEBLOCK_DEMONSTRATION_RANDOMIZE_PROBABILITY:-1}"
  --checkpoint-interval "${DODGEBLOCK_CHECKPOINT_INTERVAL:-5000000}"
)

if [[ -n "${DODGEBLOCK_DEMONSTRATIONS:-}" ]]; then
  IFS=':' read -r -a demonstrations <<< "${DODGEBLOCK_DEMONSTRATIONS}"
  for demonstration in "${demonstrations[@]}"; do
    args+=(--demonstration "$demonstration")
  done
fi

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
