#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
checkpoint_dir="${DODGEBLOCK_CHECKPOINT_DIR:-${HOME}/dodgeblock-ppo-v5/checkpoints}"
python_bin="${DODGEBLOCK_PYTHON:-python3}"
dataset="${DODGEBLOCK_DEMO_DATASET:-${HOME}/dodgeblock-demo-dataset-v5}"
mkdir -p "$checkpoint_dir"

args=(
  "$root/rl/ppo_v2.py"
  --workers "${DODGEBLOCK_WORKERS:-8}"
  --envs-per-worker "${DODGEBLOCK_ENVS_PER_WORKER:-64}"
  --total-frames "${DODGEBLOCK_TOTAL_FRAMES:-120000000}"
  --rollout "${DODGEBLOCK_ROLLOUT:-256}"
  --epochs "${DODGEBLOCK_EPOCHS:-3}"
  --minibatch "${DODGEBLOCK_MINIBATCH:-4096}"
  --device "${DODGEBLOCK_DEVICE:-cuda}"
  --checkpoint-dir "$checkpoint_dir"
  --seed "${DODGEBLOCK_SEED:-7}"
  --gamma "${DODGEBLOCK_GAMMA:-1.0}"
  --gae-lambda "${DODGEBLOCK_GAE_LAMBDA:-0.995}"
  --learning-rate "${DODGEBLOCK_LEARNING_RATE:-0.00025}"
  --learning-rate-end "${DODGEBLOCK_LEARNING_RATE_END:-0.000025}"
  --weight-decay "${DODGEBLOCK_WEIGHT_DECAY:-0.00001}"
  --clip-coef "${DODGEBLOCK_CLIP_COEF:-0.1}"
  --focus-entropy-coef-start "${DODGEBLOCK_FOCUS_ENTROPY_START:-0.003}"
  --focus-entropy-coef-end "${DODGEBLOCK_FOCUS_ENTROPY_END:-0.0001}"
  --direction-entropy-coef-start "${DODGEBLOCK_DIRECTION_ENTROPY_START:-0.002}"
  --direction-entropy-coef-end "${DODGEBLOCK_DIRECTION_ENTROPY_END:-0.0001}"
  --value-coef "${DODGEBLOCK_VALUE_COEF:-0.5}"
  --target-kl "${DODGEBLOCK_TARGET_KL:-0.03}"
  --target-height "${DODGEBLOCK_TARGET_HEIGHT:-10000}"
  --reward-mode "${DODGEBLOCK_REWARD_MODE:-target}"
  --death-penalty "${DODGEBLOCK_DEATH_PENALTY:-1.0}"
  --alive-reward "${DODGEBLOCK_ALIVE_REWARD:-0.0}"
  --cell-eval-envs 0
  --trajectory-start-probability "${DODGEBLOCK_TRAJECTORY_START_PROBABILITY:-0.5}"
  --demo-dataset "$dataset"
  --demo-seeds "${DODGEBLOCK_DEMO_SEEDS:-1-12}"
  --demo-minibatch "${DODGEBLOCK_DEMO_MINIBATCH:-1024}"
  --demo-coef-start "${DODGEBLOCK_DEMO_COEF_START:-0.1}"
  --demo-coef-end "${DODGEBLOCK_DEMO_COEF_END:-0.01}"
  --demo-focus-positive-weight "${DODGEBLOCK_DEMO_FOCUS_WEIGHT:-1.0}"
  --sticky-action-head
  --checkpoint-interval "${DODGEBLOCK_CHECKPOINT_INTERVAL:-2500000}"
)

if [[ -n "${DODGEBLOCK_TRAJECTORY_BANK_GLOB:-}" ]]; then
  while IFS= read -r bank; do
    args+=(--trajectory-bank "$bank")
  done < <(compgen -G "$DODGEBLOCK_TRAJECTORY_BANK_GLOB" | sort -V)
else
  for seed in $(seq 1 12); do
    bank="${dataset}/seed-${seed}/trajectory-bank.json.gz"
    [[ -e "$bank" ]] && args+=(--trajectory-bank "$bank")
  done
fi
if [[ " ${args[*]} " != *" --trajectory-bank "* ]]; then
  echo "no trajectory banks found under: $dataset" >&2
  exit 2
fi

if [[ -e "$checkpoint_dir/latest.pt" ]]; then
  args+=(--resume "$checkpoint_dir/latest.pt")
else
  bc_checkpoint="${DODGEBLOCK_BC_CHECKPOINT:-${HOME}/dodgeblock-bc-v5/checkpoints/best.pt}"
  if [[ ! -e "$bc_checkpoint" ]]; then
    echo "BC checkpoint does not exist: $bc_checkpoint" >&2
    exit 2
  fi
  args+=(--initialize-from "$bc_checkpoint")
fi

if [[ "${DODGEBLOCK_EVAL_AFTER_TRAIN:-1}" == "1" ]]; then
  "$python_bin" "${args[@]}"
  "$python_bin" "$root/rl/evaluate_ppo_v2.py" "$checkpoint_dir/latest.pt" \
    --workers "${DODGEBLOCK_EVAL_WORKERS:-8}" \
    --episodes "${DODGEBLOCK_EVAL_EPISODES:-256}" \
    --device "${DODGEBLOCK_DEVICE:-cuda}"
else
  exec "$python_bin" "${args[@]}"
fi
