#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
checkpoint_dir="${DODGEBLOCK_CHECKPOINT_DIR:-${HOME}/dodgeblock-ppo-v4/checkpoints}"
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
  --seed "${DODGEBLOCK_SEED:-7}"
  --gamma "${DODGEBLOCK_GAMMA:-1.0}"
  --gae-lambda "${DODGEBLOCK_GAE_LAMBDA:-0.995}"
  --learning-rate "${DODGEBLOCK_LEARNING_RATE:-0.00025}"
  --learning-rate-end "${DODGEBLOCK_LEARNING_RATE_END:-0.000025}"
  --weight-decay "${DODGEBLOCK_WEIGHT_DECAY:-0.00001}"
  --clip-coef "${DODGEBLOCK_CLIP_COEF:-0.1}"
  --focus-entropy-coef-start "${DODGEBLOCK_FOCUS_ENTROPY_START:-0.01}"
  --focus-entropy-coef-end "${DODGEBLOCK_FOCUS_ENTROPY_END:-0.0001}"
  --direction-entropy-coef-start "${DODGEBLOCK_DIRECTION_ENTROPY_START:-0.005}"
  --direction-entropy-coef-end "${DODGEBLOCK_DIRECTION_ENTROPY_END:-0.0001}"
  --value-coef "${DODGEBLOCK_VALUE_COEF:-0.5}"
  --target-kl "${DODGEBLOCK_TARGET_KL:-0.03}"
  --target-height "${DODGEBLOCK_TARGET_HEIGHT:-10000}"
  --reward-mode "${DODGEBLOCK_REWARD_MODE:-target}"
  --death-penalty "${DODGEBLOCK_DEATH_PENALTY:-1.0}"
  --shaping-cover "${DODGEBLOCK_SHAPING_COVER:-0.0}"
  --shaping-charge "${DODGEBLOCK_SHAPING_CHARGE:-0.0}"
  --svm-budget "${DODGEBLOCK_SVM_BUDGET:-0.0}"
  --svm-clip "${DODGEBLOCK_SVM_CLIP:-4.0}"
  --hazard-coef "${DODGEBLOCK_HAZARD_COEF:-0.0}"
  --action-repeat "${DODGEBLOCK_ACTION_REPEAT:-1}"
  --alive-reward "${DODGEBLOCK_ALIVE_REWARD:-0.0}"
  --cell-bank-probability "${DODGEBLOCK_CELL_BANK_PROBABILITY:-0.8}"
  --cell-heldout-fraction "${DODGEBLOCK_CELL_HELDOUT_FRACTION:-0.1}"
  --cell-band-height "${DODGEBLOCK_CELL_BAND_HEIGHT:-400}"
  --cell-eval-envs "${DODGEBLOCK_CELL_EVAL_ENVS:-16}"
  --checkpoint-interval "${DODGEBLOCK_CHECKPOINT_INTERVAL:-5000000}"
)

if [[ "${DODGEBLOCK_COMPILE:-0}" == "1" ]]; then
  args+=(--compile)
fi

# Rescue distillation: an auxiliary cross-entropy loss on verified escapes
# searched from the previous rung's own deaths.
if [[ -n "${DODGEBLOCK_DEMO_DATASET:-}" ]]; then
  args+=(
    --demo-dataset "$DODGEBLOCK_DEMO_DATASET"
    --demo-seeds "${DODGEBLOCK_DEMO_SEEDS:-1001}"
    --demo-minibatch "${DODGEBLOCK_DEMO_MINIBATCH:-1024}"
    --demo-coef-start "${DODGEBLOCK_DEMO_COEF_START:-0.5}"
    --demo-coef-end "${DODGEBLOCK_DEMO_COEF_END:-0.1}"
  )
fi

if [[ -n "${DODGEBLOCK_CELL_BANKS:-}" ]]; then
  IFS=':' read -r -a banks <<< "${DODGEBLOCK_CELL_BANKS}"
  for bank in "${banks[@]}"; do
    args+=(--cell-bank "$bank")
  done
fi
if [[ -n "${DODGEBLOCK_CELL_BANK_GLOB:-}" ]]; then
  while IFS= read -r bank; do
    args+=(--cell-bank "$bank")
  done < <(compgen -G "$DODGEBLOCK_CELL_BANK_GLOB" | sort)
fi
if [[ "${DODGEBLOCK_CELL_EVAL_ENVS:-16}" != "0" ]] &&
  [[ " ${args[*]} " != *" --cell-bank "* ]]; then
  echo "cell evaluation environments require DODGEBLOCK_CELL_BANKS or DODGEBLOCK_CELL_BANK_GLOB" >&2
  exit 2
fi

if [[ -e "$checkpoint_dir/latest.pt" ]]; then
  args+=(--resume "$checkpoint_dir/latest.pt")
elif [[ -n "${DODGEBLOCK_INITIALIZE_FROM:-}" ]]; then
  args+=(--initialize-from "$DODGEBLOCK_INITIALIZE_FROM")
fi

if [[ "${DODGEBLOCK_EVAL_AFTER_TRAIN:-0}" == "1" ]]; then
  "$python_bin" "${args[@]}"
  "$python_bin" "$root/rl/evaluate_ppo_v2.py" "$checkpoint_dir/latest.pt" \
    --workers "${DODGEBLOCK_EVAL_WORKERS:-8}" \
    --episodes "${DODGEBLOCK_EVAL_EPISODES:-256}" \
    --target-height "${DODGEBLOCK_TARGET_HEIGHT:-10000}" \
    --device "${DODGEBLOCK_DEVICE:-cuda}"
else
  exec "$python_bin" "${args[@]}"
fi
