#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${DODGEBLOCK_PYTHON:-python3}"
dataset="${DODGEBLOCK_DEMO_DATASET:-${HOME}/dodgeblock-demo-dataset-v5}"
checkpoint_dir="${DODGEBLOCK_BC_CHECKPOINT_DIR:-${HOME}/dodgeblock-bc-v5/checkpoints}"

extra_args=()
if [[ -n "${DODGEBLOCK_BC_WEIGHTS_FROM:-}" ]]; then
  extra_args+=(--weights-from "$DODGEBLOCK_BC_WEIGHTS_FROM")
fi
if [[ -n "${DODGEBLOCK_FIXED_CONTROL_INTERVAL:-}" ]]; then
  extra_args+=(
    --fixed-control-interval "$DODGEBLOCK_FIXED_CONTROL_INTERVAL"
  )
fi
if [[ -n "${DODGEBLOCK_BC_STEPS_PER_EPOCH:-}" ]]; then
  extra_args+=(--steps-per-epoch "$DODGEBLOCK_BC_STEPS_PER_EPOCH")
fi
if [[ -n "${DODGEBLOCK_CORRECTION_TRAIN_SEEDS:-}" ]]; then
  extra_args+=(
    --correction-dataset "${DODGEBLOCK_CORRECTION_DATASET:-$dataset}"
    --correction-train-seeds "$DODGEBLOCK_CORRECTION_TRAIN_SEEDS"
    --correction-fraction "${DODGEBLOCK_CORRECTION_FRACTION:-0.75}"
    --correction-repeat-coef "${DODGEBLOCK_CORRECTION_REPEAT_COEF:-0.05}"
  )
fi
if [[ -n "${DODGEBLOCK_CORRECTION_VALIDATION_SEEDS:-}" ]]; then
  extra_args+=(
    --correction-validation-seeds "$DODGEBLOCK_CORRECTION_VALIDATION_SEEDS"
  )
fi

exec "$python_bin" "$root/rl/train_bc_v5.py" \
  --dataset "$dataset" \
  --train-seeds "${DODGEBLOCK_BC_TRAIN_SEEDS:-1-12}" \
  --validation-seeds "${DODGEBLOCK_BC_VALIDATION_SEEDS:-13-16}" \
  --epochs "${DODGEBLOCK_BC_EPOCHS:-30}" \
  --batch-size "${DODGEBLOCK_BC_BATCH_SIZE:-4096}" \
  --validation-batch-size "${DODGEBLOCK_BC_VALIDATION_BATCH_SIZE:-4096}" \
  --learning-rate "${DODGEBLOCK_BC_LEARNING_RATE:-0.0003}" \
  --learning-rate-end "${DODGEBLOCK_BC_LEARNING_RATE_END:-0.00003}" \
  --weight-decay "${DODGEBLOCK_BC_WEIGHT_DECAY:-0.00001}" \
  --focus-positive-weight "${DODGEBLOCK_BC_FOCUS_WEIGHT:-1.0}" \
  --early-stop-patience "${DODGEBLOCK_BC_EARLY_STOP_PATIENCE:-5}" \
  --selection-metric "${DODGEBLOCK_BC_SELECTION_METRIC:-decision_validation}" \
  --device "${DODGEBLOCK_DEVICE:-cuda}" \
  --checkpoint-dir "$checkpoint_dir" \
  "${extra_args[@]}"
