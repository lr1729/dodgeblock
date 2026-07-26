#!/usr/bin/env bash
# Success-visitation bonus vs the measured control (rung 600 from rung-700-x13,
# 8M frames, det 0.344). Two budgets: the bonus is bounded so its TOTAL
# contribution over an episode is at most `budget` times the task reward, no
# matter what the discriminator outputs. Budget is the control surface, not a
# raw coefficient -- the coefficient was what collapsed rescue distillation.
set -uo pipefail

CODE=/home/lr/dodgeblock-v5-code
INIT=/home/lr/dodgeblock-ladder/rung-700-x13/checkpoints/latest.pt
ROOT=/home/lr/dodgeblock-svm
RESULTS=$ROOT/RESULTS.md
CONTROL=0.344

mkdir -p "$ROOT"
[ -f "$RESULTS" ] || cat > "$RESULTS" <<EOF
# Success-visitation bonus — target 600, 8M frames, init rung-700-x13
Control (no bonus): det success $CONTROL, per-layer 0.9313.
Bonus: budget * tanh(clip(discriminator log-odds)) / mean_episode_length.
Discriminator: BCE on the policy's OWN successful vs failed episode states.

EOF

run_one() {
  local name=$1 budget=$2
  local dir=$ROOT/$name
  mkdir -p "$dir/checkpoints"
  echo "[$(date +%H:%M)] launching $name budget=$budget" >> "$RESULTS"
  ( cd "$CODE" && \
    CUDA_VISIBLE_DEVICES=1 \
    DODGEBLOCK_PYTHON=/home/lr/envs/dodgeblock-rl/bin/python \
    DODGEBLOCK_WORKERS=8 DODGEBLOCK_ENVS_PER_WORKER=64 \
    DODGEBLOCK_TOTAL_FRAMES=8000000 DODGEBLOCK_TARGET_HEIGHT=600 \
    DODGEBLOCK_REWARD_MODE=target DODGEBLOCK_CELL_EVAL_ENVS=0 \
    DODGEBLOCK_COMPILE=1 DODGEBLOCK_CHECKPOINT_INTERVAL=4000000 \
    DODGEBLOCK_EVAL_AFTER_TRAIN=1 DODGEBLOCK_EVAL_EPISODES=512 \
    DODGEBLOCK_SVM_BUDGET="$budget" \
    DODGEBLOCK_CHECKPOINT_DIR="$dir/checkpoints" \
    DODGEBLOCK_INITIALIZE_FROM="$INIT" \
    DODGEBLOCK_DEVICE=cuda \
    ./rl/run-ppo-v4.sh ) >> "$dir/run.log" 2>&1

  python3 - "$dir/run.log" "$name" "$budget" "$RESULTS" "$CONTROL" <<'PY'
import json, sys
log, name, budget, results, control = sys.argv[1:6]
text = open(log).read()
index = text.rfind('"event": "evaluation"')
line = f'- **{name}** budget={budget}: '
if index < 0:
    line += 'NO EVAL (crashed)\n'
else:
    data, _ = json.JSONDecoder().raw_decode(text[text.rfind('{', 0, index):])
    success = float(data.get('target_success', 0.0))
    layers = max(1.0, float(data.get('target_height', 600)) / 40.0)
    per_layer = success ** (1.0 / layers) if success > 0 else 0.0
    # the discriminator's own health decides whether a null result means
    # "the bonus does not help" or "the features cannot separate at all"
    last = text.rfind('{"event": "progress"')
    svm = {}
    if last > 0:
        try:
            progress, _ = json.JSONDecoder().raw_decode(text[last:])
            svm = progress.get('svm') or {}
        except Exception:
            pass
    line += (f'det **{success:.3f}** (control {control}) per-layer {per_layer:.4f} '
             f'median {data.get("median_height")} '
             f'| discriminator acc {svm.get("accuracy")} sep {svm.get("separation")}\n')
open(results, 'a').write(line)
print(line.strip())
PY
}

run_one svm-half 0.5
run_one svm-full 1.5
echo "[$(date +%H:%M)] sweep complete" >> "$RESULTS"
