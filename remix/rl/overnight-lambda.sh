#!/usr/bin/env bash
# The credit-horizon sweep. The death autopsy measured the causal window of a
# death at 10-20 frames (88% of deaths escapable with 20 frames of different
# action). GAE runs at lambda=0.995 per world frame -- a 200-frame horizon, so
# credit for the fatal action is diluted across ~10-20x more frames than were
# involved. This sweeps lambda down onto the measured window.
#
# Unlike the four failed interventions above, this adds no reward, no bonus and
# no demonstrations. It only changes how existing credit is attributed.
set -uo pipefail

CODE=/home/lr/dodgeblock-v5-code
INIT=/home/lr/dodgeblock-ladder/rung-700-x13/checkpoints/latest.pt
ROOT=/home/lr/dodgeblock-lambda
RESULTS=$ROOT/RESULTS.md
CONTROL=0.344

mkdir -p "$ROOT"

# Run against a frozen copy of the tree. A sync into the live code directory
# mid-sweep once killed three arms: the deploy added a network head, and every
# arm then failed to load its own head-less checkpoint. rsync replaces files
# rather than editing them, so hard links keep this snapshot pinned to the code
# the sweep started with, at no copy cost.
SNAPSHOT=$ROOT/code
rm -rf "$SNAPSHOT"
cp -al "$CODE" "$SNAPSHOT"
CODE=$SNAPSHOT

[ -f "$RESULTS" ] || cat > "$RESULTS" <<EOF
# Credit-horizon sweep — target 600, 8M frames, init rung-700-x13
Control (lambda 0.995, horizon 200 frames): det success $CONTROL, per-layer 0.9313.
Autopsy: 88% of deaths escapable 20 frames out, 72% escapable 10 frames out.
Hypothesis: the credit horizon should match the causal window, not exceed it 10x.

EOF

run_one() {
  local name=$1 lambda=$2
  local dir=$ROOT/$name
  mkdir -p "$dir/checkpoints"
  echo "[$(date +%H:%M)] launching $name lambda=$lambda" >> "$RESULTS"
  ( cd "$CODE" && \
    CUDA_VISIBLE_DEVICES=1 \
    DODGEBLOCK_PYTHON=/home/lr/envs/dodgeblock-rl/bin/python \
    DODGEBLOCK_WORKERS=8 DODGEBLOCK_ENVS_PER_WORKER=64 \
    DODGEBLOCK_TOTAL_FRAMES=8000000 DODGEBLOCK_TARGET_HEIGHT=600 \
    DODGEBLOCK_REWARD_MODE=target DODGEBLOCK_CELL_EVAL_ENVS=0 \
    DODGEBLOCK_COMPILE=1 DODGEBLOCK_CHECKPOINT_INTERVAL=4000000 \
    DODGEBLOCK_EVAL_AFTER_TRAIN=1 DODGEBLOCK_EVAL_EPISODES=512 \
    DODGEBLOCK_GAE_LAMBDA="$lambda" \
    DODGEBLOCK_CHECKPOINT_DIR="$dir/checkpoints" \
    DODGEBLOCK_INITIALIZE_FROM="$INIT" \
    DODGEBLOCK_DEVICE=cuda \
    ./rl/run-ppo-v4.sh ) >> "$dir/run.log" 2>&1

  python3 - "$dir/run.log" "$name" "$lambda" "$RESULTS" "$CONTROL" <<'PY'
import json, sys
log, name, lam, results, control = sys.argv[1:6]
text = open(log).read()
index = text.rfind('"event": "evaluation"')
line = f'- **{name}** lambda={lam} (horizon {1/(1-float(lam)):.0f}f): '
if index < 0:
    line += 'NO EVAL (crashed)\n'
else:
    data, _ = json.JSONDecoder().raw_decode(text[text.rfind('{', 0, index):])
    success = float(data.get('target_success', 0.0))
    layers = max(1.0, float(data.get('target_height', 600)) / 40.0)
    per_layer = success ** (1.0 / layers) if success > 0 else 0.0
    line += (f'det **{success:.3f}** (control {control}) per-layer {per_layer:.4f} '
             f'median {data.get("median_height")}\n')
open(results, 'a').write(line)
print(line.strip())
PY
}

run_one lam-90 0.90
run_one lam-95 0.95
run_one lam-97 0.97
echo "[$(date +%H:%M)] sweep complete" >> "$RESULTS"

# Re-run the autopsy on the winner. If the credit horizon is the real lever,
# viability at K=10 should FALL -- the agent should stop dying in situations
# that a third of a second of any other action would have survived.
BEST=$(python3 - "$RESULTS" <<'PY'
import re, sys
best, name = -1.0, ''
for line in open(sys.argv[1]):
    match = re.search(r'\*\*(lam-\d+)\*\*.*det \*\*([0-9.]+)\*\*', line)
    if match and float(match.group(2)) > best:
        best, name = float(match.group(2)), match.group(1)
print(f'{name} {best}')
PY
)
read -r BEST_NAME BEST_SUCCESS <<< "$BEST"
echo "[$(date +%H:%M)] best $BEST_NAME det=$BEST_SUCCESS" >> "$RESULTS"

if [ -n "$BEST_NAME" ]; then
  ( cd "$CODE" && CUDA_VISIBLE_DEVICES=1 \
    /home/lr/envs/dodgeblock-rl/bin/python rl/record_trace.py \
      "$ROOT/$BEST_NAME/checkpoints/latest.pt" --episodes 256 \
      --target-height 10000 --max-frames 6000 \
      --dump-all "$ROOT/$BEST_NAME/deaths.jsonl" --out "$ROOT/$BEST_NAME/best.json" \
    && node rl/death-autopsy.mjs "$ROOT/$BEST_NAME/deaths.jsonl" \
      > "$ROOT/$BEST_NAME/autopsy.json" ) >> "$ROOT/$BEST_NAME/run.log" 2>&1
  echo '```' >> "$RESULTS"
  cat "$ROOT/$BEST_NAME/autopsy.json" >> "$RESULTS" 2>/dev/null
  echo '```' >> "$RESULTS"
fi
