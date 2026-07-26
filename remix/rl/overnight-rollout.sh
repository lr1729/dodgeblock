#!/usr/bin/env bash
# Does GAE truncation starve most of the batch?
#
# Measured: mean episode length 2333 frames, GAE segment 256 steps. The reward
# carries a hard-coded height potential (reward-v2.mjs), so for a critic that has
# fit that potential -- which is deterministic and trivially learnable -- the TD
# residual is c*(gamma-1) = 0 at every non-terminal step. All signal sits at the
# terminal. GAE resets each 256-step segment and bootstraps from V there, so a
# state gets real signal only when its episode ends inside its own segment:
# roughly 256/2333 = 11% of samples.
#
# Two knobs, and both are needed. rollout decides WHETHER the terminal is in the
# segment; lambda decides how much of it survives the decay to reach the state
# (0.995^256 = 0.28, but 0.9995^256 = 0.88).
#
# CONFOUND CONTROLLED: raising rollout at fixed env count would quadruple the
# batch and cut the number of policy updates 4x, which is a different variable.
# Envs are cut to match, so every arm sees a 131k-sample batch and ~61 updates.
set -uo pipefail

CODE=/home/lr/dodgeblock-v5-code
INIT=/home/lr/dodgeblock-ladder/rung-700-x13/checkpoints/latest.pt
ROOT=/home/lr/dodgeblock-rollout
RESULTS=$ROOT/RESULTS.md

mkdir -p "$ROOT"
SNAPSHOT=$ROOT/code
rm -rf "$SNAPSHOT"
cp -al "$CODE" "$SNAPSHOT"
CODE=$SNAPSHOT

[ -f "$RESULTS" ] || cat > "$RESULTS" <<'EOF'
# GAE truncation — target 600, 8M frames, init rung-700-x13, batch held at 131k
Control = rollout 256, lambda 0.995: det 0.344, mean height 424.38 [408.12, 441.02].
Episodes average 2333 frames, so at rollout 256 only ~11% of samples have their
episode terminal inside their own GAE segment. Primary metric is mean height with
a bootstrap CI -- det success cannot resolve differences below ~0.08 at n=512.

EOF

run_one() {
  local name=$1 rollout=$2 envs=$3 lambda=$4
  local dir=$ROOT/$name
  mkdir -p "$dir/checkpoints"
  echo "[$(date +%H:%M)] launching $name rollout=$rollout envs/worker=$envs lambda=$lambda" >> "$RESULTS"
  ( cd "$CODE" && \
    CUDA_VISIBLE_DEVICES=1 \
    DODGEBLOCK_PYTHON=/home/lr/envs/dodgeblock-rl/bin/python \
    DODGEBLOCK_WORKERS=8 DODGEBLOCK_ENVS_PER_WORKER="$envs" \
    DODGEBLOCK_ROLLOUT="$rollout" DODGEBLOCK_GAE_LAMBDA="$lambda" \
    DODGEBLOCK_TOTAL_FRAMES=8000000 DODGEBLOCK_TARGET_HEIGHT=600 \
    DODGEBLOCK_REWARD_MODE=target DODGEBLOCK_CELL_EVAL_ENVS=0 \
    DODGEBLOCK_COMPILE=1 DODGEBLOCK_CHECKPOINT_INTERVAL=4000000 \
    DODGEBLOCK_EVAL_AFTER_TRAIN=1 DODGEBLOCK_EVAL_EPISODES=2048 \
    DODGEBLOCK_CHECKPOINT_DIR="$dir/checkpoints" \
    DODGEBLOCK_INITIALIZE_FROM="$INIT" \
    DODGEBLOCK_DEVICE=cuda \
    ./rl/run-ppo-v4.sh ) >> "$dir/run.log" 2>&1

  python3 - "$dir" "$name" "$rollout" "$lambda" "$RESULTS" <<'PY'
import json, sys
from pathlib import Path
directory, name, rollout, lam, results = sys.argv[1:6]
text = (Path(directory) / 'run.log').read_text()
index = text.rfind('"event": "evaluation"')
line = f'- **{name}** rollout={rollout} lambda={lam}: '
if index < 0:
    line += 'NO EVAL (crashed)\n'
else:
    data, _ = json.JSONDecoder().raw_decode(text[text.rfind('{', 0, index):])
    reach = min(1.0, int(rollout) / 2333.0)
    line += (f'det {data.get("target_success")} | '
             f'mean height **{data.get("mean_height")}** '
             f'{data.get("mean_height_ci95")} | IQM {data.get("iqm_height")} '
             f'| ~{reach*100:.0f}% of samples reach a terminal\n')
open(results, 'a').write(line)
print(line.strip())
PY
}

#        name        rollout  envs/worker  lambda
run_one  roll-1024   1024     16           0.995
run_one  roll-1024-l 1024     16           0.9995
run_one  roll-256-l  256      64           0.9995
echo "[$(date +%H:%M)] sweep complete" >> "$RESULTS"
