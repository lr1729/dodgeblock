#!/usr/bin/env bash
# The one axis this project has never varied: how often the policy acts.
#
# Four explanation classes are closed (reward shape, estimator, representation,
# exposure). What has never changed is the policy class -- and the cheapest lever
# on it is the control interval. Tallec et al. (arXiv:1901.09732) prove the action
# gap Q(s,a) - V(s) vanishes as the control interval shrinks, and this project
# measured that fingerprint directly: held-out direction loss ~ ln(3), i.e. per
# frame the action choice barely matters. Repeating an action multiplies its
# effect and makes exploration temporally correlated -- the death autopsy's
# escapes were all found by sticky-random play with a mean hold of ~7.5 frames.
#
# BUDGET: agent steps are held at 8M, so gradient updates and GPU compute match
# the control exactly; repeat N therefore consumes N x the SIM frames. That is
# deliberate. The question is whether acting less often reaches a lower hazard,
# not whether it is more sample-efficient, and env frames are cheap CPU here.
#
# Judged against the 3-seed noise band 409.2-424.4 mean height. Also reported:
# mean episode length in sim frames, since a longer episode at equal height means
# slower play rather than better play.
set -uo pipefail

CODE=/home/lr/dodgeblock-v5-code
INIT=/home/lr/dodgeblock-ladder/rung-600-n14/checkpoints/latest.pt
ROOT=/home/lr/dodgeblock-repeat
RESULTS=$ROOT/RESULTS.md

mkdir -p "$ROOT"
SNAPSHOT=$ROOT/code
rm -rf "$SNAPSHOT"
cp -al "$CODE" "$SNAPSHOT"
CODE=$SNAPSHOT

[ -f "$RESULTS" ] || cat > "$RESULTS" <<'EOF'
# Action repeat — target 600, 8M AGENT steps, init rung-600-n14
Noise band from seeds 7/8/9 of the control (repeat 1): mean height 409.2 - 424.4.
Repeat N costs N x the sim frames at equal gradient updates; that is the design.

EOF

run_one() {
  local name=$1 repeat=$2
  local dir=$ROOT/$name
  mkdir -p "$dir/checkpoints"
  echo "[$(date +%H:%M)] launching $name action_repeat=$repeat" >> "$RESULTS"
  ( cd "$CODE" && \
    CUDA_VISIBLE_DEVICES=1 \
    DODGEBLOCK_PYTHON=/home/lr/envs/dodgeblock-rl/bin/python \
    DODGEBLOCK_WORKERS=8 DODGEBLOCK_ENVS_PER_WORKER=64 \
    DODGEBLOCK_TOTAL_FRAMES=8000000 DODGEBLOCK_TARGET_HEIGHT=600 \
    DODGEBLOCK_REWARD_MODE=target DODGEBLOCK_CELL_EVAL_ENVS=0 \
    DODGEBLOCK_COMPILE=1 DODGEBLOCK_CHECKPOINT_INTERVAL=4000000 \
    DODGEBLOCK_EVAL_AFTER_TRAIN=1 DODGEBLOCK_EVAL_EPISODES=2048 \
    DODGEBLOCK_ACTION_REPEAT="$repeat" \
    DODGEBLOCK_CHECKPOINT_DIR="$dir/checkpoints" \
    DODGEBLOCK_INITIALIZE_FROM="$INIT" \
    DODGEBLOCK_DEVICE=cuda \
    ./rl/run-ppo-v4.sh ) >> "$dir/run.log" 2>&1

  python3 - "$dir" "$name" "$repeat" "$RESULTS" <<'PY'
import json, sys
from pathlib import Path
directory, name, repeat, results = sys.argv[1:5]
text = (Path(directory) / 'run.log').read_text()
index = text.rfind('"event": "evaluation"')
line = f'- **{name}** repeat={repeat}: '
if index < 0:
    line += 'NO EVAL (crashed)\n'
else:
    data, _ = json.JSONDecoder().raw_decode(text[text.rfind('{', 0, index):])
    last = text.rfind('"fresh": {')
    length = None
    if last > 0:
        try:
            fresh, _ = json.JSONDecoder().raw_decode(text[last + len('"fresh": '):])
            length = fresh.get('mean_length')
        except Exception:
            pass
    line += (f'det {data.get("target_success")} | mean height '
             f'**{data.get("mean_height")}** {data.get("mean_height_ci95")} | '
             f'IQM {data.get("iqm_height")} | episode {length} sim frames '
             f'(control ~2333)\n')
open(results, 'a').write(line)
print(line.strip())
PY
}

run_one repeat-2 2
run_one repeat-4 4
echo "[$(date +%H:%M)] sweep complete" >> "$RESULTS"
