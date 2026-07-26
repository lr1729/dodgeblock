#!/usr/bin/env bash
# Does the objective reward speed, and does lambda=0.97 replicate?
#
# TWO INDEPENDENT QUESTIONS, deliberately not combined into one arm.
#
# (1) gamma < 1. At gamma = 1 with a terminal-only return, a 2,000-frame success
# and a 20,000-frame success are worth exactly the same. Speed is not in the
# objective. Measured consequence: 229 frames/layer against the demos' 90, a 2.5x
# gap. Discounting is the direct fix and needs no reward change -- with a
# terminal-only return, camping still pays zero, so the usual camping failure
# mode does not apply.
#
# Half-lives bracketed at 20/45/90 s. Shorter is pathological, not aggressive:
# a 2 s half-life makes a 2,333-frame success worth 1e-6 and the target becomes
# unreachable in reward terms.
#
#   half-life  gamma/frame   value of a 2333-frame success
#          2s   0.994240     0.0000   <- pathological
#          5s   0.997692     0.0046   <- pathological
#         20s   0.999423     0.2599
#         45s   0.999743     0.5494
#         90s   0.999872     0.7412
#
# The env's reward potential is discounted with the SAME gamma (ppo_v2.py passes
# discount=args.gamma to the bridge), so the potential still telescopes and the
# shaping stays policy-invariant. Without that it would degenerate into a height
# reward, which is the v1 pathology.
#
# (2) lambda = 0.97 replicate. Re-judged against the seed noise floor, it is the
# ONLY arm this project has run that looks positive: mean height 440.0
# [423.04, 455.24] against a control band of [409.2, 424.4]. It was buried under
# a "monotone worse" reading that its own third data point had already broken.
# One more seed decides whether it is real.
set -uo pipefail

CODE=/home/lr/dodgeblock-v5-code
INIT=/home/lr/dodgeblock-ladder/rung-700-x13/checkpoints/latest.pt
ROOT=/home/lr/dodgeblock-gamma
RESULTS=$ROOT/RESULTS.md

while ! grep -q 'sweep complete' /home/lr/dodgeblock-rollout/RESULTS.md 2>/dev/null; do sleep 120; done

mkdir -p "$ROOT"
SNAPSHOT=$ROOT/code
rm -rf "$SNAPSHOT"
cp -al "$CODE" "$SNAPSHOT"
CODE=$SNAPSHOT

[ -f "$RESULTS" ] || cat > "$RESULTS" <<'EOF'
# Discounted target + lambda replicate — target 600, 8M frames, init rung-700-x13
Control band from seeds 7/8 of the identical config: mean height **409.2 - 424.4**
(det 0.2935 - 0.344). Any arm inside that band is not evidence of anything.
Primary metric is mean height with a bootstrap CI, 2048 eval episodes.
Speed is the point of the gamma arms, so climb rate is reported alongside.

EOF

run_one() {
  local name=$1 gamma=$2 lambda=$3 seed=$4
  local dir=$ROOT/$name
  mkdir -p "$dir/checkpoints"
  echo "[$(date +%H:%M)] launching $name gamma=$gamma lambda=$lambda seed=$seed" >> "$RESULTS"
  ( cd "$CODE" && \
    CUDA_VISIBLE_DEVICES=1 \
    DODGEBLOCK_PYTHON=/home/lr/envs/dodgeblock-rl/bin/python \
    DODGEBLOCK_WORKERS=8 DODGEBLOCK_ENVS_PER_WORKER=64 \
    DODGEBLOCK_TOTAL_FRAMES=8000000 DODGEBLOCK_TARGET_HEIGHT=600 \
    DODGEBLOCK_REWARD_MODE=target DODGEBLOCK_CELL_EVAL_ENVS=0 \
    DODGEBLOCK_COMPILE=1 DODGEBLOCK_CHECKPOINT_INTERVAL=4000000 \
    DODGEBLOCK_EVAL_AFTER_TRAIN=1 DODGEBLOCK_EVAL_EPISODES=2048 \
    DODGEBLOCK_GAMMA="$gamma" DODGEBLOCK_GAE_LAMBDA="$lambda" \
    DODGEBLOCK_SEED="$seed" \
    DODGEBLOCK_CHECKPOINT_DIR="$dir/checkpoints" \
    DODGEBLOCK_INITIALIZE_FROM="$INIT" \
    DODGEBLOCK_DEVICE=cuda \
    ./rl/run-ppo-v4.sh ) >> "$dir/run.log" 2>&1

  python3 - "$dir" "$name" "$gamma" "$lambda" "$seed" "$RESULTS" <<'PY'
import json, sys
from pathlib import Path
directory, name, gamma, lam, seed, results = sys.argv[1:7]
text = (Path(directory) / 'run.log').read_text()
index = text.rfind('"event": "evaluation"')
line = f'- **{name}** gamma={gamma} lambda={lam} seed={seed}: '
if index < 0:
    line += 'NO EVAL (crashed)\n'
else:
    data, _ = json.JSONDecoder().raw_decode(text[text.rfind('{', 0, index):])
    # Climb rate is the point of the gamma arms: the objective change is meant to
    # buy speed, and mean height alone cannot show whether it did.
    last = text.rfind('"fresh": {')
    rate = None
    if last > 0:
        try:
            fresh, _ = json.JSONDecoder().raw_decode(text[last + len('"fresh": '):])
            if fresh.get('mean_length'):
                rate = round(fresh['mean_height'] / (fresh['mean_length'] / 60.0), 1)
        except Exception:
            pass
    line += (f'det {data.get("target_success")} | mean height '
             f'**{data.get("mean_height")}** {data.get("mean_height_ci95")} | '
             f'IQM {data.get("iqm_height")} | climb {rate} h/s (control ~10.5, demos ~26.6)\n')
open(results, 'a').write(line)
print(line.strip())
PY
}

#       name        gamma       lambda  seed
run_one gamma-20s   0.999423    0.995   7
run_one gamma-45s   0.999743    0.995   7
run_one gamma-90s   0.999872    0.995   7
run_one lam-97-s8   1.0         0.97    8
echo "[$(date +%H:%M)] sweep complete" >> "$RESULTS"
