#!/usr/bin/env bash
# Replicate action repeat = 2 at two fresh seeds.
#
# repeat-2 is the first arm in this ledger whose bootstrap CI clears the noise
# band: mean height 435.92 [427.89, 443.97] against a band top of 424.38. It also
# reached that height in the same wall time (2305 sim frames vs ~2333), so it is
# climbing faster rather than surviving longer.
#
# That is exactly what lambda=0.97 looked like at two seeds before its third came
# in flat and killed it. Paired against the same-seed control is the analysis that
# matters, since seed variance dominates:
#
#   seed 7 control 424.38 -> repeat-2 435.92 = +11.54
#
# Registered prediction: if real, seed 8 lands near 421 (control 409.22) and seed 9
# near 426 (control 414.20). Three same-signed paired differences make it the first
# confirmed effect here; one flat or negative ends it, as it ended lambda=0.97.
set -uo pipefail

CODE=/home/lr/dodgeblock-v5-code
INIT=/home/lr/dodgeblock-ladder/rung-600-n14/checkpoints/latest.pt
ROOT=/home/lr/dodgeblock-repeat-rep
RESULTS=$ROOT/RESULTS.md

while ! grep -q 'sweep complete' /home/lr/dodgeblock-repeat/RESULTS.md 2>/dev/null; do sleep 120; done

mkdir -p "$ROOT"
SNAPSHOT=$ROOT/code
rm -rf "$SNAPSHOT"
cp -al "$CODE" "$SNAPSHOT"
CODE=$SNAPSHOT

[ -f "$RESULTS" ] || cat > "$RESULTS" <<'EOF'
# action repeat = 2, replication
Same-seed controls: seed 7 = 424.38, seed 8 = 409.22, seed 9 = 414.20.
seed 7 already measured: repeat-2 = 435.92 (+11.54 paired).

EOF

run_one() {
  local seed=$1
  local dir=$ROOT/repeat-2-s${seed}
  mkdir -p "$dir/checkpoints"
  echo "[$(date +%H:%M)] launching repeat-2 seed $seed" >> "$RESULTS"
  ( cd "$CODE" && \
    CUDA_VISIBLE_DEVICES=1 \
    DODGEBLOCK_PYTHON=/home/lr/envs/dodgeblock-rl/bin/python \
    DODGEBLOCK_WORKERS=8 DODGEBLOCK_ENVS_PER_WORKER=64 \
    DODGEBLOCK_TOTAL_FRAMES=8000000 DODGEBLOCK_TARGET_HEIGHT=600 \
    DODGEBLOCK_REWARD_MODE=target DODGEBLOCK_CELL_EVAL_ENVS=0 \
    DODGEBLOCK_COMPILE=1 DODGEBLOCK_CHECKPOINT_INTERVAL=4000000 \
    DODGEBLOCK_EVAL_AFTER_TRAIN=1 DODGEBLOCK_EVAL_EPISODES=2048 \
    DODGEBLOCK_ACTION_REPEAT=2 DODGEBLOCK_SEED="$seed" \
    DODGEBLOCK_CHECKPOINT_DIR="$dir/checkpoints" \
    DODGEBLOCK_INITIALIZE_FROM="$INIT" \
    DODGEBLOCK_DEVICE=cuda \
    ./rl/run-ppo-v4.sh ) >> "$dir/run.log" 2>&1

  python3 - "$dir" "$seed" "$RESULTS" <<'PY'
import json, sys
from pathlib import Path
directory, seed, results = sys.argv[1:4]
control = {'8': 409.22, '9': 414.20}[seed]
text = (Path(directory) / 'run.log').read_text()
index = text.rfind('"event": "evaluation"')
if index < 0:
    open(results, 'a').write(f'- **repeat-2-s{seed}**: NO EVAL (crashed)\n')
    raise SystemExit
data, _ = json.JSONDecoder().raw_decode(text[text.rfind('{', 0, index):])
height = float(data.get('mean_height', 0.0))
open(results, 'a').write(
    f'- **repeat-2-s{seed}** seed={seed}: mean height **{height}** '
    f'{data.get("mean_height_ci95")} | paired vs control {control}: '
    f'{height - control:+.2f}\n')
print(seed, height, f'{height-control:+.2f}')
PY
}

run_one 8
run_one 9

python3 - "$RESULTS" <<'PY'
import re, sys, statistics, math
text = open(sys.argv[1]).read()
diffs = [11.54] + [float(m.group(1)) for m in
                   re.finditer(r'paired vs control [0-9.]+: ([+-][0-9.]+)', text)]
if len(diffs) >= 3:
    mean, sd = statistics.mean(diffs), statistics.stdev(diffs)
    se = sd / math.sqrt(len(diffs))
    lo, hi = mean - 4.303 * se, mean + 4.303 * se
    verdict = 'HOLDS at 3 seeds' if lo > 0 else 'DID NOT REPLICATE'
    open(sys.argv[1], 'a').write(
        f'\n**{len(diffs)} seeds: paired diffs {["%+.2f" % d for d in diffs]}, '
        f'mean {mean:+.2f}, 95% CI [{lo:+.2f}, {hi:+.2f}] -> {verdict}**\n')
    print(verdict)
PY
echo "[$(date +%H:%M)] complete" >> "$RESULTS"
