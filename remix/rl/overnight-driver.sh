#!/usr/bin/env bash
# Autonomous follow-up after the rollout and gamma sweeps.
#
# The rule this encodes: a single-seed arm is never a result. Seeds 7/8/9 of the
# identical config span 15.2 mean height, so the only honest response to a
# promising arm is to run it again at new seeds, not to build on it.
#
#   1. measure the hazard curve (feasibility -- independent of every sweep)
#   2. pick the best arm across both sweeps by mean height
#   3. replicate it at two fresh seeds
#   4. escalate to the ladder ONLY if its 3-seed mean clears the noise band
#
# Step 4 is what can fill a long run; if nothing replicates it stops and says so,
# which is the correct outcome and costs nothing further.
set -uo pipefail

CODE=/home/lr/dodgeblock-v5-code
INIT=/home/lr/dodgeblock-ladder/rung-700-x13/checkpoints/latest.pt
ROOT=/home/lr/dodgeblock-driver
RESULTS=$ROOT/RESULTS.md
BAND_TOP=424.38          # best of seeds 7/8/9 on the identical config

while ! grep -q 'sweep complete' /home/lr/dodgeblock-gamma/RESULTS.md 2>/dev/null; do sleep 120; done

mkdir -p "$ROOT"
SNAPSHOT=$ROOT/code
rm -rf "$SNAPSHOT"
cp -al "$CODE" "$SNAPSHOT"
CODE=$SNAPSHOT

[ -f "$RESULTS" ] || cat > "$RESULTS" <<EOF
# Autonomous follow-up
Noise band from seeds 7/8/9 of the control config: 409.2 - $BAND_TOP mean height.
An arm must beat $BAND_TOP *and then replicate at two fresh seeds* to count.

EOF

# --- 1. feasibility: how does per-layer survival behave as difficulty saturates?
echo "[$(date +%H:%M)] hazard curve on the control checkpoint" >> "$RESULTS"
( cd "$CODE" && CUDA_VISIBLE_DEVICES=1 \
  /home/lr/envs/dodgeblock-rl/bin/python rl/hazard-curve.py \
    /home/lr/dodgeblock-ladder/rung-600-n14/checkpoints/latest.pt \
    --episodes 512 > "$ROOT/hazard-control.json" ) >> "$ROOT/driver.log" 2>&1
{ echo '```'; cat "$ROOT/hazard-control.json" 2>/dev/null; echo '```'; } >> "$RESULTS"

# --- 2. best arm across both sweeps
read -r BEST_NAME BEST_HEIGHT <<< "$(python3 - <<'PY'
import re, glob
best, name = -1.0, ''
for path in glob.glob('/home/lr/dodgeblock-rollout/RESULTS.md') + \
            glob.glob('/home/lr/dodgeblock-gamma/RESULTS.md'):
    try: text = open(path).read()
    except Exception: continue
    for match in re.finditer(r'\*\*([a-z0-9-]+)\*\*[^\n]*?mean height \*\*([0-9.]+)\*\*', text):
        if float(match.group(2)) > best:
            best, name = float(match.group(2)), match.group(1)
print(f'{name} {best}')
PY
)"
echo "[$(date +%H:%M)] best arm: $BEST_NAME at $BEST_HEIGHT (band top $BAND_TOP)" >> "$RESULTS"

if [ -z "$BEST_NAME" ] || python3 -c "import sys; sys.exit(0 if $BEST_HEIGHT <= $BAND_TOP else 1)"; then
  echo "[$(date +%H:%M)] nothing beat the noise band — stopping. The sweeps are the result." >> "$RESULTS"
  exit 0
fi

# Config for each arm this project queued. Replication must reproduce the arm
# exactly except for the seed.
case "$BEST_NAME" in
  roll-1024)   ENV_ARGS=(DODGEBLOCK_ROLLOUT=1024 DODGEBLOCK_ENVS_PER_WORKER=16) ;;
  roll-1024-l) ENV_ARGS=(DODGEBLOCK_ROLLOUT=1024 DODGEBLOCK_ENVS_PER_WORKER=16 DODGEBLOCK_GAE_LAMBDA=0.9995) ;;
  roll-256-l)  ENV_ARGS=(DODGEBLOCK_GAE_LAMBDA=0.9995) ;;
  gamma-20s)   ENV_ARGS=(DODGEBLOCK_GAMMA=0.999423) ;;
  gamma-45s)   ENV_ARGS=(DODGEBLOCK_GAMMA=0.999743) ;;
  gamma-90s)   ENV_ARGS=(DODGEBLOCK_GAMMA=0.999872) ;;
  lam-97*)     ENV_ARGS=(DODGEBLOCK_GAE_LAMBDA=0.97) ;;
  *) echo "[$(date +%H:%M)] no config known for $BEST_NAME — stopping." >> "$RESULTS"; exit 0 ;;
esac

replicate() {
  local seed=$1
  local dir=$ROOT/${BEST_NAME}-s${seed}
  mkdir -p "$dir/checkpoints"
  echo "[$(date +%H:%M)] replicating $BEST_NAME at seed $seed" >> "$RESULTS"
  ( cd "$CODE" && env "${ENV_ARGS[@]}" \
    CUDA_VISIBLE_DEVICES=1 \
    DODGEBLOCK_PYTHON=/home/lr/envs/dodgeblock-rl/bin/python \
    DODGEBLOCK_WORKERS=8 \
    DODGEBLOCK_TOTAL_FRAMES=8000000 DODGEBLOCK_TARGET_HEIGHT=600 \
    DODGEBLOCK_REWARD_MODE=target DODGEBLOCK_CELL_EVAL_ENVS=0 \
    DODGEBLOCK_COMPILE=1 DODGEBLOCK_CHECKPOINT_INTERVAL=4000000 \
    DODGEBLOCK_EVAL_AFTER_TRAIN=1 DODGEBLOCK_EVAL_EPISODES=2048 \
    DODGEBLOCK_SEED="$seed" \
    DODGEBLOCK_CHECKPOINT_DIR="$dir/checkpoints" \
    DODGEBLOCK_INITIALIZE_FROM="$INIT" \
    DODGEBLOCK_DEVICE=cuda \
    ./rl/run-ppo-v4.sh ) >> "$dir/run.log" 2>&1
  python3 - "$dir" "$BEST_NAME" "$seed" "$RESULTS" <<'PY'
import json, sys
from pathlib import Path
directory, name, seed, results = sys.argv[1:5]
text = (Path(directory) / 'run.log').read_text()
index = text.rfind('"event": "evaluation"')
line = f'- **{name}-s{seed}** replicate: '
if index < 0:
    line += 'NO EVAL (crashed)\n'
else:
    data, _ = json.JSONDecoder().raw_decode(text[text.rfind('{', 0, index):])
    line += (f'det {data.get("target_success")} | mean height '
             f'**{data.get("mean_height")}** {data.get("mean_height_ci95")}\n')
open(results, 'a').write(line)
print(line.strip())
PY
}

replicate 8
replicate 9

VERDICT=$(python3 - "$RESULTS" "$BEST_HEIGHT" "$BAND_TOP" <<'PY'
import re, sys, statistics
text, first, band = open(sys.argv[1]).read(), float(sys.argv[2]), float(sys.argv[3])
values = [first] + [float(m.group(1)) for m in
                    re.finditer(r'replicate: [^\n]*mean height \*\*([0-9.]+)\*\*', text)]
mean = statistics.mean(values)
print(f'{mean:.1f} {"HOLDS" if mean > band else "DID-NOT-REPLICATE"} {len(values)}')
PY
)
read -r MEAN STATUS SEEDS <<< "$VERDICT"
echo "[$(date +%H:%M)] $BEST_NAME over $SEEDS seeds: mean height $MEAN -> **$STATUS**" >> "$RESULTS"

if [ "$STATUS" = "HOLDS" ]; then
  echo "[$(date +%H:%M)] escalating: ladder from rung 600 with $BEST_NAME's config" >> "$RESULTS"
  ( cd "$CODE" && env "${ENV_ARGS[@]}" \
    python3 rl/ladder_driver.py --bootstrap \
      --state-dir /home/lr/dodgeblock-ladder-v11 \
      --current-target 600 --current-dir "$ROOT/${BEST_NAME}-s8" \
      --current-unit none --current-frames 8000000 ) >> "$ROOT/driver.log" 2>&1
  echo "[$(date +%H:%M)] ladder seeded at /home/lr/dodgeblock-ladder-v11" >> "$RESULTS"
else
  echo "[$(date +%H:%M)] stopping. A single-seed arm that does not replicate is noise, and building on it is how the last ledger went wrong." >> "$RESULTS"
fi
