#!/usr/bin/env bash
# Finish the action-repeat picture: replicate the leader, extend the dose, and
# check whether any of it transfers to the regime the goal actually lives in.
#
# repeat-4 leads at mean height 459.34 [451.48, 467.48] and det 0.4048 but is one
# seed. repeat-2 is replicating (+11.54, +32.56 so far). The dose-response across
# 1 -> 2 -> 4 is monotone in both height and climb rate, so 8 tests whether it is
# still rising or has turned over -- the mechanism predicts a turn eventually,
# when the interval grows long enough to cost reaction time, and the death autopsy
# put the causal window of a death at 10-20 frames.
#
# The last step is the one that matters for 10k. Every gain so far is measured at
# rung 600 = the first ~40 seconds. saturated-hazard.py measures per-layer survival
# from banked cells past the 240 s saturation point, where the baseline policy sits
# at 0.6941 against the 0.9567 needed. If action repeat does not move THAT, it is a
# real effect on the easy regime and nothing more.
set -uo pipefail

CODE=/home/lr/dodgeblock-v5-code
INIT=/home/lr/dodgeblock-ladder/rung-600-n14/checkpoints/latest.pt
ROOT=/home/lr/dodgeblock-repeat-ext
RESULTS=$ROOT/RESULTS.md

while ! grep -q 'complete' /home/lr/dodgeblock-repeat-rep/RESULTS.md 2>/dev/null; do sleep 120; done

mkdir -p "$ROOT"
SNAPSHOT=$ROOT/code
rm -rf "$SNAPSHOT"
cp -al "$CODE" "$SNAPSHOT"
CODE=$SNAPSHOT

[ -f "$RESULTS" ] || cat > "$RESULTS" <<'EOF'
# Action repeat — replicate the leader, extend the dose, test transfer
Same-seed controls: seed 7 = 424.38, seed 8 = 409.22, seed 9 = 414.20.
Measured: repeat-2 435.92 (seed 7), repeat-4 459.34 (seed 7).
Saturated baseline for the transfer test: per-layer 0.6941 (needs 0.9567).

EOF

run_one() {
  local name=$1 repeat=$2 seed=$3
  local dir=$ROOT/$name
  mkdir -p "$dir/checkpoints"
  echo "[$(date +%H:%M)] launching $name repeat=$repeat seed=$seed" >> "$RESULTS"
  ( cd "$CODE" && \
    CUDA_VISIBLE_DEVICES=1 \
    DODGEBLOCK_PYTHON=/home/lr/envs/dodgeblock-rl/bin/python \
    DODGEBLOCK_WORKERS=8 DODGEBLOCK_ENVS_PER_WORKER=64 \
    DODGEBLOCK_TOTAL_FRAMES=8000000 DODGEBLOCK_TARGET_HEIGHT=600 \
    DODGEBLOCK_REWARD_MODE=target DODGEBLOCK_CELL_EVAL_ENVS=0 \
    DODGEBLOCK_COMPILE=1 DODGEBLOCK_CHECKPOINT_INTERVAL=4000000 \
    DODGEBLOCK_EVAL_AFTER_TRAIN=1 DODGEBLOCK_EVAL_EPISODES=2048 \
    DODGEBLOCK_ACTION_REPEAT="$repeat" DODGEBLOCK_SEED="$seed" \
    DODGEBLOCK_CHECKPOINT_DIR="$dir/checkpoints" \
    DODGEBLOCK_INITIALIZE_FROM="$INIT" \
    DODGEBLOCK_DEVICE=cuda \
    ./rl/run-ppo-v4.sh ) >> "$dir/run.log" 2>&1

  python3 - "$dir" "$name" "$repeat" "$seed" "$RESULTS" <<'PY'
import json, sys
from pathlib import Path
directory, name, repeat, seed, results = sys.argv[1:6]
control = {'7': 424.38, '8': 409.22, '9': 414.20}.get(seed)
text = (Path(directory) / 'run.log').read_text()
index = text.rfind('"event": "evaluation"')
if index < 0:
    open(results, 'a').write(f'- **{name}**: NO EVAL (crashed)\n')
    raise SystemExit
data, _ = json.JSONDecoder().raw_decode(text[text.rfind('{', 0, index):])
height = float(data.get('mean_height', 0.0))
last = text.rfind('"fresh": {')
length = None
if last > 0:
    try:
        fresh, _ = json.JSONDecoder().raw_decode(text[last + len('"fresh": '):])
        length = fresh.get('mean_length')
    except Exception:
        pass
climb = f'{height / (length / 60.0):.2f}' if length else '?'
open(results, 'a').write(
    f'- **{name}** repeat={repeat} seed={seed}: det {data.get("target_success")} | '
    f'mean height **{height}** {data.get("mean_height_ci95")} | climb {climb} h/s | '
    f'paired vs control {control}: {height - control:+.2f}\n')
print(name, height)
PY
}

run_one repeat-4-s8 4 8
run_one repeat-4-s9 4 9
run_one repeat-8-s7 8 7

# Transfer test on the best checkpoint by mean height.
BEST=$(python3 - "$RESULTS" <<'PY'
import re, sys
best, name = -1.0, ''
for m in re.finditer(r'\*\*([a-z0-9-]+)\*\* repeat=\d+ seed=\d+:[^\n]*mean height \*\*([0-9.]+)\*\*',
                     open(sys.argv[1]).read()):
    if float(m.group(2)) > best:
        best, name = float(m.group(2)), m.group(1)
print(name)
PY
)
if [ -n "$BEST" ]; then
  echo "[$(date +%H:%M)] transfer test on $BEST" >> "$RESULTS"
  ( cd "$CODE" && CUDA_VISIBLE_DEVICES=1 \
    /home/lr/envs/dodgeblock-rl/bin/python rl/saturated-hazard.py \
      "$ROOT/$BEST/checkpoints/latest.pt" --envs 128 --rounds 4 \
      > "$ROOT/saturated-$BEST.json" ) >> "$ROOT/driver.log" 2>&1
  python3 - "$ROOT/saturated-$BEST.json" "$BEST" "$RESULTS" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))['by_start_regime']
except Exception:
    raise SystemExit
sat = (d.get('saturated_240s_plus') or {}).get('per_layer_survival')
ramp = (d.get('ramp_60_240s') or {}).get('per_layer_survival')
verdict = 'TRANSFERS' if sat and sat > 0.75 else 'does not transfer'
open(sys.argv[3], 'a').write(
    f'\n**Transfer ({sys.argv[2]}): saturated per-layer {sat} (baseline 0.6941), '
    f'ramp {ramp} (baseline 0.6526) -> {verdict}**\n'
    f'Needed for 10k at all: 0.9567.\n')
print(verdict, sat)
PY
fi
echo "[$(date +%H:%M)] complete" >> "$RESULTS"
