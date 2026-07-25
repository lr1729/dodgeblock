#!/usr/bin/env bash
# Overnight: sweep the shaping potential's strength at rung 600 against a
# control that is already measured (det 0.344, per-layer 0.9313, from the same
# rung-700-x13 actor and the same 8M frame budget), then hand the winner to the
# ladder driver and let it climb.
#
# The sweep exists because potential-based shaping is provably optimum-preserving
# only in the limit; at finite training a strong potential can still bias
# behaviour toward camping under cover instead of climbing. Magnitude is the
# fragile axis, exactly as the demo coefficient was.
set -uo pipefail

CODE=/home/lr/dodgeblock-v5-code
INIT=/home/lr/dodgeblock-ladder/rung-700-x13/checkpoints/latest.pt
ROOT=/home/lr/dodgeblock-shaping
RESULTS=$ROOT/RESULTS.md
CONTROL=0.344

mkdir -p "$ROOT"
[ -f "$RESULTS" ] || cat > "$RESULTS" <<EOF
# Shaping sweep — target 600, 8M frames, init rung-700-x13
Control (no shaping): det success $CONTROL, per-layer 0.9313, climb 12.6 h/s.
Potential: a*cover*PHASE_COVER_WEIGHT[phase] + b*charges, telescoping.

EOF

run_one() {
  local name=$1 cover=$2 charge=$3
  local dir=$ROOT/$name
  mkdir -p "$dir/checkpoints"
  echo "[$(date +%H:%M)] launching $name cover=$cover charge=$charge" >> "$RESULTS"
  ( cd "$CODE" && \
    CUDA_VISIBLE_DEVICES=1 \
    DODGEBLOCK_PYTHON=/home/lr/envs/dodgeblock-rl/bin/python \
    DODGEBLOCK_WORKERS=8 DODGEBLOCK_ENVS_PER_WORKER=64 \
    DODGEBLOCK_TOTAL_FRAMES=8000000 DODGEBLOCK_TARGET_HEIGHT=600 \
    DODGEBLOCK_REWARD_MODE=target DODGEBLOCK_CELL_EVAL_ENVS=0 \
    DODGEBLOCK_COMPILE=1 DODGEBLOCK_CHECKPOINT_INTERVAL=4000000 \
    DODGEBLOCK_EVAL_AFTER_TRAIN=1 DODGEBLOCK_EVAL_EPISODES=512 \
    DODGEBLOCK_SHAPING_COVER="$cover" DODGEBLOCK_SHAPING_CHARGE="$charge" \
    DODGEBLOCK_CHECKPOINT_DIR="$dir/checkpoints" \
    DODGEBLOCK_INITIALIZE_FROM="$INIT" \
    DODGEBLOCK_DEVICE=cuda \
    ./rl/run-ppo-v4.sh ) >> "$dir/run.log" 2>&1

  python3 - "$dir/run.log" "$name" "$cover" "$charge" "$RESULTS" "$CONTROL" <<'PY'
import json, sys
log, name, cover, charge, results, control = sys.argv[1:7]
text = open(log).read()
index = text.rfind('"event": "evaluation"')
line = f'- **{name}** cover={cover} charge={charge}: '
if index < 0:
    line += 'NO EVAL (crashed)\n'
else:
    start = text.rfind('{', 0, index)
    data, _ = json.JSONDecoder().raw_decode(text[start:])
    success = float(data.get('target_success', 0.0))
    layers = max(1.0, float(data.get('target_height', 600)) / 40.0)
    per_layer = success ** (1.0 / layers) if success > 0 else 0.0
    # shelter-in-surge is the mechanism test: did the potential actually move
    # the behaviour it was designed to move?
    shelter = None
    pindex = text.rfind('"shelter_occupancy_by_phase"')
    if pindex > 0:
        pstart = text.rfind('{', 0, text.rfind('{"event": "progress"', 0, pindex) + 1)
        try:
            progress, _ = json.JSONDecoder().raw_decode(text[text.rfind('{"event": "progress"'):])
            shelter = (progress.get('shelter_occupancy_by_phase') or {}).get('surge')
        except Exception:
            pass
    line += (f'det **{success:.3f}** (control {control}) per-layer {per_layer:.4f} '
             f'median {data.get("median_height")} shelter_surge {shelter}\n')
open(results, 'a').write(line)
print(line.strip())
PY
}

run_one shape-a 0.01 0.005
run_one shape-b 0.03 0.015
run_one shape-c 0.10 0.050

echo "[$(date +%H:%M)] sweep complete" >> "$RESULTS"

# Hand the best setting to the ladder and let it climb for the rest of the night.
BEST=$(python3 - "$RESULTS" <<'PY'
import re, sys
best, cover, charge = -1.0, '0.0', '0.0'
for line in open(sys.argv[1]):
    match = re.search(r'cover=(\S+) charge=(\S+): det \*\*([0-9.]+)\*\*', line)
    if match and float(match.group(3)) > best:
        best, cover, charge = float(match.group(3)), match.group(1), match.group(2)
print(f'{cover} {charge} {best}')
PY
)
read -r BEST_COVER BEST_CHARGE BEST_SUCCESS <<< "$BEST"
echo "[$(date +%H:%M)] best cover=$BEST_COVER charge=$BEST_CHARGE det=$BEST_SUCCESS" >> "$RESULTS"

if python3 -c "import sys; sys.exit(0 if $BEST_SUCCESS > $CONTROL else 1)"; then
  echo "[$(date +%H:%M)] beats control — restarting ladder with shaping" >> "$RESULTS"
  python3 "$CODE/rl/ladder_driver.py" --bootstrap --state-dir /home/lr/dodgeblock-ladder-v9 \
    --current-target 600 --current-dir "$ROOT/shape-best" --current-unit none --current-frames 8000000
  echo "[$(date +%H:%M)] ladder state seeded at /home/lr/dodgeblock-ladder-v9" >> "$RESULTS"
else
  echo "[$(date +%H:%M)] no setting beat the control — shaping falsified at these magnitudes" >> "$RESULTS"
fi
