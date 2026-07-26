#!/usr/bin/env bash
# Train where the goal lives.
#
# Measured this session: the policy's mean episode is 43.5 s, it has zero exposure
# past the 240 s saturation point, and dropped into banked states there it holds
# per-layer survival 0.694 against the 0.9567 needed to reach 10k at all. Every
# A/B in this ledger optimised the first 43 seconds of a 376-second problem.
#
# This is the direct correction: mix banked starts into training so the policy
# actually experiences minutes two through six. The cell-bank machinery already
# exists (--cell-bank, --cell-bank-probability); it has simply never been pointed
# at this question, and the one prior attempt (v4) was judged on fresh-start IQM
# rather than on survival in the regime it was training for.
#
# PRE-REGISTERED. The binding metric is saturated per-layer survival from
# saturated-hazard.py, NOT target success -- at target 10000 success is ~0 for
# everyone and carries no information.
#
#   baseline (rung-600-n14):  0.6941 saturated, 0.6526 ramp
#   success  = saturated per-layer survival rises meaningfully above 0.694
#   failure  = it does not move, in which case exposure is not the deficit and
#              the regime error, while real, is not what is holding the policy back
#
# A known risk, stated in advance: v4's version of this produced a suffix
# specialist that could not start fresh. That is why fresh starts are kept at 20%
# and why the fresh-start hazard curve is re-measured afterwards -- if saturated
# survival rises while fresh collapses, the run bought nothing transferable.
set -uo pipefail

CODE=/home/lr/dodgeblock-v5-code
INIT=/home/lr/dodgeblock-ladder/rung-600-n14/checkpoints/latest.pt
ROOT=/home/lr/dodgeblock-saturated
RESULTS=$ROOT/RESULTS.md

mkdir -p "$ROOT"
SNAPSHOT=$ROOT/code
rm -rf "$SNAPSHOT"
cp -al "$CODE" "$SNAPSHOT"
CODE=$SNAPSHOT

[ -f "$RESULTS" ] || cat > "$RESULTS" <<'EOF'
# Training in the saturated regime — banked starts, target 10000, 20M frames
Baseline (rung-600-n14, trained only in minute one):
  saturated (240s+) per-layer survival **0.6941**
  ramp (60-240s)    per-layer survival **0.6526**
  needed: 0.9567 to reach 10k at all, 0.9972 for consistent 10k
Binding metric is saturated per-layer survival. Target success at 10000 is ~0 for
every policy here and carries no information.

EOF

dir=$ROOT/banked-20m
mkdir -p "$dir/checkpoints"
echo "[$(date +%H:%M)] launching banked-20m" >> "$RESULTS"
( cd "$CODE" && \
  CUDA_VISIBLE_DEVICES=1 \
  DODGEBLOCK_PYTHON=/home/lr/envs/dodgeblock-rl/bin/python \
  DODGEBLOCK_WORKERS=8 DODGEBLOCK_ENVS_PER_WORKER=64 \
  DODGEBLOCK_TOTAL_FRAMES=20000000 DODGEBLOCK_TARGET_HEIGHT=10000 \
  DODGEBLOCK_REWARD_MODE=target \
  DODGEBLOCK_CELL_BANK_GLOB='/home/lr/dodgeblock-go-explore-bank-v4/seed-*/search-checkpoint.json.gz' \
  DODGEBLOCK_CELL_BANK_PROBABILITY=0.8 DODGEBLOCK_CELL_EVAL_ENVS=16 \
  DODGEBLOCK_COMPILE=1 DODGEBLOCK_CHECKPOINT_INTERVAL=10000000 \
  DODGEBLOCK_EVAL_AFTER_TRAIN=0 \
  DODGEBLOCK_CHECKPOINT_DIR="$dir/checkpoints" \
  DODGEBLOCK_INITIALIZE_FROM="$INIT" \
  DODGEBLOCK_DEVICE=cuda \
  ./rl/run-ppo-v4.sh ) >> "$dir/run.log" 2>&1

echo "[$(date +%H:%M)] measuring saturated survival" >> "$RESULTS"
( cd "$CODE" && CUDA_VISIBLE_DEVICES=1 \
  /home/lr/envs/dodgeblock-rl/bin/python rl/saturated-hazard.py \
    "$dir/checkpoints/latest.pt" --envs 128 --rounds 4 \
    > "$dir/saturated.json" ) >> "$dir/run.log" 2>&1

# The suffix-specialist check: did fresh-start competence survive?
( cd "$CODE" && CUDA_VISIBLE_DEVICES=1 \
  /home/lr/envs/dodgeblock-rl/bin/python rl/hazard-curve.py \
    "$dir/checkpoints/latest.pt" --episodes 512 \
    > "$dir/hazard.json" ) >> "$dir/run.log" 2>&1

python3 - "$dir" "$RESULTS" <<'PY'
import json, sys
from pathlib import Path
directory, results = sys.argv[1:3]


def load(name):
    try:
        return json.loads((Path(directory) / name).read_text())
    except Exception:
        return {}


sat = load('saturated.json').get('by_start_regime', {})
haz = load('hazard.json')
saturated = (sat.get('saturated_240s_plus') or {}).get('per_layer_survival')
ramp = (sat.get('ramp_60_240s') or {}).get('per_layer_survival')
fresh = haz.get('mean_seconds')
verdict = 'NO DATA'
if saturated is not None:
    verdict = 'MOVED' if saturated > 0.72 else 'did not move'
open(results, 'a').write(
    f'- **banked-20m**: saturated per-layer **{saturated}** (baseline 0.6941), '
    f'ramp {ramp} (baseline 0.6526) -> **{verdict}**\n'
    f'    - fresh-start mean episode {fresh}s (baseline 43.5s) '
    f'— suffix-specialist check\n')
print(verdict, saturated)
PY
echo "[$(date +%H:%M)] complete" >> "$RESULTS"
