#!/usr/bin/env bash
# The auxiliary hazard head. Measured 2026-07-25: matched on height, the critic
# ranks survival at chance (AUC 0.50 / 0.52 / 0.51 / 0.54 at heights 100-400).
# It learned a progress meter, not a danger model -- P(reach target) is a
# Bernoulli label integrated over ~15 layers, and its danger component is a
# small fraction of the return variance. P(death within h frames) is densely
# labelled and locally determined, so it trains.
#
# This adds no reward and no demonstrations. It is a supervised auxiliary loss
# on the shared trunk, so unlike the four falsified interventions it cannot bias
# the objective -- it can only change what the trunk represents.
#
# Mechanism test, pre-registered: if it works, the critic's matched-height AUC
# must rise above 0.5. A success score that improves while AUC stays at chance
# means something else moved and the stated mechanism is wrong.
set -uo pipefail

CODE=/home/lr/dodgeblock-v5-code
INIT=/home/lr/dodgeblock-ladder/rung-700-x13/checkpoints/latest.pt
ROOT=/home/lr/dodgeblock-hazard
RESULTS=$ROOT/RESULTS.md
CONTROL=0.344

mkdir -p "$ROOT"
SNAPSHOT=$ROOT/code
rm -rf "$SNAPSHOT"
cp -al "$CODE" "$SNAPSHOT"
CODE=$SNAPSHOT

[ -f "$RESULTS" ] || cat > "$RESULTS" <<EOF
# Auxiliary hazard head — target 600, 8M frames, init rung-700-x13
Control (no head): det success $CONTROL, per-layer 0.9313, matched-height AUC ~0.52.
Head: P(death within {10,30,90} frames), BCE, positive-weighted, on the shared trunk.

EOF

run_one() {
  local name=$1 coef=$2
  local dir=$ROOT/$name
  mkdir -p "$dir/checkpoints"
  echo "[$(date +%H:%M)] launching $name hazard_coef=$coef" >> "$RESULTS"
  ( cd "$CODE" && \
    CUDA_VISIBLE_DEVICES=1 \
    DODGEBLOCK_PYTHON=/home/lr/envs/dodgeblock-rl/bin/python \
    DODGEBLOCK_WORKERS=8 DODGEBLOCK_ENVS_PER_WORKER=64 \
    DODGEBLOCK_TOTAL_FRAMES=8000000 DODGEBLOCK_TARGET_HEIGHT=600 \
    DODGEBLOCK_REWARD_MODE=target DODGEBLOCK_CELL_EVAL_ENVS=0 \
    DODGEBLOCK_COMPILE=1 DODGEBLOCK_CHECKPOINT_INTERVAL=4000000 \
    DODGEBLOCK_EVAL_AFTER_TRAIN=1 DODGEBLOCK_EVAL_EPISODES=512 \
    DODGEBLOCK_HAZARD_COEF="$coef" \
    DODGEBLOCK_CHECKPOINT_DIR="$dir/checkpoints" \
    DODGEBLOCK_INITIALIZE_FROM="$INIT" \
    DODGEBLOCK_DEVICE=cuda \
    ./rl/run-ppo-v4.sh ) >> "$dir/run.log" 2>&1

  # The mechanism test runs regardless of the score, so a null result can be
  # attributed: head trained but AUC flat is a different failure from head
  # never trained at all.
  ( cd "$CODE" && CUDA_VISIBLE_DEVICES=1 \
    /home/lr/envs/dodgeblock-rl/bin/python rl/critic-profile.py \
      "$dir/checkpoints/latest.pt" --episodes 384 --target-height 600 \
      > "$dir/critic.json" ) >> "$dir/run.log" 2>&1

  python3 - "$dir" "$name" "$coef" "$RESULTS" "$CONTROL" <<'PY'
import json, sys
from pathlib import Path
directory, name, coef, results, control = sys.argv[1:6]
text = (Path(directory) / 'run.log').read_text()
index = text.rfind('"event": "evaluation"')
line = f'- **{name}** coef={coef}: '
if index < 0:
    line += 'NO EVAL (crashed)\n'
else:
    data, _ = json.JSONDecoder().raw_decode(text[text.rfind('{', 0, index):])
    success = float(data.get('target_success', 0.0))
    layers = max(1.0, float(data.get('target_height', 600)) / 40.0)
    per_layer = success ** (1.0 / layers) if success > 0 else 0.0
    last = text.rfind('{"event": "progress"')
    separation = {}
    if last > 0:
        try:
            progress, _ = json.JSONDecoder().raw_decode(text[last:])
            separation = (progress.get('hazard') or {}).get('separation', {})
        except Exception:
            pass
    auc = {}
    try:
        profile = json.loads((Path(directory) / 'critic.json').read_text())
        auc = {band: entry['auc'] for band, entry
               in profile['auc_at_matched_height'].items()}
    except Exception:
        pass
    line += (f'det **{success:.3f}** (control {control}) per-layer {per_layer:.4f} '
             f'median {data.get("median_height")} | head sep {separation} '
             f'| matched-height AUC {auc}\n')
open(results, 'a').write(line)
print(line.strip())
PY
}

run_one hazard-25 0.25
run_one hazard-100 1.0
echo "[$(date +%H:%M)] sweep complete" >> "$RESULTS"
