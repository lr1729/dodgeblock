#!/usr/bin/env bash
# Hazard head AND a short credit horizon, together, from the same init and the
# same 8M frame budget as the control.
#
# The prediction being tested, registered before the run: neither half works
# alone. GAE interpolates between the Monte Carlo return (lambda -> 1, unbiased,
# smeared over 200 frames) and the one-step TD residual (lambda -> 0, sharp, and
# only as good as V). The critic measured at chance on ranking, so lam-90 alone
# sharpened credit onto a blind critic and lost (0.283 vs 0.344). The hazard head
# alone gives the trunk a danger representation but leaves the credit for it
# smeared across 200 frames.
#
# So this is the arm that should move, and if the ordering story is right it
# should beat BOTH halves, not just the control. If the combination also lands
# below 0.344, the story is wrong and the deficit is not credit assignment.
set -uo pipefail

CODE=/home/lr/dodgeblock-v5-code
INIT=/home/lr/dodgeblock-ladder/rung-700-x13/checkpoints/latest.pt
ROOT=/home/lr/dodgeblock-combined
RESULTS=$ROOT/RESULTS.md
HAZARD_RESULTS=/home/lr/dodgeblock-hazard/RESULTS.md
CONTROL=0.344

# Wait for the hazard sweep to finish so its best coefficient is known and the
# GPU is free.
while ! grep -q 'sweep complete' "$HAZARD_RESULTS" 2>/dev/null; do sleep 120; done

mkdir -p "$ROOT"
SNAPSHOT=$ROOT/code
rm -rf "$SNAPSHOT"
cp -al "$CODE" "$SNAPSHOT"
CODE=$SNAPSHOT

COEF=$(python3 - "$HAZARD_RESULTS" <<'PY'
import re, sys
best, coef = -1.0, '0.25'
for line in open(sys.argv[1]):
    match = re.search(r'coef=(\S+): det \*\*([0-9.]+)\*\*', line)
    if match and float(match.group(2)) > best:
        best, coef = float(match.group(2)), match.group(1)
print(coef)
PY
)

[ -f "$RESULTS" ] || cat > "$RESULTS" <<EOF
# Hazard head + short credit horizon — target 600, 8M frames, init rung-700-x13
Control: det $CONTROL. lam-90 alone: 0.283. Hazard coefficient carried forward: $COEF.
Prediction: the combination beats both halves, because sharp credit needs a
critic that can see danger and a danger representation needs sharp credit.

EOF

run_one() {
  local name=$1 lambda=$2
  local dir=$ROOT/$name
  mkdir -p "$dir/checkpoints"
  echo "[$(date +%H:%M)] launching $name lambda=$lambda hazard=$COEF" >> "$RESULTS"
  ( cd "$CODE" && \
    CUDA_VISIBLE_DEVICES=1 \
    DODGEBLOCK_PYTHON=/home/lr/envs/dodgeblock-rl/bin/python \
    DODGEBLOCK_WORKERS=8 DODGEBLOCK_ENVS_PER_WORKER=64 \
    DODGEBLOCK_TOTAL_FRAMES=8000000 DODGEBLOCK_TARGET_HEIGHT=600 \
    DODGEBLOCK_REWARD_MODE=target DODGEBLOCK_CELL_EVAL_ENVS=0 \
    DODGEBLOCK_COMPILE=1 DODGEBLOCK_CHECKPOINT_INTERVAL=4000000 \
    DODGEBLOCK_EVAL_AFTER_TRAIN=1 DODGEBLOCK_EVAL_EPISODES=512 \
    DODGEBLOCK_GAE_LAMBDA="$lambda" DODGEBLOCK_HAZARD_COEF="$COEF" \
    DODGEBLOCK_CHECKPOINT_DIR="$dir/checkpoints" \
    DODGEBLOCK_INITIALIZE_FROM="$INIT" \
    DODGEBLOCK_DEVICE=cuda \
    ./rl/run-ppo-v4.sh ) >> "$dir/run.log" 2>&1

  ( cd "$CODE" && CUDA_VISIBLE_DEVICES=1 \
    /home/lr/envs/dodgeblock-rl/bin/python rl/critic-profile.py \
      "$dir/checkpoints/latest.pt" --episodes 384 --target-height 600 \
      > "$dir/critic.json" ) >> "$dir/run.log" 2>&1

  # The behavioural test: if the agent learned to react, the fraction of deaths
  # that were escapable a third of a second earlier should FALL. A score that
  # rises while viability(K=20) stays at 0.88 means it got safer some other way.
  ( cd "$CODE" && CUDA_VISIBLE_DEVICES=1 \
    /home/lr/envs/dodgeblock-rl/bin/python rl/record_trace.py \
      "$dir/checkpoints/latest.pt" --episodes 256 --target-height 10000 \
      --max-frames 6000 --dump-all "$dir/deaths.jsonl" --out "$dir/best.json" \
    && node rl/death-autopsy.mjs "$dir/deaths.jsonl" > "$dir/autopsy.json" \
  ) >> "$dir/run.log" 2>&1

  python3 - "$dir" "$name" "$lambda" "$RESULTS" "$CONTROL" "$COEF" <<'PY'
import json, sys
from pathlib import Path
directory, name, lam, results, control, coef = sys.argv[1:7]
text = (Path(directory) / 'run.log').read_text()
index = text.rfind('"event": "evaluation"')
line = f'- **{name}** lambda={lam} hazard={coef}: '
if index < 0:
    line += 'NO EVAL (crashed)\n'
else:
    data, _ = json.JSONDecoder().raw_decode(text[text.rfind('{', 0, index):])
    success = float(data.get('target_success', 0.0))
    layers = max(1.0, float(data.get('target_height', 600)) / 40.0)
    per_layer = success ** (1.0 / layers) if success > 0 else 0.0
    def load(name, key=None):
        try:
            value = json.loads((Path(directory) / name).read_text())
            return value if key is None else value[key]
        except Exception:
            return {}
    auc = {band: entry.get('auc') for band, entry
           in (load('critic.json', 'auc_at_matched_height') or {}).items()}
    viability = load('autopsy.json', 'viability')
    line += (f'det **{success:.3f}** (control {control}) per-layer {per_layer:.4f} '
             f'median {data.get("median_height")}\n'
             f'    - matched-height AUC {auc}\n'
             f'    - escapable-at-K {viability} (control: 10f 0.725, 20f 0.880)\n')
open(results, 'a').write(line)
print(line.strip())
PY
}

run_one comb-95 0.95
run_one comb-90 0.90
echo "[$(date +%H:%M)] sweep complete" >> "$RESULTS"
