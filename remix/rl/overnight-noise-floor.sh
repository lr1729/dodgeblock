#!/usr/bin/env bash
# What is the noise floor of this A/B protocol?
#
# Every arm so far has been compared against ONE run of the control (det 0.344,
# rung-600-n14). Nothing establishes how much two runs of the SAME configuration
# differ, so no difference has been interpretable, including the ones I called
# falsifications.
#
# There is already a hint that the readout is worse than assumed: rung-600-n14
# and its 8M-frame continuation rung-600-x15 have statistically indistinguishable
# mean height (424.4 [408,441] vs 421.3 [405,436]) and yet report det success
# 0.344 vs 0.295. A 0.049 swing in the headline metric across checkpoints of
# equivalent competence is larger than most effects being chased.
#
# This runs the control config unchanged, varying only the training seed, and
# reports mean height with a bootstrap CI as the primary metric -- it uses every
# episode's full height instead of one success bit, so it resolves far smaller
# differences for the same compute.
set -uo pipefail

CODE=/home/lr/dodgeblock-v5-code
INIT=/home/lr/dodgeblock-ladder/rung-700-x13/checkpoints/latest.pt
ROOT=/home/lr/dodgeblock-noise-floor
RESULTS=$ROOT/RESULTS.md

# Run last: the interventions are already queued and this calibrates them.
while ! grep -q 'sweep complete' /home/lr/dodgeblock-combined/RESULTS.md 2>/dev/null; do sleep 120; done

mkdir -p "$ROOT"
SNAPSHOT=$ROOT/code
rm -rf "$SNAPSHOT"
cp -al "$CODE" "$SNAPSHOT"
CODE=$SNAPSHOT

[ -f "$RESULTS" ] || cat > "$RESULTS" <<'EOF'
# Noise floor — control config, training seed varied, nothing else changed
Reference (seed 7) = rung-600-n14: det 0.344, mean height 424.38 [408.12, 441.02].
The spread across these seeds is the smallest difference any A/B here can claim.

EOF

run_one() {
  local name=$1 seed=$2
  local dir=$ROOT/$name
  mkdir -p "$dir/checkpoints"
  echo "[$(date +%H:%M)] launching $name seed=$seed" >> "$RESULTS"
  ( cd "$CODE" && \
    CUDA_VISIBLE_DEVICES=1 \
    DODGEBLOCK_PYTHON=/home/lr/envs/dodgeblock-rl/bin/python \
    DODGEBLOCK_WORKERS=8 DODGEBLOCK_ENVS_PER_WORKER=64 \
    DODGEBLOCK_TOTAL_FRAMES=8000000 DODGEBLOCK_TARGET_HEIGHT=600 \
    DODGEBLOCK_REWARD_MODE=target DODGEBLOCK_CELL_EVAL_ENVS=0 \
    DODGEBLOCK_COMPILE=1 DODGEBLOCK_CHECKPOINT_INTERVAL=4000000 \
    DODGEBLOCK_EVAL_AFTER_TRAIN=1 DODGEBLOCK_EVAL_EPISODES=2048 \
    DODGEBLOCK_SEED="$seed" \
    DODGEBLOCK_CHECKPOINT_DIR="$dir/checkpoints" \
    DODGEBLOCK_INITIALIZE_FROM="$INIT" \
    DODGEBLOCK_DEVICE=cuda \
    ./rl/run-ppo-v4.sh ) >> "$dir/run.log" 2>&1

  python3 - "$dir" "$name" "$seed" "$RESULTS" <<'PY'
import json, sys
from pathlib import Path
directory, name, seed, results = sys.argv[1:5]
text = (Path(directory) / 'run.log').read_text()
index = text.rfind('"event": "evaluation"')
line = f'- **{name}** seed={seed}: '
if index < 0:
    line += 'NO EVAL (crashed)\n'
else:
    data, _ = json.JSONDecoder().raw_decode(text[text.rfind('{', 0, index):])
    line += (f'det {data.get("target_success")} | '
             f'mean height **{data.get("mean_height")}** '
             f'{data.get("mean_height_ci95")} | '
             f'IQM {data.get("iqm_height")} {data.get("iqm_height_ci95")}\n')
open(results, 'a').write(line)
print(line.strip())
PY
}

run_one seed-8 8
run_one seed-9 9

python3 - "$RESULTS" <<'PY'
import re, sys
values = [float(m.group(1)) for m in
          re.finditer(r'mean height \*\*([0-9.]+)\*\*', open(sys.argv[1]).read())]
values.append(424.38)  # seed 7, rung-600-n14
if len(values) >= 2:
    spread = max(values) - min(values)
    note = (f'\n**Noise floor: {len(values)} seeds of the identical config span '
            f'{spread:.1f} mean height** ({min(values):.1f}-{max(values):.1f}). '
            f'Any A/B difference smaller than this is not evidence of anything.\n')
    open(sys.argv[1], 'a').write(note)
    print(note.strip())
PY
echo "[$(date +%H:%M)] sweep complete" >> "$RESULTS"
