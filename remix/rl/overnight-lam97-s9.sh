#!/usr/bin/env bash
# Third seed for lambda = 0.97, the project's only surviving candidate.
#
# Why this is not a new direction: overnight-driver.sh already implements
# "replicate the best arm at fresh seeds", but it scans only the rollout and
# gamma result files. lambda=0.97's seed-7 run (mean height 440.0) lives in
# dodgeblock-lambda/RESULTS.md, which the driver never reads, so it sees only the
# seed-8 replicate at 421.41, judges it below the band top, and stops. This is
# the driver's own replication step with the arm correctly identified.
#
# It also corrects the ANALYSIS. Judging arms against an unpaired noise band
# (409.2-424.4) is conservative because seed variance is the dominant term. Every
# arm shares an init and an eval seed set, so comparing an arm to the control
# trained at the SAME seed removes that term:
#
#   seed 7:  control 424.38   lambda=0.97 440.00   +15.62
#   seed 8:  control 409.22   lambda=0.97 421.41   +12.19
#
# Same sign, similar magnitude, at both seeds. Seed 9's control is 414.20, so the
# prediction registered here is a lambda=0.97 result near 425-430. A third
# same-signed difference would make this the first real effect in the ledger; a
# null or negative one would end it.
set -uo pipefail

CODE=/home/lr/dodgeblock-v5-code
INIT=/home/lr/dodgeblock-ladder/rung-700-x13/checkpoints/latest.pt
ROOT=/home/lr/dodgeblock-lam97
RESULTS=$ROOT/RESULTS.md

# Chain on the driver's terminal marker, never on the process table -- a pgrep
# wait loop whose pattern matched its own command line has idled this GPU twice.
while ! grep -qE 'stopping|ladder seeded' /home/lr/dodgeblock-driver/RESULTS.md 2>/dev/null; do
  sleep 120
done

mkdir -p "$ROOT"
SNAPSHOT=$ROOT/code
rm -rf "$SNAPSHOT"
cp -al "$CODE" "$SNAPSHOT"
CODE=$SNAPSHOT

[ -f "$RESULTS" ] || cat > "$RESULTS" <<'EOF'
# lambda = 0.97, third seed
Paired against the control trained at the same seed:
  seed 7: 424.38 -> 440.00 (+15.62)
  seed 8: 409.22 -> 421.41 (+12.19)
Seed 9's control is 414.20. Registered prediction: ~425-430 if the effect is real.

EOF

dir=$ROOT/lam-97-s9
mkdir -p "$dir/checkpoints"
echo "[$(date +%H:%M)] launching lam-97 seed 9" >> "$RESULTS"
( cd "$CODE" && \
  CUDA_VISIBLE_DEVICES=1 \
  DODGEBLOCK_PYTHON=/home/lr/envs/dodgeblock-rl/bin/python \
  DODGEBLOCK_WORKERS=8 DODGEBLOCK_ENVS_PER_WORKER=64 \
  DODGEBLOCK_TOTAL_FRAMES=8000000 DODGEBLOCK_TARGET_HEIGHT=600 \
  DODGEBLOCK_REWARD_MODE=target DODGEBLOCK_CELL_EVAL_ENVS=0 \
  DODGEBLOCK_COMPILE=1 DODGEBLOCK_CHECKPOINT_INTERVAL=4000000 \
  DODGEBLOCK_EVAL_AFTER_TRAIN=1 DODGEBLOCK_EVAL_EPISODES=2048 \
  DODGEBLOCK_GAE_LAMBDA=0.97 DODGEBLOCK_SEED=9 \
  DODGEBLOCK_CHECKPOINT_DIR="$dir/checkpoints" \
  DODGEBLOCK_INITIALIZE_FROM="$INIT" \
  DODGEBLOCK_DEVICE=cuda \
  ./rl/run-ppo-v4.sh ) >> "$dir/run.log" 2>&1

python3 - "$dir" "$RESULTS" <<'PY'
import json, sys, statistics
from pathlib import Path
directory, results = sys.argv[1:3]
text = (Path(directory) / 'run.log').read_text()
index = text.rfind('"event": "evaluation"')
if index < 0:
    open(results, 'a').write('- **lam-97-s9**: NO EVAL (crashed)\n')
    raise SystemExit
data, _ = json.JSONDecoder().raw_decode(text[text.rfind('{', 0, index):])
height = float(data.get('mean_height', 0.0))
diffs = [440.00 - 424.38, 421.41 - 409.22, height - 414.20]
verdict = ('HOLDS at 3 seeds' if all(d > 0 for d in diffs)
           else 'DID NOT REPLICATE')
open(results, 'a').write(
    f'- **lam-97-s9** seed=9: det {data.get("target_success")} | mean height '
    f'**{height}** {data.get("mean_height_ci95")}\n'
    f'- paired diffs vs same-seed control: '
    f'{diffs[0]:+.2f} / {diffs[1]:+.2f} / {diffs[2]:+.2f} '
    f'(mean {statistics.mean(diffs):+.2f}) -> **{verdict}**\n')
print(verdict, f'{height:.2f}')
PY
