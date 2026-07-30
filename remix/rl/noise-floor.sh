#!/bin/bash
# Measure what the accept gate is actually resting on.
#
# Round 1 of the loop returned mean gain -0.0071 nats over three banks with an
# SE of the mean of 0.0091 -- the 0.008 threshold sits below one standard error
# of its own measurement, so accept/reject near the gate is a coin flip. Three
# quantities settle how to fix it:
#
#   A. baseline round1.pt over 8 banks, eval seed 0x5A70BE5   (widened baseline)
#   B. round-1 candidate over the same 8 banks, same seed     (paired delta, n=8)
#   C. baseline again over 4 banks, a DIFFERENT eval seed     (pure eval noise)
#
# B-A gives the true round-1 effect at n=8 instead of n=3. C-A isolates eval
# sampling noise from genuine bank-to-bank heterogeneity, which decides whether
# more banks or more episodes per bank is the cheaper way to sharpen the gate.
set -u
CODE=/home/lr/dodgeblock-v5-code
PY=/home/lr/envs/dodgeblock-rl/bin/python
BANKDIR=/home/lr/dodgeblock-go-explore-bank-v4
OUT=/home/lr/dodgeblock-distill-loop/noise-floor
BASE=/home/lr/dodgeblock-distill/round1.pt
CAND=/home/lr/dodgeblock-distill-loop/round-1/distilled.pt
ALT_SEED=99173

mkdir -p "$OUT"
cd "$CODE" || exit 1

run () {          # run <tag> <checkpoint> <bank> <seed-args...>
  local tag=$1 ckpt=$2 bank=$3; shift 3
  local dest="$OUT/$tag.json"
  [ -s "$dest" ] && { echo "[skip] $tag"; return; }
  local t0=$SECONDS
  CUDA_VISIBLE_DEVICES=1 "$PY" rl/saturated-hazard.py "$ckpt" \
      --bank "$BANKDIR/$bank/search-checkpoint.json.gz" \
      --control-interval 1 --envs 128 --rounds 4 "$@" > "$dest" 2>"$OUT/$tag.err"
  echo "[done] $tag  $((SECONDS - t0))s"
  pkill -f "env-server-v2[.]mjs" 2>/dev/null
  sleep 2
}

for b in 1 2 3 4 5 6 7 8; do run "base-seed-$b"  "$BASE" "seed-$b"; done
for b in 1 2 3 4 5 6 7 8; do run "cand-seed-$b"  "$CAND" "seed-$b"; done
for b in 1 2 3 4;         do run "alt-seed-$b"   "$BASE" "seed-$b" --seed "$ALT_SEED"; done

echo "ALL DONE"
