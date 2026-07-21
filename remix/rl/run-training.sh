#!/bin/sh
set -eu

cd "$(dirname "$0")/.."
checkpoint_dir="${HOME}/dodgeblock-rl/checkpoints"
set -- --envs 128 --rollout 256 --total-steps 200000000 --epochs 4 \
  --minibatch 1024 --threads 12 --checkpoint-dir "$checkpoint_dir"
if [ -e "$checkpoint_dir/latest.pt" ]; then
  set -- "$@" --resume "$checkpoint_dir/latest.pt"
fi
exec python rl/train.py "$@"
