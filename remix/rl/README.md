# Dodgeblock RL

The trainer uses the real deterministic JavaScript simulation. A binary Node
bridge batches environments and exposes exact terrain, falling-block, forecast,
player, and Focus state to a compact feed-forward PPO policy.

```bash
python rl/train.py --envs 128 --total-steps 50000000
```

CPU is the default on the Beelink because it outperforms the unsupported Cezanne
ROCm path for training batches. Experimental ROCm runs require
`HSA_OVERRIDE_GFX_VERSION=9.0.0` and `--device cuda`; keep minibatches at 512 or
below. Checkpoints are written atomically by step under
`~/dodgeblock-rl/checkpoints`.
