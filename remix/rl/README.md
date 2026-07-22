# DodgeBlock RL

## Recurrent replay agent

The v2 trainer is the primary experiment. It is an R2D2-style recurrent,
dueling, distributional Double-Q agent with prioritized sequence replay. It
uses the authoritative deterministic JavaScript simulation and trains only on
stable height gained. There is no survival, near-miss, material, action, or
danger shaping.

The training contract intentionally differs from assisted browser play:

- Auto Guard and gameplay checkpoint continuation are disabled.
- All 18 held horizontal/vertical/Focus combinations preserve browser input
  edges and simultaneous movement during Focus.
- Observations contain causal structured geometry, warning countdowns, stacked
  forecasts, control state, and Focus state, but never the RNG or unrevealed
  future drops.
- A training-only archive retains diverse stable states at 100-height bands.
  Evaluation always starts at height zero on held-out seeds.

```bash
python rl/train_v2.py --device cuda --workers 8 --envs-per-worker 64
python rl/evaluate_v2.py ~/dodgeblock-r2d2/checkpoints/latest.pt
```

Checkpoints include both Q networks, optimizer state, frame count, and the full
training configuration. The runner automatically resumes `latest.pt`.

## PPO baseline

The original feed-forward PPO implementation is retained as a historical
baseline. Its action/observation contract and frame-survival reward do not
match the v2 benchmark, so its scores must not be compared directly.

```bash
python rl/train.py --envs 128 --total-steps 50000000
```

CPU is the default for the PPO baseline on the Beelink because it outperforms
the unsupported Cezanne ROCm path. Its checkpoints live under
`~/dodgeblock-rl/checkpoints`.
