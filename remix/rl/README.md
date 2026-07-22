# DodgeBlock RL

## Recurrent replay agent

The v2 trainer is the primary experiment. It is an R2D2-style recurrent,
dueling, distributional Double-Q agent with prioritized sequence replay. It
uses the authoritative deterministic JavaScript simulation and trains only on
stable height gained. There is no survival, death, near-miss, material, action,
or danger shaping.

The training contract intentionally differs from assisted browser play:

- Auto Guard and gameplay checkpoint continuation are disabled.
- All 18 held horizontal/vertical/Focus combinations preserve browser input
  edges and simultaneous movement during Focus.
- Observations contain causal structured geometry, warning countdowns, stacked
  forecasts, control state, and Focus state, but never the RNG or unrevealed
  future drops.
- Fixed terrain is aligned to the authoritative block lattice. Material,
  exact horizontal block edges, collapse-branch/target state, player occupancy,
  and absolute remaining collapse time are separate learned channels rather than an
  aliased screen-space raster or a precomputed carve answer.
- Falling-block and forecast tokens use count-aware learned-query pooling. This
  preserves multiplicity and several complementary learned summaries; it is
  intentionally lighter than full pairwise token self-attention.
- Discounting follows simulation world time rather than controller ticks.
  Focus Aim therefore incurs one tenth of the ordinary temporal discount, just
  as it advances the physical world at one tenth speed. The production gamma
  of 0.99999 has a world-time reward half-life of about 69,314 frames, or 19.3
  minutes, and approximates expected final height while retaining a weak
  contraction for stable off-policy learning.
- Twenty-step distributional backups correspond to the 20 physics frames used
  by the common five-step, four-frame-repeat Atari configuration.
- A training-only archive retains diverse stable states at 100-height bands.
  Evaluation always starts at height zero on held-out seeds.
- Replay stores contiguous float observations in half precision. The default
  32,768 sequences cover about 2.6 million non-overlapping transitions while
  remaining in host RAM; simulator state and reward calculation remain full
  precision.

```bash
python rl/train_v2.py --device cuda --workers 8 --envs-per-worker 64
python rl/evaluate_v2.py ~/dodgeblock-r2d2/checkpoints/latest.pt --episodes 256
```

Checkpoints include both Q networks, optimizer state, frame count, and the full
training configuration. The runner automatically resumes `latest.pt`. Replay
and curriculum archives intentionally remain process-local, so planned restarts
should occur only at experiment boundaries.

The trainer logs wall-time fractions for actor inference, environment stepping,
sequence assembly, replay sampling, and learning, plus CUDA peak memory. Use
these measurements rather than actor count or aggregate GPU utilization when
tuning a machine. `--compile` is experimental and disabled by default because
the recurrent model has several fixed sequence shapes and compilation can have
a large cold-start cost.

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
