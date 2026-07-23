# DodgeBlock RL

## Recurrent replay agent

The recurrent v2 trainer is retained as an off-policy baseline. It is an R2D2-style recurrent,
dueling, distributional Double-Q agent with prioritized sequence replay. It
uses the authoritative deterministic JavaScript simulation and trains only on
stable height gained, with a one-block terminal penalty to anchor death as a
bad terminal boundary for bootstrapping. There is no survival, near-miss,
material, action, or danger shaping.

The training contract intentionally differs from assisted browser play:

- Auto Guard and gameplay checkpoint continuation are disabled.
- All 18 held horizontal/vertical/Focus combinations preserve browser input
  edges and simultaneous movement during Focus.
- Focus-modified actions are masked when they cannot start or maintain Focus
  Aim. This removes movement aliases that would otherwise leave Focus held
  through a dash and suppress the next deliberate press; it does not choose a
  direction or timing for the policy.
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
- Epsilon exploration holds a sampled valid action for four frames, providing
  coherent movement bursts without reducing the greedy policy's 60 Hz control.
  Actor exploration and replay sampling use independent RNG streams so replay
  ablations do not silently change the behavior-noise sequence.
- Fresh runs start with random valid actions and delayed learning. This avoids
  training the Q head against mostly unterminated, zero-reward fragments from a
  randomly initialized policy.
- The value and advantage output heads are zero-initialized, so early greedy
  preferences come from evidence in replay rather than arbitrary final-layer
  initialization.
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

## Target-policy PPO

`ppo_v2.py` trains for reliable completion of a fixed height target. Reaching
the target and dying are the only terminal outcomes. Potential-based height
shaping gives local credit but telescopes to zero on a failed fresh run and one
on a successful fresh run, so partial failure cannot become the objective.
The undiscounted production objective (`gamma=1`) maximizes target success
probability instead of preferring a faster, riskier completion.

The trainer retains the authoritative 60 Hz primitive controls. GAE decay is
measured in simulation world time, so Focus Aim no longer destroys credit ten
times faster than normal play. Entropy and learning rate anneal during training,
and a player-centered high-resolution terrain branch preserves narrow collision
geometry alongside the coarse whole-arena representation.

PPO is robustification, not trajectory discovery. First create an exact
replayable trajectory with the snapshot frontier explorer:

```bash
node rl/go-explore.mjs \
  --seed 7 \
  --target-height 10000 \
  --output-dir ~/dodgeblock-go-explore/demos
node rl/replay-demo.mjs ~/dodgeblock-go-explore/demos/demo-*.json.gz
```

The explorer uses coherent control options internally but expands them to the
same primitive input stream used by the browser. Every frontier state retains
its parent action segment, so a success produces a deterministic demonstration
from frame zero rather than a disconnected high-altitude snapshot.

Training replays the demonstration into a memory-bounded set of exact simulator
snapshots. A competence gate expands starts farther from the target only after
the current frontier is solved reliably. As the frontier moves backward, the
demonstration-start probability falls from 0.8 to 0.2 and unrevealed future
randomness rises to 100%; the remaining episodes always start from fresh seeds.
The final mixture therefore retains useful hard states without retaining the
demonstration's future sequence.

Curriculum state deliberately restarts at the easiest frontier after a process
restart. A resumed policy normally re-expands it quickly, while elapsed wall
time can never masquerade as measured competence. PPO checkpoints validate the
objective, architecture, demonstration hash, and schedule before loading.

```bash
python rl/ppo_v2.py \
  --device cuda \
  --workers 8 \
  --envs-per-worker 64 \
  --demonstration ~/dodgeblock-go-explore/demos/demo-7-10k.json.gz \
  --archive-probability 0
python rl/evaluate_ppo_v2.py \
  ~/dodgeblock-ppo-target-v3/checkpoints/latest.pt \
  --episodes 256

DODGEBLOCK_TEST_DEMONSTRATION=~/dodgeblock-go-explore/demos/demo-7-10k.json.gz \
  npm run test:rl-python
```

The historical feed-forward PPO implementation is retained separately as an
old baseline. Its action/observation contract and frame-survival reward do not
match the v2 benchmark, so its scores must not be compared directly.

```bash
python rl/train.py --envs 128 --total-steps 50000000
```

CPU is the default for the PPO baseline on the Beelink because it outperforms
the unsupported Cezanne ROCm path. Its checkpoints live under
`~/dodgeblock-rl/checkpoints`.
