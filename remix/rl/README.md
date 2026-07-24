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
- Training and evaluation start at height zero; the recurrent agent is retained
  only as a model-free baseline for the current simulator contract.
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
remains process-local, so planned restarts should occur only at experiment
boundaries.

The trainer logs wall-time fractions for actor inference, environment stepping,
sequence assembly, replay sampling, and learning, plus CUDA peak memory. Use
these measurements rather than actor count or aggregate GPU utilization when
tuning a machine. `--compile` is experimental and disabled by default because
the recurrent model has several fixed sequence shapes and compilation can have
a large cold-start cost.

## Target-policy PPO

`ppo_v2.py` now implements the v4 from-scratch target agent. Hardcore rules are
fixed: Auto Guard and checkpoint continuation are disabled. Potential shaping
telescopes to zero on failed fresh runs and one on target completion, while
`gamma=1` optimizes 10k success probability without preferring faster risk.

The policy keeps the browser's 18 primitive held-input actions, represented as
the exact autoregressive distribution
`P(Focus) P(vertical | Focus) P(horizontal | Focus, vertical)`. This preserves
same-frame control correlations while giving rare Focus decisions independent
exploration and telemetry. GAE remains measured in simulation world time.

Observations include only causal information: visible geometry and forecasts,
control timers, and remaining material counts derivable from past drops. Hidden
bag order and future RNG are never exposed. Curriculum restoration preserves
the remaining bag multiset and reshuffles only its hidden order.

Generate machine-only state banks on multiple seeds:

```bash
python rl/run_go_explore_bank.py \
  --seed-start 1 \
  --seeds 16 \
  --jobs 8 \
  --output-dir ~/dodgeblock-go-explore-bank-v4
```

The Python coordinator groups similar variants by the explorer's cell key,
holds out cell groups deterministically, tracks Beta competence evidence per
cell, balances source variants, and starts mostly near the target. Competence
propagates sampling smoothly toward lower height bands while every band and
fresh starts retain exploration probability. There is no advancement gate.
Held-out environments act deterministically and are excluded from PPO loss.
Workers only restore coordinator-selected variants and step the authoritative
simulator.

```bash
export DODGEBLOCK_CELL_BANK_GLOB="$HOME/dodgeblock-go-explore-bank-v4/seed-*/search-checkpoint.json.gz"
rl/run-ppo-v4.sh
python rl/evaluate_ppo_v2.py ~/dodgeblock-ppo-v4/checkpoints/latest.pt --episodes 256
```

PPO checkpoints include the immutable bank hashes and centralized curriculum
statistics. Changing the observation, action, objective, or held-out split
requires a new experiment directory. Adding or replacing bank files emits a
warning, restores evidence by stable cell key, and resets variant-order
counters.

To measure bounded tactical rescuability, collect sparse exact replay histories
during an evaluation and search coherent action bursts from multiple rewind
offsets against both the original future and reshuffled remaining-material
futures:

```bash
python rl/evaluate_ppo_v2.py ~/dodgeblock-ppo-v4/checkpoints/latest.pt \
  --episodes 128 \
  --death-case-dir ~/dodgeblock-rescue-cases
node rl/rescue-oracle.mjs \
  --case ~/dodgeblock-rescue-cases \
  --trials 64 \
  --horizon 360 \
  --futures 3 \
  --rewinds 1,30,60,120,240 \
  --output ~/dodgeblock-rescue-cases/report.json
```

A found original-future rescue is a demonstrated counterexample to
unavoidability. Failure to find one is inconclusive; compare rescue-rate curves
across trial budgets and horizons rather than treating the diagnostic as a
proof of the Hardcore ceiling.

```bash
python rl/train.py --envs 128 --total-steps 50000000
```

CPU is the default for the PPO baseline on the Beelink because it outperforms
the unsupported Cezanne ROCm path. Its checkpoints live under
`~/dodgeblock-rl/checkpoints`.

## V5 search distillation and robustification

The completed v4 run is retained as a negative ablation. It learned the final
400-height suffix but success probability decayed too quickly for a reverse
curriculum to reach fresh starts. V5 uses the explorer's machine-generated
actions as an annealed prior instead of asking PPO to rediscover them from
zero-return episodes.

First export exact observations from successful demonstrations. The exporter
replays every action against the authoritative simulator, verifies the final
hash, writes compact typed-array shards, and captures successful-trajectory
snapshots. A demo that does not replay exactly is rejected.

```bash
args=()
for demo in ~/dodgeblock-go-explore-bank-v4/seed-*/demo-*.json.gz; do
  args+=(--demo "$demo")
done
node rl/export-demo-dataset.mjs "${args[@]}" \
  --output-dir ~/dodgeblock-demo-dataset-v5
```

Behavior cloning uses seeds 1–12 for optimization and excludes seeds 13–16
from both gradient updates and snapshot starts. It samples source seeds
uniformly and trains the exact autoregressive PPO head with conditional
cross-entropy. A fixed mixture emphasizes opening, action-change, initial, and
Focus-positive frames without discarding the full frame distribution. Full
held-out loss remains separately reported.

```bash
export DODGEBLOCK_DEMO_DATASET="$HOME/dodgeblock-demo-dataset-v5"
rl/run-bc-v5.sh
python rl/evaluate_ppo_v2.py \
  ~/dodgeblock-bc-v5/checkpoints/best.pt \
  --episodes 256
```

Imitation metrics are necessary but not a rollout gate. Success-only PPO
should start only after the distilled policy produces meaningful closed-loop
fresh behavior. If BC has no fresh target completions, collect policy deaths
and generate search corrections before relying on target reward.

The cold-start correction stage searches from on-policy rewind states against
the original future and two common reseeded futures. Only robust candidates
are retained, and only their first second is distilled; later random search
actions are not treated as expert behavior.

```bash
python rl/evaluate_ppo_v2.py ~/dodgeblock-bc-v5/checkpoints/best.pt \
  --episodes 256 --stochastic \
  --death-case-dir ~/dodgeblock-bc-v5/deaths
node rl/rescue-oracle.mjs \
  --case ~/dodgeblock-bc-v5/deaths \
  --trials 64 --horizon 240 --futures 3 --rewinds 120 \
  --output ~/dodgeblock-bc-v5/oracle.json
node rl/export-oracle-corrections.mjs \
  --oracle ~/dodgeblock-bc-v5/oracle.json \
  --output-dir ~/dodgeblock-demo-dataset-v5 \
  --shard-seed 1001 --prefix-frames 60
```

V5 PPO initializes the policy from BC, starts half of training episodes fresh,
and samples the remainder uniformly by successful source seed and trajectory
time. It never uses arbitrary explorer archive cells. Snapshot restoration
preserves visible terrain and the remaining material multiset while
randomizing only hidden future order. Demonstration cross-entropy is evaluated
only on exact demonstration observations and anneals during robustification;
there is no global KL constraint on states where the search policy has no
demonstrated action.

```bash
rl/run-ppo-v5.sh
```

`audit-demo-trajectories.mjs` reports exact replay integrity, Focus collision
outcomes, shelter occupancy, and terrain relief without turning any of those
diagnostics into a reward:

```bash
node rl/audit-demo-trajectories.mjs \
  ~/dodgeblock-go-explore-bank-v4/seed-*/demo-*.json.gz
```
