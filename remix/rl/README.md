# DodgeBlock RL

[`RESEARCH-LOG.md`](./RESEARCH-LOG.md) records the experiment ledger: every
phase's design, registered predictions, decisive measurements, falsified
hypotheses, and the artifact index. Read it before changing the training
design; append stage results as they land.

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

The v1 trainer (`train.py`) and its v1 observation stack (`env.mjs`,
`env-server.mjs`), the R2D2 trainer (`train_v2.py`), and the v5 behaviour-cloning
trainer (`train_bc_v5.py`) were removed on 2026-07-24; their results are in
[`RESEARCH-LOG.md`](./RESEARCH-LOG.md). The live path is `ladder_driver.py` →
`run-ppo-v4.sh` → `ppo_v2.py`.

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

Raw Go-Explore actions are a weak policy target: its option proposals are
state-independent and competence comes from snapshot retries. Keep successful
trajectories as a low-level initialization and source of verified simulator
states, not as the dominant decision signal. `counterfactual-teacher.mjs`
turns a verified successful continuation into causal soft labels by replacing
each candidate action under the same known-successful suffix and matched
future randomizations:

```bash
node rl/counterfactual-teacher.mjs \
  --demo /path/to/verified-demo.json.gz \
  --output-dir ~/dodgeblock-counterfactual-v5 \
  --shard-seed 4001 --horizon 360 --futures 2
```

Correction shards are sampled separately so a small causal dataset can
dominate optimization without pretending it has the same role as trajectory
frames. Hold out correction sources independently and select checkpoints on
their loss:

```bash
export DODGEBLOCK_CORRECTION_DATASET="$HOME/dodgeblock-counterfactual-v5"
export DODGEBLOCK_CORRECTION_TRAIN_SEEDS="4001-4010"
export DODGEBLOCK_CORRECTION_VALIDATION_SEEDS="4011-4012"
export DODGEBLOCK_CORRECTION_FRACTION=0.75
export DODGEBLOCK_BC_SELECTION_METRIC=correction_validation
rl/run-bc-v5.sh
```

The trainer reports `joint_accuracy_lift_over_repeat`; a model matching the
repeat-previous baseline has not learned state-dependent decisions regardless
of its raw frame accuracy. PPO remains gated on fresh closed-loop rollouts.

For causal labels sampled at a regular cadence, train the direct action head
and remove the explorer's arbitrary burst boundary from the contract. Physics
still advances at 60 Hz; evaluation holds each decision for the saved number
of frames:

```bash
export DODGEBLOCK_CORRECTION_DATASET="$HOME/dodgeblock-counterfactual-v5-cadence2"
export DODGEBLOCK_CORRECTION_TRAIN_SEEDS="5001-5012"
export DODGEBLOCK_CORRECTION_VALIDATION_SEEDS="5013-5016"
export DODGEBLOCK_CORRECTION_FRACTION=1
export DODGEBLOCK_FIXED_CONTROL_INTERVAL=2
export DODGEBLOCK_BC_SELECTION_METRIC=correction_validation
rl/run-bc-v5.sh
```

`evaluate_ppo_v2.py` reads this cadence from the checkpoint.

The cold-start correction stage searches from on-policy rewind states against
the original future and two common reseeded futures. It branches all 18
actions over three matched continuation plans, averages each action's return,
and distills a soft action target plus the six-frame option prefix. This keeps
multiple viable escapes, teaches held-action intent, and avoids treating later
random search actions as expert behavior. Local layer progress is used only to
bootstrap the cold-start oracle; the final PPO objective remains binary 10k
success.

```bash
python rl/evaluate_ppo_v2.py ~/dodgeblock-bc-v5/checkpoints/best.pt \
  --episodes 256 --stochastic \
  --death-case-dir ~/dodgeblock-bc-v5/deaths
node rl/rescue-oracle.mjs \
  --case ~/dodgeblock-bc-v5/deaths \
  --trials 64 --horizon 240 --futures 3 --rewinds 30,60,90,120 \
  --all-candidates \
  --output ~/dodgeblock-bc-v5/oracle.json
node rl/export-oracle-corrections.mjs \
  --oracle ~/dodgeblock-bc-v5/oracle.json \
  --output-dir ~/dodgeblock-demo-dataset-v5 \
  --shard-seed 1001 --soft-targets --temperature 1 --branch-prefix 6
```

An opening-only bootstrap can use dense height reward with a 1k terminal
goal, always from a fresh game. This is a machine-learned low-level prior, not
the final objective and not a suffix curriculum:

```bash
export DODGEBLOCK_TARGET_HEIGHT=1000
export DODGEBLOCK_REWARD_MODE=height
export DODGEBLOCK_CELL_EVAL_ENVS=0
export DODGEBLOCK_TOTAL_FRAMES=20000000
rl/run-ppo-v4.sh
```

Later goal stages use `DODGEBLOCK_REWARD_MODE=target`, start from the preceding
checkpoint in a new directory, and retain fresh starts. The 10k stage is
therefore optimized against the original binary Hardcore objective.

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

## Search-in-the-loop policy iteration

The sections above are kept as a record of what did not work. V5 asked a policy
to imitate a search whose competence lived in snapshot retries rather than in
its action choices, so there was nothing state-dependent to clone. Reward
shaping failed for a separate reason recorded in the log: at gamma=1 a potential
telescopes to a constant, and the interventions that survived that objection
still added information to the reward at states where the GAE estimator
discarded it.

The current design adds no reward terms and no human data. It generates an
action-conditioned training signal by searching from the policy's own decision
states, then distils that signal back into the policy.

The unit of progress is **per-layer survival in the saturated regime**, not
score or episode length. Reaching 10,000 at all needs 0.9567 per layer;
reaching it consistently needs 0.9972. Because survival composes multiplicatively
over ~250 layers, the quantity to add up is log-hazard in nats, and every
comparison below is in nats.

Difficulty is a function of elapsed time and saturates after ~240 s, and a fresh
policy dies long before then, so saturated survival cannot be measured from
fresh starts. Start from the Go-Explore banks instead, which hold 512 verified
snapshots per seed with a median start of 309 s:

```bash
python rl/saturated-hazard.py CHECKPOINT \
  --bank ~/dodgeblock-go-explore-bank-v4/seed-1/search-checkpoint.json.gz \
  --control-interval 1 --envs 128 --rounds 4
```

This is deterministic given (checkpoint, bank, seed), so re-running it verifies
rather than resamples. Two consequences matter for any comparison built on it.
Evaluating the same checkpoint under a different `--seed` moves the answer by an
SD of 0.031 nats, while evaluating a *different* checkpoint under the same seed
moves it by 0.014 -- the second is smaller because base and candidate draw the
same cells and the same futures and most of the sampling noise cancels. That
pairing is worth 4.7x, and it decays as the two policies diverge, so a paired SD
measured on near-identical checkpoints understates the noise between checkpoints
that genuinely differ. Bank count, not episodes per bank, is the lever on
precision: the SE of a mean over n banks is 0.0142/sqrt(n).

Collection branches K lanes from each decision state under common random
numbers, rolls every lane forward under the policy, and records which first
actions survived:

```bash
python rl/search-distill-collect.py CHECKPOINT \
  --out targets.npz --samples 8000 --lanes 32 --workers 8 \
  --horizon 120 --stride 10 --control-interval 1 --max-hours 10
```

States where every lane lives or every lane dies are dropped -- a unanimous
verdict teaches nothing -- and `decision_fraction` in the output reports how
many survived that filter. It rises with K (0.309 at 32 lanes, 0.592 at 128)
while raw throughput falls, so larger K buys better-estimated targets from a
larger share of visited states at roughly 1.5x the cost per usable sample.

The target is the survival-weighted distribution over first actions, which makes
this a soft one-step improvement operator rather than best-of-K selection: the
new policy is proportional to the old one reweighted by each action's measured
survival. Training fits it with cross-entropy under a KL anchor to the frozen
source, because the targets only cover states where branches disagreed and the
anchor is what holds the rest of the policy in place:

```bash
python rl/search-distill-train.py CHECKPOINT targets.npz \
  --out distilled.pt --epochs 8 --batch 256 \
  --learning-rate 1e-4 --anchor-coef 1.0
```

**Check `anchor_kl` in the output before interpreting any evaluation of the
result.** It reports how far the update actually moved the policy, and a run
that lands far below the 0.01-0.05 nats of a normal trust region has not tested
the targets at all -- it has produced an unchanged policy, which evaluates as a
null no matter what the targets contain. The defaults above are calibrated to
land inside that range on 8000 samples; `--learning-rate 1e-5 --anchor-coef 1.0
--epochs 4` reaches 0.001 and is a no-op.

`distill_driver.py` runs the loop unattended under systemd, chaining stages on
file markers. Two rules it enforces are worth stating because both were learned
by getting them wrong. A round is accepted on the mean over many banks rather
than one, since absolute survival swings 0.1 between banks while the paired
difference is stable. And the accept threshold is sized as a multiple of the SE
of that mean -- dividing the bank spread by sqrt(n) and multiplying by the
number of standard errors required -- not as a fraction of the spread itself.
