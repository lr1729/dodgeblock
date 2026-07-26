# DodgeBlock RL Research Log

Goal: a from-scratch policy that consistently reaches 10k+ stable height on
Hardcore rules (no Auto Guard, no checkpoints) — the mark a good human reaches
through shelter play: wait for towers to form, hold cover through surges, climb
in lulls, never cut your own roof. Constraints: machine-generated data only (no
human demonstrations), no hand-shaped side rewards. 10k ≈ 250 layers, so
consistency requires ≈ 0.998 per-layer survival. This is a reliability problem
wearing a score counter.

Phase numbers below are design generations, not file names (`ppo_v2.py` hosts
phases v3–v6). Every pivot in this log was forced by a stated measurement; each
phase records what was predicted, what was measured, and the lesson that the
next phase was built from.

## Phase ledger

| Phase | When | Design | Decisive result |
| --- | --- | --- | --- |
| v1 era | ≤ Jul 22 | height-reward PPO, R2D2, macro-action PPO | 100M frames → mean ~700 (stoch 684 / det 344) |
| v3 | Jul 22–23 | success-only reward + single-demo backward corridor | gate wedged ~2.5–3.3k of 24,887 frames; fresh median 0 |
| v4 | Jul 23 | cell-bank curriculum, AR head, obs v2 | 120M frames → 3 of 24 bands; fresh det IQM 0.0 |
| v5a | Jul 23–24 | BC on explorer trajectories (9 variants + dagger1) | all flat at fresh median 40; s1k = 0 |
| v5b | Jul 24 | correction labels; counterfactual teacher (mixed, cadence) | held-out direction loss ≈ ln(3); gate failed |
| v6 | Jul 24 → | staged fresh-goal ladder (live) | stage 1 (1k bootstrap) running |

### v1 era — expected height

Objective: maximize expected final height (plus an R2D2 baseline and a
macro-action ablation). Result: mean ~700 with stochastic (684) beating
deterministic (344) — a policy leaning on noise. The mean objective literally
prefers the wrong policy at these numbers (reliable-to-2k beats 15%-to-10k in
expectation). Stack defects of the era: GAE λ decayed per controller tick
(destroying Focus credit ~10× during Aim), archive-inflated reporting, no
deterministic eval discipline. **Lesson: reliability needs a success-probability
objective; means and tails want different policies.**

### v3 — correct objective, backward corridor

Target reward: reach 10k = 1, else 0, γ = 1, potential shaping that telescopes
exactly (fresh failure return ≡ 0 — verified in telemetry). Go-Explore produced
one 10k demo on seed 7 (24,887 frames; 85 Focus presses ≈ the recharge
cadence); a competence-gated reverse curriculum walked start states backward.
Measured: tail competence real (~0.8 success from 9.5k+ starts) but the gate
wedged at ~10–13% walked back with frontier success degrading 0.66 → 0.33; fresh
episodes returned exactly 0 with policy entropy ≈ ln 9 (uniform). Implementation
defects found by review and later confirmed in logs: the gate was fragmented
per worker (8 independent gates), and the frontier measure mixed easy states
(50–100% band), inflating success and causing expand-then-wedge. **Lesson: with
a converged critic, advantages ≡ ΔP(success) — a correct objective is mute
where P ≈ 0, and shaping is value initialization, not information.**

### v4 — the clean big experiment

Pure-simulator workers + one Python coordinator; per-cell Beta competence
grouped by explorer cellKey with hash-split held-out cells; 16-seed stratified
banks (all 16 explorer seeds reached 10k); autoregressive head
P(F)·P(V|F)·P(H|F,V) over the unchanged 18 actions; observation completed
(bag-remainder counters with multiset-preserving reshuffle, jump buffer,
timeSinceJump); per-factor entropy; death/shelter/rescue telemetry. 120M
frames at ~2.9k sps.

Measured: the mechanism worked — top band 22.4% train / 16.8% held-out
(genuine generalization; EV 0.94) — and the hypothesis failed: 9,600-band 22.4%,
9,200 ~1.0%, 8,800 0.03%, everything lower 0. Three bands crossed of
twenty-four; 63% of all curriculum starts were spent on the final 10 layers to
reach ≈ 0.861/layer against the ≈ 0.998 needed. Fresh deterministic IQM 0.0;
the fresh policy was an out-of-distribution dive reflex (84.3% fast-fall).
Band-to-band transfer was worse than independent compounding predicts
(9,200 observed ~1% vs ~4.4%) — distribution shift plus inherited state debt.
Side findings: bank bands below 4,400 held only 16–32 cells; the seed-7 demo's
final band ran at just 20% overhead cover (explorer terrain is not
shelter-player terrain); Focus was spent on recharge (press-given-available
0.73) with deaths overwhelmingly at zero charges; shelter occupancy showed no
phase timing (calm 0.72 / build 0.51 / surge 0.59 / release 0.74).
**Lessons: the exponential is the objective's geometry, not a tuning artifact;
backward curricula learn the hardest regime first; cover is accumulated
capital — suffix starts train the spending, never the accumulation; "reached
alive" ≠ "competently constructed."**

### v5a — distill the trajectories

BC on the 16 successful demos (339,154 frames; seeds 1–12 train, 13–16 held
out including snapshot starts). Nine variants (v5…v5h, sticky-a): all pinned
at fresh median 40, success_1k = 0.000, nine epochs flat. Dagger1 (28 more
demos searched from BC-death query regions, replay-verified 27/28) — same
result, as predicted in advance.

Root cause, measured then confirmed by the trainer's own baselines: Go-Explore's
option proposals are drawn from constant weights — state-independent by
construction — so trajectory competence lives in snapshot-retry selection, not
in the actions. Teacher marginal entropy 1.728 nats; frame-to-frame repeat
0.870; entropy given previous action 0.581. Best BC validation 1.13 nats, and
on the final run joint accuracy sat at the repeat-previous baseline (0.858 vs
0.870) while repeating 98% of frames. **Lesson: only p(action|state) content is
distillable. A search trajectory is the residue of search, not its content.**

### v5b — distill the search's decisions

(i) Old correction labels weighted dominantly: failed — the labels themselves
carried 85% probability mass on repeat (caught in one run by the new
`joint_accuracy_lift_over_repeat` metric). (ii) Counterfactual teacher, mixed
mode (branch candidates ~6 frames, splice back into a verified successful
suffix, matched futures): corpus quality-gated well (target entropy 1.30–1.57;
demo action selected only 12–24%), but distillation first destroyed the sticky
repeat controller (decision shards repeat 9.5% vs gameplay 87%), and after
separating proposal/repeat losses it overfit 1,536 states and deployed as 100%
neutral — the retry-generated burst timing is unlearnable as a "when to decide"
signal. (iii) Fixed cadence (control interval 2, direct head, 512 evenly-spaced
states × 16 seeds; 15/16 shards replay-valid): failed the gate at median 40,
and the diagnostic closed the question — **held-out direction loss ≈ ln(3)
because the labels themselves are near-uniform. At 2-frame commitment with a
repairing suffix, Q(s,a) is flat almost everywhere: this game's decisions are
sparse in time.** Signal concentrates at critical states (imminent crush,
Focus commitments, route choices) and option-level horizons — the crisis-
filtered mixed corpus had measurably sharper labels (1.30–1.57 vs 1.73–1.79
nats) than the uniform one. The archive retains no branch outcomes, so there is
no Q-table to mine after the fact. Untested reserve: critical-state labels ×
direct head × non-tiny corpus.

### v6 — staged fresh-goal ladder (current)

The design that remains after the falsifications: plain PPO, fresh starts only,
goals staged forward. Stage 1: dense height reward with a 1k terminal
(20M frames) — confined to the opening, where measurement says the shelter
economy does not yet exist (cover availability 23–38% in the opening vs 55–76%
by 3–5k), so the known height-reward pathology has nothing to corrupt and the
short horizon aligns mean with success. Stages 2+: success-only target reward
at 2.5k → 5k → 10k, each initialized from the previous checkpoint in a fresh
directory, promoted only through deterministic success-rate gates
(promote ≈ ≥60% over 256 held-out seeds). Reward mode and goal are part of the
immutable checkpoint contract; evaluation runs at the trained goal.

Registered expectations for stage 1: deterministic success_1k in 30–80%;
below ~10% means stop and diagnose the stack, not ladder onward. Pre-registered
escalators: (i) if the 2.5k→5k stage (the shelter phase transition) stalls, or
Focus stays dead while crush deaths dominate — reintroduce the counterfactual
teacher aimed correctly: death-rewind/crisis states only, option-level
commitments (30–120 frames), closed-loop continuations, as a small auxiliary
distillation; (ii) when late-stage prefix replay dilutes sample efficiency,
add minority snapshot starts drawn from the policy's own successful
trajectories (clean provenance). At stage transitions, warm up the value head
before policy updates (reward/goal changes invalidate V). Append each stage's
gate result to this ledger.

### v6 stage ledger

**Stage 1 — dense height→1k bootstrap** (`dodgeblock-ppo-opening-1k-a`, 20M
frames, ~4.25k sps, finished 2026-07-24 ~01:50). Registered prediction: det
success_1k 30–80%, <10% = stop and diagnose. **Result: det success_1k = 0.0 —
prediction missed.** Det eval (256 seeds): median 120, IQM 102, mean 149,
p90 400, max 680, mean length 1674 frames. Training-time stochastic fresh:
median 0→200 by 13M frames then flat to 20M; p90 440; EV 0.85. Still the
project's first sustained fresh-start learning (prior fresh baseline ≈ 40).
Diagnosis from telemetry alone:

1. Deaths concentrate in surge/release phases; the first surges are the wall.
2. `shelter_occupancy_by_phase[surge]` FELL 0.36 → ~0.14 over training — the
   height objective actively trained *out* the accidental sheltering the random
   policy had. It taught reactive dodging and climbing, pointed away from the
   shelter meta. Consistent with the v1-era pathology, now measured directly.
3. Deterministic argmax lags the stochastic policy (median 120 vs 200), and the
   eval's odd 13% `focus-left` is an argmax artifact: `focus_available_fraction`
   ≈ 1–5%, so focus-left is a no-op alias of left almost always. When Focus IS
   available, press rate ≈ 0.5 (coin-flip — entropy, not strategy).

Lesson: the bootstrap's value is a competent dodging/climbing actor and a
trained trunk — not height itself. Gates on absolute success at 1k were
miscalibrated for a 20M-frame opening run.

**Stage 2 — target-500 success rung** (`dodgeblock-ppo-stage-500-a`, launched
01:52 from stage-1 actor + hand-reset critic). Critic-reset recovery cost only
~0.5M frames; by 1.4M frames stochastic median 240 exceeded stage-1's endpoint;
at 4.8M: target_success 5–7%, EV 0.76, shelter-in-surge 0.18–0.25 (up from
0.13–0.15 late in stage 1 — first hint that success pressure pushes shelter
back; this metric is the live mechanism test). Max height ≈ 600 is expected:
success terminates the episode at the target.

**Ladder automation** (`rl/ladder_driver.py`, armed 02:11 as user unit
`dodgeblock-ladder-driver`). Rungs 500 → 1000 → 1750 → 2500 → 4000 → 6500 →
10000; 20M frames/rung below 4000, 30M at/above; deterministic 512-episode
eval after each rung. Registered gates: promote at det ≥ 0.50; det in
[0.10, 0.50) → extend once (+base frames, promote at ≥ 0.35 after); det < 0.10
but stochastic (384 ep) ≥ 0.30 → extend (deterministic-degeneracy rescue);
otherwise refine to the geometric-midpoint rung (rounded to 50) initialized
from the failed actor; stop with NEEDS-ATTENTION when the midpoint lands within
1.15× of the last passed rung. Crash without an eval event → relaunch same rung
(auto-resume), max 2 retries. Caps: 30 launches (raised from 14 on day 1 — refines consume more launches than budgeted), 26h wall. Final rung passing →
1024-episode confirmation eval → LADDER-COMPLETE. State and morning-readable
ledger: desktop `~/dodgeblock-ladder/{ladder-state.json,RESULTS.md,driver.log}`.

Registered deviation: rungs after stage 2 carry actor AND critic (no reset,
no warmup) — target→target transitions keep the value family (P(reach T) →
P(reach T′)), both rewards live in [−1, 1], and stage 2 measured the transition
cost at ~0.5M frames, cheaper than reset-plus-warmup machinery. The
value-warmup note above stands only for reward-family *changes*.

Registered predictions (written 02:15, before outcomes): 500 promotes,
possibly via one extend; 1000 likely promotes; the first wall is expected at
1750–2500 where shelter becomes mandatory. Escalator (i) fires iff a rung
≤ 2500 hits refine with shelter-in-surge still ≤ 0.2.

### Re-evaluation, 2026-07-24 evening — the speed reframe

A ladder-wide audit produced three findings that change the theory of what the
policy must learn, without changing the ladder skeleton.

**1. Difficulty is a function of TIME, so climbing fast IS a survival strategy.**
`naturalSpawnRate = 1 + 4.2(1−e^(−s/75)) + 0.12√min` saturates near 5.2–5.6
after ~4 minutes, so per-frame hazard is roughly pace-independent, and total
exposure ≈ hazard × time-to-target. Halving climb time halves log-failure.
Measured support:

| run | climb rate | det success | shelter-in-surge |
| --- | --- | --- | --- |
| stage-1 (height reward) | 8.0 h/s | — | 0.15 |
| rung-500-x3 | 8.3 h/s | 0.111 | 0.25 |
| rung-500-x7 | 10.2 h/s | 0.224 | 0.20 |
| rung-500-n9 | 12.0 h/s | 0.330 | 0.14 |
| rung-500-x10 | 12.1 h/s | 0.389 | 0.19 |
| go-explore demos (all 16 reached 10k) | **24.3 h/s** | — | ~0.20 cover |

Competence rises monotonically with climb rate; shelter occupancy is flat and
uncorrelated. The 10k existence proofs we own are **speedruns at ~24 h/s with
~20% overhead cover**, not shelter play. The shelter meta is a human strategy
under human reaction limits; nothing measured says it is the optimum for a
frame-accurate agent. Consequence: **escalator (i)'s trigger is re-registered
from "shelter-in-surge ≤ 0.2" to "per-layer survival stalls across two
consecutive rung attempts"** — absence of sheltering is not evidence of a
problem. Climb rate is now a tracked per-rung metric.

**2. Per-layer survival is the invariant; rung success is a proxy.** With
40 px layers, passing a rung at 50% demands: 500 → 0.927, 1000 → 0.973,
2500 → 0.989, **10000 → 0.9972**. Measured at rung 500 across four attempts:
0.838 → 0.890 → 0.928 → 0.927 — improving, then flat. The project needs a
~25× reduction in per-layer hazard from here, and the fixed-height sequence is
the honest read on whether frames alone buy it. The driver now logs
`per_layer=` and `climb=` on every gate line. Registered prediction: at
per-layer 0.93, rung 1000 lands ≈ 0.93^25 ≈ 0.16 → extend, then likely refine
to ~700.

**3. Rungs were 85% wasted.** Within-rung success saturates by ~3M frames;
the remaining 17M bought +0.02–0.07 (n9 0.28@3M → 0.33@20M; 400-n8
0.45@3M → 0.50@20M; 300-x5 0.48@3M → 0.50@20M). Base rung length cut
20M → 8M (12M at/above 4000); the extend path still buys more frames where the
tail actually matters. Expected ~3× rungs per hour. Registered prediction: 8M
rungs land within ~0.05 of 20M rungs at the same target.

**Focus economy — corrected reading.** The policy is not hoarding charges: it
*burns them on arrival*. `focus_available_fraction` ≈ 0.010–0.014 with
`press_given_available` 0.47–0.68, i.e. ~4 charges earned per episode and each
spent within a few frames, so deaths occur at zero charges (matching v4).
Focus is a ~3-lives-per-10-layers resource being spent at near-random times —
the largest identified untapped hazard reduction, and a natural target for
SIL amplification rather than shaping.

**Focus/dash audit (`rl/audit-trace.mjs`, rung-500 actor, 1520-height episode).**
Mechanism verified sound, usage verified worthless:

- 15 press-edges → 15 aims → 15 dashes, 0 wasted; Auto Guard 0 fires (hardcore).
- The focus logit is correctly masked when no charge is usable, and aim
  direction is re-steerable every aim frame (`updateFocusDirection`), with
  world-scale-correct credit (`gamma ** world_scale`), so neither perception
  nor credit plumbing explains the behaviour.
- Charges spent 15 of ~15.7 earned, **mean 7.5 frames held before spending**:
  dash *timing* is set by the recharge clock, not by state — the same
  state-independence pathology found in the v5 teacher, now self-inflicted.
- Dashes are aimed **upward 10/11** (intent learned) but produce nothing:
  mean height gain in the 90 frames after a dash **21.8 vs 30.4 baseline
  (lift −8.6)**, 8 of 11 gain exactly zero, 2 blocks shattered per 15 dashes.
  A dash rises ~50 px into empty air; height only counts *stable* height, so
  it lands back where it started.

Interpretation: the policy learned a ritual ("press when charged, aim up"),
not a tool. The user's two real uses — dash for traversal, and dash to break
towers into shelter — are unlearned, and the second is currently *unlearnable*:
cover forms at 3–5k, the policy dies at ~1.5k, so no training state exists
where breaking a tower pays. Fixing focus strategy now would repeat the v4
error of training the spending before the accumulation. Registered: re-audit
focus at rung ≥ 2500 (towers exist there); if dashes are still ritual, SIL is
the first lever, and only then will there be successful held-charge episodes
worth amplifying. `dash_effectiveness.lift_over_baseline` is the tracked
number.

**Rung 1000 failed as predicted, and the ladder self-corrected.** Det 0.021
(prediction: ≈0.16 — the direction was right, the magnitude optimistic),
stochastic 0.031, **per-layer 0.858 — below rung 500's 0.927**. Per-layer hazard
*rises* with height because difficulty tracks elapsed time, so the ladder aims
at a moving bar. Symptom of the v3 signal desert recurring: at 2% success the
gradient vanishes and the policy drifts — median height *regressed* 440 → 360
and climb rate 12.1 → 8.9 h/s versus the rung-500 actor it started from.
The driver refined to 700 unattended. **Registered read: scale alone is not
sufficient** — more frames at a rung stop helping (3M saturation) well before
the rung is solved, and the per-layer bar rises faster than frames close it.

**Adaptive rung sizing (replaces the tabulated ladder).** A fixed table cannot
know the step a policy can take; per-layer survival can. After each pass the
driver now solves for the height whose *predicted* success is AIM_SUCCESS=0.35
given measured per-layer survival, bounded to [1.08x, 1.5x]. From the measured
rung-500 pass the derived schedule is 500 → 550 → 800 → 1200 → 1800 → 2700 →
4050 → 6100 → 9150 → 10000: conservative while survival is weak, accelerating
as it compounds — nine rungs instead of a table that burned one on a x2 jump.
A refine now replaces the queue rather than restoring the failed target.

**Hardware headroom audit (2026-07-24 evening).** Half-idle VRAM is not idle
compute: the 3090 sits at 95% utilization and the update is compute-bound
(`collect_fraction` ≈ 0.15, i.e. 85% of wall-clock is the PPO update).
Micro-benchmark at the live minibatch (4096), contending with the live run:

| variant | ms/step | samples/s | peak VRAM |
| --- | --- | --- | --- |
| eager (current) | 556.8 | 7.4k | 11.1 GiB |
| channels_last | 539.2 | 7.6k | 11.1 GiB |
| torch.compile | 357.0 | 11.5k | 4.7 GiB |

`torch.compile` looks like a free 1.56× at half the memory — **and it is
unusable**: a 1.5M-frame end-to-end smoke run with `--compile` produced
`policy_loss/value_loss/kl/entropy = NaN` from the first update. Isolation:
the focus-mask path is NOT the cause (forward is NaN-free compiled and eager,
masked and unmasked), advantage normalisation is guarded (`+1e-8`), and the
eager rung-1000 run with the *same* initialisation, target, and zero-completed-
episode first window logs `policy_loss −0.00259`. Compile alone flips it.
**Root cause found and fixed (2026-07-24 night).** Bisection: every submodule
compiles cleanly on its own, so the fault only appears in the whole-graph
compile; NaN is confined to the six `forecasts.token` parameters while the
identically-typed `falling.token` is clean; empty token sets are NOT the
trigger. The discriminator was precision — compiled+bf16 breaks, compiled+fp32
is clean, and compiled+bf16 with *only the forecast encoder* in fp32 is clean.
Diagnosis: eager `LayerNorm` upcasts its reduction to fp32 internally; the
fused bf16 kernel Inductor generates for this small token dimension does not,
and the resulting instability surfaces as NaN gradients in the backward.
Fix: `TokenEncoder.forward` runs the set encoder in fp32 (`r2d2.py`), which
costs ~0.1% of model FLOPs and makes compiled and eager numerics agree.
Measured after the fix, batch 4096: eager 752 ms/step vs compiled 387 ms/step —
**1.94x** — clean over 20 optimiser steps, peak VRAM 11.3 -> 4.8 GiB.
Two false leads are recorded because they cost time: a `-inf` max-pool in the
same module was a genuine latent backward hazard (now a finite `-1e4` sentinel)
but not this bug, and an early NaN probe sampled *masked* actions, making
`log_prob` hit the `-1e9` fill and producing meaningless losses in both arms —
synthetic probes must sample from the policy's own distribution.
Per-submodule compilation was also measured and rejected: correct but *slower*
than eager (4.9k vs 7.7k samples/s), because graph breaks cost more than the
remaining fusion saves. **Enabled on the ladder 2026-07-24 22:49** after an end-to-end smoke run
(1.5M frames, compiled) came back with finite losses, EV 0.49 -> 0.82, median
0 -> 400, and a checkpoint the gate evaluator loads normally (state_dict keys
survive because `nn.Module.compile()` is used, not the wrapping form).
**Measured on the live ladder: 6190 sps compiled vs 4215 sps eager — 1.47x
end-to-end.** `collect_fraction` stays ~0.14, so collection is not yet the
bottleneck and worker count was left at 8; halving it would buy only ~1.08x.

Larger minibatches were not pursued:
the GPU is already saturated at 4096, and bigger batches buy fewer optimiser
steps per frame. The real throughput win was scheduling, not hardware — cutting
rungs from 20M to 8M frames is ~3×, which dwarfs any of the above.

**Dead-code cleanup, 2026-07-24 (audited before deleting).** Removed the v1
trainer and its whole v1 observation stack (`train.py`, `env.mjs`,
`env-server.mjs`), the R2D2 trainer (`train_v2.py`), the v5 behaviour-cloning
trainer (`train_bc_v5.py`), five launcher scripts, four unit files pointing at
a deploy path that no longer exists, and a dead helper duplicated in two
modules — 1,566 lines. Everything removed was verified to have no importer on
the live path first.

Deliberately kept: `go-explore.mjs` and its runner (they produced the only 10k
existence proofs the project owns), `rescue-oracle.mjs` (reachable from the
live evaluator's `--death-case-dir`), `counterfactual-teacher.mjs` (registered
escalator (i)), and the trace tooling (`record_trace.py`, `render_trace.mjs`,
`audit-trace.mjs`).

Deferred deliberately, because it is risky while a ladder is in flight: the
`ppo_v2.py` de-branching (sticky head, demo/imitation path, trajectory bank,
held-out cell eval), collapsing `cell_bank.py` — which is *live-degenerate*,
not dead, since `CellBankCoordinator` is constructed unconditionally and
`FRESH_CELL_ID` classifies every episode — and moving `TokenEncoder`,
`ResidualBlock`, `OBSERVATION_KEYS` and `interquartile_mean` out of `r2d2.py`
(~480 of its 565 lines are dead, but those four symbols are on the live path,
including the gate evaluator's import chain). The hazard is `training_contract()`:
it is compared by exact dict equality on `--resume`, which is precisely the
driver's crash-retry path, and it still contains four inert `cell_*` keys.
Removing them invalidates every checkpoint on disk for resume, and
`TRAINING_CONTRACT_VERSION` is not a migration mechanism — it is just another
compared key. Do this at a ladder boundary, keeping the contract byte-identical.

**Registered next experiment (one flag, no code):** `--gamma` per world frame
is already plumbed; γ = 1 gives the objective *zero* time preference even
though exposure time is now the measured hazard driver. A/B a discount with
~60–120 s half-life against γ = 1 at a fixed rung. This is a discount, not a
shaped reward, so it stays inside the no-shaping constraint.

### Method map — upgrades not yet used (reviewed 2026-07-24)

Ranked against measured constraints. None are churn-now; each has a trigger.

- **Strict upgrade, apply at a stage boundary:** value head as classifier
  (two-hot / HL-Gauss CE — Farebrother et al. 2024). Our values ARE success
  probabilities and we live near 0 and 1, where MSE gradients are worst.
- **Likely upgrade #1 — Self-Imitation Learning (Oh et al. 2018), escalator
  (iii), FIRST response at the wall:** advantage-weighted CE replay of the
  policy's own successful episodes. At 5–15% rung success PPO uses each rare
  success once; SIL amplifies them — exactly the mechanism needed to grow the
  first shelter-won successes. Own-data only (constraint-clean), ~50 lines.
- **Likely upgrade #2 — auxiliary predictive heads (UNREAL-style):**
  self-supervised crush-within-k / cover-within-k heads shape the trunk toward
  danger-vs-cover geometry without touching reward. Trigger: shelter transition
  stalls and SIL alone doesn't break it.
- **Architectural option — goal-conditioned policy + hindsight relabeling
  (UVFA/HER):** the continuous ladder; every death at h is success for g < h.
  The rebuild-from-scratch design. Trigger: rung-hopping gets brittle above
  ~2.5k. Not a bolt-on (on-policy mismatch; success-terminal episodes).
- **Already registered:** critical-state search distillation = escalator (i).
- Escalation order at the wall: SIL → aux heads → search labels (cheapest
  first).

Considered and rejected on our own measurements: curiosity/novelty bonuses
(failure is strategic valuation, not coverage — shelter states are visited at
0.14–0.36 occupancy but not preferred; procedural terrain saturates novelty);
potential shaping on cover (injects the strategy we want emergent; v3 lesson:
shaping = value init); learned HRL (v5b interface lessons; search-at-crises
gets the benefit); MuZero-class per-decision search (restore 2.16 ms vs 60 Hz —
arithmetic-dead; the sim is already the exact model); off-policy swaps/R2D2
(sim fast, learner-bound, PPO stable); PBT (one GPU); difficulty randomization
(the benchmark is fixed Hardcore). Observability audited: phase one-hot,
pressure, block rate, elapsed time, Focus economy, bag remainders all in the
state vector — shelter timing is fully observable; no obs gap.

**Rung 500 result + driver incident (2026-07-24 morning).** Det eval at 20M:
target_success 0.111, median 200, p90 520, max 640 — extend band, and the
det-vs-stochastic median gap closed (200 ≈ parity; stage-1's argmax degeneracy
faded under the success objective). The driver correctly chose EXTEND at 03:13
but implemented it as resume-with-larger-total — `total_frames` is part of the
immutable training contract, so all three relaunch attempts died in <60s on the
contract check and the driver stopped with NEEDS-ATTENTION. Cost: ~6.6h idle
GPU (03:17–09:53). Fix: extension is now a fresh base-frames run in a new
directory initialized from the rung's actor (schedule re-anneals — consistent
with per-rung re-annealing everywhere else); the contract stays untouched.
The check caught a real contract violation — the bug was mine, the guard
earned its keep. Extension relaunched 09:56 as `rung-500-x3`; post-extend
gate: promote at det ≥ 0.35, else refine to ~300.

### Escalator (i) — on-policy critical-state rescue distillation (registered 2026-07-24 23:30, before any result)

**Trigger fired.** Per-layer survival stalled across consecutive attempts at two
targets: 500 went 0.838 -> 0.890 -> 0.928 -> 0.927 (hazard -34%, -36%, then 0%),
and 700 went 0.892 -> 0.914 over a full extension. Requirement rises faster than
achievement: 0.946 needed at 500, 0.961 at 700, 0.973 at 1000, 0.9972 at 10k
against a measured asymptote near 0.92.

**Why sparse reward cannot close this.** At 10k one layer of hazard is worth
dP ~ 0.003 while the return is Bernoulli with sigma ~ 0.5 — an SNR of 0.006 per
sample, so resolving a single action distinction needs order 250,000 samples.
The simulator answers the same question at the same state in ~0.5 s of CPU.
That ratio is the entire argument for injecting search labels, and it explains
why the policy learns short-credit-path skills (dodging, which kills in under a
second) but not long-credit-path ones (banking a Focus charge for a crisis,
positional discipline).

**Design.** Between rungs: roll out the CURRENT policy, capture deaths with
rewind snapshots, run the rescue search at those states only, and distil the
escapes as an auxiliary cross-entropy loss alongside PPO (the existing
`--demo-dataset` path, whose shard format already carries a soft target over
all 18 actions). Every difference from the three v5b failures is deliberate and
measured: on-policy states instead of stale corrections (which were 85%
repeat-biased), critical states instead of fixed cadence (whose labels were
~ln 3, uniform), the direct head instead of the sticky head (the interface
defect), auxiliary loss on top of PPO instead of standalone BC, and a large
refreshed corpus instead of 1,536 states.

**Registered A/B.** Control = rung 600 from the rung-700-x13 actor, no
distillation (running from 23:14). Treatment = same target, same
initialisation, same frame budget, plus the rescue corpus. One variable.

**Registered predictions, written before the corpus exists:**
1. Rescue search finds an escape at t-30 for >= 60% of sampled deaths (the v4
   oracle measured ~1.0 on its sample; a lower on-policy rate is expected).
2. Treatment beats control on det success at rung 600 by >= 0.08 absolute,
   equivalently per-layer +0.02 (hazard -20%). Below that = falsified.
3. A majority of found escapes use Focus, given the policy currently spends
   charges within 7.5 frames of earning them and dies at zero.
4. Risk to watch: corpus overfitting (v5b overfit 1,536 states) and the
   auxiliary loss fighting PPO. Mitigations: >= 10k labelled states, small
   demo coefficient, per-rung refresh.

**Corpus built 2026-07-24 23:51 (results against the registered predictions).**
1,024 on-policy deaths captured from the rung-700-x13 actor (stochastic, free-run
to death), searched at rewinds 30/60/120 with 128 trials x 3 futures — 3,072
evaluations in 85 s on 12 cores (0.03 core-seconds per evaluation).

- **Prediction 1 VALIDATED, and beaten.** Original-future rescue rate
  0.828 / 0.838 / 0.864 at rewind 30 / 60 / 120; all-futures-robust
  0.751 / 0.754 / 0.780, against a registered threshold of >= 0.60. Rescue rate
  *rises* with rewind depth, so intervention up to 2 s before death is
  learnable, not only last-instant escape.
- Corpus: **2,340 verified escapes -> 140,400 label rows**, 0 rejected on
  re-verification (v5b overfit 1,536 states; this is ~90x larger).
- **Prediction 3 FALSIFIED, informatively.** Only **0.4%** of escapes use Focus
  (0.02% of label frames); the mix is up-left 0.22, up-right 0.20, neutral 0.10,
  left 0.09. Cause: the search only offers focus options when a charge exists,
  and the policy *dies at zero charges* (it spends each within ~7.5 frames of
  earning it). **Rescue distillation therefore cannot teach the Focus economy
  at all** — the error is ~10 s upstream of the crisis it labels. Focus needs a
  separate mechanism; escalator (i) will teach positional escape only.
- Process note: the treatment's early fresh-median looked like the v5b collapse
  (40 vs an initialised ~440) but the control shows the same shape (0, 20, 120).
  It is a censoring artifact — at low frame counts only short episodes have
  completed, biasing the median down in both arms. Do not read fresh medians
  before ~1M frames.

Two defects the corpus exposed and fixed: the exporter aborted the whole run if
one record failed re-verification (now skips and truncates, reporting the
count), and the trainer sampled rescue shards with `decision_weighted=True`,
whose opening/initial/switch pools are trajectory-order concepts meaningless
for concatenated escape prefixes — ~30% of the auxiliary gradient was landing on
whichever escapes were exported first.

**Control measured:** rung 600 from the rung-700-x13 actor, 8M frames, no
distillation -> det 0.344, per-layer 0.9313, climb 12.6 h/s. Treatment
(identical init, target, and budget; demo_coef 0.3 -> 0.05) launched 23:57.

**What would falsify the whole escalator:** distillation lifts held-out
crisis accuracy but not det success — meaning the policy can be taught to
escape crises it is already in, while still walking into them. That outcome
sends the work to positional/avoidance signal instead of rescue.

### Escalator (i) first result, and what the demonstrations actually do (2026-07-25)

**Prediction 2 FALSIFIED at demo_coef 0.3 -> 0.05.** Against the matched control
(rung 600 from the rung-700-x13 actor, 8M frames, det 0.344), the treatment
collapsed: at 1.7M frames control succ 0.234 / median 400 vs treatment 0.002 /
120; at 2.5M control 0.246 / 400 vs treatment 0.000 / 80. Imitation accuracy
climbed to 0.83 the whole time — the policy fit the crisis labels while its
ordinary play fell apart. This is the v5b failure reproduced: an auxiliary loss
of comparable gradient scale on an exclusively-crisis state distribution drags
global behaviour toward escape actions. Run stopped at 2.4M. A retry an order
of magnitude weaker is the obvious next setting, but the coefficient is now
known to be the fragile axis.
*Process note: an earlier read of this run as "not a collapse" was wrong — it
compared against rung-600-x15, which has a different initialisation. Only the
matched control is admissible.*

**Demonstration audit — the contrast that matters.** Same tooling
(`audit-trace.mjs`) run on the seed-7 go-explore demonstration (10,000 height,
22,589 frames, replay verified by finalHash):

| metric | our rung-600 policy | 10k demonstration |
| --- | --- | --- |
| climb rate | 12.6 h/s | **26.6 h/s** |
| charges spent | 15 of ~15.7 | 85 of ~86.3 |
| **frames a charge is held before spending** | **7.5** | **69.3** |
| fraction of frames holding a charge | 0.023 | **0.261** |
| **blocks shattered by dash** | **2** | **31** |
| action mix | down-family 0.43 | up-family **0.72** |

Three corrections follow:
1. **The `dash_effectiveness.lift_over_baseline` metric does not discriminate.**
   The demonstration scores -6.9 and reached 10k; our policy scores -8.6. Height
   gained after a dash is the wrong measure because dashes are spent on
   shattering and defence, not climbing. Earlier text calling the policy's
   dashes "worthless" over-read this number. The discriminating statistics are
   charge-holding time (9x), charge availability (11x) and shatters (15x).
2. **The user's stated strategy is what the successful demonstrations do.**
   31 shatters means dash-into-blocks is a core part of a 10k run — breaking
   structure, exactly the "break towers to make shelter" play described. Our
   policy does essentially none of it.
3. **The v4 fast-fall pathology persists in milder form**: the policy spends 43%
   of frames on down-family actions while the demonstration spends 72% on
   up-family ones.

This also explains why rescue distillation cannot reach the Focus problem: it
labels states that already have zero charges. Charge banking is a *policy-wide*
habit, not a crisis decision, and no crisis-state label can teach it.

### v7 proposal — potential-based occupancy shaping (designed 2026-07-25)

**The measurement that decides the design.** Replaying the seed-7 10k
demonstration through the sim's own shelter test (a fixed, non-faulting block
overhead overlapping the player by >= 6 px):

| | our rung-600 policy | 10k demonstration |
| --- | --- | --- |
| sheltered, all frames | — | 0.399 |
| **sheltered during surge** | **0.13-0.22** | **0.427** |
| sheltered during release | — | 0.471 |
| sheltered during calm | — | 0.340 |
| fraction of frames holding a charge | 0.023 | 0.261 |
| sheltered while holding a charge | — | 0.423 |
| blocks shattered by dash | 2 | 31 |
| up-family action share | 0.43 down-family | 0.72 |

The demonstration shelters most during surge and release and least during calm —
"hold cover through surges, climb in lulls", measured in machine-generated data.
This supersedes the v4-era note that the demo ran at ~20% cover, which was
measured on the final band only and is not representative.

**Why this changes the design.** Every gap above is a property of *state
occupancy*, not of the action taken in a state. That is exactly why BC on these
demonstrations failed (v5a: their actions are search residue, state-independent
by construction) and why crisis-state rescue labels cannot help (they label
states that already have zero charges). Occupancy is the distillable content of
a search trajectory; actions are not.

**Design.** Keep the success-target ladder, adaptive rung sizing and compile
exactly as they are. Add potential-based shaping F = Phi(s') - Phi(s) with

    Phi(s) = a * cover(s) * risk(phase) + b * charges(s)

- `cover(s)` is the existing shelter test; `risk(phase)` is taken from the
  demonstration's own phase profile (surge/release high, calm low), not from
  intuition.
- `b * charges(s)` makes *spending* a charge cost something. The policy
  currently spends for free within 7.5 frames of earning; the demonstration
  holds ~69 frames and converts charges into 31 shatters.
- Phi excludes absolute height and elapsed time. Rewarding height is the v1-era
  pathology (mean objective prefers the wrong policy) and would recreate it.

**Why this is not the shaping we rejected.** Potential-based shaping is
policy-invariant (Ng, Harada & Russell 1999): with gamma = 1 it telescopes to
Phi(end) - Phi(start), so the optimal policy is provably unchanged and the
success objective still decides. It does not add information — it redistributes
credit, which is precisely the measured bottleneck (dP ~ 0.003 per layer against
sigma ~ 0.5 requires ~250k samples per action distinction; shaping shortens the
path from "take cover now" to "survive the surge three seconds from now").
The v3 lesson stands and is not contradicted: shaping cannot create signal where
P(success) ~ 0. We are no longer there — rung 600 sits at 0.34.

**Registered falsification (write results here):**
1. If shelter-in-surge rises toward ~0.4 **and** per-layer survival breaks its
   0.92 asymptote -> the shelter hypothesis and the design are both supported.
2. If shelter-in-surge rises but per-layer survival does not -> **the shelter
   meta itself is not what makes 10k reachable**, and the demonstration's cover
   is incidental. That would be the single most valuable negative result
   available to this project.
3. If shelter-in-surge does not move, the potential is too weak or the policy
   cannot reach cover from where it plays; escalate to a learned state-only
   discriminator over local features (POfD / GAIfO, Kang et al. 2018;
   Torabi et al. 2018) instead of a hand-specified Phi.

**Retained as secondary:** the rescue corpus (2,340 verified escapes, 85 s to
rebuild) retried at a coefficient an order of magnitude below the 0.3 that
collapsed the policy. It addresses recovery, which is a real but smaller part
of the gap.

### v8 — search as teacher: the architecture requirement, measured (2026-07-25)

Goal restated by the user: not a TAS and not mechanical perfection, but
*emergent strategic decision making* — a Move-37-class choice that gives up
something obvious for a non-obvious payoff, learned rather than engineered.
A TAS is explicitly not wanted, and would be useless anyway: it conditions on
future spawns the live policy cannot see, so its decisions are unlearnable
(the "learning by cheating" failure mode).

`rl/beam-search.mjs`: option-level beam search over the exact simulator,
candidates scored across reseeded futures. Decisions at option granularity
because held-out direction loss at 2-frame commitment measured ~ln(3).

**Result — naive search is a worse teacher than the policy it would teach.**

| searcher / policy | climb rate | note |
| --- | --- | --- |
| beam, greedy height, horizon 15, real future | 9.6 h/s | reached 480 in 3,000 frames |
| beam, greedy height, horizon 45, 2 futures | 5.6 h/s | beam extinguished at 400 |
| trained PPO policy (rung 600) | 12.6 h/s | median 480 |
| go-explore demonstration | 26.6 h/s | search residue, wanders |

Two findings, both structural:

1. **A greedy searcher is myopic and loses to the policy.** Scoring by height
   gained over an option window climbs into danger; the policy's success-trained
   critic (EV 0.94) encodes risk the searcher has no access to. This is the
   AlphaGo lesson in miniature: MCTS without a value network is weak. **Search
   is only a useful teacher when guided by the learned value function** — which
   is exactly the ingredient this project has never had. Go-explore proposed
   options from constant weights; PPO had no search; the ExIt loop was never
   real on either side.
2. **Open-loop plans cannot be future-robust here.** Requiring survival across
   reseeded spawn patterns extinguishes the beam: no fixed 45-frame action
   sequence survives arbitrary futures. A fair plan must be a *closed-loop
   policy*, not a sequence — so the search must evaluate candidates by handing
   the continuation to the policy plus its value estimate, not by rolling a
   committed sequence.

**Consequent design (the real ExIt loop).** Move the search into Python so the
network is inside the loop: restore a root snapshot into many envs, expand
option candidates in lockstep through the existing batched bridge, evaluate the
leaves with V(s) rather than rolling to death, keep the top-B, replan at each
option boundary. Cost from measured primitives (~28k env-steps/s aggregate,
value call ~5 ms): a 512-branch expansion of a 15-frame option is ~0.3 s, so a
15,000-frame episode is ~5 minutes of search. Requires one bounded change to
`env-server-v2.mjs` / `v2_bridge.py`: restore an arbitrary snapshot into env i
(today only pre-supplied bank cells can be restored).

**Why this is the design that could produce strategy rather than mechanics.**
A decision is strategic exactly when its payoff lies beyond the horizon you can
simulate cheaply — give up climbing now, sit under cover, survive the surge.
Greedy search cannot see it; a value-guided search can, because V already
scores "how likely am I to reach the target from here". The shelter meta is a
Move-37-class behaviour by this definition, we have measured that the
demonstrations do it (0.427 sheltered during surge) and the policy does not
(0.13-0.22), and value-guided search is the only mechanism proposed so far that
could find it *and* hand it back to the policy.

**Registered before building:** if value-guided search does not beat the policy
from matched states, ExIt has no teacher and this whole direction is dead —
that is the first measurement to take, before any distillation is wired.

### CORRECTION to the v8 search result (2026-07-25, later)

The first v8 table was measured with a fatal bug in my own harness: **`dones`
fires on SUCCESS as well as death** — reaching the target ends the episode — and
`died |= dones` therefore marked *winning* branches dead and pruned them.
`search_success` was structurally incapable of being non-zero; the 0.00 column
was guaranteed by construction, not measured. Fixed (success is now a terminal
win with value 1, and `died = dones & ~successes`).

Re-measured after the fix, target 600, beam 1, 6-frame options, same harness
whose policy control reproduces the known result:

| controller | median height | success |
| --- | --- | --- |
| **the policy itself** | **560** | **0.44** |
| value-guided search | 60 | 0.00 |
| random option selection | 20 | 0.00 |
| height-greedy search | 0 | 0.00 |

**The corrected conclusion is narrower and more interesting than the one it
replaces.** The critic is NOT useless: V-guided beats random 3x and beats
height-greedy outright, so it carries real ranking signal. But every search
variant is ~10x worse than the policy. The right statement is therefore:
*short-horizon greedy improvement over this critic is not a policy improvement* —
not "search cannot work here".

The theory says exactly this should happen. Greedy improvement is bounded by
2*gamma*epsilon/(1-gamma) in value error epsilon; **we run gamma = 1, where that
bound is infinite.** Undiscounted, 3,000-step horizon, per-action value
differences far below the critic's error is the textbook pathological case. The
policy's action distribution encodes a whole multi-step behaviour learned over
hundreds of millions of frames; one-step argmax over a noisy V discards it.

Untested and still open: deeper search (beam >> 1 with long horizons), a
calibrated/ensembled critic, or pessimistic (lower-confidence-bound) selection
instead of argmax. Search is not established as dead — one-step greedy over
*this* critic is.

### v8 result — value-guided search is worse than the policy (2026-07-25)

Built `rl/value_search.py` (Python-side beam search; leaves scored by the
trained critic) on a new simulator primitive: action 253/252 SAVE/RESTORE
against a per-worker slot table, so snapshots never cross the wire. The
primitive is verified bitwise (restore -> replay 200 actions -> every packet
field identical, frame counter unchanged). Two non-obvious requirements found by
ablation: restore must also carry `previousAction` (it is part of the
observation) and `episode` (or replays diverge after an in-branch death).

**Measured at target 600, against the same checkpoint whose policy scores
det 0.344:**

| searcher | success | median height |
| --- | --- | --- |
| greedy on height (JS beam, 15-frame options) | — | 400 @ 9.6 h/s |
| value-guided, 15-frame options | 0.00 | 140 |
| value-guided, 6-frame options | 0.00 | 200 |
| value-guided, 3-frame options | 0.00 | 80 |
| **value-guided, 1-frame (pure greedy on Q)** | **0.00** | **40** |
| **the policy itself, same harness (control)** | **0.44** | **560** |

**Process note, and the reason the control existed.** The first control run
returned 0.0 for the policy too, which would have been reported as "search is
dead" on a broken harness. The fault was mine: the env resets in the same step
it dies, so `current_heights` already holds the NEXT episode's zero — the height
at death lives in `heights`. After the fix the control reproduces the known
policy performance, and only then are the search numbers admissible.

**What this means.** Greedy w.r.t. a one-step Q is exactly the case the policy
improvement theorem covers: it cannot be worse than the policy unless the Q
estimate is wrong. It is the *worst* row in the table. So the critic, despite
EV 0.94, cannot rank neighbouring states. EV measures fit to returns on the
policy's own distribution; it says nothing about action-level resolution. With
18 near-identical successors, argmax selects the critic's ERROR, and compounding
that for thousands of frames walks a trajectory adversarial to the value
function. This is the same dP ~ 0.003 against sigma ~ 0.5 wall seen from the
planning side, and it is consistent with the earlier ln(3) result: **search
cannot exploit a flat, noisy Q — it amplifies value error.**

It also explains Go-Explore: its power is *retries*, not per-decision quality,
which is exactly the v5a finding that competence lives in retry-selection. A
live policy cannot retry.

**Consequence for the design.** ExIt/AlphaZero-style search-as-teacher is dead
here unless the critic's action-level resolution improves by orders of
magnitude. That elevates external dense signal from optional to necessary:
when the critic cannot resolve action differences, shaping is the only source
of a usable gradient. Implemented accordingly in `ppo_v2.py`:
`--shaping-cover` and `--shaping-charge` add F = Phi(s') - Phi(s) with
Phi = a * cover * PHASE_COVER_WEIGHT[phase] + b * charges, where the phase
weights are the measured demonstration cover profile. Potential-based, so the
optimum is provably unchanged (Ng, Harada & Russell 1999); terminal potential is
0 so dying is never rewarded, and the carry-forward is correct across the
env's same-step auto-reset.

### Literature review — three findings that change the plan (2026-07-25)

1. **Marginal occupancy matching is behaviourally underdetermined.** Agents with
   identical state visitation can behave oppositely (Burnwal et al.,
   "Learning from Observation: A Survey of Recent Advances", arXiv:2509.19379).
   Matching cover-fraction 0.427 does not pin down behaviour — "shelter then
   climb" and "climb then cower" share a marginal. **The v7 potential
   `Phi = a*cover(s)*risk(phase)` is provably underdetermined and should be
   redefined over cover-ENTRY transitions, not the cover marginal.** The sweep
   now running uses the marginal form; treat its result as a lower bound.
2. **The escalator-(i) collapse was structural, not a coefficient problem.**
   DEMO3 (arXiv:2503.01837, ICML 2025) bounds the auxiliary term as
   `r + beta*tanh(delta)` with beta chosen so the bonus *cannot* push one
   stage's shaped reward into another's range — structurally incapable of
   overriding the sparse task reward. **Bound the bonus; do not tune its
   weight.** Directly transplantable: our 250 layers are DEMO3's stages.
3. **A demonstration-free replacement for the hand-designed potential.**
   SVM (arXiv:2606.23640, 2026) trains a discriminator by plain BCE between the
   agent's OWN successful and unsuccessful episodes and adds a clipped log-odds
   bonus. No demonstrations, so it sidesteps the survivorship and search-residue
   problems entirely; positives are realizable by construction; the occupancy
   structure is discovered rather than hand-encoded. Our 0.34 success rate
   supplies both classes in healthy proportion. Caveat: its guarantee assumes
   determinism, and our sim is deterministic only given a seed the policy cannot
   see.

Also struck on evidence: the DICE family (O-DICE, arXiv:2402.00348, shows
ValueDICE *underperforming plain BC* on single-trajectory imitation), and
discriminator gradient penalty as a default stabiliser (DecompGAIL,
arXiv:2510.06913 — GP enforces smoothness, but collision-like death conditions
need sharp decision boundaries).

**Revised priority: SVM with a DEMO3-style bounded bonus** — self-generated
success/failure discriminator, clipped log-odds, refreshed per rung, no
demonstrations and no hand-specified potential.

### v7 shaping — FALSIFIED, and the reason is structural (2026-07-25 03:31)

Sweep at target 600, 8M frames each, same init as the measured control
(det 0.344, per-layer 0.9313):

| arm | cover / charge | det success | per-layer | shelter-in-surge |
| --- | --- | --- | --- | --- |
| control | none | **0.344** | 0.9313 | ~0.14-0.19 |
| shape-a | 0.01 / 0.005 | 0.285 | 0.9198 | 0.178 |
| shape-b | 0.03 / 0.015 | 0.291 | 0.9210 | 0.200 |
| shape-c | 0.10 / 0.050 | 0.322 | 0.9273 | 0.191 |

Every arm is BELOW control, and — decisively — **the potential did not move the
behaviour it was designed to move.** In the strongest arm shelter-in-surge went
0.094 -> 0.190 -> 0.140 over training: no sustained shift, against a
demonstration value of 0.427. This is registered falsification case 3.

**Why it could never have worked, and this generalises.** With gamma = 1 the
shaping contribution to an episode's return telescopes to
Phi(terminal) - Phi(s_0), and terminal potential is 0 by construction, so the
total is **-Phi(s_0): a constant**. Constants are absorbed by the advantage
baseline. Potential-based shaping therefore *cannot* change the ranking of
trajectories by return — it can only redistribute credit within an episode via
bootstrapped TD/GAE, a second-order effect that an already-good critic
(EV 0.94) largely provides anyway, and which here was swamped by added variance.

**The property that made it principled is exactly what made it useless.**
Policy invariance (Ng, Harada & Russell 1999) guarantees the optimum is
unchanged; at gamma = 1 with zero terminal potential it also guarantees the
*return* is unchanged up to a constant. To actually shift behaviour the bonus
must be biased — it must change what is optimal. That is precisely what the
literature's current answers do: SVM's clipped log-odds (arXiv:2606.23640) and
DEMO3's bounded `beta*tanh` (arXiv:2503.01837), where the bonus is *bounded* so
it cannot override the task reward, rather than *potential-shaped* so it cannot
change anything.

Consequence: **do not reach for potential-based shaping in an undiscounted
success-only objective again.** The remaining live design is a bounded,
self-generated bonus: a discriminator trained by BCE on the agent's own
successful vs unsuccessful episodes, added as a clipped log-odds term with a
DEMO3-style bound, refreshed per rung. No demonstrations, no hand-specified
potential, and structurally incapable of the coefficient collapse that killed
the rescue distillation.

## Falsified hypotheses

1. Expected-height optimization converges to reliable completion (v1).
2. Success-only credit propagates backward along a demo corridor (v3).
3. …or through an any-state cell-bank curriculum at practical scale (v4).
4. Explorer trajectory actions carry distillable competence (v5a, ×9 + dagger1).
5. Repeat-biased correction labels work if weighted dominantly (v5b).
6. Uniformly-sampled fine-cadence counterfactuals are informative (v5b).
7. Retry-burst timing is learnable as a decision boundary (v5b).

## Measured facts about the game

- Difficulty is a function of time, near-plateau after ~4 min
  (`naturalSpawnRate` body τ = 75 s, tail 0.12·√min).
- 10k ≈ 250 layers ≈ 7 min at the demo pace (~0.61 layers/s); consistency
  requires ≈ 0.998/layer.
- Hazard horizon ≈ information horizon ≈ 1–2 s (18-frame telegraph plus fall
  time): shallow search is locally near-optimal; recurrence unnecessary once
  counters complete the information set.
- Q is flat at fine cadence almost everywhere; decisions are sparse in time.
- Cover availability: 23–38% in the opening → 55–76% by 3–5k (storm
  saturation). Shelter is an economy of accumulated capital; it can only be
  learned from fresh play.
- Phase cycle 26 s; camera allows ~35 s of camping: the environment itself
  forces the shelter/climb rhythm.
- Sim throughput (single core): ~6.3k steps/s full-demo replay, ~2.6k late-game;
  snapshot 0.22 ms, restore 2.16 ms.
- Deaths at measured (low) heights are overwhelmingly rescuable at 30-frame
  rewinds — failures are policy error, not unfairness. The at-altitude Hardcore
  ceiling remains unmeasured pending a competent fresh policy.
- The 2.2M-parameter network is not the bottleneck: it generalized across
  held-out cells immediately whenever real signal existed.

## Process rules that earned their keep

- Deterministic held-out gates before spend: PPO was never launched on a failed
  prior (three times).
- Metrics carry their own baselines (`lift_over_repeat`, marginal and
  repeat-previous): one epoch of the right metric replaced eight blind
  variants.
- Register predictions before results land; log them here.
- Verify before load-bearing: stale explorer statuses, 1/28 unreplayable
  dagger trace, 1/16 invalid cadence shard were each caught by verification.
- One unforced error for the record: BC-on-trajectories was revived against an
  earlier correct warning (the human-data analogy does not transfer to
  survivorship-filtered noise). Cost ≈ one day.

## Artifact index

Laptop `~/dodgeblock` — the git repo (push: `lr1729`; `origin` is the old
upstream lineage, read-only). Desktop `~/dodgeblock-v5-code` — rsync deploy
target (not a git checkout; pin to commits at the next stage boundary).

desktop-linux (RTX 3090 = CUDA 1, 290 W; the RTX PRO 6000 runs an unrelated
vLLM server — leave it alone):

- `~/dodgeblock-ppo-target-v3/` — v3 run + checkpoints (negative ablation).
- `~/dodgeblock-ppo-v4/` — v4 120M run, final eval.
- `~/dodgeblock-bc-v5*/` — nine BC variants; `model-selection/` holds per-epoch
  fresh evals; `sticky-a` also holds dagger harvests.
- `~/dodgeblock-demo-dataset-v5[/-dagger1]/` — exported demo datasets.
- `~/dodgeblock-counterfactual-v5-a/`, `…-cadence2/` — counterfactual corpora
  (mixed / cadence) with per-shard manifests.
- `~/dodgeblock-v5-diagnostics/` — dagger searches, correction rounds 1–6,
  per-death rescue reports (`results/`).
- `~/dodgeblock-ppo-opening-1k-a/` — v6 stage 1 (complete; det eval in run.log).
- `~/dodgeblock-ppo-stage-500-a/` — v6 stage 2, first ladder rung.
- `~/dodgeblock-ladder/` — ladder driver state, per-rung dirs
  (`rung-<target>-n<k>/`), `RESULTS.md` decision ledger, `driver.log`.

beelink-arch: `~/dodgeblock-go-explore-bank-v4/` (16 seeds: demos + stratified
search checkpoints), `~/dodgeblock-go-explore-bank-v5-rerun/`,
`~/dodgeblock-v5-diagnostics/`, code copies in `~/dodgeblock-v4-explorer/` and
`~/dodgeblock-v5-code/`.

## v9 — the death autopsy inverts the diagnosis (2026-07-25)

### SVM falsified, cleanly

| arm | budget | det success | per-layer | discriminator |
|---|---|---|---|---|
| control | — | **0.344** | 0.9313 | — |
| svm-half | 0.5 | 0.301 | 0.9230 | acc 0.757, sep 1.54 |
| svm-full | 1.5 | 0.281 | 0.9189 | acc 0.751, sep 1.70 |

The discriminator was **healthy** — accuracy 0.76, log-odds separation 1.5–1.7, far
above the 0.11 the smoke test warned about. So the pre-registered escape hatch
("weak features") is closed: the features separate fine and the bonus still hurt,
monotonically in budget. Dose-response with the wrong sign is the strongest form
this falsification could take.

Why, in hindsight: positives come from the *current* policy's own successes, so
the bonus rewards resembling what the policy already does. That is self-imitation,
which accelerates convergence to the existing mode. The measured problem is a
*plateau* — premature convergence is precisely the wrong medicine.

Four reward-side doors are now closed: scale, search-as-teacher, potential
shaping, self-generated visitation. Every one of them assumed the deficit was
strategic. None of them tested that assumption.

### The autopsy

Replay 200 real deaths, snapshot K frames before each, and search for any action
sequence surviving H=120 more frames. `restore` is exact, so every rollout faces
the identical falling blocks: this asks "was there an escape from THIS situation".
Search is 64 sticky-random rollouts — a *flailing* controller, not an oracle.

| K (frames before death) | 2 | 5 | 10 | 20 | 40 | 80 | 160 |
|---|---|---|---|---|---|---|---|
| fraction escapable | 0.015 | 0.195 | **0.725** | **0.880** | 0.920 | 0.940 | 0.990 |

Causes: squished 148, fell 52.

Read it in one direction only: a found escape *proves* viability; no escape found
does not prove doom. So these are lower bounds — the true escapability is higher.

**88% of deaths were escapable with 20 frames (1/3 s) of different action, and
random flailing finds the escape.** The agent is not walking into unavoidable
traps. It is not failing at strategy, positioning, or exploration. It is failing
to react inside a third of a second, in situations where noise would have lived.

This inverts the premise of every intervention above, and explains all four
failures at once: they taught strategy to an agent whose deficit is reflex.

### The mismatch this exposes

GAE runs at lambda = 0.995 per world frame -> effective credit horizon
1/(1-lambda) = **200 frames**. The measured causal window of a death is
**10-20 frames**. Credit for the fatal action is diluted across ~10-20x more
frames than were causally involved.

That is exactly the measured Q-flatness (held-out direction loss ~ ln 3): the
signal is not absent, it is smeared. At gamma = 1 with terminal-only reward,
V(s) = P(success from s), so delta_t = V(s_{t+1}) - V(s_t) is *precisely* the
local change in survival probability — the sharp, correctly-attributed credit
signal. Large lambda throws that away in favour of a high-variance Monte Carlo
return; small lambda recovers it.

lambda = 0.95 -> horizon 20 frames. lambda = 0.90 -> horizon 10 frames. Both
match the measured window. The usual objection to small lambda — early in
training V is garbage, so bootstrapping on it is biased — does not apply here:
every rung initialises from a trained critic.

One knob, directly implied by the first real diagnostic run on this project.

### Correction: the critic does not carry the ranking signal I credited it with

The v8 entry read the beam-search result (V-guided median 60 vs random 20) as
"the critic carries real ranking signal, but short-horizon greedy over it is
worse than the policy", and attributed the gap to the gamma = 1 greedy bound
2*gamma*eps/(1-gamma) being infinite. The simpler explanation is now measured
and it supersedes that one: **there is no ranking signal to search over.**

Matched on height, AUC of V against eventual survival:

| height | 100 | 200 | 300 | 400 |
|---|---|---|---|---|
| AUC | 0.503 | 0.524 | 0.513 | 0.539 |

A first pass pooling every visited state gave AUC 0.43, which looks like an
*anti*-correlated critic. That number is confounded by dwell time — an episode
that lingers near the target contributes many high-V states and still dies — and
sampling one state per episode at matched progress removes it. The honest
reading is chance, not inversion.

So V is a progress meter. It tracks how far along the episode is and nothing
else; at equal height it cannot tell a safe position from a fatal one. Its mean
is well calibrated to the marginal success rate (-0.10 mean against an implied
-0.36) while its ordering is uninformative. This explains the Q-flatness
directly: advantages built from a state-blind critic cannot discriminate actions.

### The credit horizon alone is not the fix

| arm | lambda | credit horizon | det success |
|---|---|---|---|
| control | 0.995 | 200 frames | **0.344** |
| lam-90 | 0.90 | 10 frames | 0.283 |

lam-95 and lam-97 pending.

lambda = 0.90 matches the measured causal window and still loses. In hindsight
the two measurements are coupled and I should have seen it before spending the
run: GAE interpolates between the Monte Carlo return (lambda = 1, unbiased,
smeared) and the one-step TD residual (lambda = 0, sharp, and only as good as V).
With V at chance, shrinking lambda trades a high-variance but *correct* signal
for a low-variance but *uninformative* one. Sharpening credit onto a blind
critic sharpens noise.

That makes the ordering non-optional:

1. Give the trunk a danger representation (auxiliary hazard head).
2. Only then shorten the credit horizon onto the causal window.

Neither should work alone, and lam-90 is the first half of that prediction.
The hazard sweep runs at the control lambda precisely so its effect is measured
independently; its pre-registered mechanism test is the matched-height AUC, not
the score. If AUC rises while the score does not, the indicated next run is the
combination, not abandonment.

### Process note

Three arms of the lambda sweep died because a deploy landed in the live code
directory mid-run: the change added a network head, and every arm then failed to
load its own head-less checkpoint. lam-90's 8.1M frames of training survived and
only its evaluation was lost. Sweeps now `cp -al` the tree and run from that
snapshot -- rsync replaces files rather than editing them, so hard links pin a
running sweep to the code it started with at no copy cost. Checkpoint loading
now also tolerates a missing hazard head, and says so when it resets the
optimiser rather than restoring misaligned moments.
