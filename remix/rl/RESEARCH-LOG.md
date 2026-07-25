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
