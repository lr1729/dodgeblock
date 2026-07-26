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

---

## STATUS — 2026-07-26

### Where the project actually stands

Not at the goal, and the goal is far. Per-layer survival plateaus near **0.92**;
10k on Hardcore needs **0.9972**. Best deterministic evaluation at rung 600 is
**0.344**, and the ladder self-terminated there rather than climbing.

What changed tonight is not the number. It is that the diagnosis is now grounded
in measurement instead of assumption, and the assumption it replaced was wrong.

### The five falsified interventions

| # | intervention | result vs control 0.344 | why it failed |
|---|---|---|---|
| 1 | more frames | rungs saturate by ~3M | not data-limited |
| 2 | search as teacher (rescue distillation) | collapse at coef 0.3, failing at 0.05 | taught strategy; deficit is reflex |
| 3 | potential-based shaping | 0.285 / 0.291 / 0.322 | at gamma=1 it telescopes to a constant |
| 4 | success-visitation bonus (SVM) | 0.301 / 0.281 | self-imitation accelerates the existing mode |
| 5 | shorter credit horizon (GAE lambda) | 0.283 (0.90) / 0.307 (0.95) | sharpens credit onto a blind critic |

Interventions 1-4 all assumed the agent's deficit was **strategic** — that it did
not know where to go or what to build. None of them tested that assumption. It
was false.

### What the two diagnostics found

**The agent dies in recoverable situations.** Replaying 200 real deaths,
snapshotting K frames before each, and searching from the exact restore point:

| K frames before death | 2 | 5 | 10 | 20 | 40 | 80 | 160 |
|---|---|---|---|---|---|---|---|
| fraction escapable | 0.015 | 0.195 | **0.725** | **0.880** | 0.920 | 0.940 | 0.990 |

88% of deaths had an escape available a third of a second earlier, found by
*sticky-random flailing* — not an oracle. A found escape proves viability, so
these are lower bounds. Causes: squished 148, fell 52.

**The critic cannot see it coming.** Matched on height, AUC of V against eventual
survival is **0.503 / 0.524 / 0.513 / 0.539** at heights 100/200/300/400. The
critic learned a progress meter: it reads off how far along the episode is, and
at equal progress cannot rank a safe position above a fatal one. Its mean is
well calibrated to the marginal success rate while its ordering is noise.

That single fact explains the whole pattern. PPO's actor learns from advantages
built on V. A state-blind V yields state-blind advantages, which is exactly the
measured Q-flatness (held-out direction loss ~ ln 3), and it is why every attempt
to add information to the *reward* failed: the channel that would carry it was
broken.

### Two corrections to my own earlier conclusions

1. **"The critic carries real ranking signal."** Recorded in v8 after V-guided
   beam search beat random (median 60 vs 20), and attributed to the gamma=1
   greedy bound. Wrong. Matched-height AUC is chance — there was no ranking
   signal to search over, which explains the beam result more simply.

2. **"The critic is anti-correlated with survival" (AUC 0.43).** My own first
   pass, pooling every visited state. Confounded by dwell time: an episode that
   lingers near the target contributes many high-V states and still dies.
   Sampling one state per episode at matched progress gives chance, not
   inversion. I caught this before acting on it.

### The credit-horizon result, read honestly

lambda 0.90 -> 0.283, lambda 0.95 -> 0.307, lambda 0.995 -> 0.344 (lam-97
pending). Monotone: every shortening of the credit horizon made things worse.

This is the *opposite* of what the autopsy alone predicted, and it is consistent.
GAE interpolates between the Monte Carlo return (lambda -> 1, unbiased, smeared
over 200 frames) and the one-step TD residual (lambda -> 0, sharp, and only as
good as V). With V at chance, the Monte Carlo return is the only real signal in
the estimator, and shrinking lambda discards it. Sharpening credit onto a blind
critic sharpens noise.

So the two measurements are coupled, and the ordering is forced:

1. Give the trunk a danger representation.
2. Only then shorten the credit horizon onto the causal window.

### What is running

An auxiliary head predicting P(death within {10, 30, 90} frames), trained by
weighted BCE on labels the rollout already contains. It adds no reward and no
demonstrations, so unlike interventions 2-4 it cannot bias the objective — it can
only change what the shared trunk represents.

Smoke test, 700k frames: loss 0.677 -> 0.638, separation **0.538 / 0.479 / 0.285**
for horizons 10 / 30 / 90. Strongest at the shortest horizon, which is the
signature the design predicted: near-term death is locally determined, long-term
death is not. This is the well-posed problem that P(reach target) is not.

Queued: hazard sweep (coef 0.25, 1.0) at control lambda, then the combination
(hazard + lambda 0.95 / 0.90). Pre-registered mechanism tests, so a null result
is attributable rather than ambiguous:

- **matched-height AUC must rise above 0.5.** If the score improves while AUC
  stays at chance, the stated mechanism is wrong and something else moved.
- **viability at K=20 must fall below 0.880.** If the agent learned to react,
  fewer of its deaths should be ones a third of a second would have saved.

Prediction on record: neither half works alone — lam-90 and lam-95 are the first
half of that confirmed — and the combination should beat *both halves*, not just
the control. If the combination also lands below 0.344, the credit-assignment
story is wrong and the deficit is not credit assignment at all.

### Honest assessment of the 10k goal

10k is reachable in principle: Go-Explore found 10k trajectories, so the game is
survivable at saturated difficulty. The obstacle is not the environment.

But the arithmetic is unchanged and unforgiving. Going from 0.92 to 0.9972
per-layer is not a tuning problem; at 10k the difference between two actions is
~0.003 in survival probability against a Bernoulli sigma of 0.5, which is
~250,000 samples per action distinction. No amount of the current signal buys
that. The reason to fix the critic is not that it closes the gap — it is that a
critic which ranks states at chance cannot climb *any* part of it, and every
method that assumed otherwise has now been measured and has failed.

---

## v10 — the v9 diagnosis was wrong, and the measurement that killed it (2026-07-26)

Adversarial review plus one new measurement retired the plan from the previous
section before it consumed the GPU. Recording it in full, because two of the
errors are mine and one had been asserted in three separate documents.

### 1. The reward is not terminal-only

`reward-v2.mjs`, `targetReward`, contains a hard-coded height potential:

```js
const beforePotential = Math.min(1, beforeHeight / targetHeight);
const afterPotential  = dead || success ? 0 : Math.min(1, afterHeight / targetHeight);
return (success ? 1 : 0) + Math.pow(discount, worldScale) * afterPotential - beforePotential;
```

`critic-profile.py`, the v9 log entry, and the hazard-head comment all assert a
terminal-only reward and derive `V(s) = P(success|s)` from it. The correct
statement is `V(s) = P(success|s) - Phi(s)`, `Phi = height/target`.

The *objective* is unharmed -- the potential telescopes and `Phi(s_0) = 0`, so the
episode return is exactly `1[success]`. Confirmed in the logs, where
`mean_return 0.286` equals `target_success 0.286` to three digits. But two
downstream readings were wrong:

- The pooled AUC of 0.43 has an exact mechanical explanation and needs no
  dwell-time story: `V` is anti-monotone in height *by construction*, while
  survivors visit high heights more. I reached the right conclusion (match on
  height) via reasoning that was not the real cause.
- Every place the log treats `V` as a calibrated `P(success)` is off by `Phi`.

### 2. The critic is not underfit — the outcome is not predictable

The v9 fix assumed AUC 0.5 was a defect. That assumes a better answer exists.
`outcome-probe.py` tests it: freeze the trunk, capture the 384-d representation at
each episode's first crossing of a height band, label with eventual survival,
train a supervised probe, split by EPISODE, score held out.

| height | 100 | 200 | 300 | 400 |
|---|---|---|---|---|
| probe AUC | 0.494 | 0.467 | 0.500 | 0.469 |
| critic AUC | 0.535 | 0.549 | 0.549 | 0.556 |
| test episodes | 586 | 517 | 385 | 312 |

A probe trained directly on outcome labels cannot beat chance, and the critic
narrowly *outperforms* it. **At matched height the eventual outcome is not a
function of the state.** The critic is at or near optimal, and "a progress meter
blind to danger" was wrong: it is a progress meter because progress is what is
predictable.

Read alongside the hazard smoke test -- the same trunk separates P(death within
10 frames) at 0.538 -- the pair is jointly diagnostic and neither number is
noise: **short-horizon hazard is state-determined; long-horizon success is not.**

Consequence: the auxiliary hazard head cannot work through the stated mechanism.
It would train well and change nothing, because there is no ranking for a better
`V` to express. The queued hazard and hazard x lambda runs were cancelled.

### 3. The credit horizon result is not monotone

lam-97 finished after the v9 entry was written: det **0.357**, above the control's
0.344.

| lambda | 0.90 | 0.95 | 0.97 | 0.995 (control) |
|---|---|---|---|---|
| det success | 0.283 | 0.307 | **0.357** | 0.344 |

The v9 entry called this monotone on two points plus the control. It is not. The
supportable claim is narrower: **lambda at or below 0.95 hurts; 0.97 and 0.995
are indistinguishable.** The "sharpening credit onto a blind critic" story
predicted monotonicity and does not survive its own third data point.

### 4. None of the A/Bs are statistically significant

At n=512 and p ~ 0.34, SE(one arm) = 0.021 and SE(a difference) = 0.030, so a
difference needs |d| > 0.058 to clear 95%, or > 0.080 with a Bonferroni
correction over the seven arms sharing one control.

| arm | shape-a | shape-b | shape-c | svm-half | svm-full | lam-90 | lam-95 | lam-97 |
|---|---|---|---|---|---|---|---|---|
| diff from control | +0.059 | +0.053 | +0.022 | +0.043 | +0.063 | +0.061 | +0.037 | -0.013 |
| z | 1.99 | 1.79 | 0.74 | 1.45 | 2.12 | 2.05 | 1.25 | -0.44 |

Minimum detectable difference at 80% power: **0.083**. Every effect this project
has interpreted is smaller than what the protocol can resolve. "Falsified" was
too strong throughout; the supportable statement is **"no intervention produced a
detectable improvement"**, which is a weaker and different claim.

Worse, all arms and the control run seed 7, so run-to-run variance of the
*identical* configuration has never been measured. There is a hint it is large:
rung-600-n14 and its continuation rung-600-x15 have statistically
indistinguishable mean height (424.4 [408,441] vs 421.3 [405,436]) and report det
success 0.344 vs 0.295 -- a 0.049 swing in the headline metric across checkpoints
of equivalent competence.

**Protocol changes:** mean height with a bootstrap CI becomes the primary metric
(it uses every episode's full height rather than one success bit, and the two
runs above show it is far more stable); evaluation goes to 2048 episodes; and a
noise-floor sweep varying only the training seed now runs before any further
interventions are interpreted.

### 5. The estimator, not the reward and not the representation

Measured: mean episode length **2333 frames**; GAE segment **256 steps**.

For a critic that has fit the height potential -- deterministic and trivially
learnable -- the TD residual at a non-terminal step is

  delta_t = [gamma*Phi_{t+1} - Phi_t] + gamma*(c - Phi_{t+1}) - (c - Phi_t) = c*(gamma - 1) = 0 at gamma = 1

so all signal sits at the terminal step. GAE restarts every 256-step segment and
bootstraps from `V` at the boundary, so a state receives real signal only when its
episode terminates inside its own segment: **~256/2333 = 11% of samples.** Within
that segment, lambda=0.995 decays the terminal credit to 0.28 across 256 steps.

This is arithmetic, not inference, and it is independent of whether the critic is
underfit. It also explains what the previous five interventions have in common:
all of them added information to the *reward*, at states where the estimator
discards it.

Queued (batch held at 131k samples and ~61 updates per arm, so rollout length is
not confounded with update count): rollout 1024 x lambda 0.995, rollout 1024 x
lambda 0.9995, rollout 256 x lambda 0.9995.

### 6. Where this points

The two live measurements now say opposite-looking things that are in fact
consistent, and together they name the estimator:

- The **state** effect is nil: at matched height nothing predicts the outcome.
- The **action** effect is real: 72-88% of deaths had an escape 10-20 frames out.

That is exactly the regime where paired action contrasts are the right estimator
and GAE is the wrong one. `A(s,a)` can be sharp while `V(s)` is flat, and GAE
estimates `A` through a +-1 terminal return whose noise swamps it.

The asset this project has and has never used for estimation: an exact simulator
with bitwise-verified `restore` and no reseed, so two rollouts from one snapshot
face identical falling blocks. Under common random numbers the contrast
`1[survive|a] - 1[survive|a']` has variance `P(discordant) - delta^2` rather than
~0.25 -- a large reduction in samples per action distinction, and unbiased,
because the rollouts are the agent's own policy from its own visited states.
This is Monte-Carlo policy *evaluation*, not the search-as-teacher and
distillation approaches that already failed (those were policy improvement and
behaviour cloning respectively, and carried the corresponding pathologies).

Its screening property also addresses the measured Q-flatness directly: where all
branches agree, the state is not a decision point and can be dropped; where they
disagree, it is. That is a measured crisis detector rather than a heuristic one.

Not yet built. The truncation sweep and the noise floor come first, because both
are cheap and both gate the interpretation of everything after them.

### Feasibility, quantified from the search budget (2026-07-26)

All 14 go-explore seeds reached 10k; median cost **64,237 restore-rollouts**. That
number converts the goal into a ladder, because a causal policy gets exactly one
attempt where the search got 64,237:

| per-layer survival | P(reach 10k) | episodes per success |
|---|---|---|
| 0.92 (measured now) | 8.8e-10 | 1,130,000,000 |
| 0.9400 | 1.9e-07 | 5,224,000 |
| **0.9567** | 1.6e-05 | **64,000** |
| 0.9700 | 4.9e-04 | 2,028 |
| 0.9900 | 8.1e-02 | 12 |
| 0.9972 | 4.96e-01 | 2 |

A one-shot policy at **p = 0.9567** reaches 10k about as often as the entire
64,237-iteration search does. That is a **1.8x hazard reduction** from where we
are (0.080 -> 0.043) -- a real milestone, and far short of the 29x needed for the
stated goal of *consistent* 10k. The project has been quoting only the 29x.

### The regime we have never measured

Every A/B in this ledger is at target 600 = 15 layers, which the agent covers in
roughly the first 40 seconds. A 10k run is 22,548 frames = **6.3 minutes**, and
difficulty saturates at ~4 minutes, so **~36% of a 10k trajectory is at saturated
difficulty** and essentially none of rung 600 is.

So the measured per-layer survival of 0.92 comes from the easy part of the curve,
and is being extrapolated to a regime it never sampled. Per-layer survival at
saturated difficulty is the quantity that actually decides feasibility, and it has
never been isolated. It is measurable directly: start the policy from go-explore
snapshots at high height and late elapsed time, and estimate the hazard there.

If saturated-difficulty per-layer survival is near 0.92, the 1.8x milestone is
plausible and the goal is a long climb. If it is materially worse -- and the
autopsy's "squished 148 / fell 52" split at low difficulty gives no information
about this -- then the distance is larger than any number in this log.

## v11 — the noise floor lands, and CRN is measured before it is built (2026-07-26)

### The noise floor changes most of the ledger

Same config, seed only:

| seed | det success | mean height (95% CI) | IQM |
|---|---|---|---|
| 7 | 0.3440 | **424.38** [408.12, 441.02] | 458.44 |
| 8 | 0.2935 | **409.22** [401.13, 417.31] | 433.36 |
| 9 | 0.3110 | **414.20** [405.84, 422.36] | 442.70 |

**Span 15.2 mean height and 0.051 det from the seed alone.** Every effect this
project has interpreted is smaller than that.

Re-judging every arm against that band (all evaluated at target 600):

| arm | mean height | verdict |
|---|---|---|
| lam-97 | **440.0** [423.04, 455.24] | **above band** |
| rung-600-n14 (seed 7) | 424.4 | control |
| shape-c | 422.2 | inside |
| seed-8 | 409.2 | control |
| svm-half | 408.8 | inside |
| lam-95 | 405.2 | below, CI overlaps |
| shape-a / shape-b | 402.5 / 399.6 | below, CI overlaps |
| svm-full | 387.3 [370.0, 404.45] | below, CI clears |
| lam-90 | 380.1 [362.65, 396.72] | below, CI clears |

**Only the two most aggressive settings clear the noise. Everything else was
seed noise being read as an effect.** And lam-97 is the one arm this project has
ever run that looks positive -- buried under a "monotone worse" reading that its
own third data point had already broken. A second seed is queued.

### The gamma bracket was wrong and is corrected

I proposed 2s/5s/20s half-lives. The value of a 2333-frame success:

| half-life | 2s | 5s | 20s | 45s | 90s |
|---|---|---|---|---|---|
| success worth | 0.0000 | 0.0046 | 0.2599 | 0.5494 | 0.7412 |

2s and 5s collapse the return to nothing and would produce pathological urgency,
not speed. Swept at **20/45/90s** instead. The env's potential is discounted with
the same gamma (`ppo_v2.py` passes `discount=args.gamma`), so it still telescopes
and the shaping stays policy-invariant; without that it would silently become a
height reward.

### CRN measured before building: 2.7x, not 50-200x

`crn-probe.py` branches 32 forced first actions from one restored snapshot and
rolls the policy 90 frames. Because `restoreSlot` does not reseed, every branch
faces bit-identical blocks, so any outcome difference is caused by the action.

| quantity | measured |
|---|---|
| pairwise discordance | 0.064 |
| independent contrast variance | 0.171 |
| **implied sample reduction** | **2.7x** |
| decision-point rate, all states | 0.233 |
| decision-point rate, dangerous states (survival < 0.5) | **1.00** |

The projected 50-200x assumed ~1% discordance and a p near 0.5. Both are wrong
here: discordance is 6.4%, and branch survival is 0.906 so the independent
variance is 0.171, not ~0.5. **The paired trick buys under 3x.** Recorded before
any of it was built, which is the point of running the probe first.

The screen is the more valuable half and it is a stronger result: **in 77% of
states all 18 actions give the identical outcome.** The true advantage there is
zero, so PPO spends the large majority of its gradient where nothing is
learnable -- and in exactly those states the entropy bonus is the only remaining
force, which is a mechanism for the observed jitter rather than a metaphor for it.

Caveats on the number: the 90-frame horizon may hide action effects that only
appear later, and with 32 lanes over 18 actions 14 actions are duplicated, which
biases discordance down by ~3% relative (0.064 -> 0.066 corrected). Neither moves
the order of magnitude.

### Consequence

The direction is not "CRN gives cheap low-variance advantages" -- it gives 2.7x.
It is "**77% of the batch is dead weight and can be identified for free**". Those
are different projects: the first is a variance argument, the second is a
sampling and behaviour argument, and only the second plausibly touches the
jitter the goal is about.

### Process: self-matching pgrep idled the GPU (again)

The rollout sweep was chained behind the noise floor with

```
while pgrep -f overnight-noise-floor.sh >/dev/null && pgrep -f ppo_v2 >/dev/null; do sleep 60; done
```

The waiting shell's own command line contains *both* patterns, so both pgreps
matched the waiter itself and the loop could never exit. The GPU sat idle from
seed-9's completion until it was caught. This exact failure -- a wait-loop pgrep
matching its own command line -- is already recorded once in this project, and
the chained sweeps that use a **file marker** (`grep -q 'sweep complete'`) have
never had the problem. Chain on artifacts, not on process tables.

### Truncation falsified; the lambda curve is an inverted U (2026-07-26)

| arm | rollout | lambda | mean height | vs noise band 409.2-424.4 |
|---|---|---|---|---|
| roll-1024 | 1024 | 0.995 | 418.83 [410.55, 426.7] | inside — no effect |
| roll-1024-l | 1024 | 0.9995 | **373.81** [365.7, 382.3] | clearly below, CI clears |
| roll-256-l | 256 | 0.9995 | **398.83** [390.86, 407.09] | below, CI clears |

lambda 0.9995 loses at BOTH segment lengths (398.8 at 256, 373.8 at 1024), so the
harm belongs to lambda and not to a lambda x rollout interaction. Extending the
segment on top of it makes things worse still, which is the opposite of what the
truncation account predicts.

Longer segments alone do nothing; longer segments plus lambda -> 1 are markedly
worse. roll-1024-l was named in advance as the arm that actually tests the
hypothesis, since credit cannot reach a state without both a segment that
contains the terminal *and* a lambda that survives the decay to it. It got both
and lost 45 points of mean height.

So the "~11% of samples see a terminal" arithmetic is correct and **behaviourally
inert**. Fixing the reach does not help, because pushing lambda toward 1 turns
GAE into the raw Monte-Carlo return -- maximum variance, no bootstrap -- and the
variance costs more than the reach buys.

Collecting every lambda measured at rollout 256:

| lambda | 0.90 | 0.95 | 0.97 | 0.995 (control) |
|---|---|---|---|---|
| mean height | 380.1 | 405.2 | **440.0** | 409.2-424.4 (3 seeds) |

That is a smooth inverted U with a peak at 0.97, and roll-1024-l's 373.8 at
lambda 0.9995 extends the right-hand fall. **Both neighbours of 0.97 are worse**,
which is a much weaker coincidence than a lone high point would be — the v9
entry called this same arm's neighbourhood "monotone worse" on two points.

It is still one seed. lam-97-s8 is queued in the gamma sweep and the driver
replicates the best arm at two further seeds, so it will have three before it is
believed.

### Discounting did not buy speed (2026-07-26)

| arm | half-life | mean height | climb rate |
|---|---|---|---|
| control | — (gamma=1) | 409.2-424.4 (3 seeds) | ~10.5 h/s |
| gamma-20s | 20 s | 409.59 [401.5, 417.36] | **10.1 h/s** |
| 10k demos | — | — | 26.6 h/s |

Mean height sits at the very bottom of the noise band, so no effect there. The
mechanism test is climb rate, and it did not move -- if anything it fell.

This matters more than the score. A 20 s half-life makes a 2,333-frame success
worth 0.26 of an instant one, which is a large time preference, and it is the
STRONGEST discount in the bracket: 45 s and 90 s discount less, so if the
strongest fails to move climb rate the weaker two are very unlikely to.

The hypothesis was that gamma = 1 makes the objective indifferent to speed and
that discounting would therefore buy it. The premise is still true -- it is a
property of the return -- but the conclusion does not follow, and the honest
alternative is now live: **the policy may already be at its speed/safety
optimum**, in which case the 2.5x gap to the demos is not slack.

That would fit the demos being the surviving tail of ~64,000 retries. A search
that can retry is free to be reckless; the fast paths are the ones that happened
to live. A causal one-life policy travelling at 26.6 h/s might die almost always,
which would make part of the observed slowness correct risk management rather
than bumbling.

One seed, two arms outstanding. Recorded now because the prediction (45 s and 90 s
should move climb rate even less) is falsifiable before they land.

**The prediction held. The full bracket is null.**

| arm | half-life | mean height | climb rate |
|---|---|---|---|
| control | gamma = 1 | 409.2-424.4 (3 seeds) | ~10.5 h/s |
| gamma-20s | 20 s | 409.59 [401.5, 417.36] | 10.1 h/s |
| gamma-45s | 45 s | 402.40 [394.22, 410.18] | 10.3 h/s |
| gamma-90s | 90 s | 404.88 [396.72, 412.74] | 9.8 h/s |
| 10k demos | — | — | 26.6 h/s |

Three discount strengths spanning a 3.5x range of time preference. Not one of
them increased climb rate; all three came in slightly *slower* than gamma = 1, and
all three land at or below the bottom of the noise band on mean height. The
individual climb-rate gaps (9.8-10.3 against 10.5) are small enough to be noise on
their own -- what is not noise is that there is no hint of a gain anywhere in a
bracket built to produce one.

**So gamma < 1 is falsified as a route to faster play.** The premise survives
untouched: at gamma = 1 the return genuinely is indifferent to how long a success
takes. Adding a time preference simply does not make the policy travel faster,
which means slowness was not the objective's fault.

That leaves the alternative stated above as the better-supported reading: the
policy is close to its own speed/safety frontier, and the 2.5x gap to the demos is
not slack to be recovered by asking for speed. If the agent were merely idling,
any of three discounts should have moved it.

Consequence for the stated goal: "fast, smooth, expert-looking" play is not
reachable by reweighting the objective. Either the frontier itself has to move --
better reactions, which is the action-conditioned credit thread -- or the target
speed is wrong because it was read off trajectories that survived 64,000 retries.

## v12 — the hazard curve, and why every A/B so far measured the wrong 40 seconds

512 uncensored episodes (target set to 1,000,000 so nothing ends by succeeding).
**512 deaths. Mean peak height 491.9. Mean episode length 43.5 seconds.**

| elapsed | exposure (s) | deaths | survival/s | implied per-layer |
|---|---|---|---|---|
| 0-60 s | 21,059 | 410 | 0.9805 | **0.928** |
| 60-120 s | 1,222 | 101 | 0.9173 | **0.720** |
| 120-180 s | 16 | 1 | 0.9387 | 0.786 |
| >=240 s (saturated) | **0** | 0 | — | **unmeasurable** |

Two things fall out, and both are worse than anything previously in this log.

**1. Per-layer survival is not a constant, and it collapses with elapsed time.**
The 0.92 this project has quoted for months is the *first-minute* figure. In the
second minute it is 0.72. Every feasibility number derived from a flat 0.92 --
including the ladder in v11 that put "reach 10k at all" at a 1.8x hazard
reduction -- assumed a constant that does not exist. The real requirement is to
*sustain* 0.9567 for 376 seconds, against a policy whose hazard already
quadruples between minute one and minute two.

**2. The saturated regime cannot be measured from this policy at all.** Difficulty
saturates at ~240 s; the mean episode is 43.5 s and the longest barely touch 180 s.
Exposure past 240 s is exactly zero. So the quantity named in v11 as "the number
that actually decides feasibility" is not merely unmeasured -- it is unreachable
by rolling this policy forward, and always was.

And the framing error is now quantified. Rung 600 is reached in roughly the first
40 seconds; the mean episode is 43.5 seconds. **Every A/B in this ledger -- all
seven interventions, the noise floor, the rollout sweep, the gamma bracket --
optimised the first 43 seconds of a 376-second problem**, in the region where
difficulty has barely begun to ramp. That is not a subtle sampling issue: minute
two is a different game, with 4x the per-layer hazard, and no run in this project
has ever trained in it.

Consequence for the gamma result: the bracket asked for speed in the regime where
speed matters least. Its null stands as measured, but it does not generalise to
the regime the goal lives in.

Measuring the saturated regime requires *starting* there. The go-explore bank
holds snapshots at high height, hence at late elapsed time, and the restore path
is exact -- so the measurement is available, it simply has never been run.

### lambda = 0.97 did not replicate (2026-07-26)

| seed | control | lambda=0.97 | paired diff |
|---|---|---|---|
| 7 | 424.38 | 440.00 | +15.62 |
| 8 | 409.22 | 421.41 | +12.19 |
| 9 | 414.20 | **412.87** [404.67, 421.08] | **-1.33** |

mean +8.83, sd 8.96, SE 5.17, t(2) = 1.71, 95% CI **[-13.44, +31.09]**. The
interval spans zero. The prediction registered before the run -- 425 to 430 if
the effect were real -- was wrong; the third seed came in flat.

So the last live thread from the sweeps closes, and **nothing in this ledger is a
confirmed positive.** Every intervention run against a properly measured control
is null or negative:

| direction | verdict |
|---|---|
| more frames | saturates by 3M |
| behaviour cloning / distillation from search | collapse |
| potential-based shaping | no-op by construction at gamma = 1 |
| success-visitation bonus | negative, dose-dependent |
| auxiliary hazard head | cancelled: the probe showed no signal to extract |
| GAE credit horizon (lambda) | inverted U, peak did not replicate |
| GAE truncation (rollout length) | null alone, harmful with lambda -> 1 |
| discounted target (gamma < 1) | null across a 3.5x bracket, no speed gain |

The two-seed result looked like the strongest signal the project had produced,
and it was reported that way. It was seed noise with a consistent sign across two
draws, which is exactly what a 15-point noise band makes likely and exactly what
the replication gate exists to catch. Cost: one 25-minute run.

The pattern across the whole ledger is now hard to miss: **every intervention has
targeted the reward, the estimator, or the objective, and all of them were
evaluated in the first 43 seconds of a 376-second problem.** The hazard curve says
that regime has 4x lower hazard than the next minute and zero exposure to the one
the goal lives in. Whatever is or is not true of these knobs at rung 600, none of
it has been tested where it matters.

## v13 — the saturated regime, measured at last (2026-07-26)

The hazard curve said the policy never reaches the 240-second saturation point, so
the deciding quantity could not be measured by rolling it forward. It can be
measured by *starting* there: the go-explore banks hold 512 snapshots spanning
frames 0-22,315 (median 18,531 = 309 s) and the env restores a chosen cell exactly.

Policy dropped into banked states, 512 episodes per band, uncensored:

| start regime | cells | deaths | layers gained | mean survival | per-layer |
|---|---|---|---|---|---|
| 60-240 s (ramp) | 23 | 512 | 1474 | 11.1 s | **0.6526** |
| 240 s+ (saturated) | 485 | 512 | 1674 | 7.9 s | **0.6941** |

**Saturated is not worse than ramp.** Both bands are equally off-distribution --
both were built by search -- so the comparison between them is internally
controlled for that confound, and it shows essentially no difficulty penalty past
240 s. That is what a difficulty function saturating in time predicts.

Now put it beside the on-distribution curve. The policy's own minute-2 states give
0.720; search-built states at comparable times give 0.65-0.69. **Those are close.**
So distribution shift is not what makes the banked number low -- by minute two the
regime is simply hard, whoever built the state.

### The real distance to the goal

| regime | per-layer | hazard | x reduction for 10k-at-all | for consistent 10k |
|---|---|---|---|---|
| minute 1 | 0.928 | 0.072 | 1.7x | 26x |
| minute 2 | 0.720 | 0.280 | **6.5x** | 100x |
| 240 s+ (banked) | 0.694 | 0.306 | **7.1x** | 109x |

v11 quoted "1.8x hazard reduction to reach 10k at all". That number is the
minute-one figure and it is not the binding one: a 10k run is 376 seconds, so what
must be *sustained* is the minute-two-and-beyond rate. The honest requirement is a
**~7x hazard reduction sustained across six minutes**, and ~110x for the goal as
originally stated. Both v11 numbers were optimistic by roughly 4x.

### What this reframes

The policy's hazard rises 0.928 -> 0.720 between minute one and minute two on its
own trajectory, while banked states show no ramp-to-saturated difference. Those
two facts together point away from the spawn-rate ramp as the driver and toward
**the state the policy is in by minute two** -- which, since difficulty has
saturated, is largely the pile it has built for itself.

That is the first evidence in this ledger pointing at pile management rather than
reaction speed, and it is the one hypothesis that matches what a strong human
player reports doing: breaking towers to shape cover rather than climbing whatever
is in front of them. It is not established -- the two measurements come from
different state distributions and only their *difference* is controlled -- but it
is now the best-supported live account of the plateau.

### Training in the saturated regime does not help (2026-07-26)

20M frames, 80% banked starts, target 10000, from rung-600-n14.

| metric | baseline | after | |
|---|---|---|---|
| saturated (240s+) per-layer | 0.6941 | **0.7146** | flat |
| ramp (60-240s) per-layer | 0.6526 | **0.5854** | worse |
| fresh-start mean episode | 43.5 s | **20.5 s** | **halved** |

The pre-registered suffix-specialist check fired: fresh-start competence lost half
its episode length while the target metric moved by 0.02, which is inside what a
15-point noise band on a related metric would suggest is nothing.

**And the obvious excuse does not apply.** Cell training reported
`successes 3754 / completions 24474 = 15.3%`, so the policy was not in a signal
desert -- it had abundant reward. It simply spent it learning to finish from cells
that already sit a layer or two below 10,000 (median cell height 7,920, median
progress per episode 40 = one layer) rather than learning to survive the regime.

So **exposure is not the deficit.** The regime error identified in v12 is real as a
*measurement* problem -- every A/B in this ledger scored the wrong 43 seconds --
but correcting the *training* distribution does not correct the capability. The
policy cannot learn to survive minutes two through six even when trained there
with a working reward signal.

That closes the fourth class of explanation. The plateau is not:

| class | closed by |
|---|---|
| reward shape | shaping, SVM, BC/distillation (5 arms) |
| estimator | lambda sweep, rollout truncation, gamma bracket |
| representation | outcome probe: no extractable signal at matched height |
| exposure | this run |

What has never been varied is the policy class itself -- 2.2M parameters,
feedforward, one decision per world frame, an 18-way factored head. Every
experiment in this ledger has changed what the network is *told*; none has changed
what it *is*, or how often it acts.
