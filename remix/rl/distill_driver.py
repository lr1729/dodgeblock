#!/usr/bin/env python3
"""Unattended search-distillation loop.

v25 established the mechanism and round 1 confirmed it moves the goal metric:
saturated per-layer survival rose on all three banks (0.8213->0.8305,
0.7747->0.8155, 0.8230->0.8372), a mean of -0.0265 nats/layer from one round of
8000 search targets. The whole 96M-frame ladder bought 0.109 nats, so one round
bought roughly a quarter of that.

The mechanism should compound: search finds the rare good continuation,
distillation moves it into the mode, and the next round's search starts from a
better prior. This runs that loop unattended so it does not depend on anyone
launching each round by hand -- which is exactly how the GPU sat idle for 11
hours after round 1 landed.

Each round:

    collect  K-branch survival targets from the current best checkpoint
    distil   them in, anchored by KL to that checkpoint
    evaluate saturated per-layer survival on every bank in BANKS
    gate     accept if mean log-hazard improved, else retry once with more
             samples, else stop

Gating on the MEAN over banks, in nats, is deliberate. v22 measured that absolute
per-layer survival swings 0.10 between banks -- wider than any effect this project
has produced -- while the log-hazard difference is stable at +/-0.017. A
single-bank gate would be reading bank difficulty, not progress.

State lives in <state-dir>/distill-state.json; every decision is appended to
<state-dir>/RESULTS.md. Chains on file markers and runs each stage under
systemd-run, because this codebase has now lost work three separate ways to
process-table checks and to shells that did not outlive their ssh session.
"""
import argparse
import json
import math
import subprocess
import time
from datetime import datetime
from pathlib import Path

CODE_DIR = Path.home() / 'dodgeblock-v5-code'
PYTHON_BIN = Path.home() / 'envs/dodgeblock-rl/bin/python'
BANK_DIR = Path.home() / 'dodgeblock-go-explore-bank-v4'
# All sixteen. The paired per-bank SD is 0.0142 nats, so the SE of the mean is
# 0.0142/sqrt(n) -- 0.0082 at three banks, 0.0036 at sixteen. A three-bank gate
# could not resolve any effect this loop is capable of producing. Bank count is
# the only lever that matters here: re-running a bank under a different eval seed
# moves it by 0.031, but base and candidate share cells and futures under the
# same seed, so the paired difference is already 4.7x quieter and more episodes
# per bank buy far less than the unpaired spread suggests.
BANKS = tuple(f'seed-{index}' for index in range(1, 17))

SAMPLES = 8000
RETRY_SAMPLES = 16000
LANES = 32
# Measured 2026-07-29: 8 workers -> 1136 samples/h, 16 workers -> 1737 samples/h
# while contending with a live round, so >=1.53x and a floor. The collector is
# latency-bound on bridge round-trips, which are per-step-across-all-envs, so
# extra trajectories harvest more contested states at nearly the same cost. Load
# average was 2.61 of 20 cores at 8 workers. At 16 a round finishes its full
# sample budget in ~4.6h instead of truncating on the wall guard.
# Reverted to 8 on 2026-07-30 after the box became unreachable (sshd and all
# userspace dead, kernel still answering ping) shortly after a switch to 16 and
# had to be power-cycled. No mechanism was ever established -- systemd cgroups
# should reap stage children, 16 processes on 20 cores should not wedge
# anything, and an unrelated vLLM server had already died on its own beforehand.
# The condition set then was to revisit once a full round completed clean, and
# one has: round 1 ran collect, train and eval end to end, followed by ~7 h of
# continuous eval sweeps at load 2-5 of 20 cores with no incident. Restored to 16
# because a block is 4 rounds and the difference is 4 blocks in the remaining
# budget rather than 7. `ensure_clean` now runs before every stage.
WORKERS = 16
HORIZON = 120
STRIDE = 10
# Calibrated 2026-07-30 against the achieved KL, not guessed. The previous
# settings (1e-5, anchor 1.0, 4 epochs) ended 0.00095 nats from the source
# policy -- roughly 30x below the bottom of a normal trust region -- so the round
# they produced was a no-op and its null said nothing about the targets. A
# ladder on fixed targets reached 0.0157 / 0.0538 / 0.0917 nats at (1e-4, 1.0),
# (3e-4, 0.3) and (3e-4, 0.0).
#
# These are the (3e-4, 0.3) arm. It is not the largest step, and on eight
# selection banks it was not even the best-looking arm; (1e-4, 1.0) scored higher
# there and then REVERSED SIGN on eight held-out banks, pooling to -0.001. This
# arm held its sign on both halves (+0.0094 select, +0.0067 held out) and pools
# to +0.0080 +/- 0.0055 over sixteen banks. Anchor stays above zero because the
# targets only cover states where branches disagreed; the anchor is what holds
# the other two thirds of the policy in place.
EPOCHS = 8
BATCH = 256
LEARNING_RATE = 3e-4
ANCHOR_COEF = 0.3
# The collector is latency-bound on per-frame GPU round-trips, not CPU-bound
# (measured: 42 min CPU over 115 min wall). With the post-v26 fixed collector
# and 8 workers, the observed rate is ~800-1150 samples/hour. A fixed 5h wall
# silently turns both the base and retry rounds into similarly sized partial
# batches, so the "retry with more samples" branch would not actually increase
# evidence. Scale the wall with requested samples while retaining a hard cap per
# stage so a broken collector cannot stall the loop indefinitely.
COLLECT_HOURS_PER_8K = 10.0
COLLECT_MIN_HOURS = 5.0
EVAL_ENVS = 128
EVAL_ROUNDS = 4

# Distillation only ever sees saturated states, so it can improve late-game
# survival while quietly wrecking the fresh prefix -- exactly the split the
# hazard head showed (7.6x collapse on the rung metric, 0.016 on saturated).
# Every accepted round is therefore also scored from fresh starts, and a large
# regression stops the loop even if saturated survival improved.
FRESH_TARGET = 1150
FRESH_EPISODES = 1024
# round1.pt's OWN fresh play, measured 2026-07-30: 751.8, CI [730.7, 774.2].
# This was 814.92, which is the 24M control -- a different lineage. Every
# candidate was therefore scored against a policy it is not descended from, and
# the resulting "regressed to 0.92x baseline" readings were measuring lineage,
# not damage: the no-op candidate scored 764.5 while sitting 0.001 nats from its
# parent. All three step-ladder arms land inside the CI above, so a real step
# does not hurt the prefix.
FRESH_BASELINE_MEAN = 751.8
FRESH_REGRESSION_LIMIT = 0.80    # fraction of baseline mean height tolerated

# Gate on BLOCKS, not rounds. One round buys about +0.008 nats against an SE of
# 0.0055, which is z=1.5 -- a per-round gate set high enough to be meaningful
# would reject most genuine rounds, and one set low enough to pass them would be
# accepting noise. Four accumulated rounds move roughly 4x as far against the
# same measurement, so the block is what can actually be resolved, and verifying
# once per block costs a quarter of the eval time. Rounds inside a block are
# accepted unconditionally; the block is what can be rolled back.
VERIFY_EVERY = 4
MIN_BLOCK_GAIN = 0.016
MAX_ROUNDS = 40
WALL_HOURS = 168.0
POLL_SECONDS = 120


def now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def log(message):
    print(f'[{now()}] {message}', flush=True)


class DistillLoop:
    def __init__(self, state_dir):
        self.state_dir = Path(state_dir)
        self.state_path = self.state_dir / 'distill-state.json'
        self.results_path = self.state_dir / 'RESULTS.md'
        self.state = json.loads(self.state_path.read_text())
        # `best_checkpoint` is the working head and moves every round;
        # `verified_checkpoint` is the last one a block measurement stood behind
        # and is what a failed block falls back to. A state file written before
        # block gating has only the former.
        self.state.setdefault('verified_checkpoint',
                              self.state['best_checkpoint'])
        self.state.setdefault('verified_nats', self.state['best_nats'])
        self.state.setdefault('block_misses', 0)

    def save(self):
        temporary = self.state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.state, indent=2))
        temporary.replace(self.state_path)

    def record(self, line):
        with self.results_path.open('a') as handle:
            handle.write(f'- {now()} {line}\n')
        log(line)

    def stop(self, status, reason):
        self.state.update(done=True, status=status)
        self.save()
        self.record(f'**{status}** — {reason}')

    def ensure_clean(self):
        """No stray env-servers before a stage starts.

        Stage units are cgroup-scoped so systemd should reap their children, but
        this loop lost a box to an undiagnosed resource exhaustion and the cost
        of being wrong here is another power cycle. Cheap, idempotent, and it
        makes the assumption checkable instead of assumed.
        """
        probe = subprocess.run(
            ['pgrep', '-c', '-f', 'env-server-v2[.]mjs'],
            capture_output=True, text=True,
        )
        stray = int(probe.stdout.strip() or 0)
        if stray:
            self.record(f'found {stray} stray env-server processes before a '
                        f'stage; killing them')
            subprocess.run(['pkill', '-f', 'env-server-v2[.]mjs'], check=False)
            time.sleep(10)

    def run_stage(self, unit, command, marker):
        """Run one stage to completion, detected by its marker file."""
        self.ensure_clean()
        marker = Path(marker)
        if marker.exists():
            marker.unlink()
        subprocess.run(
            ['systemd-run', '--user', '--collect', f'--unit={unit}',
             '/usr/bin/bash', '-lc', f'{command} && touch {marker}'],
            capture_output=True, text=True, check=False,
        )
        # systemd-run returns before the unit is fully up, so a unit that reads
        # 'activating' is alive, not failed. Treating anything != 'active' as
        # failure would abort every round in its first second.
        alive = {'active', 'activating', 'reloading', 'deactivating'}
        time.sleep(10)
        while not marker.exists():
            probe = subprocess.run(
                ['systemctl', '--user', 'is-active', unit],
                capture_output=True, text=True,
            )
            if probe.stdout.strip() not in alive:
                # Unit is gone. Give the filesystem a moment for the marker the
                # command may have written just before exiting, then report.
                time.sleep(10)
                return marker.exists()
            time.sleep(POLL_SECONDS)
        return True

    def evaluate(self, checkpoint, tag):
        """Mean log-hazard across banks; lower is better."""
        nats = {}
        for bank in BANKS:
            out = self.state_dir / f'{tag}-{bank}.json'
            unit = f'distill-eval-{tag}-{bank}'.replace('_', '-')
            command = (
                f'cd {CODE_DIR} && CUDA_VISIBLE_DEVICES=1 {PYTHON_BIN} '
                f'rl/saturated-hazard.py {checkpoint} '
                f'--bank {BANK_DIR}/{bank}/search-checkpoint.json.gz '
                f'--control-interval 1 --envs {EVAL_ENVS} '
                f'--rounds {EVAL_ROUNDS} > {out}'
            )
            if not self.run_stage(unit, command, self.state_dir / f'.{unit}.done'):
                return None
            try:
                payload = json.loads(out.read_text())
                survival = payload['by_start_regime'][
                    'saturated_240s_plus']['per_layer_survival']
            except Exception:
                return None
            if not survival:
                return None
            nats[bank] = -math.log(survival)
        return nats

    def fresh_eval(self, checkpoint, tag):
        """Mean height from fresh starts; guards the prefix distillation ignores."""
        out = self.state_dir / f'{tag}-fresh.json'
        unit = f'distill-fresh-{tag}'.replace('_', '-')
        command = (
            f'cd {CODE_DIR} && CUDA_VISIBLE_DEVICES=1 {PYTHON_BIN} '
            f'rl/evaluate_ppo_v2.py {checkpoint} --episodes {FRESH_EPISODES} '
            f'--target-height {FRESH_TARGET} --control-interval 1 > {out}'
        )
        if not self.run_stage(unit, command, self.state_dir / f'.{unit}.done'):
            return None
        try:
            return float(json.loads(out.read_text())['mean_height'])
        except Exception:
            return None

    def round(self):
        index = self.state['round']
        base = self.state['best_checkpoint']
        samples = RETRY_SAMPLES if self.state.get('block_misses') else SAMPLES
        collect_hours = max(
            COLLECT_MIN_HOURS,
            COLLECT_HOURS_PER_8K * samples / SAMPLES,
        )
        work = self.state_dir / f'round-{index}'
        work.mkdir(parents=True, exist_ok=True)
        dataset = work / 'targets.npz'
        candidate = work / 'distilled.pt'

        collect = (
            f'cd {CODE_DIR} && CUDA_VISIBLE_DEVICES=1 {PYTHON_BIN} '
            f'rl/search-distill-collect.py {base} --out {dataset} '
            f'--samples {samples} --lanes {LANES} --workers {WORKERS} '
            f'--horizon {HORIZON} --stride {STRIDE} --control-interval 1 '
            f'--max-hours {collect_hours:.2f} '
            f'> {work}/collect.json 2> {work}/collect.log'
        )
        if not self.run_stage(f'distill-collect-{index}', collect,
                              work / '.collect.done'):
            self.stop('NEEDS-ATTENTION', f'round {index} collect produced no marker')
            return
        try:
            collect_info = json.loads((work / 'collect.json').read_text())
            actual_samples = int(collect_info.get('samples', samples))
        except Exception:
            actual_samples = samples

        train = (
            f'cd {CODE_DIR} && CUDA_VISIBLE_DEVICES=1 {PYTHON_BIN} '
            f'rl/search-distill-train.py {base} {dataset} --out {candidate} '
            f'--epochs {EPOCHS} --batch {BATCH} '
            f'--learning-rate {LEARNING_RATE} --anchor-coef {ANCHOR_COEF} '
            f'> {work}/train.json'
        )
        if not self.run_stage(f'distill-train-{index}', train, work / '.train.done'):
            self.stop('NEEDS-ATTENTION', f'round {index} train produced no marker')
            return

        # Adopt every round so the next one searches from the improved policy.
        # A single round cannot be resolved against its own noise, so nothing is
        # decided here; the block boundary below is where a result exists.
        self.state['best_checkpoint'] = str(candidate)
        self.state['round'] += 1

        fresh = self.fresh_eval(candidate, f'round-{index}')
        fresh_note = 'fresh=n/a'
        if fresh is not None:
            fresh_note = f'fresh={fresh:.1f} ({fresh / FRESH_BASELINE_MEAN:.2f}x base)'
        if (fresh is not None
                and fresh < FRESH_REGRESSION_LIMIT * FRESH_BASELINE_MEAN):
            self.state['best_checkpoint'] = self.state['verified_checkpoint']
            self.record(
                f'round {index}: {fresh_note} → **ROLLBACK** (fresh prefix '
                f'regressed past {FRESH_REGRESSION_LIMIT:.0%} of baseline)')
            self.stop('FRESH-REGRESSION',
                      'distillation damaged fresh play; needs a prefix term or '
                      'mixed-state targets')
            return

        if index % VERIFY_EVERY:
            self.record(
                f'round {index}: {fresh_note} → adopted unverified '
                f'(samples={actual_samples}/{samples}); '
                f'block verifies at round {index + VERIFY_EVERY - index % VERIFY_EVERY}')
            self.save()
            return

        after = self.evaluate(candidate, f'round-{index}-eval')
        if after is None:
            self.stop('NEEDS-ATTENTION', f'round {index} evaluation incomplete')
            return
        before = self.state['verified_nats']
        gain = sum(before[b] for b in BANKS) / len(BANKS) - \
            sum(after[b] for b in BANKS) / len(BANKS)
        block = f'rounds {index - VERIFY_EVERY + 1}-{index}'

        if gain >= MIN_BLOCK_GAIN:
            self.state.update(verified_checkpoint=str(candidate),
                              verified_nats=after, block_misses=0)
            self.state['history'].append(
                {'block': block, 'gain_nats': gain, 'nats': after,
                 'fresh_mean_height': fresh, 'checkpoint': str(candidate),
                 'samples': actual_samples})
            self.record(
                f'{block}: mean gain {gain:+.4f} nats over {len(BANKS)} banks | '
                f'{fresh_note} → **PROMOTE**')
            self.save()
            return

        # The block did not clear the gate, so the last verified checkpoint is
        # the best thing that exists and the rounds since are discarded.
        self.state['best_checkpoint'] = self.state['verified_checkpoint']
        self.state['block_misses'] = self.state.get('block_misses', 0) + 1
        if self.state['block_misses'] < 2:
            self.record(
                f'{block}: mean gain {gain:+.4f} nats over {len(BANKS)} banks | '
                f'{fresh_note} → **ROLLBACK**, retrying the block at '
                f'{RETRY_SAMPLES} samples/round')
            self.save()
            return
        self.record(
            f'{block}: mean gain {gain:+.4f} nats over {len(BANKS)} banks | '
            f'{fresh_note} → **PLATEAU** — two consecutive blocks below '
            f'{MIN_BLOCK_GAIN} nats')
        self.stop('PLATEAU', 'search distillation stopped compounding; raise K, '
                             'or move to option-level continuations or margin '
                             'training')

    def run(self):
        log(f'driver up: round={self.state["round"]} '
            f'best={self.state["best_checkpoint"]}')
        while not self.state['done']:
            if self.state['round'] >= MAX_ROUNDS:
                self.stop('DONE', f'round cap {MAX_ROUNDS} reached')
                break
            elapsed = (time.time() - self.state['started_at']) / 3600.0
            if elapsed > WALL_HOURS:
                self.stop('WALL-REACHED', f'{elapsed:.1f}h elapsed')
                break
            self.round()
        log('driver exiting')


def bootstrap(state_dir, checkpoint, nats):
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / 'distill-state.json'
    if path.exists():
        log(f'state exists at {path}, not overwriting')
        return
    path.write_text(json.dumps({
        'round': 1,
        'best_checkpoint': str(checkpoint),
        'best_nats': nats,
        'verified_checkpoint': str(checkpoint),
        'verified_nats': nats,
        'block_misses': 0,
        'history': [],
        'started_at': time.time(),
        'done': False,
        'status': 'running',
    }, indent=2))
    results = state_dir / 'RESULTS.md'
    if not results.exists():
        results.write_text(
            '# Search-distillation loop\n\n'
            f'Rounds are adopted unverified and gated in blocks of '
            f'{VERIFY_EVERY}: a block promotes when mean log-hazard over '
            f'{len(BANKS)} banks improves by >= {MIN_BLOCK_GAIN} nats/layer, '
            'otherwise it rolls back to the last verified checkpoint. One round '
            'moves about +0.008 nats against an SE of 0.0055, so a single round '
            'is not resolvable against its own noise and a block is; verifying '
            'per block is also a quarter of the eval cost. Gating on the mean in '
            'nats because absolute survival swings 0.10 between banks while the '
            'paired difference has SD 0.0142.\n\n')
    log(f'bootstrapped {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-dir', default=str(Path.home() / 'dodgeblock-distill-loop'))
    parser.add_argument('--bootstrap', action='store_true')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--nats', default='',
                        help='JSON of bank->nats for the starting checkpoint')
    args = parser.parse_args()
    if args.bootstrap:
        bootstrap(args.state_dir, args.checkpoint, json.loads(args.nats))
        return
    DistillLoop(args.state_dir).run()


if __name__ == '__main__':
    main()
