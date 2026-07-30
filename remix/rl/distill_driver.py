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
BANKS = ('seed-1', 'seed-3', 'seed-5')

SAMPLES = 8000
RETRY_SAMPLES = 16000
LANES = 32
HORIZON = 120
STRIDE = 10
EPOCHS = 4
BATCH = 256
LEARNING_RATE = 1e-5
ANCHOR_COEF = 1.0
EVAL_ENVS = 128
EVAL_ROUNDS = 4

# A round must beat the base by more than bank-to-bank noise. v22 measured the
# log-hazard spread at +/-0.017 nats across banks for a fixed policy pair, so
# anything under half that is not evidence.
MIN_NATS_GAIN = 0.008
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

    def run_stage(self, unit, command, marker):
        """Run one stage to completion, detected by its marker file."""
        marker = Path(marker)
        if marker.exists():
            marker.unlink()
        subprocess.run(
            ['systemd-run', '--user', '--collect', f'--unit={unit}',
             '/usr/bin/bash', '-lc', f'{command} && touch {marker}'],
            capture_output=True, text=True, check=False,
        )
        while not marker.exists():
            probe = subprocess.run(
                ['systemctl', '--user', 'is-active', unit],
                capture_output=True, text=True,
            )
            if probe.stdout.strip() != 'active' and not marker.exists():
                time.sleep(5)
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

    def round(self):
        index = self.state['round']
        base = self.state['best_checkpoint']
        samples = RETRY_SAMPLES if self.state['retrying'] else SAMPLES
        work = self.state_dir / f'round-{index}'
        work.mkdir(parents=True, exist_ok=True)
        dataset = work / 'targets.npz'
        candidate = work / 'distilled.pt'

        collect = (
            f'cd {CODE_DIR} && CUDA_VISIBLE_DEVICES=1 {PYTHON_BIN} '
            f'rl/search-distill-collect.py {base} --out {dataset} '
            f'--samples {samples} --lanes {LANES} --workers 8 '
            f'--horizon {HORIZON} --stride {STRIDE} --control-interval 1 '
            f'> {work}/collect.json'
        )
        if not self.run_stage(f'distill-collect-{index}', collect,
                              work / '.collect.done'):
            self.stop('NEEDS-ATTENTION', f'round {index} collect produced no marker')
            return

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

        after = self.evaluate(candidate, f'round-{index}-eval')
        if after is None:
            self.stop('NEEDS-ATTENTION', f'round {index} evaluation incomplete')
            return
        before = self.state['best_nats']
        gain = sum(before[b] for b in BANKS) / len(BANKS) - \
            sum(after[b] for b in BANKS) / len(BANKS)
        detail = ' '.join(
            f'{b}:{math.exp(-after[b]):.4f}' for b in BANKS)

        self.state['round'] += 1
        if gain >= MIN_NATS_GAIN:
            self.state.update(best_checkpoint=str(candidate), best_nats=after,
                              retrying=False)
            self.state['history'].append(
                {'round': index, 'gain_nats': gain, 'nats': after,
                 'checkpoint': str(candidate)})
            self.record(
                f'round {index}: {detail} | mean gain {gain:+.4f} nats '
                f'→ **ACCEPT** (samples={samples})')
        elif not self.state['retrying']:
            self.state['retrying'] = True
            self.record(
                f'round {index}: {detail} | mean gain {gain:+.4f} nats '
                f'→ **RETRY** at {RETRY_SAMPLES} samples (below '
                f'{MIN_NATS_GAIN} threshold)')
        else:
            self.state['retrying'] = False
            self.record(
                f'round {index}: {detail} | mean gain {gain:+.4f} nats '
                f'→ **PLATEAU** — two consecutive rounds below threshold')
            self.stop('PLATEAU', 'search distillation stopped improving; the '
                                 'ceiling curve says raise K or move to margin')
            return
        self.save()

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
        'retrying': False,
        'history': [],
        'started_at': time.time(),
        'done': False,
        'status': 'running',
    }, indent=2))
    results = state_dir / 'RESULTS.md'
    if not results.exists():
        results.write_text(
            '# Search-distillation loop\n\n'
            f'Accept a round when mean log-hazard over {list(BANKS)} improves by '
            f'>= {MIN_NATS_GAIN} nats/layer. Gating on the mean in nats because '
            'absolute survival swings 0.10 between banks (v22) while the '
            'log-hazard difference is stable to +/-0.017.\n\n')
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
