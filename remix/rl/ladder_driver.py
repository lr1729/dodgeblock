#!/usr/bin/env python3
"""Unattended driver for the v6 fresh-goal PPO ladder.

Chains success-target PPO stages (rungs) overnight: waits for the current
rung's systemd unit to exit, reads the deterministic post-train eval from its
run.log, applies pre-registered gates, and launches the next rung initialized
from the previous actor. Registered rules (RESEARCH-LOG.md, 2026-07-24):

  det target_success >= 0.50                 -> promote to next rung
  0.10 <= det < 0.50                         -> extend once (+base frames);
                                                after extension >= 0.35 promotes
  det < 0.10 and stochastic (384 ep) >= 0.30 -> extend once (det degeneracy)
  det < 0.10 and stochastic < 0.30           -> refine: geometric-midpoint rung,
                                                initialized from the failed actor
  refined rung within 15% of last pass       -> stop, NEEDS-ATTENTION
  unit exit without an eval event            -> relaunch same rung (auto-resume),
                                                max 2 retries

State lives in <state-dir>/ladder-state.json; every decision is appended to
<state-dir>/RESULTS.md. The driver never kills a running unit.
"""
import argparse
import json
import math
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

RUNGS = [500, 1000, 1750, 2500, 4000, 6500, 10000]
PROMOTE = 0.50
EXTEND_LOW = 0.10
POST_EXTEND_PROMOTE = 0.35
STOCH_RESCUE = 0.30
MIN_RUNG_RATIO = 1.15
MAX_LAUNCHES = 14
MAX_RETRIES = 2
WALL_HOURS = 26.0
POLL_SECONDS = 60
DET_EVAL_EPISODES = 512
STOCH_EVAL_EPISODES = 384

CODE_DIR = Path.home() / 'dodgeblock-v5-code'
PYTHON_BIN = Path.home() / 'envs/dodgeblock-rl/bin/python'

TRAIN_ENV = (
    'CUDA_VISIBLE_DEVICES=1 '
    f'DODGEBLOCK_PYTHON={PYTHON_BIN} '
    'DODGEBLOCK_WORKERS=8 DODGEBLOCK_ENVS_PER_WORKER=64 '
    'DODGEBLOCK_REWARD_MODE=target DODGEBLOCK_CELL_EVAL_ENVS=0 '
    'DODGEBLOCK_CHECKPOINT_INTERVAL=2500000 '
    'DODGEBLOCK_EVAL_AFTER_TRAIN=1 '
    f'DODGEBLOCK_EVAL_EPISODES={DET_EVAL_EPISODES} '
    'DODGEBLOCK_DEVICE=cuda'
)


def now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def log(message):
    print(f'[{now()}] {message}', flush=True)


def base_frames(target):
    return 30_000_000 if target >= 4000 else 20_000_000


def round50(value):
    return int(round(value / 50.0) * 50)


def unit_running(unit):
    result = subprocess.run(
        ['systemctl', '--user', 'is-active', unit],
        capture_output=True, text=True,
    )
    return result.stdout.strip() in ('active', 'activating', 'reloading')


def parse_last_event(path, event):
    """Return the last (possibly pretty-printed) JSON object in `path` whose
    first key marks it as `event`, or None."""
    try:
        content = Path(path).read_text()
    except OSError:
        return None
    marker = f'"event": "{event}"'
    index = content.rfind(marker)
    if index < 0:
        return None
    start = content.rfind('{', 0, index)
    if start < 0:
        return None
    try:
        parsed, _ = json.JSONDecoder().raw_decode(content[start:])
        return parsed
    except json.JSONDecodeError:
        return None


class Ladder:
    def __init__(self, state_dir):
        self.state_dir = Path(state_dir)
        self.state_path = self.state_dir / 'ladder-state.json'
        self.results_path = self.state_dir / 'RESULTS.md'
        self.state = json.loads(self.state_path.read_text())

    def save(self):
        temporary = self.state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.state, indent=2))
        temporary.replace(self.state_path)

    def append_result(self, line):
        with self.results_path.open('a') as handle:
            handle.write(f'- {now()} {line}\n')
        log(f'RESULT {line}')

    def stop(self, status, reason):
        self.state['done'] = True
        self.state['status'] = status
        self.save()
        self.append_result(f'**{status}** — {reason}')
        log(f'ladder stopped: {status} ({reason})')

    # ---- launching ------------------------------------------------------

    def launch(self, target, run_dir, total_frames, init_from=None):
        if self.state['launches'] >= MAX_LAUNCHES:
            self.stop('NEEDS-ATTENTION', f'launch cap {MAX_LAUNCHES} reached')
            return False
        elapsed_hours = (time.time() - self.state['started_at']) / 3600.0
        if elapsed_hours > WALL_HOURS:
            self.stop('WALL-REACHED', f'{elapsed_hours:.1f}h elapsed, no new launches')
            return False
        run_dir = Path(run_dir)
        (run_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
        unit = f"dodgeblock-ladder-t{target}-n{self.state['launches']}"
        parts = [
            f'cd {CODE_DIR} &&',
            TRAIN_ENV,
            f'DODGEBLOCK_TOTAL_FRAMES={total_frames}',
            f'DODGEBLOCK_TARGET_HEIGHT={target}',
            f'DODGEBLOCK_CHECKPOINT_DIR={run_dir}/checkpoints',
        ]
        if init_from:
            parts.append(f'DODGEBLOCK_INITIALIZE_FROM={init_from}')
        parts.append(f'rl/run-ppo-v4.sh >> {run_dir}/run.log 2>&1')
        command = ' '.join(parts)
        result = subprocess.run(
            ['systemd-run', '--user', '--collect', f'--unit={unit}',
             '/usr/bin/bash', '-lc', command],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            self.stop('NEEDS-ATTENTION', f'systemd-run failed: {result.stderr.strip()}')
            return False
        self.state['launches'] += 1
        self.state['current'] = {
            'target': target,
            'dir': str(run_dir),
            'unit': unit,
            'total_frames': total_frames,
            'extended': self.state.get('current', {}).get('extended', False)
            if str(run_dir) == self.state.get('current', {}).get('dir') else False,
            'retries': self.state.get('current', {}).get('retries', 0)
            if str(run_dir) == self.state.get('current', {}).get('dir') else 0,
        }
        self.save()
        log(f'launched {unit}: target={target} frames={total_frames} '
            f'dir={run_dir} init={init_from or "resume"}')
        return True

    def launch_next_pending(self, init_from):
        target = self.state['pending'][0]
        run_dir = self.state_dir / f"rung-{target}-n{self.state['launches']}"
        return self.launch(target, run_dir, base_frames(target), init_from=init_from)

    # ---- evaluation -----------------------------------------------------

    def stochastic_success(self, checkpoint, target):
        log(f'running stochastic eval: {checkpoint} target={target}')
        result = subprocess.run(
            [str(PYTHON_BIN), 'rl/evaluate_ppo_v2.py', str(checkpoint),
             '--workers', '8', '--episodes', str(STOCH_EVAL_EPISODES),
             '--target-height', str(target), '--device', 'cuda', '--stochastic'],
            capture_output=True, text=True, cwd=CODE_DIR,
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': '1'},
        )
        index = result.stdout.rfind('"event": "evaluation"')
        if index < 0:
            log(f'stochastic eval produced no result: {result.stderr[-500:]}')
            return None
        start = result.stdout.rfind('{', 0, index)
        parsed, _ = json.JSONDecoder().raw_decode(result.stdout[start:])
        return parsed.get('target_success')

    # ---- gate logic -----------------------------------------------------

    def resolve_current(self):
        current = self.state['current']
        target = current['target']
        run_dir = Path(current['dir'])
        run_log = run_dir / 'run.log'
        evaluation = parse_last_event(run_log, 'evaluation')
        checkpoint = run_dir / 'checkpoints/latest.pt'

        if evaluation is None or int(evaluation.get('training_frames', 0)) < current['total_frames']:
            if current['retries'] >= MAX_RETRIES:
                self.stop('NEEDS-ATTENTION',
                          f'rung {target}: no eval after {MAX_RETRIES} retries ({run_log})')
                return
            current['retries'] += 1
            self.save()
            log(f'rung {target}: unit exited without eval, retry {current["retries"]} (auto-resume)')
            self.launch(target, run_dir, current['total_frames'])
            return

        success = float(evaluation.get('target_success', 0.0))
        progress = parse_last_event(run_log, 'progress') or {}
        shelter = (progress.get('shelter_occupancy_by_phase') or {}).get('surge')
        summary = (
            f'rung {target}: det success={success:.3f} '
            f'median={evaluation.get("median_height")} p90={evaluation.get("p90_height")} '
            f'max={evaluation.get("max_height")} mean_len={evaluation.get("mean_length")} '
            f'shelter_surge={shelter} frames={current["total_frames"]}'
        )

        if success >= (POST_EXTEND_PROMOTE if current['extended'] else PROMOTE):
            self.promote(current, evaluation, summary)
            return

        if not current['extended']:
            if success >= EXTEND_LOW:
                self.extend(current, summary, f'det {success:.3f} in extend band')
                return
            stochastic = self.stochastic_success(checkpoint, target)
            if stochastic is not None and stochastic >= STOCH_RESCUE:
                self.extend(current, summary,
                            f'det {success:.3f} but stochastic {stochastic:.3f} (degeneracy)')
                return
            self.refine(current, summary,
                        f'det {success:.3f}, stochastic {stochastic}')
            return

        self.refine(current, summary, f'det {success:.3f} after extension')

    def promote(self, current, evaluation, summary):
        target = current['target']
        checkpoint = Path(current['dir']) / 'checkpoints/latest.pt'
        self.state['passed'].append({
            'target': target,
            'checkpoint': str(checkpoint.resolve()),
            'det_success': float(evaluation.get('target_success', 0.0)),
            'median_height': evaluation.get('median_height'),
            'p90_height': evaluation.get('p90_height'),
        })
        if target in self.state['pending']:
            self.state['pending'].remove(target)
        self.append_result(f'{summary} → **PROMOTE**')
        if not self.state['pending']:
            self.confirm_and_finish(checkpoint)
            return
        self.save()
        self.launch_next_pending(init_from=str(checkpoint.resolve()))

    def extend(self, current, summary, reason):
        current['extended'] = True
        current['retries'] = 0
        total = current['total_frames'] + base_frames(current['target'])
        self.append_result(f'{summary} → **EXTEND** to {total} ({reason})')
        self.save()
        self.launch(current['target'], current['dir'], total)

    def refine(self, current, summary, reason):
        failed_target = current['target']
        last_passed = self.state['passed'][-1]['target'] if self.state['passed'] else 200
        midpoint = round50(math.sqrt(last_passed * failed_target))
        if midpoint <= last_passed * MIN_RUNG_RATIO:
            self.append_result(f'{summary} → **STOP** ({reason})')
            self.stop('NEEDS-ATTENTION',
                      f'rung {failed_target} failed and midpoint {midpoint} too close '
                      f'to last pass {last_passed}')
            return
        if self.state['pending'] and self.state['pending'][0] == failed_target:
            self.state['pending'].insert(0, midpoint)
        else:
            self.state['pending'].insert(0, failed_target)
            self.state['pending'].insert(0, midpoint)
        self.append_result(f'{summary} → **REFINE** to rung {midpoint} ({reason})')
        self.save()
        failed_checkpoint = Path(current['dir']) / 'checkpoints/latest.pt'
        self.launch_next_pending(init_from=str(failed_checkpoint.resolve()))

    def confirm_and_finish(self, checkpoint):
        log('final rung passed; running 1024-episode confirmation eval')
        result = subprocess.run(
            [str(PYTHON_BIN), 'rl/evaluate_ppo_v2.py', str(checkpoint),
             '--workers', '8', '--episodes', '1024',
             '--target-height', '10000', '--device', 'cuda'],
            capture_output=True, text=True, cwd=CODE_DIR,
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': '1'},
        )
        (self.state_dir / 'final-eval.json').write_text(result.stdout)
        self.stop('LADDER-COMPLETE', f'confirmation eval saved (checkpoint {checkpoint})')

    # ---- main loop ------------------------------------------------------

    def run(self):
        log(f'driver up: pending={self.state["pending"]} '
            f'current={self.state["current"]["unit"]}')
        while not self.state['done']:
            current = self.state['current']
            if unit_running(current['unit']):
                time.sleep(POLL_SECONDS)
                continue
            log(f'unit {current["unit"]} exited; resolving')
            time.sleep(5)
            self.resolve_current()
        log('driver exiting')


def bootstrap(state_dir, current_target, current_dir, current_unit, current_frames):
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / 'ladder-state.json'
    if state_path.exists():
        log(f'state exists at {state_path}, not overwriting')
        return
    state = {
        'pending': [rung for rung in RUNGS if rung >= current_target],
        'passed': [],
        'current': {
            'target': current_target,
            'dir': current_dir,
            'unit': current_unit,
            'total_frames': current_frames,
            'extended': False,
            'retries': 0,
        },
        'launches': 0,
        'started_at': time.time(),
        'done': False,
        'status': 'running',
    }
    state_path.write_text(json.dumps(state, indent=2))
    results = state_dir / 'RESULTS.md'
    if not results.exists():
        results.write_text(
            '# Ladder results\n\n'
            f'Gates: promote det>={PROMOTE} (post-extend >={POST_EXTEND_PROMOTE}); '
            f'extend det in [{EXTEND_LOW},{PROMOTE}) or stochastic>={STOCH_RESCUE}; '
            'else refine to geometric midpoint; stop when midpoint within '
            f'{MIN_RUNG_RATIO}x of last pass.\n\n'
        )
    log(f'bootstrapped {state_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-dir', default=str(Path.home() / 'dodgeblock-ladder'))
    parser.add_argument('--bootstrap', action='store_true',
                        help='create initial state around an in-flight rung, then exit')
    parser.add_argument('--current-target', type=int, default=500)
    parser.add_argument('--current-dir', default=str(Path.home() / 'dodgeblock-ppo-stage-500-a'))
    parser.add_argument('--current-unit', default='dodgeblock-ppo-stage-500-a')
    parser.add_argument('--current-frames', type=int, default=20_000_000)
    parser.add_argument('--simulate', metavar='RUN_LOG',
                        help='parse RUN_LOG, print the eval + gate decision, exit')
    args = parser.parse_args()

    if args.simulate:
        evaluation = parse_last_event(args.simulate, 'evaluation')
        print(json.dumps(evaluation, indent=2) if evaluation else 'no evaluation event')
        if evaluation:
            success = float(evaluation.get('target_success', 0.0))
            if success >= PROMOTE:
                print(f'decision: PROMOTE ({success:.3f})')
            elif success >= EXTEND_LOW:
                print(f'decision: EXTEND ({success:.3f})')
            else:
                print(f'decision: stochastic eval then EXTEND or REFINE ({success:.3f})')
        return

    if args.bootstrap:
        bootstrap(args.state_dir, args.current_target, args.current_dir,
                  args.current_unit, args.current_frames)
        return

    Ladder(args.state_dir).run()


if __name__ == '__main__':
    main()
