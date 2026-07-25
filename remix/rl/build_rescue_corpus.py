#!/usr/bin/env python3
"""Build a rescue-distillation corpus from a policy's own deaths.

Deaths captured by `evaluate_ppo_v2.py --death-case-dir` are rewound, searched
in parallel for an escape that survives the full horizon in the original future
AND in every reseeded future, then exported as a demo shard the trainer
consumes through `--demo-dataset`. Only verified escapes are exported; the
utility-scored soft-label family is deliberately not used (see RESEARCH-LOG).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def log(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def chunk(items, count):
    buckets = [[] for _ in range(count)]
    for index, item in enumerate(items):
        buckets[index % count].append(item)
    return [bucket for bucket in buckets if bucket]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deaths', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 4) - 4))
    parser.add_argument('--rewinds', default='30,60,120')
    parser.add_argument('--trials', type=int, default=128)
    parser.add_argument('--horizon', type=int, default=360)
    parser.add_argument('--futures', type=int, default=3)
    parser.add_argument('--prefix-frames', type=int, default=60)
    parser.add_argument('--shard-seed', type=int, default=1001)
    parser.add_argument('--limit', type=int, default=0, help='cap death cases (0 = all)')
    args = parser.parse_args()

    deaths = sorted(Path(args.deaths).glob('*.json.gz'))
    if args.limit:
        deaths = deaths[:args.limit]
    if not deaths:
        sys.exit(f'no death cases in {args.deaths}')

    output = Path(args.output)
    chunks_dir, reports_dir = output / 'chunks', output / 'reports'
    for directory in (chunks_dir, reports_dir):
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True)

    buckets = chunk(deaths, args.workers)
    log(f'{len(deaths)} deaths -> {len(buckets)} search workers '
        f'(rewinds {args.rewinds}, {args.trials} trials, {args.futures} futures)')

    processes = []
    for index, bucket in enumerate(buckets):
        bucket_dir = chunks_dir / f'chunk-{index:03d}'
        bucket_dir.mkdir()
        for case in bucket:
            (bucket_dir / case.name).symlink_to(case.resolve())
        processes.append(subprocess.Popen(
            ['node', str(ROOT / 'rl/rescue-oracle.mjs'),
             '--case', str(bucket_dir),
             '--rewinds', args.rewinds,
             '--trials', str(args.trials),
             '--horizon', str(args.horizon),
             '--futures', str(args.futures),
             '--seed', str(0x5E5C0E17 + index),
             '--output', str(reports_dir / f'report-{index:03d}.json')],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE))

    started = time.perf_counter()
    for index, process in enumerate(processes):
        _out, error = process.communicate()
        if process.returncode:
            sys.exit(f'search worker {index} failed: {error.decode()[-400:]}')
    log(f'search finished in {time.perf_counter() - started:.0f}s')

    # Aggregate the rescue-rate curves at the full trial budget.
    totals = {}
    robust_cases = 0
    for report_path in sorted(reports_dir.glob('*.json')):
        report = json.loads(report_path.read_text())
        robust_cases += sum(1 for case in report['cases'] if case.get('robust'))
        for curve in report['curves']:
            if curve['budget'] != max(c['budget'] for c in report['curves']):
                continue
            entry = totals.setdefault(curve['rewind'], {'cases': 0, 'original': 0.0, 'robust': 0.0})
            entry['cases'] += curve['cases']
            entry['original'] += curve['originalRescueRate'] * curve['cases']
            entry['robust'] += curve['robustRescueRate'] * curve['cases']
    for rewind in sorted(totals):
        entry = totals[rewind]
        log(f'  rewind {rewind:4d}: original {entry["original"] / entry["cases"]:.3f} '
            f'robust {entry["robust"] / entry["cases"]:.3f}  (n={entry["cases"]})')
    log(f'verified escapes available: {robust_cases}')

    shard_dir = output / 'shard'
    shutil.rmtree(shard_dir, ignore_errors=True)
    result = subprocess.run(
        ['node', str(ROOT / 'rl/export-oracle-corrections.mjs'),
         '--oracle', str(reports_dir),
         '--output-dir', str(shard_dir),
         '--shard-seed', str(args.shard_seed),
         '--prefix-frames', str(args.prefix_frames)],
        cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        sys.exit(f'export failed: {result.stderr[-600:]}')

    manifest = json.loads((shard_dir / f'seed-{args.shard_seed}' / 'manifest.json').read_text())
    log(f'shard: {manifest["frames"]} frames from {manifest["correctionCases"]} escapes '
        f'({manifest.get("skippedRecords", 0)} skipped on re-verification)')
    log(f'train with: --demo-dataset {shard_dir} --demo-seeds {args.shard_seed}')


if __name__ == '__main__':
    main()
