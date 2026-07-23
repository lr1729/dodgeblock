#!/usr/bin/env python3
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import subprocess


def run_seed(args, seed):
    output = Path(args.output_dir) / f'seed-{seed}'
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / 'search-checkpoint.json.gz'
    command = [
        'node',
        str(Path(__file__).with_name('go-explore.mjs')),
        '--seed', str(seed),
        '--search-seed', str((seed ^ args.search_seed_salt) & 0xFFFF_FFFF),
        '--target-height', str(args.target_height),
        '--iterations', str(args.iterations),
        '--explore-frames', str(args.explore_frames),
        '--archive-capacity', str(args.archive_capacity),
        '--checkpoint-interval', str(args.checkpoint_interval),
        '--output-dir', str(output),
    ]
    if args.resume and checkpoint.exists():
        command.extend(['--resume', str(checkpoint)])
    with (output / 'explorer.log').open('a') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    return seed, result.returncode, checkpoint


def main():
    parser = argparse.ArgumentParser(
        description='Run independent machine-only Go-Explore cell-bank searches.',
    )
    parser.add_argument('--seed-start', type=int, default=1)
    parser.add_argument('--seeds', type=int, default=16)
    parser.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 4) // 2))
    parser.add_argument('--search-seed-salt', type=int, default=0x6F2D_4B19)
    parser.add_argument('--target-height', type=int, default=10_000)
    parser.add_argument('--iterations', type=int, default=2_000_000)
    parser.add_argument('--explore-frames', type=int, default=240)
    parser.add_argument('--archive-capacity', type=int, default=2048)
    parser.add_argument('--checkpoint-interval', type=int, default=50_000)
    parser.add_argument('--output-dir', default='rl/go-explore-bank')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    seeds = range(args.seed_start, args.seed_start + args.seeds)
    failures = []
    with ThreadPoolExecutor(max_workers=min(args.jobs, args.seeds)) as executor:
        futures = [executor.submit(run_seed, args, seed) for seed in seeds]
        for future in as_completed(futures):
            seed, returncode, checkpoint = future.result()
            status = 'complete' if returncode == 0 else 'search-exhausted'
            print(f'seed={seed} status={status} checkpoint={checkpoint}', flush=True)
            if not checkpoint.exists():
                failures.append(seed)
    if failures:
        raise SystemExit(f'missing checkpoints for seeds: {failures}')


if __name__ == '__main__':
    main()
