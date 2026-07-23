#!/usr/bin/env python3
import argparse
from collections import deque
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch

from r2d2 import (
    PrioritizedSequenceReplay,
    PersistentEpsilonExplorer,
    RecurrentQNetwork,
    SequenceAssembler,
    ema_update,
    exploration_rates,
    greedy_actions,
    learn_batch,
    sample_valid_actions,
    tensor_observation,
)
from v2_bridge import ParallelEnvBridge


def arguments():
    parser = argparse.ArgumentParser(description='Train the DodgeBlock recurrent replay agent.')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--envs-per-worker', type=int, default=64)
    parser.add_argument('--total-frames', type=int, default=2_000_000_000)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--hidden-size', type=int, default=512)
    parser.add_argument('--quantiles', type=int, default=51)
    parser.add_argument('--burn-in', type=int, default=40)
    parser.add_argument('--unroll', type=int, default=80)
    parser.add_argument('--n-step', type=int, default=20)
    parser.add_argument('--gamma', type=float, default=0.99999)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--replay-capacity', type=int, default=32768)
    parser.add_argument('--minimum-replay', type=int, default=1024)
    parser.add_argument('--replay-ratio', type=float, default=2.0)
    parser.add_argument('--priority-alpha', type=float, default=0.6)
    parser.add_argument('--priority-beta-start', type=float, default=0.4)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--target-tau', type=float, default=0.002)
    parser.add_argument('--exploration-hold', type=int, default=4)
    parser.add_argument('--epsilon-max', type=float, default=0.6)
    parser.add_argument('--epsilon-min', type=float, default=0.02)
    parser.add_argument('--random-warmup-frames', type=int, default=1_000_000)
    parser.add_argument('--learning-start-frames', type=int, default=1_000_000)
    parser.add_argument('--death-penalty', type=float, default=1.0)
    parser.add_argument('--checkpoint-dir', default=str(Path.home() / 'dodgeblock-r2d2/checkpoints'))
    parser.add_argument('--checkpoint-interval', type=int, default=10_000_000)
    parser.add_argument('--log-interval', type=float, default=20.0)
    parser.add_argument('--resume')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--compile-mode', default='default')
    return parser.parse_args()


def atomic_checkpoint(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)
    latest = path.parent / 'latest.pt'
    latest.unlink(missing_ok=True)
    latest.symlink_to(path.name)


def window_stats(records):
    if not records:
        return {}
    heights = np.asarray([item[0] for item in records], np.float64)
    lengths = np.asarray([item[1] for item in records], np.float64)
    return {
        'episodes': len(records),
        'mean_height': round(float(heights.mean()), 1),
        'median_height': round(float(np.median(heights)), 1),
        'p90_height': round(float(np.percentile(heights, 90)), 1),
        'max_height': round(float(heights.max()), 1),
        'mean_length': round(float(lengths.mean()), 1),
        'success_1k': round(float(np.mean(heights >= 1_000)), 3),
        'success_2_5k': round(float(np.mean(heights >= 2_500)), 3),
        'success_5k': round(float(np.mean(heights >= 5_000)), 3),
        'success_10k': round(float(np.mean(heights >= 10_000)), 3),
    }


def main():
    args = arguments()
    if not 0 < args.gamma <= 1:
        raise ValueError('--gamma must be in (0, 1]')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')

    online = RecurrentQNetwork(args.hidden_size, args.quantiles).to(device)
    target = RecurrentQNetwork(args.hidden_size, args.quantiles).to(device)
    target.load_state_dict(online.state_dict())
    target.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        online.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, eps=1e-5,
    )
    frames = 0
    learner_updates = 0
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        online.load_state_dict(saved['online'])
        target.load_state_dict(saved['target'])
        optimizer.load_state_dict(saved['optimizer'])
        frames = int(saved['frames'])
        learner_updates = int(saved.get('learner_updates', 0))
    if args.compile:
        online.compile(mode=args.compile_mode, dynamic=False)
        target.compile(mode=args.compile_mode, dynamic=False)

    env_count = args.workers * args.envs_per_worker
    bridge = ParallelEnvBridge(
        args.workers,
        args.envs_per_worker,
        args.seed,
        death_penalty=args.death_penalty,
    )
    assembler = SequenceAssembler(env_count, args.burn_in, args.unroll, args.n_step)
    replay = PrioritizedSequenceReplay(args.replay_capacity, args.priority_alpha)
    replay_rng = np.random.default_rng(args.seed ^ 0x5EED5EED)
    actor_rng = np.random.default_rng(args.seed ^ 0xAC710)
    epsilons = exploration_rates(env_count, args.epsilon_max, args.epsilon_min)
    explorer = PersistentEpsilonExplorer(epsilons, args.exploration_hold, actor_rng)
    fresh_episodes = deque(maxlen=500)
    curriculum_episodes = deque(maxlen=500)
    recent_losses = deque(maxlen=200)
    recent_grad_norms = deque(maxlen=200)
    recent_q = deque(maxlen=200)
    timing = {name: 0.0 for name in ('inference', 'environment', 'assembly', 'sampling', 'learning')}
    update_budget = 0.0
    stop = False
    checkpoint_dir = Path(args.checkpoint_dir)
    next_checkpoint = ((frames // args.checkpoint_interval) + 1) * args.checkpoint_interval

    def request_stop(_signum, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    packet = bridge.read()
    assembler.initialize(packet)
    hidden = online.initial_hidden(env_count, device)
    started = time.time()
    last_log_time = started
    last_log_frames = frames
    amp = not args.no_amp
    parameter_count = sum(parameter.numel() for parameter in online.parameters())
    half_life = None if args.gamma == 1 else round(np.log(0.5) / np.log(args.gamma), 1)
    print(json.dumps({
        'event': 'start',
        'frames': frames,
        'device': str(device),
        'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'envs': env_count,
        'workers': args.workers,
        'parameters': parameter_count,
        'gamma_per_world_frame': args.gamma,
        'discount_half_life_world_frames': half_life,
        'n_step': args.n_step,
        'replay_capacity': args.replay_capacity,
        'exploration_hold': args.exploration_hold,
        'epsilon_max': args.epsilon_max,
        'epsilon_min': args.epsilon_min,
        'random_warmup_frames': args.random_warmup_frames,
        'learning_start_frames': args.learning_start_frames,
        'death_penalty': args.death_penalty,
        'compiled': args.compile,
    }), flush=True)

    try:
        while frames < args.total_frames and not stop:
            phase_started = time.perf_counter()
            online.eval()
            if frames < args.random_warmup_frames:
                actions = sample_valid_actions(packet['state'], actor_rng)
            else:
                observation = tensor_observation({
                    key: packet[key] for key in ('terrain', 'skyline', 'falling', 'forecasts', 'state')
                }, device)
                with torch.inference_mode(), torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=amp and device.type == 'cuda',
                ):
                    quantiles, hidden = online(observation, hidden)
                    greedy = greedy_actions(quantiles, observation).cpu().numpy()
                actions = explorer.select(greedy, packet['state'])
            timing['inference'] += time.perf_counter() - phase_started
            phase_started = time.perf_counter()
            packet = bridge.step(actions)
            explorer.reset(packet['dones'])
            timing['environment'] += time.perf_counter() - phase_started
            frames += env_count
            done_indices = np.flatnonzero(packet['dones'])
            if len(done_indices):
                hidden = hidden.clone()
                hidden[:, done_indices] = 0
                for index in done_indices:
                    record = (float(packet['heights'][index]), int(packet['lengths'][index]))
                    if packet['episode_starts'][index] > 0:
                        curriculum_episodes.append(record)
                    else:
                        fresh_episodes.append(record)

            phase_started = time.perf_counter()
            for sequence in assembler.append(actions, packet):
                replay.add(sequence)
            timing['assembly'] += time.perf_counter() - phase_started

            if len(replay) >= args.minimum_replay and frames >= args.learning_start_frames:
                update_budget += env_count * args.replay_ratio / (args.batch_size * args.unroll)
                while update_budget >= 1:
                    progress = min(1.0, frames / max(1, args.total_frames))
                    beta = args.priority_beta_start + (1 - args.priority_beta_start) * progress
                    phase_started = time.perf_counter()
                    indices, weights, batch = replay.sample(args.batch_size, beta, replay_rng)
                    timing['sampling'] += time.perf_counter() - phase_started
                    online.train()
                    phase_started = time.perf_counter()
                    result = learn_batch(
                        online,
                        target,
                        optimizer,
                        batch,
                        weights,
                        args.burn_in,
                        args.unroll,
                        args.n_step,
                        args.gamma,
                        device,
                        amp=amp,
                    )
                    timing['learning'] += time.perf_counter() - phase_started
                    replay.update_priorities(indices, result['priorities'])
                    ema_update(target, online, args.target_tau)
                    recent_losses.append(result['loss'])
                    recent_grad_norms.append(result['grad_norm'])
                    recent_q.append(result['q_mean'])
                    learner_updates += 1
                    update_budget -= 1

            now = time.time()
            if now - last_log_time >= args.log_interval:
                interval_sps = (frames - last_log_frames) / (now - last_log_time)
                stats = {
                    'event': 'progress',
                    'frames': frames,
                    'sps': round(interval_sps, 1),
                    'updates': learner_updates,
                    'replay': len(replay),
                    'loss': round(float(np.mean(recent_losses)), 5) if recent_losses else None,
                    'grad_norm': round(float(np.mean(recent_grad_norms)), 3) if recent_grad_norms else None,
                    'q_mean': round(float(np.mean(recent_q)), 3) if recent_q else None,
                    'fresh': window_stats(fresh_episodes),
                    'curriculum': window_stats(curriculum_episodes),
                }
                if device.type == 'cuda':
                    stats['gpu_memory_gib'] = {
                        'allocated': round(torch.cuda.memory_allocated(device) / 2**30, 2),
                        'reserved': round(torch.cuda.memory_reserved(device) / 2**30, 2),
                        'peak': round(torch.cuda.max_memory_allocated(device) / 2**30, 2),
                    }
                timed = sum(timing.values())
                stats['timing'] = {
                    name: round(value / timed, 3) if timed else 0
                    for name, value in timing.items()
                }
                print(json.dumps(stats), flush=True)
                last_log_time = now
                last_log_frames = frames
                timing = {name: 0.0 for name in timing}

            if frames >= next_checkpoint:
                path = checkpoint_dir / f'r2d2-{frames:012d}.pt'
                atomic_checkpoint(path, {
                    'online': online.state_dict(),
                    'target': target.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'frames': frames,
                    'learner_updates': learner_updates,
                    'args': vars(args),
                })
                print(json.dumps({'event': 'checkpoint', 'frames': frames, 'path': str(path)}), flush=True)
                next_checkpoint += args.checkpoint_interval
    finally:
        final_path = checkpoint_dir / f'r2d2-{frames:012d}.pt'
        atomic_checkpoint(final_path, {
            'online': online.state_dict(),
            'target': target.state_dict(),
            'optimizer': optimizer.state_dict(),
            'frames': frames,
            'learner_updates': learner_updates,
            'args': vars(args),
        })
        bridge.close()
        elapsed = time.time() - started
        print(json.dumps({
            'event': 'stop', 'frames': frames, 'updates': learner_updates,
            'elapsed_seconds': round(elapsed, 1), 'checkpoint': str(final_path),
        }), flush=True)


if __name__ == '__main__':
    main()
