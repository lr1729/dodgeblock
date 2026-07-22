#!/usr/bin/env python3
import argparse
import json

import numpy as np
import torch

from r2d2 import RecurrentQNetwork, interquartile_mean, tensor_observation
from v2_bridge import ParallelEnvBridge


OBSERVATION_KEYS = ('terrain', 'skyline', 'falling', 'forecasts', 'state')


def bootstrap_interval(values, statistic, rng, samples=2_000):
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    estimates = np.asarray([statistic(values[index]) for index in indices])
    low, high = np.percentile(estimates, (2.5, 97.5))
    return [round(float(low), 2), round(float(high), 2)]


def main():
    parser = argparse.ArgumentParser(description='Evaluate a DodgeBlock v2 policy on held-out seeds.')
    parser.add_argument('checkpoint')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--episodes', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0xD06E_B10C)
    parser.add_argument('--target-height', type=float, default=10_000)
    parser.add_argument('--max-frames', type=int, default=300_000)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--no-amp', action='store_true')
    args = parser.parse_args()
    if args.episodes % args.workers:
        raise ValueError('--episodes must be divisible by --workers')

    device = torch.device(args.device)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = saved.get('args', {})
    model = RecurrentQNetwork(config.get('hidden_size', 512), config.get('quantiles', 51)).to(device)
    model.load_state_dict(saved['online'])
    model.eval()
    bridge = ParallelEnvBridge(args.workers, args.episodes // args.workers, args.seed, archive_probability=0)
    packet = bridge.read()
    hidden = model.initial_hidden(args.episodes, device)
    active = np.ones(args.episodes, dtype=bool)
    heights = np.zeros(args.episodes, np.float32)
    lengths = np.zeros(args.episodes, np.int64)
    outcomes = np.full(args.episodes, 'timeout', dtype=object)
    amp = not args.no_amp

    try:
        for _frame in range(args.max_frames):
            observation = tensor_observation({key: packet[key] for key in OBSERVATION_KEYS}, device)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp and device.type == 'cuda',
            ):
                quantiles, hidden = model(observation, hidden)
                actions = quantiles.float().mean(dim=-1).argmax(dim=-1).cpu().numpy().astype(np.uint8)
            actions[~active] = 0
            packet = bridge.step(actions)
            lengths[active] += 1

            succeeded = active & (packet['current_heights'] >= args.target_height)
            heights[succeeded] = packet['current_heights'][succeeded]
            outcomes[succeeded] = 'success'
            active[succeeded] = False

            died = active & packet['dones'].astype(bool)
            heights[died] = packet['heights'][died]
            lengths[died] = packet['lengths'][died]
            outcomes[died] = 'death'
            active[died] = False
            if not active.any():
                break

        heights[active] = packet['current_heights'][active]
    finally:
        bridge.close()

    bootstrap_rng = np.random.default_rng(args.seed ^ 0xB0057A9)
    result = {
        'checkpoint': args.checkpoint,
        'training_frames': int(saved.get('frames', 0)),
        'episodes': args.episodes,
        'seed': args.seed,
        'mean_height': round(float(heights.mean()), 2),
        'mean_height_ci95': bootstrap_interval(heights, np.mean, bootstrap_rng),
        'median_height': round(float(np.median(heights)), 2),
        'iqm_height': round(interquartile_mean(heights), 2),
        'iqm_height_ci95': bootstrap_interval(heights, interquartile_mean, bootstrap_rng),
        'p90_height': round(float(np.percentile(heights, 90)), 2),
        'max_height': round(float(heights.max()), 2),
        'mean_length': round(float(lengths.mean()), 2),
        'success_1k': round(float(np.mean(heights >= 1_000)), 4),
        'success_2_5k': round(float(np.mean(heights >= 2_500)), 4),
        'success_5k': round(float(np.mean(heights >= 5_000)), 4),
        'success_10k': round(float(np.mean(heights >= 10_000)), 4),
        'deaths': int(np.sum(outcomes == 'death')),
        'timeouts': int(np.sum(outcomes == 'timeout')),
    }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
