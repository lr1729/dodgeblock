#!/usr/bin/env python3
import argparse
import json

import numpy as np
import torch

from evaluate_v2 import ACTION_NAMES, bootstrap_interval
from ppo_v2 import (
    ActorCriticNetwork,
    AutoregressiveActionDistribution,
    STICKY_MODEL_ARCHITECTURE,
    StickyActorCriticNetwork,
    packet_observation,
    tensor_observation,
)
from r2d2 import interquartile_mean
from v2_bridge import ParallelEnvBridge


def main():
    parser = argparse.ArgumentParser(description='Evaluate a DodgeBlock v2 PPO policy.')
    parser.add_argument('checkpoint')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--episodes', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0xD06E_B10C)
    parser.add_argument('--target-height', type=float, default=10_000)
    parser.add_argument('--max-frames', type=int, default=300_000)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--stochastic', action='store_true')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--death-case-dir', default='')
    parser.add_argument(
        '--control-interval',
        type=int,
        default=0,
        help='Physics frames per policy decision; zero uses the checkpoint contract.',
    )
    args = parser.parse_args()
    if args.episodes % args.workers:
        raise ValueError('--episodes must be divisible by --workers')

    device = torch.device(args.device)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_interval = int(saved.get('args', {}).get('fixed_control_interval', 0))
    control_interval = args.control_interval or saved_interval or 1
    if control_interval <= 0:
        raise ValueError('--control-interval must be positive')
    network_class = (
        StickyActorCriticNetwork
        if saved.get('model_architecture') == STICKY_MODEL_ARCHITECTURE
        else ActorCriticNetwork
    )
    agent = network_class().to(device)
    agent.load_state_dict(saved['agent'])
    agent.eval()
    bridge = ParallelEnvBridge(
        args.workers,
        args.episodes // args.workers,
        args.seed,
        target_height=args.target_height,
        reward_mode='target',
        death_case_dir=args.death_case_dir,
    )
    packet = bridge.read()
    active = np.ones(args.episodes, dtype=bool)
    heights = np.zeros(args.episodes, np.float32)
    lengths = np.zeros(args.episodes, np.int64)
    outcomes = np.full(args.episodes, 'timeout', dtype=object)
    action_counts = np.zeros(len(ACTION_NAMES), np.int64)
    decision_counts = np.zeros(len(ACTION_NAMES), np.int64)
    held_actions = np.zeros(args.episodes, dtype=np.uint8)
    amp = not args.no_amp

    try:
        for _frame in range(args.max_frames):
            if _frame % control_interval == 0:
                observation = tensor_observation(packet_observation(packet), device)
                with torch.inference_mode(), torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=amp and device.type == 'cuda',
                ):
                    logits, _value = agent(observation)
                    distribution = AutoregressiveActionDistribution(
                        logits,
                        observation,
                    )
                    if args.stochastic:
                        held_actions = (
                            distribution.sample().cpu().numpy().astype(np.uint8)
                        )
                    else:
                        held_actions = (
                            distribution.mode().cpu().numpy().astype(np.uint8)
                        )
                decision_counts += np.bincount(
                    held_actions[active],
                    minlength=len(ACTION_NAMES),
                )
            actions = held_actions.copy()
            action_counts += np.bincount(actions[active], minlength=len(ACTION_NAMES))
            actions[~active] = 254
            packet = bridge.step(actions)
            lengths[active] += 1

            succeeded = active & packet['successes'].astype(bool)
            heights[succeeded] = packet['heights'][succeeded]
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
    print(json.dumps({
        'event': 'evaluation',
        'checkpoint': args.checkpoint,
        'training_frames': int(saved.get('frames', 0)),
        'episodes': args.episodes,
        'seed': args.seed,
        'stochastic': args.stochastic,
        'control_interval': control_interval,
        'target_height': args.target_height,
        'target_success': round(float(np.mean(outcomes == 'success')), 4),
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
        'action_fractions': {
            name: round(float(count / max(1, action_counts.sum())), 4)
            for name, count in zip(ACTION_NAMES, action_counts)
            if count
        },
        'decision_fractions': {
            name: round(float(count / max(1, decision_counts.sum())), 4)
            for name, count in zip(ACTION_NAMES, decision_counts)
            if count
        },
    }, indent=2))


if __name__ == '__main__':
    main()
