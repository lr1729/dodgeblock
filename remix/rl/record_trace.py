#!/usr/bin/env python3
"""Roll out a policy deterministically and dump the best episode as a
replayable action trace (seed is recomputable from bridge seed + env index,
so `render_trace.mjs` can reproduce the episode exactly in the JS sim)."""
import argparse
import base64
import json

import numpy as np
import torch

from ppo_v2 import (
    ActorCriticNetwork,
    load_agent_state,
    AutoregressiveActionDistribution,
    STICKY_MODEL_ARCHITECTURE,
    StickyActorCriticNetwork,
    packet_observation,
    tensor_observation,
)
from v2_bridge import ParallelEnvBridge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--episodes', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0xD06E_B10C)
    parser.add_argument('--target-height', type=float, default=10_000)
    parser.add_argument('--max-frames', type=int, default=100_000)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--stochastic', action='store_true')
    parser.add_argument('--out', default='trace.json')
    parser.add_argument('--dump-all', help='also write every episode as JSONL '
                                           '(deaths included, for autopsy)')
    args = parser.parse_args()

    device = torch.device(args.device)
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    control_interval = int(saved.get('args', {}).get('fixed_control_interval', 0)) or 1
    network_class = (
        StickyActorCriticNetwork
        if saved.get('model_architecture') == STICKY_MODEL_ARCHITECTURE
        else ActorCriticNetwork
    )
    agent = network_class().to(device)
    load_agent_state(agent, saved['agent'])
    agent.eval()

    bridge = ParallelEnvBridge(
        1, args.episodes, args.seed,
        target_height=args.target_height,
        reward_mode='target',
    )
    packet = bridge.read()
    active = np.ones(args.episodes, dtype=bool)
    heights = np.zeros(args.episodes, np.float32)
    lengths = np.zeros(args.episodes, np.int64)
    outcomes = np.full(args.episodes, 'timeout', dtype=object)
    held_actions = np.zeros(args.episodes, dtype=np.uint8)
    action_rows = []

    try:
        for frame in range(args.max_frames):
            if frame % control_interval == 0:
                observation = tensor_observation(packet_observation(packet), device)
                with torch.inference_mode(), torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == 'cuda',
                ):
                    logits, _value, _hazard = agent(observation)
                    distribution = AutoregressiveActionDistribution(logits, observation)
                    sampled = (
                        distribution.sample() if args.stochastic
                        else distribution.mode()
                    )
                    held_actions = sampled.cpu().numpy().astype(np.uint8)
            actions = held_actions.copy()
            action_rows.append(actions.copy())
            actions[~active] = 254
            packet = bridge.step(actions)
            lengths[active] += 1

            succeeded = active & packet['successes'].astype(bool)
            heights[succeeded] = packet['heights'][succeeded]
            outcomes[succeeded] = 'success'
            active[succeeded] = False
            died = active & packet['dones'].astype(bool)
            heights[died] = packet['heights'][died]
            outcomes[died] = 'death'
            active[died] = False
            if not active.any():
                break
        heights[active] = packet['current_heights'][active]
    finally:
        bridge.close()

    best = int(np.argmax(heights))
    length = int(lengths[best])
    actions_bytes = bytes(bytearray(int(row[best]) for row in action_rows[:length]))
    trace = {
        'checkpoint': args.checkpoint,
        'bridge_seed': args.seed,
        'env_index': best,
        'episode': 0,
        'stochastic': args.stochastic,
        'frames': length,
        'height': float(heights[best]),
        'outcome': str(outcomes[best]),
        'target_height': args.target_height,
        'actions': base64.b64encode(actions_bytes).decode(),
        'height_percentiles': {
            'median': float(np.median(heights)),
            'p90': float(np.percentile(heights, 90)),
            'max': float(heights.max()),
        },
    }
    if args.dump_all:
        with open(args.dump_all, 'w') as handle:
            for index in range(args.episodes):
                length = int(lengths[index])
                actions_bytes = bytes(bytearray(
                    int(row[index]) for row in action_rows[:length]))
                handle.write(json.dumps({
                    'bridge_seed': args.seed,
                    'env_index': index,
                    'episode': 0,
                    'frames': length,
                    'height': float(heights[index]),
                    'outcome': str(outcomes[index]),
                    'actions': base64.b64encode(actions_bytes).decode(),
                }) + '\n')

    with open(args.out, 'w') as handle:
        json.dump(trace, handle)
    print(json.dumps({key: trace[key] for key in
                      ('env_index', 'frames', 'height', 'outcome', 'height_percentiles')}))


if __name__ == '__main__':
    main()
