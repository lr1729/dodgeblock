#!/usr/bin/env python3
"""Per-layer survival in the regime the goal actually lives in.

The hazard curve showed this policy dies at a mean of 43.5 seconds and has
exactly zero exposure past the 240-second point where difficulty saturates. So
the quantity that decides whether 10k is reachable cannot be measured by rolling
the policy forward -- it never gets there.

It can be measured by *starting* there. The go-explore banks hold 512 snapshots
per seed spanning frames 0 to 22,315, with a median of 18,531 (309 seconds), and
the env restores a chosen cell exactly. Dropping the policy into a saturated-
difficulty state and measuring how long it lasts gives the number directly.

One honest caveat, stated up front: these states were built by a search, not by
this policy, so the pile configuration is off-distribution for it. That biases the
result DOWNWARD -- part of any measured hazard is unfamiliarity rather than
difficulty. It is therefore a lower bound on achievable survival, which is the
useful direction: if survival here is already close to what is needed, the goal is
live; if it is far below, the gap is at least that large.

Reported against the references the ledger uses: 0.928 measured in minute one,
0.9567 needed to reach 10k at all, 0.9972 for consistent 10k.
"""
import argparse
import gzip
import json

import numpy as np
import torch

from ppo_v2 import (
    ActorCriticNetwork,
    checkpoint_control_interval,
    AutoregressiveActionDistribution,
    STICKY_MODEL_ARCHITECTURE,
    StickyActorCriticNetwork,
    load_agent_state,
    packet_observation,
    tensor_observation,
)
from v2_bridge import ParallelEnvBridge

LAYER = 40.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--bank', default='/home/lr/dodgeblock-go-explore-bank-v4/seed-1/search-checkpoint.json.gz')
    parser.add_argument('--envs', type=int, default=128)
    parser.add_argument('--rounds', type=int, default=4,
                        help='how many times to refill all lanes with fresh cells')
    parser.add_argument('--max-frames', type=int, default=6000)
    parser.add_argument('--seed', type=int, default=0x5A7_0BE5)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--stochastic', action='store_true')
    args = parser.parse_args()

    with gzip.open(args.bank) as handle:
        entries = json.load(handle)['entries']
    # Variant ids are indices into the bank's entry list, in order.
    banded = {
        'fresh_0_60s': [i for i, e in enumerate(entries) if e['frame'] < 3600],
        'ramp_60_240s': [i for i, e in enumerate(entries) if 3600 <= e['frame'] < 14400],
        'saturated_240s_plus': [i for i, e in enumerate(entries) if e['frame'] >= 14400],
    }
    heights = {i: float(e['height']) for i, e in enumerate(entries)}

    device = torch.device(args.device)
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    network_class = (
        StickyActorCriticNetwork
        if saved.get('model_architecture') == STICKY_MODEL_ARCHITECTURE
        else ActorCriticNetwork
    )
    agent = network_class().to(device)
    load_agent_state(agent, saved['agent'])
    agent.eval()
    interval = checkpoint_control_interval(saved)

    # The bank declares targetHeight 10000 and the env refuses a mismatch; that
    # also guarantees no episode ends by succeeding, so the estimate is uncensored.
    bridge = ParallelEnvBridge(
        8, max(1, args.envs // 8), args.seed,
        target_height=10_000,
        reward_mode='target',
        cell_banks=[args.bank],
    )
    rng = np.random.default_rng(0)
    report = {}

    try:
        for band, pool in banded.items():
            if len(pool) < 8:
                report[band] = {'cells': len(pool), 'skipped': True}
                continue
            layers = 0.0
            deaths = 0
            frames_alive = 0
            for _round in range(args.rounds):
                chosen = rng.choice(pool, size=args.envs, replace=True)
                packet = bridge.reset(np.asarray(chosen, np.int32))
                start = np.array([heights[int(c)] for c in chosen], np.float64)
                peak = start.copy()
                active = np.ones(args.envs, dtype=bool)
                held = np.zeros(args.envs, np.uint8)
                for frame in range(args.max_frames):
                    # Hold each action for the interval the checkpoint was trained
                    # at; re-sampling every frame measures the policy off its own
                    # control timescale.
                    if frame % interval == 0:
                        observation = tensor_observation(packet_observation(packet), device)
                        with torch.inference_mode(), torch.autocast(
                            device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == 'cuda',
                        ):
                            logits, _value, _hazard = agent(observation)
                            distribution = AutoregressiveActionDistribution(logits, observation)
                            sampled = (distribution.sample() if args.stochastic
                                       else distribution.mode())
                        held = sampled.cpu().numpy().astype(np.uint8)
                    actions = held.copy()
                    actions[~active] = 254
                    live = packet['current_heights']
                    peak[active] = np.maximum(peak[active], live[active])
                    frames_alive += int(active.sum())
                    packet = bridge.step(actions)
                    finished = active & packet['dones'].astype(bool)
                    deaths += int(finished.sum())
                    active &= ~finished
                    if not active.any():
                        break
                layers += float(np.maximum(0.0, peak - start).sum()) / LAYER
            survival = 1.0 - deaths / layers if layers >= 1 else None
            report[band] = {
                'cells': len(pool),
                'episodes': args.envs * args.rounds,
                'deaths': deaths,
                'layers_gained': round(layers, 1),
                'mean_seconds_survived': round(frames_alive / max(1, deaths) / 60.0, 1),
                'per_layer_survival': None if survival is None else round(survival, 4),
            }
    finally:
        bridge.close()

    print(json.dumps({
        'checkpoint': args.checkpoint,
        'bank': args.bank,
        'control_interval': interval,
        'by_start_regime': report,
        'reference': {
            'minute_one_measured': 0.928,
            'needed_to_reach_10k_at_all': 0.9567,
            'needed_for_consistent_10k': 0.9972,
        },
        'caveat': 'bank states were built by search, so they are off-distribution '
                  'for this policy; the measured survival is a lower bound',
    }, indent=2))


if __name__ == '__main__':
    main()
