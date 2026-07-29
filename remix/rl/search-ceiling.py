#!/usr/bin/env python3
"""How much of 10k play is already inside this policy's own distribution?

policy-regret.py measured the ceiling on choosing ONE action well: about 1.8x
hazard, landing near 0.90 per layer against the 0.9972 the goal needs. That
bounds the entire paired-advantage direction, and it is not enough. The open
question is where the remaining ~35x lives, and it has exactly two answers that
imply different projects:

  A. 10k play IS in the policy's distribution, just rarely sampled. Then search
     over the policy finds it, and the design is search-in-the-loop policy
     iteration -- AlphaZero-shaped, which an exact-restore simulator suits
     perfectly, and which go-explore has already shown terminates on this game.

  B. 10k play is NOT in the distribution at any depth. Then no estimator, no
     lookahead and no amount of branch rollouts help, because there is nothing
     to find. The leverage has to move to margin -- not entering states that
     demand a perfect escape -- which is a different objective, not a better
     gradient.

This distinguishes them. From banked saturated states, along the policy's own
trajectory, clone into 18 lanes: lane 0 runs the deterministic policy, lanes
1-17 run independent stochastic samples. `restoreSlot` does no reseed, so all 18
face bit-identical blocks and every difference is a difference in chosen actions.
Report survival for the deterministic policy and for best-of-K as K grows.

    best-of-K climbing steeply toward 0.9567+  ->  reading A
    best-of-K flattening well below it         ->  reading B

Best-of-K is a retry statistic and a live policy gets one attempt -- it is not
achievable survival. It is the ceiling of what search over this distribution
could ever reach, which is the quantity that picks the design.
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

DEFAULT_LANES = 18
SATURATION_FRAME = 14400
HORIZONS = (30, 90, 240)
K_VALUES = (1, 2, 4, 8, 16, 32, 64, 127)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--bank', default='/home/lr/dodgeblock-go-explore-bank-v4/'
                                          'seed-1/search-checkpoint.json.gz')
    parser.add_argument('--probe-points', type=int, default=800)
    parser.add_argument('--stride', type=int, default=25)
    parser.add_argument('--horizon', type=int, default=max(HORIZONS))
    parser.add_argument('--seed', type=int, default=0x5EA5_C11)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--control-interval', type=int, default=0)
    parser.add_argument('--lanes', type=int, default=DEFAULT_LANES,
                        help='K is bounded by lanes-1; all lanes share one '
                             'worker because the slot table is per worker')
    args = parser.parse_args()
    LANES = args.lanes
    horizons = tuple(sorted({h for h in HORIZONS if h <= args.horizon}
                            | {args.horizon}))

    with gzip.open(args.bank) as handle:
        entries = json.load(handle)['entries']
    pool = [i for i, e in enumerate(entries) if e['frame'] >= SATURATION_FRAME]

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
    interval = args.control_interval or checkpoint_control_interval(saved)

    bridge = ParallelEnvBridge(
        1, LANES, args.seed,
        target_height=10_000, reward_mode='target', cell_banks=[args.bank],
    )
    slots = np.zeros(LANES, np.int64)
    rng = np.random.default_rng(0)

    def act(packet):
        """Lane 0 deterministic, lanes 1+ sampled. One forward for both."""
        observation = tensor_observation(packet_observation(packet), device)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=device.type == 'cuda',
        ):
            logits, _value, _hazard = agent(observation)
            distribution = AutoregressiveActionDistribution(logits, observation)
            sampled = distribution.sample()
            mode = distribution.mode()
        actions = sampled.cpu().numpy().astype(np.uint8)
        actions[0] = mode.cpu().numpy().astype(np.uint8)[0]
        return actions

    records = []
    try:
        probes = 0
        while probes < args.probe_points:
            cell = int(rng.choice(pool))
            packet = bridge.reset(np.full(LANES, cell, np.int32))
            held = act(packet)
            frame = 0
            alive_real = True
            while alive_real and probes < args.probe_points:
                if frame % args.stride == 0:
                    packet = bridge.save_slots(slots)
                    packet = bridge.restore_slots(slots)
                    alive = np.ones(LANES, dtype=bool)
                    survival = {}
                    branch_held = None
                    for step in range(args.horizon):
                        if step % interval == 0 or branch_held is None:
                            branch_held = act(packet)
                        actions = branch_held.copy()
                        actions[~alive] = 254
                        packet = bridge.step(actions)
                        alive &= ~packet['dones'].astype(bool)
                        for horizon in horizons:
                            if step == horizon - 1:
                                survival[horizon] = alive.copy()
                    for horizon in horizons:
                        survival.setdefault(horizon, alive.copy())
                    records.append({h: survival[h].copy() for h in horizons})
                    probes += 1
                    packet = bridge.restore_slots(slots)
                    held = act(packet)

                if frame % interval == 0 and frame % args.stride != 0:
                    held = act(packet)
                packet = bridge.step(held)
                alive_real = not bool(packet['dones'][0])
                frame += 1
    finally:
        bridge.close()

    # Lane 0 is deterministic; lanes 1..17 are the stochastic pool best-of-K
    # draws from. Averaging over draws rather than taking the first K removes
    # the arbitrary lane ordering.
    draw_rng = np.random.default_rng(7)
    report = {}
    for horizon in horizons:
        stochastic = np.array([r[horizon][1:] for r in records])
        deterministic = np.array([r[horizon][0] for r in records])
        entry = {'deterministic_survives': round(float(deterministic.mean()), 4)}
        for k in K_VALUES:
            if k > stochastic.shape[1]:
                continue
            hits = []
            for _ in range(64):
                pick = draw_rng.permutation(stochastic.shape[1])[:k]
                hits.append(stochastic[:, pick].any(axis=1).mean())
            entry[f'best_of_{k}'] = round(float(np.mean(hits)), 4)
        report[f'horizon_{horizon}'] = entry

    print(json.dumps({
        'checkpoint': args.checkpoint,
        'bank': args.bank,
        'control_interval': interval,
        'probes': len(records),
        'by_horizon': report,
        'reading': 'best-of-K climbing toward 0.9567+ means 10k play is inside '
                   'this distribution and search finds it (build search-in-the-'
                   'loop); flattening well below means it is not there at any '
                   'depth and the leverage is margin, not better gradients',
        'reference': {
            'saturated_per_layer_now': 0.8213,
            'perfect_one_step_ceiling': 0.90,
            'needed_to_reach_10k_at_all': 0.9567,
            'needed_for_consistent_10k': 0.9972,
        },
    }, indent=2))


if __name__ == '__main__':
    main()
