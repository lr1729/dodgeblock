#!/usr/bin/env python3
"""Does shelter actually nullify hazard, and what does holding it cost?

The audit of a winning 10k trajectory (go-explore seed 7, 24,887 frames) shows it
spends 41% of frames sheltered -- 47% above height 4000, with individual bands at
85-98%. The trained policy sits near 18%. That gap is the most concrete behavioural
difference found so far between "reaches 10k" and "dies at 800", so it is worth
knowing whether it is causal before anything is built on it.

Shelter is a solid overhead block (env-server-v2.mjs isSheltered: a fixed,
non-faulting block whose underside is above the player, overlapping >= 6 units of
its width). The obvious story is that it blocks the squish deaths that are 47% of
this policy's terminations. If true, time-in-shelter converts directly into
log-hazard, and the lever is enormous.

But shelter is not free. Hiding under an overhang is not climbing, and the metric
that decides 10k is survival *per layer gained*, not per second. A policy that
shelters 95% of the time and never ascends has perfect per-frame survival and
never finishes. So this measures both halves and reports the quantity that
actually composes:

    per-layer hazard in a condition = deaths in it / (height gained in it / 40)

Both conditions are measured from the same rollouts out of banked saturated cells,
so difficulty is matched. The counterfactual at the end holds both conditional
hazards fixed and varies only the mix, which is the question a training change
would be trying to move.

Honest limits: conditioning on shelter is not intervening on it. The policy chooses
when to shelter, so sheltered frames are also frames it judged safe to be in, and
part of any measured difference is selection rather than protection. The
counterfactual is therefore an upper bound on what re-mixing alone would buy.
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
SATURATION_FRAME = 14400


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--bank', default='/home/lr/dodgeblock-go-explore-bank-v4/'
                                          'seed-1/search-checkpoint.json.gz')
    parser.add_argument('--envs', type=int, default=128)
    parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--max-frames', type=int, default=6000)
    parser.add_argument('--seed', type=int, default=0x5A7_0BE5)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--stochastic', action='store_true')
    parser.add_argument('--control-interval', type=int, default=0)
    args = parser.parse_args()

    with gzip.open(args.bank) as handle:
        entries = json.load(handle)['entries']
    pool = [i for i, e in enumerate(entries) if e['frame'] >= SATURATION_FRAME]
    heights = {i: float(e['height']) for i, e in enumerate(entries)}
    if len(pool) < 8:
        raise SystemExit('bank has too few saturated cells')

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
        8, max(1, args.envs // 8), args.seed,
        target_height=10_000,
        reward_mode='target',
        cell_banks=[args.bank],
    )
    rng = np.random.default_rng(0)

    # [unsheltered, sheltered]
    frames = np.zeros(2, np.int64)
    deaths = np.zeros(2, np.int64)
    gained = np.zeros(2, np.float64)

    try:
        for _round in range(args.rounds):
            chosen = rng.choice(pool, size=args.envs, replace=True)
            packet = bridge.reset(np.asarray(chosen, np.int32))
            active = np.ones(args.envs, dtype=bool)
            held = np.zeros(args.envs, np.uint8)
            previous_height = np.array(
                [heights[int(c)] for c in chosen], np.float64)
            for frame in range(args.max_frames):
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
                packet = bridge.step(actions)

                # step_sheltered is written before the sim advances and dones
                # covers that same transition, so both describe one frame.
                shelter = packet['step_sheltered'].astype(bool)
                current = packet['current_heights'].astype(np.float64)
                climbed = np.maximum(0.0, current - previous_height)
                previous_height = np.where(active, current, previous_height)
                finished = active & packet['dones'].astype(bool)

                for state in (0, 1):
                    mask = active & (shelter == bool(state))
                    frames[state] += int(mask.sum())
                    gained[state] += float(climbed[mask].sum())
                    deaths[state] += int((finished & mask).sum())

                active &= ~finished
                if not active.any():
                    break
    finally:
        bridge.close()

    def report(state):
        layers = gained[state] / LAYER
        per_frame = deaths[state] / max(1, frames[state])
        per_layer = deaths[state] / layers if layers >= 1 else None
        return {
            'frames': int(frames[state]),
            'occupancy': round(float(frames[state] / max(1, frames.sum())), 4),
            'deaths': int(deaths[state]),
            'layers_gained': round(float(layers), 1),
            'height_per_frame': round(float(gained[state] / max(1, frames[state])), 4),
            'hazard_per_frame': round(float(per_frame), 6),
            'per_layer_survival': (None if per_layer is None
                                   else round(float(max(0.0, 1.0 - per_layer)), 4)),
        }

    unsheltered, sheltered = report(0), report(1)
    occupancy = frames[1] / max(1, frames.sum())

    # Counterfactual: hold both conditional hazards and both climb rates fixed and
    # vary only the mix. Deaths and layers both scale with time spent in each
    # condition, so the mix determines the composite per-layer hazard.
    def composite(target_occupancy):
        weights = np.array([1.0 - target_occupancy, target_occupancy])
        rate = np.array([gained[s] / max(1, frames[s]) for s in (0, 1)])
        hazard = np.array([deaths[s] / max(1, frames[s]) for s in (0, 1)])
        layers = (weights * rate).sum() / LAYER
        if layers <= 0:
            return None
        return round(float(max(0.0, 1.0 - (weights * hazard).sum() / layers)), 4)

    print(json.dumps({
        'checkpoint': args.checkpoint,
        'bank': args.bank,
        'control_interval': interval,
        'saturated_cells': len(pool),
        'measured': {
            'shelter_occupancy': round(float(occupancy), 4),
            'unsheltered': unsheltered,
            'sheltered': sheltered,
        },
        'counterfactual_per_layer_survival_by_occupancy': {
            f'{target:.2f}': composite(target)
            for target in (0.0, 0.18, 0.41, 0.47, 0.60, 0.80, 0.95, 1.0)
        },
        'reference': {
            'winning_10k_trajectory_occupancy': 0.4112,
            'winning_10k_occupancy_above_4000': 0.4694,
            'needed_to_reach_10k_at_all': 0.9567,
            'needed_for_consistent_10k': 0.9972,
        },
        'caveat': 'conditioning on shelter is not intervening on it; the policy '
                  'chooses when to shelter, so part of any gap is selection, and '
                  'the counterfactual is an upper bound on re-mixing alone',
    }, indent=2))


if __name__ == '__main__':
    main()
