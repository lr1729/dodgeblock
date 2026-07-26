#!/usr/bin/env python3
"""Do paired action contrasts actually collapse the variance, and are decision
points findable?

The proposed direction is to build PPO advantages from branched rollouts under
common random numbers instead of from GAE over a noisy terminal return. Two
assumptions carry it, and both are cheap to test before writing the estimator:

  A. VARIANCE. `restoreSlot` performs no reseed, so two rollouts from one
     snapshot face bit-identical falling blocks. The contrast
     1[survive|a] - 1[survive|a'] then has variance P(discordant) - delta^2
     rather than ~2*p*(1-p) ~ 0.5 for independent samples. If discordance is a
     few percent, samples-per-distinction drops by one to two orders of
     magnitude. If it is ~50%, the futures are not actually shared and the whole
     direction is void.

  B. SCREENING. The project measured Q-flatness (held-out direction loss ~ ln 3)
     and concluded decisions are sparse in time. Branching detects that for free:
     if every action from a state gives the same outcome, it is not a decision
     point and can be dropped. This reports how sparse they actually are, which
     sets the subsampling rate the real estimator would need.

Reports both, plus the same numbers restricted to states near a death, since
that is where the autopsy says the action effect lives.
"""
import argparse
import json

import numpy as np
import torch

from ppo_v2 import (
    ActorCriticNetwork,
    AutoregressiveActionDistribution,
    STICKY_MODEL_ARCHITECTURE,
    StickyActorCriticNetwork,
    load_agent_state,
    packet_observation,
    tensor_observation,
)
from v2_bridge import ParallelEnvBridge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--states', type=int, default=200,
                        help='how many probe states to branch from')
    parser.add_argument('--lanes', type=int, default=32,
                        help='branches per state; each is one forced first action')
    parser.add_argument('--horizon', type=int, default=90,
                        help='frames to roll each branch before scoring survival')
    parser.add_argument('--settle', type=int, default=240,
                        help='frames to advance between probe states')
    parser.add_argument('--seed', type=int, default=0x0C12_0BE5)
    parser.add_argument('--target-height', type=float, default=600)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

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

    # One worker: the slot table is per worker, so every lane must live in it for
    # a single saved state to be restored into all of them.
    bridge = ParallelEnvBridge(
        1, args.lanes, args.seed,
        target_height=args.target_height,
        reward_mode='target',
    )
    packet = bridge.read()

    def act(packet, stochastic=True):
        observation = tensor_observation(packet_observation(packet), device)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=device.type == 'cuda',
        ):
            logits, _value, _hazard = agent(observation)
            distribution = AutoregressiveActionDistribution(logits, observation)
            sampled = distribution.sample() if stochastic else distribution.mode()
        return sampled.cpu().numpy().astype(np.uint8)

    discordant_pairs = total_pairs = 0
    decision_states = scored_states = 0
    survival_spread = []
    near_death = {'decision': 0, 'states': 0}

    try:
        for index in range(args.states):
            # Advance lane 0 to a fresh state. All lanes run in lockstep from the
            # same seed, so they are identical anyway; only lane 0 is saved.
            for _ in range(args.settle):
                packet = bridge.step(act(packet))
                if packet['dones'].any():
                    break

            # Snapshot lane 0, then clone it into every lane.
            slots = np.zeros(args.lanes, np.int64)
            packet = bridge.save_slots(slots)
            packet = bridge.restore_slots(slots)

            # Each lane takes a different forced first action, then follows the
            # policy. Identical restore => identical falling blocks across lanes,
            # so any difference in outcome is caused by the forced action.
            forced = np.arange(args.lanes, dtype=np.uint8) % 18
            packet = bridge.step(forced)
            alive = ~packet['dones'].astype(bool)
            for _ in range(args.horizon - 1):
                actions = act(packet)
                actions[~alive] = 254
                packet = bridge.step(actions)
                alive &= ~packet['dones'].astype(bool)

            survived = alive.astype(np.float64)
            scored_states += 1
            spread = float(survived.mean())
            survival_spread.append(spread)
            # Pairwise discordance across the branches of THIS state.
            ones = int(survived.sum())
            zeros = args.lanes - ones
            discordant_pairs += ones * zeros
            total_pairs += args.lanes * (args.lanes - 1) // 2
            if 0 < ones < args.lanes:
                decision_states += 1
                if spread < 0.5:
                    near_death['decision'] += 1
            if spread < 0.5:
                near_death['states'] += 1
    finally:
        bridge.close()

    discordance = discordant_pairs / max(1, total_pairs)
    spread = np.array(survival_spread)
    # Independent sampling of a Bernoulli difference has variance ~2p(1-p);
    # the paired contrast has variance ~P(discordant). The ratio is the factor
    # by which samples-per-distinction falls.
    mean_p = float(spread.mean())
    independent = 2 * mean_p * (1 - mean_p)
    reduction = independent / discordance if discordance > 0 else float('inf')

    print(json.dumps({
        'checkpoint': args.checkpoint,
        'states_scored': scored_states,
        'lanes_per_state': args.lanes,
        'horizon_frames': args.horizon,
        'mean_branch_survival': round(mean_p, 4),
        'decision_point_rate': round(decision_states / max(1, scored_states), 4),
        'pairwise_discordance': round(discordance, 5),
        'independent_contrast_variance': round(independent, 5),
        'implied_sample_reduction': round(reduction, 1),
        'dangerous_states': {
            'count': near_death['states'],
            'decision_point_rate': round(
                near_death['decision'] / max(1, near_death['states']), 4),
        },
        'reading': 'discordance near 0.5 means the futures are not shared and the '
                   'direction is void; a low decision_point_rate is the screen '
                   'the real estimator would subsample on',
    }, indent=2))


if __name__ == '__main__':
    main()
