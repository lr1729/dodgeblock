#!/usr/bin/env python3
"""When an escape exists, does the policy take it?

Everything measured so far says the deficit is action selection: saturated states
are escapable (94.5% within 64 sticky-random tries), the network predicts its own
death 10-30 frames out at AUC 0.76-0.83, and it dies at a mean of 10.7 seconds
anyway. The proposed fix is a critical-state-screened paired-advantage estimator.
That is a week of building, and it is worth knowing the ceiling first.

This measures the ceiling directly. Along the policy's own trajectory out of
banked saturated cells, every `stride` frames:

    snapshot -> clone into 18 lanes -> force a different action in each ->
    let the POLICY drive all 18 for `horizon` frames -> record who lived

`restoreSlot` performs no reseed, so all 18 lanes face bit-identical falling
blocks. Any difference in outcome was caused by the forced first action, and the
continuation is the policy's own, so this is exactly the quantity a one-step
lookahead over the current policy would have access to.

The headline is REGRET:

    P(the policy's own action dies | at least one action survives)

    high -> at critical states an escape existed, was reachable by the policy's
            own continuation, and the policy chose past it. Better credit
            assignment at those states is precisely the missing piece, and the
            regret rate bounds what it can buy.
    low  -> when the policy dies, every action dies. The commitment was made
            earlier, one-step advantages have nothing to fix, and the leverage
            has to move to margin -- not entering those states at all.

Reported at several horizons, because the two readings separate with lead time: a
state that is lost at 30 frames but winnable at 90 was decided before the probe.
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

ACTIONS = 18
SATURATION_FRAME = 14400
HORIZONS = (30, 60, 90)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--bank', default='/home/lr/dodgeblock-go-explore-bank-v4/'
                                          'seed-1/search-checkpoint.json.gz')
    parser.add_argument('--probe-points', type=int, default=1500)
    parser.add_argument('--stride', type=int, default=15,
                        help='frames between probes along the trajectory')
    parser.add_argument('--horizon', type=int, default=max(HORIZONS))
    parser.add_argument('--seed', type=int, default=0x9E37_79B9)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--control-interval', type=int, default=0)
    args = parser.parse_args()
    # Record at the requested horizon too, not only the defaults -- regret grows
    # with lead time, so a longer rollout that reported nothing past 90 would
    # answer a different question than the one it cost.
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

    # One worker: the slot table is per worker, so all 18 lanes must share it for
    # a single snapshot to be restored into every one of them.
    bridge = ParallelEnvBridge(
        1, ACTIONS, args.seed,
        target_height=10_000, reward_mode='target', cell_banks=[args.bank],
    )
    slots = np.zeros(ACTIONS, np.int64)
    rng = np.random.default_rng(0)

    def policy_actions(packet, stochastic=False):
        observation = tensor_observation(packet_observation(packet), device)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=device.type == 'cuda',
        ):
            logits, _value, _hazard = agent(observation)
            distribution = AutoregressiveActionDistribution(logits, observation)
            sampled = distribution.sample() if stochastic else distribution.mode()
        return sampled.cpu().numpy().astype(np.uint8)

    # survived[horizon] accumulates per-probe booleans
    records = []

    try:
        probes = 0
        while probes < args.probe_points:
            cell = int(rng.choice(pool))
            packet = bridge.reset(np.full(ACTIONS, cell, np.int32))
            held = policy_actions(packet)
            frame = 0
            alive_real = True
            while alive_real and probes < args.probe_points:
                if frame % args.stride == 0:
                    # Snapshot the (identical) lanes, then branch.
                    packet = bridge.save_slots(slots)
                    packet = bridge.restore_slots(slots)
                    chosen = int(policy_actions(packet)[0])

                    forced = np.arange(ACTIONS, dtype=np.uint8)
                    packet = bridge.step(forced)
                    alive = ~packet['dones'].astype(bool)
                    survival = {}
                    for step in range(1, args.horizon):
                        actions = policy_actions(packet)
                        actions[~alive] = 254
                        packet = bridge.step(actions)
                        alive &= ~packet['dones'].astype(bool)
                        for horizon in horizons:
                            if step == horizon - 1:
                                survival[horizon] = alive.copy()
                    for horizon in horizons:
                        survival.setdefault(horizon, alive.copy())

                    records.append({
                        'chosen': chosen,
                        'survival': {h: survival[h].copy() for h in horizons},
                    })
                    probes += 1

                    # Put every lane back on the shared trajectory.
                    packet = bridge.restore_slots(slots)
                    held = np.full(ACTIONS, chosen, np.uint8)

                if frame % interval == 0 and frame % args.stride != 0:
                    held = policy_actions(packet)
                packet = bridge.step(held)
                alive_real = not bool(packet['dones'][0])
                frame += 1
    finally:
        bridge.close()

    report = {}
    for horizon in horizons:
        chosen_lives = np.array([r['survival'][horizon][r['chosen']] for r in records])
        any_lives = np.array([r['survival'][horizon].any() for r in records])
        count_lives = np.array([int(r['survival'][horizon].sum()) for r in records])
        # Critical states: not everything survives, so the action matters.
        critical = count_lives < ACTIONS
        rescuable = any_lives & ~chosen_lives
        report[f'horizon_{horizon}'] = {
            'probes': int(len(records)),
            'policy_action_survives': round(float(chosen_lives.mean()), 4),
            'some_action_survives': round(float(any_lives.mean()), 4),
            'mean_surviving_actions_of_18': round(float(count_lives.mean()), 2),
            'critical_state_rate': round(float(critical.mean()), 4),
            'regret_rate_overall': round(float(rescuable.mean()), 4),
            'regret_given_escape_exists': (
                round(float(rescuable.sum() / max(1, any_lives.sum())), 4)),
            'regret_given_critical': (
                round(float(rescuable[critical].mean()), 4) if critical.any() else None),
        }

    print(json.dumps({
        'checkpoint': args.checkpoint,
        'bank': args.bank,
        'control_interval': interval,
        'stride': args.stride,
        'by_horizon': report,
        'reading': 'regret_given_escape_exists is the ceiling on one-step action '
                   'selection: the fraction of survivable situations the policy '
                   'walks out of. High means credit assignment at critical states '
                   'is the missing piece; low means the commitment happened '
                   'earlier and the leverage is margin, not advantages.',
        'reference': {
            'saturated_per_layer_now': 0.8213,
            'needed_to_reach_10k_at_all': 0.9567,
            'needed_for_consistent_10k': 0.9972,
        },
    }, indent=2))


if __name__ == '__main__':
    main()
