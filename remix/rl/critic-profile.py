#!/usr/bin/env python3
"""Does the critic see death coming, and how early?

The autopsy put the causal window of a death at 10-20 frames. Shrinking the GAE
horizon onto that window only helps if the critic actually resolves it: at
gamma = 1 with terminal-only reward V(s) = P(success from s), so the credit
signal delta_t = V(s_{t+1}) - V(s_t) is sharp exactly when V drops at the moment
survival probability drops.

This profiles V against frames-before-death. Two outcomes, two different fixes:

- V falls across the causal window (K ~ 10-20): the signal is present and the
  GAE horizon is what smears it. Shrinking lambda is the fix.
- V falls only in the last 2-3 frames, or not at all: the critic is blind to
  local danger and no lambda recovers a signal that was never encoded. The fix
  is representational -- an auxiliary head predicting near-term death, which is
  densely labelled and needs no reward change.
"""
import argparse
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

OFFSETS = [1, 2, 3, 5, 8, 12, 20, 30, 45, 60, 90, 120, 180]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--episodes', type=int, default=256)
    parser.add_argument('--seed', type=int, default=0xD06E_B10C)
    parser.add_argument('--target-height', type=float, default=10_000)
    parser.add_argument('--max-frames', type=int, default=4000)
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

    bridge = ParallelEnvBridge(
        1, args.episodes, args.seed,
        target_height=args.target_height,
        reward_mode='target',
    )
    packet = bridge.read()
    active = np.ones(args.episodes, dtype=bool)
    values = [[] for _ in range(args.episodes)]
    died = np.zeros(args.episodes, dtype=bool)
    # V at the first crossing of each height band. Pooling every visited state
    # confounds the ranking with dwell time -- an episode that lingers near the
    # target contributes many high-V states and still dies. Sampling one state
    # per episode at matched progress removes that.
    bands = [100, 200, 300, 400]
    crossed = {band: np.full(args.episodes, np.nan) for band in bands}

    try:
        for _frame in range(args.max_frames):
            observation = tensor_observation(packet_observation(packet), device)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == 'cuda',
            ):
                logits, value, _hazard = agent(observation)
                actions = AutoregressiveActionDistribution(
                    logits, observation).mode().cpu().numpy().astype(np.uint8)
            value = value.float().cpu().numpy()
            live = packet['current_heights']
            for index in np.flatnonzero(active):
                values[index].append(float(value[index]))
                for band in bands:
                    if np.isnan(crossed[band][index]) and live[index] >= band:
                        crossed[band][index] = value[index]
            actions[~active] = 254
            packet = bridge.step(actions)

            finished = active & packet['dones'].astype(bool)
            success = packet['successes'].astype(bool)
            died |= finished & ~success
            active &= ~finished
            if not active.any():
                break
    finally:
        bridge.close()

    # V as a function of frames-before-death, per episode then averaged, so a
    # few long episodes cannot dominate the profile.
    profile, drops = {}, {}
    tracks = [np.array(values[i]) for i in np.flatnonzero(died)
              if len(values[i]) > max(OFFSETS)]
    for offset in OFFSETS:
        at = np.array([track[-offset] for track in tracks])
        profile[offset] = round(float(at.mean()), 4)
    baseline = profile[max(OFFSETS)]
    for offset in OFFSETS:
        drops[offset] = round(float(baseline - profile[offset]), 4)

    # Where does the fall actually happen? Report the offset by which each
    # fraction of the total decline has already occurred.
    total = drops[min(OFFSETS)]
    reached = {}
    for fraction in (0.25, 0.5, 0.75, 0.9):
        hit = next((o for o in sorted(OFFSETS, reverse=True)
                    if total > 0 and drops[o] >= fraction * total), None)
        reached[f'{int(fraction * 100)}%'] = hit

    # Is the critic a function of the state at all, or has it collapsed to the
    # marginal success rate? Two tests. Spread: a constant critic has none.
    # Ranking: AUC of V over visited states against whether that episode ended
    # in death -- 0.5 means V carries no information about the outcome, however
    # well calibrated its mean happens to be.
    every = np.concatenate([np.array(v) for v in values if v])
    labels = np.concatenate([np.full(len(values[i]), died[i], bool)
                             for i in range(args.episodes) if values[i]])
    survivors, doomed = every[~labels], every[labels]
    if len(survivors) and len(doomed):
        order = np.argsort(every)
        ranks = np.empty(len(every), float)
        ranks[order] = np.arange(1, len(every) + 1)
        auc = ((ranks[~labels].sum() - len(survivors) * (len(survivors) + 1) / 2)
               / (len(survivors) * len(doomed)))
    else:
        auc = None

    def rank_auc(scores, labels):
        """P(score of a survivor > score of a doomed one)."""
        good, bad = scores[~labels], scores[labels]
        if not len(good) or not len(bad):
            return None
        order = np.argsort(scores)
        ranks = np.empty(len(scores), float)
        ranks[order] = np.arange(1, len(scores) + 1)
        return float((ranks[~labels].sum() - len(good) * (len(good) + 1) / 2)
                     / (len(good) * len(bad)))

    matched = {}
    for band in bands:
        seen = ~np.isnan(crossed[band])
        value = rank_auc(crossed[band][seen], died[seen])
        matched[band] = {
            'episodes': int(seen.sum()),
            'auc': None if value is None else round(value, 4),
        }

    print(json.dumps({
        'checkpoint': args.checkpoint,
        'auc_at_matched_height': matched,
        'deaths_profiled': len(tracks),
        'value_by_frames_before_death': profile,
        'decline_from_180f': drops,
        'frames_before_death_at_which_decline_reached': reached,
        'value_spread': {
            'mean': round(float(every.mean()), 4),
            'std': round(float(every.std()), 4),
            'p1': round(float(np.percentile(every, 1)), 4),
            'p50': round(float(np.percentile(every, 50)), 4),
            'p99': round(float(np.percentile(every, 99)), 4),
        },
        'marginal_success_implied_value': round(float(2 * (1 - died.mean()) - 1), 4),
        'auc_value_predicts_survival': None if auc is None else round(float(auc), 4),
    }, indent=2))


if __name__ == '__main__':
    main()
