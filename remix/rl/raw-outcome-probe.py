#!/usr/bin/env python3
"""Is imminent death predictable from the observation, or only absent from the trunk?

outcome-probe.py trained a probe on the network's frozen 384-d hidden and got AUC
0.494-0.500 against a critic at 0.535-0.556. It concluded that "no function of this
observation predicts the outcome" and cancelled the auxiliary hazard head on that
basis. The head has sat implemented and wired to --hazard-coef, defaulted to 0.0,
ever since.

That conclusion does not follow from that measurement. The probe reads the frozen
trunk -- its own docstring says so -- so a trunk that never learned to encode
danger returns chance whether or not the observation contains the signal. "Not
computed by this network" and "not computable from this input" are different
claims, and only the first was tested. The probe also sampled height bands 100-400,
which at 14 h/s is 7-29 seconds: the flat opening, where this policy rarely dies
and where difficulty has not begun to ramp.

This settles the actual question. Same rollouts, same labels, same episode-level
split, two probes fit side by side:

    raw    -- on the full observation the network receives
    trunk  -- on the frozen 384-d hidden, reproducing the original measurement

and it does so from banked SATURATED cells, the regime that decides 10k.

    raw >> trunk ~ 0.5  ->  the signal is in the input and the trunk discards it.
                            The hazard head is exactly the right medicine and was
                            cancelled on a misreading.
    raw ~ trunk ~ 0.5   ->  imminent death genuinely is not a function of the
                            current observation. No auxiliary loss on this input
                            helps, and the leverage has to move earlier in time --
                            to not entering the state at all.

Labels are "dies within K frames" for each K in HORIZONS, matching the horizons the
hazard head was built for. Splitting by episode is what keeps the probe from
memorising a trajectory it has already seen; splitting by frame would let adjacent
frames of one death leak across the split and inflate every number here.
"""
import argparse
import gzip
import json

import numpy as np
import torch
from torch import nn

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

SATURATION_FRAME = 14400
HORIZONS = (10, 30, 90)


def rank_auc(scores, labels):
    """P(score of a doomed state > score of a surviving one)."""
    positive = int(labels.sum())
    negative = int((~labels).sum())
    if not positive or not negative:
        return None
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels].sum() - positive * (positive + 1) / 2)
                 / (positive * negative))


def fit_probe(train_x, train_y, test_x, test_y, device, epochs=200, hidden=256):
    """Small MLP, class-balanced, early-stopped on the held-out split."""
    model = nn.Sequential(
        nn.Linear(train_x.shape[1], hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, 1),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    positive = float(train_y.sum())
    negative = float(len(train_y) - positive)
    if not positive or not negative:
        return None
    pos_weight = torch.tensor([negative / positive], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_x = train_x.to(device)
    train_y = train_y.to(device)
    test_x = test_x.to(device)
    best = 0.5
    batch = 4096
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(train_x), device=device)
        for start in range(0, len(train_x), batch):
            index = permutation[start:start + batch]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(train_x[index]).squeeze(-1), train_y[index])
            loss.backward()
            optimizer.step()
        if epoch % 10 == 9:
            model.eval()
            with torch.inference_mode():
                scores = model(test_x).squeeze(-1).float().cpu().numpy()
            auc = rank_auc(scores, test_y)
            if auc is not None and abs(auc - 0.5) > abs(best - 0.5):
                best = auc
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--bank', default='/home/lr/dodgeblock-go-explore-bank-v4/'
                                          'seed-1/search-checkpoint.json.gz')
    parser.add_argument('--envs', type=int, default=128)
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--max-frames', type=int, default=1200)
    parser.add_argument('--subsample', type=int, default=4,
                        help='keep every Nth frame; adjacent frames are nearly '
                             'identical and only inflate the sample count')
    parser.add_argument('--max-samples', type=int, default=120_000)
    parser.add_argument('--seed', type=int, default=0x5A7_0BE5)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--control-interval', type=int, default=0)
    args = parser.parse_args()

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
        8, max(1, args.envs // 8), args.seed,
        target_height=10_000, reward_mode='target', cell_banks=[args.bank],
    )
    rng = np.random.default_rng(0)

    raw_rows, trunk_rows, episode_rows, frame_rows = [], [], [], []
    death_frame = {}
    episode_counter = 0

    try:
        for round_index in range(args.rounds):
            chosen = rng.choice(pool, size=args.envs, replace=True)
            packet = bridge.reset(np.asarray(chosen, np.int32))
            episode_ids = np.arange(episode_counter, episode_counter + args.envs)
            episode_counter += args.envs
            active = np.ones(args.envs, dtype=bool)
            held = np.zeros(args.envs, np.uint8)
            for frame in range(args.max_frames):
                observation = tensor_observation(packet_observation(packet), device)
                with torch.inference_mode(), torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16,
                    enabled=device.type == 'cuda',
                ):
                    hidden = agent.encode(observation)
                    logits, _value, _hazard = agent(observation)
                    distribution = AutoregressiveActionDistribution(logits, observation)
                    sampled = distribution.mode()
                if frame % interval == 0:
                    held = sampled.cpu().numpy().astype(np.uint8)

                if frame % args.subsample == 0 and len(raw_rows) < args.max_samples:
                    keep = np.flatnonzero(active)
                    flat = torch.cat([
                        observation[key].flatten(1).float()
                        for key in sorted(observation)
                    ], dim=1)
                    raw_rows.append(flat[keep].half().cpu())
                    trunk_rows.append(hidden[keep].float().half().cpu())
                    episode_rows.append(episode_ids[keep].copy())
                    frame_rows.append(np.full(len(keep), frame, np.int32))

                actions = held.copy()
                actions[~active] = 254
                packet = bridge.step(actions)
                finished = active & packet['dones'].astype(bool)
                for index in np.flatnonzero(finished):
                    death_frame[int(episode_ids[index])] = frame
                active &= ~finished
                if not active.any():
                    break
    finally:
        bridge.close()

    raw = torch.cat(raw_rows).float()
    trunk = torch.cat(trunk_rows).float()
    episodes = np.concatenate(episode_rows)
    frames = np.concatenate(frame_rows)

    # Standardise; raw features live on wildly different scales.
    raw = (raw - raw.mean(0)) / raw.std(0).clamp_min(1e-6)
    trunk = (trunk - trunk.mean(0)) / trunk.std(0).clamp_min(1e-6)

    unique = np.unique(episodes)
    split_rng = np.random.default_rng(12345)
    split_rng.shuffle(unique)
    test_ids = set(unique[:len(unique) // 4].tolist())
    is_test = np.array([e in test_ids for e in episodes])

    report = {}
    for horizon in HORIZONS:
        # Positive iff this episode died and did so within `horizon` frames.
        died_in = np.array([
            death_frame.get(int(e), 1 << 30) - int(f) for e, f in zip(episodes, frames)
        ])
        labels = (died_in >= 0) & (died_in < horizon)
        if labels.sum() < 50 or (~labels).sum() < 50:
            report[f'within_{horizon}'] = {'skipped': 'too few of one class'}
            continue
        label_tensor = torch.tensor(labels, dtype=torch.float32)
        entry = {
            'positives': int(labels.sum()),
            'positive_rate': round(float(labels.mean()), 4),
        }
        for name, features in (('raw_observation', raw), ('frozen_trunk', trunk)):
            entry[name] = fit_probe(
                features[~is_test], label_tensor[~is_test],
                features[is_test], labels[is_test], device,
            )
            if entry[name] is not None:
                entry[name] = round(entry[name], 4)
        report[f'within_{horizon}'] = entry

    print(json.dumps({
        'checkpoint': args.checkpoint,
        'bank': args.bank,
        'regime': 'saturated cells only (frame >= 14400)',
        'control_interval': interval,
        'samples': int(len(episodes)),
        'episodes': int(len(unique)),
        'test_episodes': len(test_ids),
        'raw_feature_dim': int(raw.shape[1]),
        'auc_by_horizon': report,
        'reading': 'raw >> trunk means the signal is in the input and the trunk '
                   'discards it, so the auxiliary hazard head was cancelled on a '
                   'misreading; raw ~ trunk ~ 0.5 means imminent death is not a '
                   'function of the current observation and the leverage is earlier',
        'prior': {
            'outcome_probe_trunk_auc_bands_100_400': [0.494, 0.467, 0.500, 0.469],
            'critic_auc_bands_100_400': [0.535, 0.556],
        },
    }, indent=2))


if __name__ == '__main__':
    main()
