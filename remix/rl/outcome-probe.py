#!/usr/bin/env python3
"""Is the critic underfit, or is the outcome simply not predictable?

The critic ranks survival at chance once matched on height (AUC 0.50-0.54). That
was read as a defect, and an auxiliary danger head was designed to fix it. But
the reading assumes a better answer exists. If the outcome genuinely is not
determined by the observation at that height -- the threat is stochastic and
arrives faster than position protects against -- then chance is the CORRECT
answer, the critic is fine, and the whole repair is aimed at nothing.

This settles it. Collect states at matched height, label each with whether its
episode survived, and train a probe on exactly those states. Split by EPISODE so
the probe cannot memorise a trajectory it has already seen. Then compare the
probe's held-out AUC against the critic's AUC on the identical test states.

    probe >> critic ~ 0.5  ->  the information is there and the critic missed it.
                              Underfit. The auxiliary head is the right medicine.
    probe ~ critic ~ 0.5   ->  no function of this observation predicts the
                              outcome. The critic is already optimal and the
                              danger head will train beautifully and change
                              nothing.

The probe reads the frozen trunk, so it asks the narrower and more useful
question: is the signal already computed by the network and merely unused by the
value head?
"""
import argparse
import json

import numpy as np
import torch
from torch import nn

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

BANDS = [100, 200, 300, 400]


def rank_auc(scores, labels):
    """P(score of a survivor > score of a doomed one)."""
    good = int((~labels).sum())
    bad = int(labels.sum())
    if not good or not bad:
        return None
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[~labels].sum() - good * (good + 1) / 2) / (good * bad))


def fit_probe(features, labels, device, epochs=300, hidden=256):
    """Small MLP, heavily regularised -- the question is whether ANY signal is
    there, so underfitting would answer the wrong question and overfitting would
    answer it dishonestly. Held-out AUC is what gets reported either way."""
    x = torch.as_tensor(features, device=device, dtype=torch.float32)
    y = torch.as_tensor(labels, device=device, dtype=torch.float32)
    model = nn.Sequential(
        nn.LayerNorm(x.shape[1]),
        nn.Linear(x.shape[1], hidden), nn.SiLU(), nn.Dropout(0.2),
        nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(0.2),
        nn.Linear(hidden, 1),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    positive = float(y.sum())
    weight = (len(y) - positive) / max(1.0, positive)
    for _ in range(epochs):
        model.train()
        logits = model(x).squeeze(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, y, pos_weight=torch.tensor(weight, device=device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--episodes', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=0xD06E_B10C)
    parser.add_argument('--target-height', type=float, default=600)
    parser.add_argument('--max-frames', type=int, default=4000)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--stochastic', action='store_true',
                        help='sample actions, so repeated seeds do not give a '
                             'single deterministic outcome per episode')
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
    died = np.zeros(args.episodes, dtype=bool)
    hidden_size = 384
    captured = {band: np.zeros((args.episodes, hidden_size), np.float32)
                for band in BANDS}
    critic = {band: np.full(args.episodes, np.nan, np.float32) for band in BANDS}
    seen = {band: np.zeros(args.episodes, bool) for band in BANDS}

    try:
        for _frame in range(args.max_frames):
            observation = tensor_observation(packet_observation(packet), device)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=device.type == 'cuda',
            ):
                trunk = agent.encode(observation)
                value = agent.critic(trunk).squeeze(-1)
                logits, _value, _hazard = agent(observation)
                distribution = AutoregressiveActionDistribution(logits, observation)
                sampled = (distribution.sample() if args.stochastic
                           else distribution.mode())
            actions = sampled.cpu().numpy().astype(np.uint8)
            trunk = trunk.float().cpu().numpy()
            value = value.float().cpu().numpy()
            live = packet['current_heights']
            for band in BANDS:
                fresh = np.flatnonzero(active & ~seen[band] & (live >= band))
                if len(fresh):
                    captured[band][fresh] = trunk[fresh]
                    critic[band][fresh] = value[fresh]
                    seen[band][fresh] = True
            actions[~active] = 254
            packet = bridge.step(actions)
            finished = active & packet['dones'].astype(bool)
            died |= finished & ~packet['successes'].astype(bool)
            active &= ~finished
            if not active.any():
                break
    finally:
        bridge.close()

    # Split by episode, not by state: an episode contributes at most one row per
    # band, so a random split over episodes leaks nothing across the boundary.
    rng = np.random.default_rng(0)
    holdout = rng.random(args.episodes) < 0.3

    report = {}
    for band in BANDS:
        index = np.flatnonzero(seen[band])
        if len(index) < 200:
            report[band] = {'episodes': int(len(index)), 'skipped': True}
            continue
        train = index[~holdout[index]]
        test = index[holdout[index]]
        labels_train, labels_test = died[train], died[test]
        if labels_train.sum() < 20 or (~labels_train).sum() < 20 \
                or labels_test.sum() < 20 or (~labels_test).sum() < 20:
            report[band] = {'episodes': int(len(index)), 'skipped': 'class imbalance'}
            continue
        model = fit_probe(captured[band][train], labels_train.astype(np.float32), device)
        with torch.no_grad():
            scores = model(torch.as_tensor(
                captured[band][test], device=device, dtype=torch.float32)
            ).squeeze(-1).cpu().numpy()
        report[band] = {
            'episodes': int(len(index)),
            'test_episodes': int(len(test)),
            'death_rate': round(float(labels_test.mean()), 3),
            'probe_auc': round(rank_auc(scores, labels_test), 4),
            'critic_auc': round(rank_auc(critic[band][test], labels_test), 4),
        }

    print(json.dumps({
        'checkpoint': args.checkpoint,
        'stochastic': args.stochastic,
        'overall_death_rate': round(float(died.mean()), 4),
        'auc_at_matched_height': report,
        'reading': 'probe >> critic means underfit; probe ~ critic ~ 0.5 means '
                   'the outcome is not a function of this observation',
    }, indent=2))


if __name__ == '__main__':
    main()
