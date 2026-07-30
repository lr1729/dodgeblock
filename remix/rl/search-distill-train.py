#!/usr/bin/env python3
"""Distil search-improved targets back into the policy.

The other half of the loop v25 argued for. `search-distill-collect.py` produced
survival-weighted distributions over first actions at states where the branches
disagreed; this moves them into the policy's mode.

Loss is cross-entropy from the soft target to the policy's joint 18-way
log-probabilities, which the autoregressive head exposes directly, plus a KL
anchor to the frozen original policy on the same states.

The anchor is not decoration. The hazard-head experiment (v25) is the cautionary
case in this codebase: an auxiliary objective at coefficient 0.5 trained
beautifully -- separation 0.689 at ten frames -- and cost 7.6x on the rung metric,
because nothing held the policy near a working one. Here the targets cover only
the ~15% of states where branches disagreed, so without an anchor the other 85%
are free to drift arbitrarily while the loss looks excellent. Report both terms
so a collapse is visible in the log rather than only in the eval afterwards.

Evaluate the result with saturated-hazard.py across bank seeds. That is the goal
metric, and per v22 it must be compared bank-for-bank: absolute levels swing 0.10
between banks, which is wider than any effect measured so far.
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
    tensor_observation,
)

NON_OBSERVATION = {'targets', 'survival_spread'}


def build(saved, device):
    network_class = (
        StickyActorCriticNetwork
        if saved.get('model_architecture') == STICKY_MODEL_ARCHITECTURE
        else ActorCriticNetwork
    )
    agent = network_class().to(device)
    load_agent_state(agent, saved['agent'])
    return agent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('dataset')
    parser.add_argument('--out', required=True)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--anchor-coef', type=float, default=1.0,
                        help='KL to the frozen original policy; 0 disables the '
                             'trust region and invites the hazard-head failure')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device)
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    agent = build(saved, device)
    anchor = build(saved, device)
    anchor.eval()
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)

    data = np.load(args.dataset)
    targets = torch.tensor(data['targets'], dtype=torch.float32)
    observation_np = {k: data[k] for k in data.files if k not in NON_OBSERVATION}
    count = len(targets)
    optimizer = torch.optim.AdamW(agent.parameters(), lr=args.learning_rate,
                                  weight_decay=1e-5)

    def batch_observation(index):
        return tensor_observation(
            {k: v[index] for k, v in observation_np.items()}, device)

    history = []
    for epoch in range(args.epochs):
        agent.train()
        order = np.random.default_rng(epoch).permutation(count)
        distil_total = anchor_total = batches = 0.0
        dropped_total = 0.0
        dropped_rows = 0
        for start in range(0, count, args.batch):
            index = order[start:start + args.batch]
            observation = batch_observation(index)
            target = targets[index].to(device)

            logits, _value, _hazard = agent(observation)
            logprobs = AutoregressiveActionDistribution(
                logits, observation).joint_logprobs
            # A target can carry mass on an action the stored observation marks
            # illegal -- measured at 2 rows in 101, 0.13 total mass, cause not
            # yet established (a focus state feature flipping within the step is
            # the suspect). One such entry contributes 6.7e7 to the loss via the
            # -1e9 focus mask, so drop that mass and renormalise rather than let
            # a rounding-level disagreement dominate every gradient.
            legal = logprobs > -50.0
            masked = target * legal
            dropped = 1.0 - masked.sum(dim=-1)
            keep = masked.sum(dim=-1) > 0
            renormalised = masked / masked.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            distil = -((renormalised * logprobs).sum(dim=-1)[keep]).mean()
            dropped_total += float(dropped.sum())
            dropped_rows += int((dropped > 1e-6).sum())

            with torch.inference_mode():
                anchor_logits, _v, _h = anchor(observation)
                anchor_logprobs = AutoregressiveActionDistribution(
                    anchor_logits, observation).joint_logprobs
            anchor_probs = anchor_logprobs.exp().clone()
            anchor_kl = (anchor_probs * (anchor_logprobs.clone() - logprobs)
                         ).sum(dim=-1).mean()

            loss = distil + args.anchor_coef * anchor_kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()
            distil_total += float(distil)
            anchor_total += float(anchor_kl)
            batches += 1
        history.append({
            'epoch': epoch,
            'distil_ce': round(distil_total / batches, 5),
            'anchor_kl': round(anchor_total / batches, 5),
            'illegal_target_mass_dropped': round(dropped_total, 4),
            'rows_with_dropped_mass': dropped_rows,
        })
        print(json.dumps(history[-1]), flush=True)

    saved['agent'] = agent.state_dict()
    saved.setdefault('args', {})['distilled_from'] = args.dataset
    torch.save(saved, args.out)
    print(json.dumps({
        'out': args.out,
        'samples': int(count),
        'epochs': args.epochs,
        'anchor_coef': args.anchor_coef,
        'history': history,
        'next': 'evaluate with saturated-hazard.py across bank seeds; compare '
                'bank-for-bank, absolute levels swing 0.10 between banks',
    }, indent=2))


if __name__ == '__main__':
    main()
