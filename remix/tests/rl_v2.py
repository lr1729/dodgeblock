#!/usr/bin/env python3
import base64
import gzip
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rl'))

from r2d2 import (  # noqa: E402
    PrioritizedSequenceReplay,
    RecurrentQNetwork,
    SequenceAssembler,
    TokenEncoder,
    learn_batch,
    tensor_observation,
    world_discounts,
)
from ppo_v2 import ActorCriticNetwork, packet_observation, tensor_observation as ppo_tensor_observation  # noqa: E402
from v2_bridge import ParallelEnvBridge  # noqa: E402


def main():
    bridge = ParallelEnvBridge(1, 2, 123, archive_probability=0)
    packet = bridge.read()
    assert np.all(packet['world_scales'] == 1)
    assembler = SequenceAssembler(2, burn_in=4, unroll=8, n_step=3)
    assembler.initialize(packet)
    sequences = []
    terminal_returns = []
    try:
        for _ in range(2_000):
            packet = bridge.step(np.zeros(2, np.uint8))
            sequences.extend(assembler.append(np.zeros(2, np.uint8), packet))
            for index in np.flatnonzero(packet['dones']):
                terminal_returns.append((packet['returns'][index], packet['heights'][index]))
            if sequences and terminal_returns:
                break
    finally:
        bridge.close()

    assert sequences
    assert terminal_returns
    for episode_return, height in terminal_returns:
        assert abs(episode_return - (height / 40 - 1)) < 1e-5
    for sequence in sequences:
        assert np.all(sequence['world_scales'][sequence['valid'].astype(bool)] > 0)
        terminal = np.flatnonzero(sequence['dones'] * sequence['valid'])
        if len(terminal):
            last_valid = np.flatnonzero(sequence['valid'])[-1]
            assert terminal[-1] == last_valid

    replay = PrioritizedSequenceReplay(8)
    for sequence in sequences[:4]:
        replay.add(sequence)
    while len(replay) < 2:
        replay.add(sequences[0])
    _, weights, batch = replay.sample(2, 0.4, np.random.default_rng(1))

    online = RecurrentQNetwork(hidden_size=64, quantiles=11)
    target = RecurrentQNetwork(hidden_size=64, quantiles=11)
    target.load_state_dict(online.state_dict())
    target.requires_grad_(False)
    optimizer = torch.optim.AdamW(online.parameters(), lr=1e-4)
    result = learn_batch(
        online, target, optimizer, batch, weights,
        burn_in=4, unroll=8, n_step=3, gamma=0.999,
        device=torch.device('cpu'), amp=False,
    )
    assert np.isfinite(result['loss'])
    assert np.all(np.isfinite(result['priorities']))

    first = {key: value[:, 0] for key, value in tensor_observation(batch['observations'], torch.device('cpu')).items()}
    quantiles, hidden = online(first)
    assert quantiles.shape == (2, 18, 11)
    assert hidden.shape == (1, 2, 64)

    token_encoder = TokenEncoder(features=4, count=4, output=8, queries=2)
    one = torch.zeros(1, 4, 4)
    two = torch.zeros(1, 4, 4)
    one[:, 0] = torch.tensor([0.2, 0.3, 0.4, 1.0])
    two[:, :2] = torch.tensor([0.2, 0.3, 0.4, 1.0])
    with torch.no_grad():
        assert not torch.allclose(token_encoder(one), token_encoder(two))
    scales = torch.tensor([1.0, 0.1])
    discounts = world_discounts(0.99999, scales)
    assert discounts[1] > discounts[0]
    assert torch.isclose(discounts[1].pow(10), discounts[0], atol=1e-6)

    ppo = ActorCriticNetwork()
    ppo_observation = ppo_tensor_observation(packet_observation(packet), torch.device('cpu'))
    with torch.no_grad():
        logits, values = ppo(ppo_observation)
    assert logits.shape == (2, 18)
    assert values.shape == (2,)
    assert torch.all(torch.isfinite(logits))
    assert torch.all(torch.isfinite(values))

    demonstration = os.environ.get('DODGEBLOCK_TEST_DEMONSTRATION')
    if demonstration:
        with gzip.open(demonstration, 'rt') as source:
            payload = json.load(source)
        final_action = base64.b64decode(payload['actions'])[-1]
        target_bridge = ParallelEnvBridge(
            1,
            2,
            456,
            archive_probability=0,
            target_height=payload['targetHeight'],
            reward_mode='target',
            demonstrations=[demonstration],
            demonstration_probability=1,
            demonstration_probability_end=1,
            demonstration_snapshot_capacity=8,
            reverse_curriculum_initial_frames=1,
            demonstration_randomize_probability=0,
        )
        try:
            target_bridge.read()
            result = target_bridge.step(np.full(2, final_action, np.uint8))
        finally:
            target_bridge.close()
        assert np.all(result['dones'] == 1)
        assert np.all(result['successes'] == 1)
        assert np.all(result['heights'] >= payload['targetHeight'])
        assert np.all(result['current_heights'] < payload['targetHeight'])

    print('ok recurrent replay, PPO forward pass, terminal rewards, and target curriculum are valid')


if __name__ == '__main__':
    main()
