#!/usr/bin/env python3
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rl'))

from r2d2 import (  # noqa: E402
    PrioritizedSequenceReplay,
    RecurrentQNetwork,
    SequenceAssembler,
    learn_batch,
    tensor_observation,
)
from v2_bridge import ParallelEnvBridge  # noqa: E402


def main():
    bridge = ParallelEnvBridge(1, 2, 123, archive_probability=0)
    packet = bridge.read()
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
        assert abs(episode_return - height / 40) < 1e-5
    for sequence in sequences:
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
    print('ok recurrent replay, terminal sequences, and score reward are valid')


if __name__ == '__main__':
    main()
