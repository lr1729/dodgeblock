#!/usr/bin/env python3
import base64
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

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
from cell_bank import CellBankCoordinator  # noqa: E402
from imitation_v5 import autoregressive_imitation_loss  # noqa: E402
from ppo_v2 import (  # noqa: E402
    ActorCriticNetwork,
    AutoregressiveActionDistribution,
    packet_observation,
    tensor_observation as ppo_tensor_observation,
)
from trajectory_bank import TrajectoryStartCoordinator  # noqa: E402
from v2_bridge import ParallelEnvBridge  # noqa: E402


def main():
    bridge = ParallelEnvBridge(1, 2, 123)
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
        distribution = AutoregressiveActionDistribution(logits, ppo_observation)
    assert logits[0].shape == (2, 2)
    assert logits[1].shape == (2, 2, 3)
    assert logits[2].shape == (2, 2, 3, 3)
    assert distribution.joint_logprobs.shape == (2, 18)
    assert torch.allclose(
        distribution.joint_logprobs.exp().sum(dim=-1),
        torch.ones(2),
    )
    assert values.shape == (2,)
    assert all(torch.all(torch.isfinite(component)) for component in logits)
    assert torch.all(torch.isfinite(values))
    conditional_logits = (
        torch.tensor([[0.0, 3.0], [3.0, 0.0]]),
        torch.tensor([
            [[8.0, 0.0, 0.0], [0.0, 0.0, 8.0]],
            [[0.0, 8.0, 0.0], [8.0, 0.0, 0.0]],
        ]),
        torch.zeros(2, 2, 3, 3),
    )
    conditional_logits[2][0, 1, 2, 1] = 8
    conditional_logits[2][1, 0, 1, 2] = 8
    conditional = AutoregressiveActionDistribution(
        conditional_logits,
        ppo_observation,
    )
    assert conditional.mode().tolist() == [16, 5]
    imitation_observation = {
        key: value[:2].clone()
        for key, value in ppo_observation.items()
    }
    imitation_observation['state'][:, 10] = 1
    imitation_observation['state'][:, 12] = 0
    imitation_observation['state'][:, 14] = 0
    imitation_observation['state'][:, 19] = 0
    imitation_actions = torch.tensor([16, 5])
    imitation_loss, imitation_metrics = autoregressive_imitation_loss(
        conditional_logits,
        imitation_observation,
        imitation_actions,
    )
    assert torch.isfinite(imitation_loss)
    assert imitation_metrics['joint_accuracy'] == 1

    with tempfile.TemporaryDirectory() as temporary:
        synthetic_bank = Path(temporary) / 'synthetic-bank.json.gz'
        with gzip.open(synthetic_bank, 'wt') as target:
            json.dump({
                'version': 1,
                'targetHeight': 10_000,
                'seed': 99,
                'entries': [
                    {
                        'key': f'band-{height}',
                        'height': height,
                        'previousAction': 0,
                        'snapshot': {},
                    }
                    for height in (0, 400, 800)
                ],
            }, target)
        wave = CellBankCoordinator(
            [synthetic_bank],
            target_height=10_000,
            seed=3,
            probability=1,
            heldout_fraction=0,
            band_height=400,
        )
        initial_weights, initial_competences = wave._training_band_weights()
        assert all(value == 0 for value in initial_competences.values())
        initial_total = sum(initial_weights.values())
        assert initial_weights[2] / initial_total > 0.8
        assert initial_weights[1] > 0
        top_variant = next(
            item.variant_id for item in wave.variants if item.height == 800
        )
        for _ in range(20):
            wave.record_start(top_variant)
            wave.record_result(top_variant, True)
        learned_weights, competences = wave._training_band_weights()
        assert competences[2] > 0.7
        assert learned_weights[1] > initial_weights[1] * 20
        assert learned_weights[1] > learned_weights[0]
        wave.sample_clock = 10**12
        assert wave._cell_weight('band-0') <= 2.25

        trajectory_paths = []
        for seed in (4, 8):
            path = Path(temporary) / f'trajectory-{seed}.json.gz'
            with gzip.open(path, 'wt') as target:
                json.dump({
                    'version': 1,
                    'targetHeight': 10_000,
                    'seed': seed,
                    'entries': [
                        {
                            'key': f'seed-{seed}-frame-{frame}',
                            'frame': frame,
                            'height': frame / 2,
                            'previousAction': 0,
                            'snapshot': {},
                        }
                        for frame in range(seed, seed + 3)
                    ],
                }, target)
            trajectory_paths.append(path)
        trajectory = TrajectoryStartCoordinator(
            trajectory_paths,
            seed=12,
            probability=1,
        )
        selected = [trajectory.select() for _ in range(100)]
        selected_seeds = {
            trajectory.variants[variant_id]['seed']
            for variant_id in selected
        }
        assert selected_seeds == {4, 8}
        assert trajectory.select(heldout=True) == -1
        trajectory.record_start(selected[0])
        trajectory.record_result(selected[0], True)
        saved_trajectory = trajectory.state_dict()
        restored_trajectory = TrajectoryStartCoordinator(
            trajectory_paths,
            seed=99,
            probability=1,
        )
        restored_trajectory.load_state_dict(saved_trajectory)
        assert restored_trajectory.metrics()['successes'] == 1
        assert restored_trajectory.select() == trajectory.select()

        subprocess.run([
            'node',
            str(Path(__file__).resolve().parents[1] / 'rl' / 'go-explore.mjs'),
            '--seed', '17',
            '--iterations', '1',
            '--checkpoint-interval', '1',
            '--output-dir', temporary,
        ], check=False, stdout=subprocess.DEVNULL)
        bank = str(Path(temporary) / 'search-checkpoint.json.gz')
        coordinator = CellBankCoordinator(
            [bank],
            target_height=10_000,
            seed=9,
            probability=1,
            heldout_fraction=0,
        )
        assert len(coordinator.variants) == 1
        selected = coordinator.select()
        assert selected == 0
        coordinator.record_start(selected)
        coordinator.record_result(selected, True)
        assert coordinator.metrics()['success_rate'] == 1

        bank_bridge = ParallelEnvBridge(
            1,
            1,
            456,
            target_height=10_000,
            reward_mode='target',
            cell_banks=[bank],
        )
        try:
            bank_bridge.read()
            reset_packet = bank_bridge.reset(np.zeros(1, np.int32))
            assert reset_packet['current_cell_ids'][0] == 0
            for _ in range(2_000):
                result = bank_bridge.step(
                    np.zeros(1, np.uint8),
                    np.zeros(1, np.int32),
                )
                if result['dones'][0]:
                    break
            else:
                raise AssertionError('fixture environment did not terminate')
        finally:
            bank_bridge.close()
        assert result['episode_cell_ids'][0] == 0
        assert result['current_cell_ids'][0] == 0
        assert np.allclose(result['state'][0, 32:36], 1)

        death_dir = Path(temporary) / 'deaths'
        capture_bridge = ParallelEnvBridge(
            1,
            1,
            987,
            death_case_dir=death_dir,
        )
        try:
            capture_bridge.read()
            for _ in range(3_000):
                capture_packet = capture_bridge.step(np.zeros(1, np.uint8))
                assert capture_packet['step_phases'].shape == (1,)
                assert capture_packet['step_sheltered'].shape == (1,)
                if capture_packet['dones'][0]:
                    break
            else:
                raise AssertionError('capture fixture did not terminate')
        finally:
            capture_bridge.close()
        death_files = list(death_dir.glob('*.json.gz'))
        assert death_files
        with gzip.open(death_files[0], 'rt') as source:
            death_case = json.load(source)
        assert death_case['version'] == 2
        assert death_case['history']['actions']
        oracle = subprocess.run([
            'node',
            str(Path(__file__).resolve().parents[1] / 'rl' / 'rescue-oracle.mjs'),
            '--case', str(death_files[0]),
            '--trials', '1',
            '--horizon', '64',
            '--futures', '1',
            '--rewinds', '1,30',
        ], check=True, capture_output=True, text=True)
        oracle_result = json.loads(oracle.stdout)
        assert oracle_result['version'] == 2
        assert oracle_result['evaluationCount'] >= 1

    demonstration = os.environ.get('DODGEBLOCK_TEST_DEMONSTRATION')
    if demonstration:
        with gzip.open(demonstration, 'rt') as source:
            payload = json.load(source)
        final_action = base64.b64decode(payload['actions'])[-1]
        assert final_action < 18

    print('ok recurrent replay, autoregressive PPO, cell banks, and terminal rewards are valid')


if __name__ == '__main__':
    main()
