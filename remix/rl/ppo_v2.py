#!/usr/bin/env python3
import argparse
from collections import Counter, deque
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical

from cell_bank import CellBankCoordinator, FRESH_CELL_ID
from svm import SuccessBonus
from demo_dataset import DemoDataset
from imitation_v5 import (
    autoregressive_imitation_loss,
    sticky_joint_logprobs,
    tensor_demo_batch,
)
from r2d2 import (
    ACTION_COUNT,
    FALLING_COUNT,
    FALLING_FEATURES,
    FORECAST_COUNT,
    FORECAST_FEATURES,
    OBSERVATION_KEYS,
    ResidualBlock,
    SKYLINE_SIZE,
    STATE_SIZE,
    TERRAIN_COLS,
    TERRAIN_ROWS,
    TokenEncoder,
)
from v2_bridge import ParallelEnvBridge
from trajectory_bank import TrajectoryStartCoordinator

TRAINING_CONTRACT_VERSION = 2
MODEL_ARCHITECTURE = 'ppo-v4-autoregressive-cell-bank-1'
STICKY_MODEL_ARCHITECTURE = 'ppo-v5-sticky-autoregressive-1'
DEATH_CAUSE_NAMES = {0: 'success', 1: 'fell', 2: 'squished'}
PHASE_NAMES = {0: 'unknown', 1: 'opening', 2: 'calm', 3: 'build', 4: 'surge', 5: 'release'}
# How much a 10k demonstration shelters in each phase, measured 2026-07-25 by
# replaying it through the sim's own cover test. Used as the phase weighting of
# the shaping potential so the target comes from data, not from intuition.
PHASE_COVER_WEIGHT = np.array([0.0, 0.0, 0.72, 0.90, 0.91, 1.0], np.float32)
FOCUS_STATE_INDEX = 10


def parse_seeds(value):
    result = []
    for part in value.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            start, stop = (int(item) for item in part.split('-', 1))
            result.extend(range(start, stop + 1))
        else:
            result.append(int(part))
    return sorted(set(result))




def focus_action_mask(observation):
    state = observation['state']
    aiming = state[..., 12] > 0
    can_press = (
        (state[..., 10] > 0) &
        (state[..., 14] <= 0) &
        (state[..., 19] <= 0)
    )
    return torch.stack((torch.ones_like(aiming), aiming | can_press), dim=-1)


# Credit horizons for the auxiliary hazard head. 10 and 30 straddle the
# causal window the death autopsy measured (72% of deaths escapable 10
# frames out, 88% at 20); 90 gives a slower-moving positional signal.
HAZARD_HORIZONS = (10, 30, 90)


def load_agent_state(agent, state):
    """Load a checkpoint that may predate the auxiliary hazard head.

    The head is zero-initialised and purely auxiliary, so restoring it fresh is
    correct. Every other mismatch is still an error -- silently tolerating those
    is how a checkpoint contract rots.
    """
    missing, unexpected = agent.load_state_dict(state, strict=False)
    missing = [key for key in missing if not key.startswith('hazard.')]
    if missing or unexpected:
        raise RuntimeError(
            f'checkpoint mismatch: missing={missing} unexpected={list(unexpected)}')


def hazard_labels(deaths, dones):
    """Per-state labels for "does this episode die within h frames".

    Walks backwards accumulating the distance to the next death, resetting at
    every episode boundary. A state whose window runs past the end of the
    rollout without resolving is masked out rather than labelled negative --
    otherwise the tail of every rollout would teach "safe" indiscriminately.
    """
    steps, envs = deaths.shape
    far = float(steps + max(HAZARD_HORIZONS) + 1)
    distance = torch.full((steps + 1, envs), far)
    unresolved = torch.ones((steps + 1, envs), dtype=torch.bool)
    for t in reversed(range(steps)):
        died = deaths[t] > 0
        ended = dones[t] > 0
        distance[t] = torch.where(died, 1.0, torch.where(ended, far, 1 + distance[t + 1]))
        unresolved[t] = ~died & ~ended & unresolved[t + 1]
    distance, unresolved = distance[:steps], unresolved[:steps]

    remaining = (steps - torch.arange(steps)).unsqueeze(-1).float()
    targets, valid = [], []
    for horizon in HAZARD_HORIZONS:
        hit = distance <= horizon
        targets.append(hit.float())
        # Known if the death landed inside the window, or the episode ended, or
        # the window fits entirely within what we recorded.
        valid.append(hit | ~unresolved | (remaining >= horizon))
    targets = torch.stack(targets, dim=-1)
    valid = torch.stack(valid, dim=-1).float()
    # Deaths are rare per frame; without reweighting the head converges to
    # "always safe" and contributes no gradient to the trunk.
    positives = (targets * valid).sum(dim=(0, 1))
    counted = valid.sum(dim=(0, 1))
    pos_weight = ((counted - positives) / positives.clamp(min=1)).clamp(1.0, 100.0)
    return targets, valid, pos_weight


class AutoregressiveActionDistribution:
    def __init__(self, logits, observation):
        self.sticky = len(logits) == 4
        if self.sticky:
            (
                self.joint_logprobs,
                self.repeat_logprobs,
                _previous,
            ) = sticky_joint_logprobs(logits, observation)
            repeat_logits, focus_logits, vertical_logits, horizontal_logits = logits
            del repeat_logits
        else:
            focus_logits, vertical_logits, horizontal_logits = logits
        self.focus = Categorical(
            logits=focus_logits.masked_fill(~focus_action_mask(observation), -1e9)
        )
        self.vertical_logits = vertical_logits
        self.horizontal_logits = horizontal_logits
        focus_logprob = self.focus.logits
        vertical_logprob = torch.log_softmax(vertical_logits, dim=-1)
        horizontal_logprob = torch.log_softmax(horizontal_logits, dim=-1)
        if not self.sticky:
            self.joint_logprobs = (
                focus_logprob.unsqueeze(-1).unsqueeze(-1) +
                vertical_logprob.unsqueeze(-1) +
                horizontal_logprob
            ).flatten(start_dim=-3)
        self.joint = Categorical(logits=self.joint_logprobs)

    def sample(self):
        return self.joint.sample()

    def mode(self):
        return self.joint_logprobs.argmax(dim=-1)

    def log_prob(self, actions):
        return self.joint.log_prob(actions)

    def entropies(self):
        focus_probability = self.focus.probs
        vertical = Categorical(logits=self.vertical_logits)
        vertical_entropy = (
            focus_probability * vertical.entropy()
        ).sum(dim=-1)
        vertical_probability = vertical.probs
        horizontal_entropy = Categorical(logits=self.horizontal_logits).entropy()
        horizontal_entropy = (
            focus_probability.unsqueeze(-1) *
            vertical_probability *
            horizontal_entropy
        ).sum(dim=(-2, -1))
        return {
            'repeat': (
                -(self.repeat_logprobs.exp() * self.repeat_logprobs).sum(dim=-1)
                if self.sticky else torch.zeros_like(self.focus.entropy())
            ),
            'focus': self.focus.entropy(),
            'vertical': vertical_entropy,
            'horizontal': horizontal_entropy,
        }


class ActorCriticNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.terrain_material = nn.Embedding(4, 8)
        self.terrain = nn.Sequential(
            nn.Conv2d(15, 32, 3, padding=1),
            ResidualBlock(32),
            nn.Conv2d(32, 64, 3, stride=(2, 2), padding=1),
            ResidualBlock(64),
            nn.Conv2d(64, 64, 3, stride=(2, 1), padding=1),
            ResidualBlock(64),
            nn.AdaptiveAvgPool2d((4, 13)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 13, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
        )
        self.local_terrain = nn.Sequential(
            nn.Conv2d(15, 32, 3, padding=1),
            ResidualBlock(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            ResidualBlock(32),
            nn.Flatten(),
            nn.Linear(32 * 8 * 10, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
        )
        self.skyline = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2), nn.SiLU(),
            nn.Conv1d(16, 32, 5, stride=2, padding=2), nn.SiLU(),
            nn.AdaptiveAvgPool1d(13), nn.Flatten(),
            nn.Linear(32 * 13, 64), nn.LayerNorm(64), nn.SiLU(),
        )
        self.falling = TokenEncoder(FALLING_FEATURES, FALLING_COUNT)
        self.forecasts = TokenEncoder(FORECAST_FEATURES, FORECAST_COUNT)
        self.state = nn.Sequential(
            nn.LayerNorm(STATE_SIZE),
            nn.Linear(STATE_SIZE, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
        )
        self.body = nn.Sequential(
            nn.Linear(256 + 128 + 64 + 128 + 128 + 128, 384),
            nn.LayerNorm(384),
            nn.SiLU(),
            nn.Linear(384, 384),
            nn.SiLU(),
        )
        self.focus_actor = nn.Linear(384, 2)
        self.vertical_actor = nn.Linear(384, 2 * 3)
        self.horizontal_actor = nn.Linear(384, 2 * 3 * 3)
        self.critic = nn.Linear(384, 1)
        # Auxiliary: P(death within h frames), one logit per horizon. The critic
        # measured at chance (AUC 0.50-0.54) once matched on height -- it learned
        # a progress meter, not a danger model, because P(reach target) is a
        # Bernoulli label integrated over ~15 layers of noise and its danger
        # component is a small fraction of the return variance. Near-term death
        # is densely labelled and locally determined, so it trains, and it forces
        # the shared trunk to encode the hazard the actor needs.
        self.hazard = nn.Linear(384, len(HAZARD_HORIZONS))
        for actor in (self.focus_actor, self.vertical_actor, self.horizontal_actor):
            nn.init.zeros_(actor.weight)
            nn.init.zeros_(actor.bias)
        nn.init.zeros_(self.critic.weight)
        nn.init.zeros_(self.critic.bias)

    def encode(self, observation):
        terrain_codes = observation['terrain'].long()
        material = self.terrain_material(terrain_codes & 3)
        terrain_flags = torch.stack((
            (terrain_codes >> 2) & 1,
            (terrain_codes >> 3) & 1,
            (terrain_codes >> 4) & 1,
            (terrain_codes >> 5) & 1,
            (terrain_codes >> 6) & 1,
            (terrain_codes >> 7) & 1,
            (((terrain_codes >> 8) & 255) / 60).clamp(max=1),
        ), dim=-1).to(material.dtype)
        terrain = torch.cat((material, terrain_flags), dim=-1)
        terrain = terrain.permute(0, 3, 1, 2)
        state = observation['state'].float()
        padded = F.pad(terrain, (10, 10, 0, 0))
        player_columns = (state[..., 0] * TERRAIN_COLS).long().clamp(0, TERRAIN_COLS - 1) + 10
        offsets = torch.arange(-10, 10, device=terrain.device)
        local_columns = player_columns.unsqueeze(-1) + offsets
        local_indices = local_columns[:, None, None, :].expand(-1, terrain.shape[1], 16, -1)
        local = padded[:, :, 5:21, :].gather(3, local_indices)
        skyline = (observation['skyline'].float().unsqueeze(1) - 128) / 16
        falling = observation['falling'].float()
        forecasts = observation['forecasts'].float()
        return self.body(torch.cat((
            self.terrain(terrain),
            self.local_terrain(local),
            self.skyline(skyline),
            self.falling(falling),
            self.forecasts(forecasts),
            self.state(state),
        ), dim=-1))

    def forward(self, observation):
        hidden = self.encode(observation)
        batch_shape = hidden.shape[:-1]
        logits = (
            self.focus_actor(hidden),
            self.vertical_actor(hidden).reshape(*batch_shape, 2, 3),
            self.horizontal_actor(hidden).reshape(*batch_shape, 2, 3, 3),
        )
        return logits, self.critic(hidden).squeeze(-1), self.hazard(hidden)


class StickyActorCriticNetwork(ActorCriticNetwork):
    def __init__(self):
        super().__init__()
        self.repeat_actor = nn.Linear(384, 2)
        nn.init.zeros_(self.repeat_actor.weight)
        nn.init.zeros_(self.repeat_actor.bias)

    def forward(self, observation):
        hidden = self.encode(observation)
        batch_shape = hidden.shape[:-1]
        logits = (
            self.repeat_actor(hidden),
            self.focus_actor(hidden),
            self.vertical_actor(hidden).reshape(*batch_shape, 2, 3),
            self.horizontal_actor(hidden).reshape(*batch_shape, 2, 3, 3),
        )
        return logits, self.critic(hidden).squeeze(-1), self.hazard(hidden)


def arguments():
    parser = argparse.ArgumentParser(description='Train a PPO actor-critic on DodgeBlock v2.')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--envs-per-worker', type=int, default=64)
    parser.add_argument('--total-frames', type=int, default=100_000_000)
    parser.add_argument('--rollout', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--minibatch', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--gae-lambda', type=float, default=0.995)
    parser.add_argument('--learning-rate', type=float, default=2.5e-4)
    parser.add_argument('--learning-rate-end', type=float, default=2.5e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--clip-coef', type=float, default=0.1)
    parser.add_argument('--focus-entropy-coef-start', type=float, default=0.01)
    parser.add_argument('--focus-entropy-coef-end', type=float, default=0.0001)
    parser.add_argument('--direction-entropy-coef-start', type=float, default=0.005)
    parser.add_argument('--direction-entropy-coef-end', type=float, default=0.0001)
    parser.add_argument('--value-coef', type=float, default=0.5)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--target-kl', type=float, default=0.03)
    parser.add_argument('--target-height', type=float, default=10_000)
    parser.add_argument(
        '--reward-mode',
        choices=('target', 'height'),
        default='target',
    )
    parser.add_argument('--death-penalty', type=float, default=1.0)
    parser.add_argument('--shaping-cover', type=float, default=0.0)
    parser.add_argument('--shaping-charge', type=float, default=0.0)
    parser.add_argument('--svm-budget', type=float, default=0.0,
                        help='total bounded bonus per episode, as a fraction of '
                             'the task reward; 0 disables')
    parser.add_argument('--hazard-coef', type=float, default=0.0,
                        help='weight of the auxiliary near-term-death head; '
                             'shapes the shared trunk, does not change the reward')
    parser.add_argument('--svm-clip', type=float, default=4.0)
    parser.add_argument('--svm-epochs', type=int, default=2)
    parser.add_argument('--alive-reward', type=float, default=0.0)
    parser.add_argument('--cell-bank', action='append', default=[])
    parser.add_argument('--cell-bank-probability', type=float, default=0.8)
    parser.add_argument('--cell-heldout-fraction', type=float, default=0.1)
    parser.add_argument('--cell-band-height', type=float, default=400)
    parser.add_argument('--cell-eval-envs', type=int, default=0)
    parser.add_argument('--trajectory-bank', action='append', default=[])
    parser.add_argument('--trajectory-start-probability', type=float, default=0.5)
    parser.add_argument('--demo-dataset')
    parser.add_argument('--demo-seeds', default='1-12')
    parser.add_argument('--demo-minibatch', type=int, default=1024)
    parser.add_argument('--demo-coef-start', type=float, default=0.0)
    parser.add_argument('--demo-coef-end', type=float, default=0.0)
    parser.add_argument('--demo-focus-positive-weight', type=float, default=1.0)
    parser.add_argument('--sticky-action-head', action='store_true')
    parser.add_argument('--checkpoint-dir', default=str(Path.home() / 'dodgeblock-ppo-v2/checkpoints'))
    parser.add_argument('--checkpoint-interval', type=int, default=5_000_000)
    parser.add_argument('--log-interval', type=float, default=20.0)
    parser.add_argument('--resume')
    parser.add_argument('--initialize-from')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--compile-mode', default='default')
    return parser.parse_args()


def packet_observation(packet):
    return {key: packet[key] for key in OBSERVATION_KEYS}


def tensor_observation(observation, device):
    return {key: torch.as_tensor(value, device=device) for key, value in observation.items()}


def allocate_rollout(steps, envs, pin_memory=False):
    return {
        'terrain': torch.empty((steps, envs, TERRAIN_ROWS, TERRAIN_COLS), dtype=torch.uint16, pin_memory=pin_memory),
        'skyline': torch.empty((steps, envs, SKYLINE_SIZE), dtype=torch.uint8, pin_memory=pin_memory),
        'falling': torch.empty((steps, envs, FALLING_COUNT, FALLING_FEATURES), dtype=torch.float16, pin_memory=pin_memory),
        'forecasts': torch.empty((steps, envs, FORECAST_COUNT, FORECAST_FEATURES), dtype=torch.float16, pin_memory=pin_memory),
        'state': torch.empty((steps, envs, STATE_SIZE), dtype=torch.float32, pin_memory=pin_memory),
        'actions': torch.empty((steps, envs), dtype=torch.long, pin_memory=pin_memory),
        'logprobs': torch.empty((steps, envs), dtype=torch.float32, pin_memory=pin_memory),
        'rewards': torch.empty((steps, envs), dtype=torch.float32, pin_memory=pin_memory),
        'dones': torch.empty((steps, envs), dtype=torch.float32, pin_memory=pin_memory),
        # Death specifically, not termination: reaching the target also ends
        # an episode, and a success is not a hazard.
        'deaths': torch.empty((steps, envs), dtype=torch.float32, pin_memory=pin_memory),
        'world_scales': torch.empty((steps, envs), dtype=torch.float32, pin_memory=pin_memory),
        'values': torch.empty((steps, envs), dtype=torch.float32, pin_memory=pin_memory),
    }


def store_observation(storage, t, observation):
    storage['terrain'][t].copy_(torch.from_numpy(observation['terrain']))
    storage['skyline'][t].copy_(torch.from_numpy(observation['skyline']))
    storage['falling'][t].copy_(torch.from_numpy(observation['falling']).to(torch.float16))
    storage['forecasts'][t].copy_(torch.from_numpy(observation['forecasts']).to(torch.float16))
    storage['state'][t].copy_(torch.from_numpy(observation['state']))


def flatten_observations(storage):
    return {
        'terrain': storage['terrain'].flatten(0, 1),
        'skyline': storage['skyline'].flatten(0, 1),
        'falling': storage['falling'].flatten(0, 1),
        'forecasts': storage['forecasts'].flatten(0, 1),
        'state': storage['state'].flatten(0, 1),
    }


def minibatch_observation(flat, indices, device):
    return {key: value[indices].to(device, non_blocking=True) for key, value in flat.items()}


def atomic_checkpoint(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)
    latest = path.parent / 'latest.pt'
    latest.unlink(missing_ok=True)
    latest.symlink_to(path.name)


def training_contract(args):
    contract = {
        'version': TRAINING_CONTRACT_VERSION,
        'model_architecture': (
            STICKY_MODEL_ARCHITECTURE
            if args.sticky_action_head else MODEL_ARCHITECTURE
        ),
        'action_count': ACTION_COUNT,
        'terrain_shape': [TERRAIN_ROWS, TERRAIN_COLS],
        'target_height': args.target_height,
        'reward_mode': args.reward_mode,
        'death_penalty': args.death_penalty,
        'alive_reward': args.alive_reward,
        'shaping_cover': args.shaping_cover,
        'shaping_charge': args.shaping_charge,
        'gamma': args.gamma,
        'gae_lambda': args.gae_lambda,
        'total_frames': args.total_frames,
        'rollout': args.rollout,
        'epochs': args.epochs,
        'minibatch': args.minibatch,
        'learning_rate': args.learning_rate,
        'learning_rate_end': args.learning_rate_end,
        'focus_entropy_coef_start': args.focus_entropy_coef_start,
        'focus_entropy_coef_end': args.focus_entropy_coef_end,
        'direction_entropy_coef_start': args.direction_entropy_coef_start,
        'direction_entropy_coef_end': args.direction_entropy_coef_end,
        'cell_bank_probability': args.cell_bank_probability,
        'cell_heldout_fraction': args.cell_heldout_fraction,
        'cell_band_height': args.cell_band_height,
        'cell_eval_envs': args.cell_eval_envs,
    }
    if args.trajectory_bank:
        contract.update({
            'start_sampler': 'successful-trajectory-v5',
            'trajectory_start_probability': args.trajectory_start_probability,
            'demo_seeds': args.demo_seeds,
            'demo_minibatch': args.demo_minibatch,
            'demo_coef_start': args.demo_coef_start,
            'demo_coef_end': args.demo_coef_end,
            'demo_focus_positive_weight': args.demo_focus_positive_weight,
        })
    return contract


def checkpoint_payload(
    agent,
    optimizer,
    coordinator,
    frames,
    args,
    contract,
    bank_contract,
    demo_contract,
):
    payload = {
        'agent': agent.state_dict(),
        'optimizer': optimizer.state_dict(),
        'frames': frames,
        'model_architecture': contract['model_architecture'],
        'args': vars(args),
        'training_contract': contract,
        'cell_bank_contract': bank_contract,
        'cell_coordinator': coordinator.state_dict(),
    }
    if args.trajectory_bank:
        payload['trajectory_bank_contract'] = bank_contract
    if demo_contract:
        payload['demo_dataset_contract'] = demo_contract
    return payload


def window_stats(records):
    if not records:
        return {}
    heights = np.asarray([item[0] for item in records], np.float64)
    progress = np.asarray([item[1] for item in records], np.float64)
    returns = np.asarray([item[2] for item in records], np.float64)
    lengths = np.asarray([item[3] for item in records], np.float64)
    successes = np.asarray([item[4] for item in records], np.bool_)
    return {
        'episodes': len(records),
        'mean_height': round(float(heights.mean()), 1),
        'mean_progress': round(float(progress.mean()), 1),
        'median_progress': round(float(np.median(progress)), 1),
        'p90_progress': round(float(np.percentile(progress, 90)), 1),
        'max_progress': round(float(progress.max()), 1),
        'median_height': round(float(np.median(heights)), 1),
        'p90_height': round(float(np.percentile(heights, 90)), 1),
        'max_height': round(float(heights.max()), 1),
        'mean_return': round(float(returns.mean()), 3),
        'mean_length': round(float(lengths.mean()), 1),
        'target_success': round(float(successes.mean()), 3),
        'success_1k': round(float(np.mean(heights >= 1_000)), 3),
        'success_2_5k': round(float(np.mean(heights >= 2_500)), 3),
        'success_5k': round(float(np.mean(heights >= 5_000)), 3),
        'success_10k': round(float(np.mean(heights >= 10_000)), 3),
    }


def explained_variance(prediction, target):
    target_var = torch.var(target)
    if target_var <= 1e-8:
        return float('nan')
    return float((1 - torch.var(target - prediction) / target_var).item())


def main():
    args = arguments()
    if args.cell_bank and args.trajectory_bank:
        raise ValueError('--cell-bank and --trajectory-bank are mutually exclusive')
    if args.resume and args.initialize_from:
        raise ValueError('--resume and --initialize-from are mutually exclusive')
    if args.trajectory_bank and args.cell_eval_envs:
        raise ValueError('trajectory starts do not use held-out evaluation environments')
    if (args.demo_coef_start or args.demo_coef_end) and not args.demo_dataset:
        raise ValueError('nonzero demonstration coefficients require --demo-dataset')
    if args.demo_dataset and args.demo_minibatch <= 0:
        raise ValueError('--demo-minibatch must be positive')
    if args.death_penalty < 0 or args.alive_reward < 0:
        raise ValueError('reward coefficients cannot be negative')
    env_count = args.workers * args.envs_per_worker
    training_env_count = env_count - args.cell_eval_envs
    if not 0 <= args.cell_eval_envs < env_count:
        raise ValueError('--cell-eval-envs must be smaller than the total environment count')
    if args.cell_eval_envs and not args.cell_bank:
        raise ValueError('--cell-eval-envs requires at least one --cell-bank')
    if args.minibatch > args.rollout * training_env_count:
        raise ValueError('--minibatch cannot exceed rollout batch size')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')

    contract = training_contract(args)
    bank_paths = args.trajectory_bank or args.cell_bank
    bank_contract = CellBankCoordinator.file_contract(bank_paths)
    if args.trajectory_bank:
        coordinator = TrajectoryStartCoordinator(
            args.trajectory_bank,
            seed=args.seed ^ 0xC311_BA4C,
            probability=args.trajectory_start_probability,
            band_height=args.cell_band_height,
        )
    else:
        coordinator = CellBankCoordinator(
            args.cell_bank,
            target_height=args.target_height,
            seed=args.seed ^ 0xC311_BA4C,
            probability=args.cell_bank_probability,
            heldout_fraction=args.cell_heldout_fraction,
            band_height=args.cell_band_height,
        )
    demo_dataset = (
        DemoDataset(args.demo_dataset, parse_seeds(args.demo_seeds))
        if args.demo_dataset else None
    )
    demo_contract = demo_dataset.contract() if demo_dataset else None
    demo_rng = np.random.default_rng(args.seed ^ 0xD3A0_5EED)
    network_class = (
        StickyActorCriticNetwork
        if args.sticky_action_head else ActorCriticNetwork
    )
    agent = network_class().to(device)
    optimizer = torch.optim.AdamW(agent.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, eps=1e-5)
    frames = 0
    if args.initialize_from:
        initialized = torch.load(
            args.initialize_from,
            map_location=device,
            weights_only=False,
        )
        load_agent_state(agent, initialized['agent'])
        print(json.dumps({
            'event': 'initialize',
            'checkpoint': str(Path(args.initialize_from).resolve()),
            'stage': initialized.get('stage'),
            'source_samples': initialized.get('samples'),
        }), flush=True)
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        saved_contract = dict(saved.get('training_contract', {}))
        legacy_banks = saved_contract.pop('cell_banks', None)
        if saved_contract != contract:
            raise ValueError(
                'checkpoint training contract does not match this run; '
                'use a new checkpoint directory for a changed objective or architecture'
            )
        saved_banks = saved.get(
            'trajectory_bank_contract',
            saved.get('cell_bank_contract', legacy_banks),
        )
        current_banks = bank_contract
        banks_changed = saved_banks != current_banks
        if banks_changed:
            print(json.dumps({
                'event': 'cell_bank_change',
                'saved_banks': saved_banks,
                'current_banks': current_banks,
                'message': (
                    'cell evidence was restored by stable cell key; '
                    'variant sampling counters were reset'
                ),
            }), flush=True)
        load_agent_state(agent, saved['agent'])
        if 'hazard.weight' in saved['agent']:
            optimizer.load_state_dict(saved['optimizer'])
        else:
            # A checkpoint predating the hazard head has fewer parameters than
            # the model now has, so its optimiser state cannot be mapped back.
            # Adam moments re-warm within a few hundred steps; say so rather
            # than restoring something misaligned.
            print(json.dumps({
                'event': 'optimizer_reset',
                'reason': 'checkpoint predates the hazard head',
            }), flush=True)
        if saved.get('cell_coordinator'):
            coordinator.load_state_dict(
                saved['cell_coordinator'],
                load_variant_starts=not banks_changed,
            )
        frames = int(saved.get('frames', 0))
        if saved.get('demo_dataset_contract') != demo_contract:
            raise ValueError('checkpoint demonstration dataset does not match this run')
    if args.compile:
        agent.compile(mode=args.compile_mode, dynamic=False)

    bridge = ParallelEnvBridge(
        args.workers,
        args.envs_per_worker,
        args.seed,
        target_height=args.target_height,
        discount=args.gamma,
        reward_mode=args.reward_mode,
        death_penalty=args.death_penalty,
        alive_reward=args.alive_reward,
        cell_banks=bank_paths,
    )
    packet = bridge.read()
    eval_mask = np.zeros(env_count, dtype=bool)
    if args.cell_eval_envs:
        eval_mask[-args.cell_eval_envs:] = True
    training_mask = ~eval_mask
    eval_indices_device = torch.as_tensor(
        np.flatnonzero(eval_mask),
        dtype=torch.long,
        device=device,
    )
    pending_reset_ids = np.asarray([
        coordinator.select(heldout=bool(eval_mask[index]))
        for index in range(env_count)
    ], np.int32)
    packet = bridge.reset(pending_reset_ids)
    for variant_id in packet['current_cell_ids']:
        coordinator.record_start(int(variant_id))
    pending_reset_ids = np.asarray([
        coordinator.select(heldout=bool(eval_mask[index]))
        for index in range(env_count)
    ], np.int32)
    fresh_episodes = deque(maxlen=500)
    curriculum_episodes = deque(maxlen=500)
    heldout_episodes = deque(maxlen=500)
    checkpoint_dir = Path(args.checkpoint_dir)
    next_checkpoint = ((frames // args.checkpoint_interval) + 1) * args.checkpoint_interval
    stop = False

    def request_stop(_signum, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    started = time.time()
    last_log = started
    last_log_frames = frames
    amp = not args.no_amp
    autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == 'cuda')
    action_counts = np.zeros(ACTION_COUNT, np.int64)
    focus_counts = np.zeros(2, np.int64)
    focus_available_steps = 0
    focus_presses_available = 0
    vertical_counts = np.zeros(3, np.int64)
    horizontal_counts = np.zeros(3, np.int64)
    death_causes = Counter()
    death_phases = Counter()
    death_focus = Counter()
    phase_steps = Counter()
    sheltered_steps = Counter()
    recent = deque(maxlen=100)
    parameter_count = sum(parameter.numel() for parameter in agent.parameters())
    half_life = None if args.gamma == 1 else round(np.log(0.5) / np.log(args.gamma), 1)
    rollout = allocate_rollout(args.rollout, env_count, pin_memory=device.type == 'cuda')
    print(json.dumps({
        'event': 'start',
        'frames': frames,
        'device': str(device),
        'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'envs': env_count,
        'workers': args.workers,
        'parameters': parameter_count,
        'rollout': args.rollout,
        'minibatch': args.minibatch,
        'epochs': args.epochs,
        'gamma_per_world_frame': args.gamma,
        'discount_half_life_world_frames': half_life,
        'target_height': args.target_height,
        'reward_mode': args.reward_mode,
        'death_penalty': args.death_penalty,
        'alive_reward': args.alive_reward,
        'gae_lambda_per_world_frame': args.gae_lambda,
        'focus_entropy_coef_start': args.focus_entropy_coef_start,
        'focus_entropy_coef_end': args.focus_entropy_coef_end,
        'direction_entropy_coef_start': args.direction_entropy_coef_start,
        'direction_entropy_coef_end': args.direction_entropy_coef_end,
        'learning_rate_start': args.learning_rate,
        'learning_rate_end': args.learning_rate_end,
        'cell_banks': args.cell_bank,
        'trajectory_banks': args.trajectory_bank,
        'start_sampler': (
            'successful-trajectory-v5' if args.trajectory_bank else 'cell-v4'
        ),
        'start_probability': (
            args.trajectory_start_probability
            if args.trajectory_bank else args.cell_bank_probability
        ),
        'cell_training': coordinator.metrics(False),
        'cell_heldout': coordinator.metrics(True),
        'cell_eval_envs': args.cell_eval_envs,
        'demo_dataset': args.demo_dataset,
        'demo_seeds': parse_seeds(args.demo_seeds) if demo_dataset else [],
        'demo_frames': demo_dataset.frames if demo_dataset else 0,
        'demo_coef_start': args.demo_coef_start,
        'demo_coef_end': args.demo_coef_end,
        'compiled': args.compile,
    }), flush=True)

    shaping = bool(args.shaping_cover or args.shaping_charge)
    previous_potential = np.zeros(env_count, np.float32)
    # Success-visitation bonus: features are the state vector plus the cover
    # flag, so nothing about the policy network or its contract changes.
    svm = (SuccessBonus(STATE_SIZE + 1, device, args.svm_budget, args.svm_clip,
                        epochs=args.svm_epochs)
           if args.svm_budget > 0 else None)
    svm_features = np.zeros((args.rollout, env_count, STATE_SIZE + 1), np.float32)
    svm_labels = np.full((args.rollout, env_count), -1, np.int8)
    svm_episode_start = np.zeros(env_count, np.int32)
    recent_lengths = deque(maxlen=256)
    try:
        while frames < args.total_frames and not stop:
            collect_started = time.perf_counter()
            agent.eval()
            for t in range(args.rollout):
                observation_np = packet_observation(packet)
                store_observation(rollout, t, observation_np)
                observation = tensor_observation(observation_np, device)
                with torch.inference_mode(), autocast:
                    logits, value, _hazard = agent(observation)
                    distribution = AutoregressiveActionDistribution(logits, observation)
                    action = distribution.sample()
                    if eval_indices_device.numel():
                        action[eval_indices_device] = distribution.mode()[eval_indices_device]
                    logprob = distribution.log_prob(action)
                action_np = action.cpu().numpy().astype(np.uint8)
                training_actions = action_np[training_mask]
                training_state = observation_np['state'][training_mask]
                focus_available = (
                    ((training_state[:, 10] > 0) &
                     (training_state[:, 14] <= 0) &
                     (training_state[:, 19] <= 0)) |
                    (training_state[:, 12] > 0)
                )
                focus_available_steps += int(np.sum(focus_available))
                focus_presses_available += int(np.sum(
                    (training_actions >= 9) & focus_available
                ))
                action_counts += np.bincount(training_actions, minlength=ACTION_COUNT)
                focus_counts += np.bincount(training_actions // 9, minlength=2)
                vertical_counts += np.bincount((training_actions % 9) // 3, minlength=3)
                horizontal_counts += np.bincount(training_actions % 3, minlength=3)
                rollout['actions'][t].copy_(action.cpu())
                rollout['logprobs'][t].copy_(logprob.float().cpu())
                rollout['values'][t].copy_(value.float().cpu())
                packet = bridge.step(action_np, pending_reset_ids)
                for phase in np.unique(packet['step_phases'][training_mask]):
                    if phase:
                        phase_mask = training_mask & (packet['step_phases'] == phase)
                        phase_steps[int(phase)] += int(np.sum(phase_mask))
                        sheltered_steps[int(phase)] += int(np.sum(
                            packet['step_sheltered'][phase_mask]
                        ))
                rewards = packet['rewards']
                if svm is not None:
                    features = np.concatenate((
                        packet['state'],
                        packet['step_sheltered'].astype(np.float32)[:, None],
                    ), axis=1)
                    svm_features[t] = features
                    rewards = rewards + svm.bonus(features)
                    finished_now = packet['dones'].astype(bool)
                    if finished_now.any():
                        outcome = packet['successes'].astype(np.int8)
                        for env in np.flatnonzero(finished_now):
                            svm_labels[svm_episode_start[env]:t + 1, env] = outcome[env]
                            svm_episode_start[env] = t + 1
                        recent_lengths.extend(
                            packet['lengths'][finished_now].tolist())
                if shaping:
                    # F = Phi(s') - Phi(s). Potential-based, so the optimal
                    # policy is provably unchanged; it only shortens the credit
                    # path from "take cover now" to "survive the surge in three
                    # seconds", which sparse terminal reward cannot resolve
                    # (dP ~ 0.003 per layer against sigma ~ 0.5).
                    potential = (
                        args.shaping_cover
                        * packet['step_sheltered'].astype(np.float32)
                        * PHASE_COVER_WEIGHT[np.clip(packet['step_phases'], 0, 5)]
                        + args.shaping_charge
                        * packet['state'][:, FOCUS_STATE_INDEX]
                    ).astype(np.float32)
                    # A terminal state has potential 0 so the telescoping sum
                    # stays exact and dying is never rewarded. The env resets in
                    # the same step it dies, so `potential` already describes the
                    # NEXT episode's first state and is the right carry-forward.
                    finished = packet['dones'].astype(bool)
                    rewards = rewards + np.where(finished, 0.0, potential) - previous_potential
                    previous_potential = potential
                rollout['rewards'][t].copy_(torch.from_numpy(np.asarray(rewards, np.float32)))
                rollout['dones'][t].copy_(torch.from_numpy(packet['dones']).float())
                rollout['deaths'][t].copy_(torch.from_numpy(
                    packet['dones'].astype(bool) & ~packet['successes'].astype(bool)
                ).float())
                rollout['world_scales'][t].copy_(torch.from_numpy(packet['world_scales']))
                done_indices = np.flatnonzero(packet['dones'])
                for index in done_indices:
                    source_cell = int(packet['episode_cell_ids'][index])
                    success = bool(packet['successes'][index])
                    coordinator.record_result(source_cell, success)
                    coordinator.record_start(int(packet['current_cell_ids'][index]))
                    record = (
                        float(packet['heights'][index]),
                        float(packet['heights'][index] - packet['episode_starts'][index]),
                        float(packet['returns'][index]),
                        int(packet['lengths'][index]),
                        success,
                    )
                    if eval_mask[index]:
                        heldout_episodes.append(record)
                    elif source_cell != FRESH_CELL_ID:
                        curriculum_episodes.append(record)
                    else:
                        fresh_episodes.append(record)
                    death_causes[int(packet['death_causes'][index])] += 1
                    death_phases[int(packet['death_phases'][index])] += 1
                    death_focus[int(packet['death_focus'][index])] += 1
                    pending_reset_ids[index] = coordinator.select(
                        heldout=bool(eval_mask[index])
                    )
                frames += training_env_count
            collect_seconds = time.perf_counter() - collect_started
            if svm is not None:
                # Fit on the episodes that terminated inside this rollout; the
                # bonus applied above came from the previous fit, so the
                # discriminator never scores the batch it was just trained on.
                labelled = svm_labels >= 0
                if labelled.any():
                    svm.set_episode_length(
                        np.mean(recent_lengths) if recent_lengths else args.rollout)
                    svm.fit(svm_features[labelled], svm_labels[labelled].astype(np.float32))
                svm_labels.fill(-1)
                svm_episode_start.fill(0)

            with torch.inference_mode(), autocast:
                _, next_value, _hazard = agent(tensor_observation(packet_observation(packet), device))
                next_value = next_value.float().cpu()

            advantages = torch.zeros_like(rollout['rewards'])
            last_gae = torch.zeros(env_count)
            for t in reversed(range(args.rollout)):
                bootstrap = next_value if t == args.rollout - 1 else rollout['values'][t + 1]
                nonterminal = 1 - rollout['dones'][t]
                discount = torch.pow(torch.full_like(rollout['world_scales'][t], args.gamma), rollout['world_scales'][t])
                delta = rollout['rewards'][t] + discount * bootstrap * nonterminal - rollout['values'][t]
                trace_discount = torch.pow(
                    torch.full_like(rollout['world_scales'][t], args.gamma * args.gae_lambda),
                    rollout['world_scales'][t],
                )
                last_gae = delta + trace_discount * nonterminal * last_gae
                advantages[t] = last_gae
            returns = advantages + rollout['values']
            hazard_targets, hazard_valid, hazard_pos_weight = hazard_labels(
                rollout['deaths'], rollout['dones'])
            flat_observation = flatten_observations(rollout)
            flat_actions = rollout['actions'].flatten()
            flat_logprobs = rollout['logprobs'].flatten()
            flat_advantages = advantages.flatten()
            flat_returns = returns.flatten()
            flat_values = rollout['values'].flatten()
            flat_hazard_targets = hazard_targets.flatten(0, 1)
            flat_hazard_valid = hazard_valid.flatten(0, 1)
            training_indices = np.flatnonzero(
                np.tile(training_mask, args.rollout)
            )
            training_index_tensor = torch.as_tensor(training_indices, dtype=torch.long)
            training_advantages = flat_advantages[training_index_tensor]
            flat_advantages[training_index_tensor] = (
                (training_advantages - training_advantages.mean()) /
                (training_advantages.std() + 1e-8)
            )

            batch_size = len(training_indices)
            indices = training_indices.copy()
            update_started = time.perf_counter()
            training_progress = min(1.0, frames / max(1, args.total_frames))
            focus_entropy_coef = (
                args.focus_entropy_coef_start +
                training_progress *
                (args.focus_entropy_coef_end - args.focus_entropy_coef_start)
            )
            direction_entropy_coef = (
                args.direction_entropy_coef_start +
                training_progress *
                (args.direction_entropy_coef_end - args.direction_entropy_coef_start)
            )
            demo_coef = (
                args.demo_coef_start +
                training_progress * (args.demo_coef_end - args.demo_coef_start)
            )
            learning_rate = (
                args.learning_rate +
                training_progress * (args.learning_rate_end - args.learning_rate)
            )
            for group in optimizer.param_groups:
                group['lr'] = learning_rate
            policy_losses = []
            value_losses = []
            focus_entropies = []
            vertical_entropies = []
            horizontal_entropies = []
            repeat_entropies = []
            approx_kls = []
            clip_fracs = []
            grad_norms = []
            imitation_losses = []
            imitation_accuracies = []
            hazard_losses = []
            hazard_separations = []
            agent.train()
            for _epoch in range(args.epochs):
                np.random.shuffle(indices)
                for start in range(0, batch_size, args.minibatch):
                    mb = torch.as_tensor(indices[start:start + args.minibatch], dtype=torch.long)
                    observation = minibatch_observation(flat_observation, mb, device)
                    with autocast:
                        logits, new_value, hazard_logits = agent(observation)
                        distribution = AutoregressiveActionDistribution(logits, observation)
                        new_logprob = distribution.log_prob(flat_actions[mb].to(device, non_blocking=True))
                        component_entropy = distribution.entropies()
                        repeat_entropy = component_entropy['repeat'].mean()
                        focus_entropy = component_entropy['focus'].mean()
                        vertical_entropy = component_entropy['vertical'].mean()
                        horizontal_entropy = component_entropy['horizontal'].mean()
                        logratio = new_logprob - flat_logprobs[mb].to(device, non_blocking=True)
                        ratio = logratio.exp()
                        mb_advantages = flat_advantages[mb].to(device, non_blocking=True)
                        policy_loss_1 = -mb_advantages * ratio
                        policy_loss_2 = -mb_advantages * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef)
                        policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()
                        old_value = flat_values[mb].to(device, non_blocking=True)
                        target_return = flat_returns[mb].to(device, non_blocking=True)
                        clipped_value = old_value + (new_value - old_value).clamp(-args.clip_coef, args.clip_coef)
                        value_loss = 0.5 * torch.max(
                            (new_value - target_return).square(),
                            (clipped_value - target_return).square(),
                        ).mean()
                        hazard_loss = None
                        if args.hazard_coef:
                            mb_hazard_target = flat_hazard_targets[mb].to(device, non_blocking=True)
                            mb_hazard_valid = flat_hazard_valid[mb].to(device, non_blocking=True)
                            elementwise = F.binary_cross_entropy_with_logits(
                                hazard_logits.float(),
                                mb_hazard_target,
                                pos_weight=hazard_pos_weight.to(device),
                                reduction='none',
                            )
                            hazard_loss = (
                                (elementwise * mb_hazard_valid).sum() /
                                mb_hazard_valid.sum().clamp(min=1.0)
                            )
                            with torch.no_grad():
                                probability = torch.sigmoid(hazard_logits.float())
                                positive = mb_hazard_target * mb_hazard_valid
                                negative = (1 - mb_hazard_target) * mb_hazard_valid
                                hazard_separations.append((
                                    (probability * positive).sum(0) / positive.sum(0).clamp(min=1) -
                                    (probability * negative).sum(0) / negative.sum(0).clamp(min=1)
                                ).cpu().numpy())
                                hazard_losses.append(float(hazard_loss.item()))
                        loss = (
                            policy_loss +
                            args.value_coef * value_loss -
                            focus_entropy_coef * focus_entropy -
                            direction_entropy_coef * (
                                repeat_entropy +
                                vertical_entropy +
                                horizontal_entropy
                            )
                        )
                        if hazard_loss is not None:
                            loss = loss + args.hazard_coef * hazard_loss
                        imitation_loss = None
                        if demo_dataset and demo_coef:
                            # Rescue shards are a concatenation of independent
                            # escape prefixes, so the trajectory-order sampler
                            # (opening / initial / switch pools) is meaningless
                            # here and would aim ~30% of the auxiliary gradient
                            # at whichever rescues happened to be exported first.
                            demo_batch = demo_dataset.sample(
                                args.demo_minibatch,
                                demo_rng,
                                decision_weighted=False,
                            )
                            demo_observation, demo_actions, demo_targets = tensor_demo_batch(
                                demo_batch,
                                device,
                            )
                            demo_logits, _demo_value, _demo_hazard = agent(demo_observation)
                            imitation_loss, imitation_metrics = (
                                autoregressive_imitation_loss(
                                    demo_logits,
                                    demo_observation,
                                    demo_actions,
                                    targets=demo_targets,
                                    focus_positive_weight=(
                                        args.demo_focus_positive_weight
                                    ),
                                )
                            )
                            loss = loss + demo_coef * imitation_loss
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = ((ratio - 1).abs() > args.clip_coef).float().mean()
                    policy_losses.append(float(policy_loss.item()))
                    value_losses.append(float(value_loss.item()))
                    focus_entropies.append(float(focus_entropy.item()))
                    vertical_entropies.append(float(vertical_entropy.item()))
                    horizontal_entropies.append(float(horizontal_entropy.item()))
                    repeat_entropies.append(float(repeat_entropy.item()))
                    approx_kls.append(float(approx_kl.item()))
                    clip_fracs.append(float(clip_frac.item()))
                    grad_norms.append(float(grad_norm))
                    if imitation_loss is not None:
                        imitation_losses.append(float(imitation_loss.item()))
                        imitation_accuracies.append(
                            float(imitation_metrics['joint_accuracy'])
                        )
                if args.target_kl and approx_kls and approx_kls[-1] > args.target_kl:
                    break
            update_seconds = time.perf_counter() - update_started
            recent.append({
                'policy_loss': np.mean(policy_losses),
                'value_loss': np.mean(value_losses),
                'focus_entropy': np.mean(focus_entropies),
                'vertical_entropy': np.mean(vertical_entropies),
                'horizontal_entropy': np.mean(horizontal_entropies),
                'repeat_entropy': np.mean(repeat_entropies),
                'kl': np.mean(approx_kls),
                'clip_frac': np.mean(clip_fracs),
                'grad_norm': np.mean(grad_norms),
                'imitation_loss': (
                    np.mean(imitation_losses) if imitation_losses else 0.0
                ),
                'imitation_accuracy': (
                    np.mean(imitation_accuracies) if imitation_accuracies else 0.0
                ),
                'hazard_loss': np.mean(hazard_losses) if hazard_losses else 0.0,
                'hazard_separation': (
                    np.mean(hazard_separations, axis=0) if hazard_separations
                    else np.zeros(len(HAZARD_HORIZONS))
                ),
                'explained_variance': explained_variance(
                    flat_values[training_index_tensor],
                    flat_returns[training_index_tensor],
                ),
                'collect_seconds': collect_seconds,
                'update_seconds': update_seconds,
            })

            now = time.time()
            if now - last_log >= args.log_interval:
                total_actions = max(1, action_counts.sum())
                stats = {
                    'event': 'progress',
                    'frames': frames,
                    'sps': round((frames - last_log_frames) / (now - last_log), 1),
                    'fresh': window_stats(fresh_episodes),
                    'curriculum': window_stats(curriculum_episodes),
                    'heldout': window_stats(heldout_episodes),
                    'cell_training': coordinator.metrics(False),
                    'cell_heldout': coordinator.metrics(True),
                    'policy_loss': round(float(np.mean([item['policy_loss'] for item in recent])), 5),
                    'value_loss': round(float(np.mean([item['value_loss'] for item in recent])), 5),
                    'svm': svm.metrics if svm is not None else {},
                    'hazard': {
                        'loss': round(float(np.mean([
                            item['hazard_loss'] for item in recent])), 5),
                        'separation': {
                            str(horizon): round(float(value), 4) for horizon, value
                            in zip(HAZARD_HORIZONS, np.mean(
                                [item['hazard_separation'] for item in recent], axis=0))
                        },
                    } if args.hazard_coef else {},
                    'imitation_accuracy': round(float(np.mean([
                        item['imitation_accuracy'] for item in recent
                    ])), 4),
                    'imitation_loss': round(float(np.mean([
                        item['imitation_loss'] for item in recent
                    ])), 5),
                    'demo_coef': round(float(demo_coef), 6),
                    'entropy': {
                        component: round(float(np.mean([
                            item[f'{component}_entropy'] for item in recent
                        ])), 3)
                        for component in (
                            'repeat',
                            'focus',
                            'vertical',
                            'horizontal',
                        )
                    },
                    'kl': round(float(np.mean([item['kl'] for item in recent])), 5),
                    'clip_frac': round(float(np.mean([item['clip_frac'] for item in recent])), 3),
                    'grad_norm': round(float(np.mean([item['grad_norm'] for item in recent])), 3),
                    'explained_variance': round(float(np.mean([item['explained_variance'] for item in recent])), 3),
                    'entropy_coef': {
                        'focus': round(float(focus_entropy_coef), 6),
                        'direction': round(float(direction_entropy_coef), 6),
                    },
                    'learning_rate': round(float(learning_rate), 8),
                    'collect_fraction': round(float(np.mean([
                        item['collect_seconds'] / (item['collect_seconds'] + item['update_seconds'])
                        for item in recent
                    ])), 3),
                    'action_fractions': {
                        str(index): round(float(count / total_actions), 4)
                        for index, count in enumerate(action_counts)
                        if count
                    },
                    'action_factor_fractions': {
                        'focus': [
                            round(float(count / max(1, focus_counts.sum())), 4)
                            for count in focus_counts
                        ],
                        'vertical': [
                            round(float(count / max(1, vertical_counts.sum())), 4)
                            for count in vertical_counts
                        ],
                        'horizontal': [
                            round(float(count / max(1, horizontal_counts.sum())), 4)
                            for count in horizontal_counts
                        ],
                    },
                    'focus_available_fraction': round(
                        focus_available_steps /
                        max(1, int(action_counts.sum())),
                        4,
                    ),
                    'focus_press_given_available': round(
                        focus_presses_available /
                        max(1, focus_available_steps),
                        4,
                    ),
                    'terminal_causes': {
                        DEATH_CAUSE_NAMES.get(code, str(code)): count
                        for code, count in sorted(death_causes.items())
                    },
                    'terminal_phases': {
                        PHASE_NAMES.get(code, str(code)): count
                        for code, count in sorted(death_phases.items())
                    },
                    'terminal_focus': {
                        str(charges): count
                        for charges, count in sorted(death_focus.items())
                    },
                    'shelter_occupancy_by_phase': {
                        PHASE_NAMES.get(code, str(code)): round(
                            sheltered_steps[code] / max(1, phase_steps[code]),
                            4,
                        )
                        for code in sorted(phase_steps)
                    },
                }
                if device.type == 'cuda':
                    stats['gpu_memory_gib'] = {
                        'allocated': round(torch.cuda.memory_allocated(device) / 2**30, 2),
                        'reserved': round(torch.cuda.memory_reserved(device) / 2**30, 2),
                        'peak': round(torch.cuda.max_memory_allocated(device) / 2**30, 2),
                    }
                print(json.dumps(stats), flush=True)
                last_log = now
                last_log_frames = frames
                action_counts.fill(0)
                focus_counts.fill(0)
                focus_available_steps = 0
                focus_presses_available = 0
                vertical_counts.fill(0)
                horizontal_counts.fill(0)
                death_causes.clear()
                death_phases.clear()
                death_focus.clear()
                phase_steps.clear()
                sheltered_steps.clear()

            if frames >= next_checkpoint:
                path = checkpoint_dir / f'ppo-v2-{frames:012d}.pt'
                atomic_checkpoint(
                    path,
                    checkpoint_payload(
                        agent,
                        optimizer,
                        coordinator,
                        frames,
                        args,
                        contract,
                        bank_contract,
                        demo_contract,
                    ),
                )
                print(json.dumps({'event': 'checkpoint', 'frames': frames, 'path': str(path)}), flush=True)
                next_checkpoint += args.checkpoint_interval
    finally:
        final_path = checkpoint_dir / f'ppo-v2-{frames:012d}.pt'
        atomic_checkpoint(
            final_path,
            checkpoint_payload(
                agent,
                optimizer,
                coordinator,
                frames,
                args,
                contract,
                bank_contract,
                demo_contract,
            ),
        )
        bridge.close()
        print(json.dumps({
            'event': 'stop',
            'frames': frames,
            'elapsed_seconds': round(time.time() - started, 1),
            'checkpoint': str(final_path),
        }), flush=True)


if __name__ == '__main__':
    main()
