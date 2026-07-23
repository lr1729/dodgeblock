#!/usr/bin/env python3
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical

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
    valid_action_mask,
)
from v2_bridge import ParallelEnvBridge

TRAINING_CONTRACT_VERSION = 1
MODEL_ARCHITECTURE = 'ppo-v2-local-terrain-1'


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
        self.actor = nn.Linear(384, ACTION_COUNT)
        self.critic = nn.Linear(384, 1)
        nn.init.zeros_(self.actor.weight)
        nn.init.zeros_(self.actor.bias)
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
        logits = self.actor(hidden).masked_fill(~valid_action_mask(observation), -1e9)
        return logits, self.critic(hidden).squeeze(-1)


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
    parser.add_argument('--entropy-coef-start', type=float, default=0.01)
    parser.add_argument('--entropy-coef-end', type=float, default=0.0001)
    parser.add_argument('--value-coef', type=float, default=0.5)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--target-kl', type=float, default=0.03)
    parser.add_argument('--archive-probability', type=float, default=0.25)
    parser.add_argument('--archive-capacity', type=int, default=2048)
    parser.add_argument('--death-penalty', type=float, default=1.0)
    parser.add_argument('--alive-reward', type=float, default=0.001)
    parser.add_argument('--target-height', type=float, default=10_000)
    parser.add_argument('--reward-mode', choices=('height', 'target'), default='target')
    parser.add_argument('--demonstration', action='append', default=[])
    parser.add_argument('--demonstration-probability', type=float, default=0.8)
    parser.add_argument('--demonstration-probability-end', type=float, default=0.2)
    parser.add_argument('--demonstration-snapshot-capacity', type=int, default=256)
    parser.add_argument('--reverse-curriculum-initial-frames', type=int, default=60)
    parser.add_argument('--demonstration-randomize-probability', type=float, default=1.0)
    parser.add_argument('--checkpoint-dir', default=str(Path.home() / 'dodgeblock-ppo-v2/checkpoints'))
    parser.add_argument('--checkpoint-interval', type=int, default=5_000_000)
    parser.add_argument('--log-interval', type=float, default=20.0)
    parser.add_argument('--resume')
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
    demonstrations = []
    for filename in args.demonstration:
        path = Path(filename).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        demonstrations.append({
            'path': str(path),
            'sha256': digest,
        })
    return {
        'version': TRAINING_CONTRACT_VERSION,
        'model_architecture': MODEL_ARCHITECTURE,
        'action_count': ACTION_COUNT,
        'terrain_shape': [TERRAIN_ROWS, TERRAIN_COLS],
        'target_height': args.target_height,
        'reward_mode': args.reward_mode,
        'gamma': args.gamma,
        'gae_lambda': args.gae_lambda,
        'total_frames': args.total_frames,
        'rollout': args.rollout,
        'epochs': args.epochs,
        'minibatch': args.minibatch,
        'learning_rate': args.learning_rate,
        'learning_rate_end': args.learning_rate_end,
        'entropy_coef_start': args.entropy_coef_start,
        'entropy_coef_end': args.entropy_coef_end,
        'demonstrations': demonstrations,
        'demonstration_probability': args.demonstration_probability,
        'demonstration_probability_end': args.demonstration_probability_end,
        'demonstration_snapshot_capacity': args.demonstration_snapshot_capacity,
        'reverse_curriculum_initial_frames': args.reverse_curriculum_initial_frames,
        'demonstration_randomize_probability': args.demonstration_randomize_probability,
    }


def checkpoint_payload(agent, optimizer, frames, args, contract):
    return {
        'agent': agent.state_dict(),
        'optimizer': optimizer.state_dict(),
        'frames': frames,
        'args': vars(args),
        'training_contract': contract,
    }


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
    if args.minibatch > args.rollout * args.workers * args.envs_per_worker:
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
    agent = ActorCriticNetwork().to(device)
    optimizer = torch.optim.AdamW(agent.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, eps=1e-5)
    frames = 0
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        if saved.get('training_contract') != contract:
            raise ValueError(
                'checkpoint training contract does not match this run; '
                'use a new checkpoint directory for a changed objective or architecture'
            )
        agent.load_state_dict(saved['agent'])
        optimizer.load_state_dict(saved['optimizer'])
        frames = int(saved.get('frames', 0))
    if args.compile:
        agent.compile(mode=args.compile_mode, dynamic=False)

    env_count = args.workers * args.envs_per_worker
    bridge = ParallelEnvBridge(
        args.workers,
        args.envs_per_worker,
        args.seed,
        archive_probability=args.archive_probability,
        archive_capacity=args.archive_capacity,
        death_penalty=args.death_penalty,
        alive_reward=args.alive_reward,
        target_height=args.target_height,
        discount=args.gamma,
        reward_mode=args.reward_mode,
        demonstrations=args.demonstration,
        demonstration_probability=args.demonstration_probability if args.demonstration else 0,
        demonstration_probability_end=args.demonstration_probability_end,
        demonstration_snapshot_capacity=args.demonstration_snapshot_capacity,
        reverse_curriculum_initial_frames=args.reverse_curriculum_initial_frames,
        demonstration_randomize_probability=args.demonstration_randomize_probability,
    )
    packet = bridge.read()
    fresh_episodes = deque(maxlen=500)
    curriculum_episodes = deque(maxlen=500)
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
        'archive_probability': args.archive_probability,
        'death_penalty': args.death_penalty,
        'alive_reward': args.alive_reward,
        'target_height': args.target_height,
        'reward_mode': args.reward_mode,
        'gae_lambda_per_world_frame': args.gae_lambda,
        'entropy_coef_start': args.entropy_coef_start,
        'entropy_coef_end': args.entropy_coef_end,
        'learning_rate_start': args.learning_rate,
        'learning_rate_end': args.learning_rate_end,
        'demonstrations': args.demonstration,
        'demonstration_probability': args.demonstration_probability if args.demonstration else 0,
        'demonstration_probability_end': args.demonstration_probability_end,
        'demonstration_snapshot_capacity': args.demonstration_snapshot_capacity,
        'reverse_curriculum_initial_frames': args.reverse_curriculum_initial_frames,
        'demonstration_randomize_probability': args.demonstration_randomize_probability,
        'compiled': args.compile,
    }), flush=True)

    try:
        while frames < args.total_frames and not stop:
            collect_started = time.perf_counter()
            agent.eval()
            for t in range(args.rollout):
                observation_np = packet_observation(packet)
                store_observation(rollout, t, observation_np)
                observation = tensor_observation(observation_np, device)
                with torch.inference_mode(), autocast:
                    logits, value = agent(observation)
                    distribution = Categorical(logits=logits)
                    action = distribution.sample()
                    logprob = distribution.log_prob(action)
                action_np = action.cpu().numpy().astype(np.uint8)
                action_counts += np.bincount(action_np, minlength=ACTION_COUNT)
                rollout['actions'][t].copy_(action.cpu())
                rollout['logprobs'][t].copy_(logprob.float().cpu())
                rollout['values'][t].copy_(value.float().cpu())
                packet = bridge.step(action_np)
                rollout['rewards'][t].copy_(torch.from_numpy(packet['rewards']))
                rollout['dones'][t].copy_(torch.from_numpy(packet['dones']).float())
                rollout['world_scales'][t].copy_(torch.from_numpy(packet['world_scales']))
                done_indices = np.flatnonzero(packet['dones'])
                for index in done_indices:
                    record = (
                        float(packet['heights'][index]),
                        float(packet['heights'][index] - packet['episode_starts'][index]),
                        float(packet['returns'][index]),
                        int(packet['lengths'][index]),
                        bool(packet['successes'][index]),
                    )
                    if packet['episode_starts'][index] > 0:
                        curriculum_episodes.append(record)
                    else:
                        fresh_episodes.append(record)
                frames += env_count
            collect_seconds = time.perf_counter() - collect_started

            with torch.inference_mode(), autocast:
                _, next_value = agent(tensor_observation(packet_observation(packet), device))
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
            flat_observation = flatten_observations(rollout)
            flat_actions = rollout['actions'].flatten()
            flat_logprobs = rollout['logprobs'].flatten()
            flat_advantages = advantages.flatten()
            flat_returns = returns.flatten()
            flat_values = rollout['values'].flatten()
            flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

            batch_size = flat_actions.numel()
            indices = np.arange(batch_size)
            update_started = time.perf_counter()
            training_progress = min(1.0, frames / max(1, args.total_frames))
            entropy_coef = (
                args.entropy_coef_start +
                training_progress * (args.entropy_coef_end - args.entropy_coef_start)
            )
            learning_rate = (
                args.learning_rate +
                training_progress * (args.learning_rate_end - args.learning_rate)
            )
            for group in optimizer.param_groups:
                group['lr'] = learning_rate
            policy_losses = []
            value_losses = []
            entropies = []
            approx_kls = []
            clip_fracs = []
            grad_norms = []
            agent.train()
            for _epoch in range(args.epochs):
                np.random.shuffle(indices)
                for start in range(0, batch_size, args.minibatch):
                    mb = torch.as_tensor(indices[start:start + args.minibatch], dtype=torch.long)
                    observation = minibatch_observation(flat_observation, mb, device)
                    with autocast:
                        logits, new_value = agent(observation)
                        distribution = Categorical(logits=logits)
                        new_logprob = distribution.log_prob(flat_actions[mb].to(device, non_blocking=True))
                        entropy = distribution.entropy().mean()
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
                        loss = policy_loss + args.value_coef * value_loss - entropy_coef * entropy
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = ((ratio - 1).abs() > args.clip_coef).float().mean()
                    policy_losses.append(float(policy_loss.item()))
                    value_losses.append(float(value_loss.item()))
                    entropies.append(float(entropy.item()))
                    approx_kls.append(float(approx_kl.item()))
                    clip_fracs.append(float(clip_frac.item()))
                    grad_norms.append(float(grad_norm))
                if args.target_kl and approx_kls and approx_kls[-1] > args.target_kl:
                    break
            update_seconds = time.perf_counter() - update_started
            recent.append({
                'policy_loss': np.mean(policy_losses),
                'value_loss': np.mean(value_losses),
                'entropy': np.mean(entropies),
                'kl': np.mean(approx_kls),
                'clip_frac': np.mean(clip_fracs),
                'grad_norm': np.mean(grad_norms),
                'explained_variance': explained_variance(flat_values, flat_returns),
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
                    'policy_loss': round(float(np.mean([item['policy_loss'] for item in recent])), 5),
                    'value_loss': round(float(np.mean([item['value_loss'] for item in recent])), 5),
                    'entropy': round(float(np.mean([item['entropy'] for item in recent])), 3),
                    'kl': round(float(np.mean([item['kl'] for item in recent])), 5),
                    'clip_frac': round(float(np.mean([item['clip_frac'] for item in recent])), 3),
                    'grad_norm': round(float(np.mean([item['grad_norm'] for item in recent])), 3),
                    'explained_variance': round(float(np.mean([item['explained_variance'] for item in recent])), 3),
                    'entropy_coef': round(float(entropy_coef), 6),
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

            if frames >= next_checkpoint:
                path = checkpoint_dir / f'ppo-v2-{frames:012d}.pt'
                atomic_checkpoint(
                    path,
                    checkpoint_payload(agent, optimizer, frames, args, contract),
                )
                print(json.dumps({'event': 'checkpoint', 'frames': frames, 'path': str(path)}), flush=True)
                next_checkpoint += args.checkpoint_interval
    finally:
        final_path = checkpoint_dir / f'ppo-v2-{frames:012d}.pt'
        atomic_checkpoint(
            final_path,
            checkpoint_payload(agent, optimizer, frames, args, contract),
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
