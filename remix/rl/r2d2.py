#!/usr/bin/env python3
import math

import numpy as np
import torch
from torch import nn

from v2_bridge import (
    ACTION_COUNT,
    FALLING_COUNT,
    FALLING_FEATURES,
    FORECAST_COUNT,
    FORECAST_FEATURES,
    SKYLINE_SIZE,
    STATE_SIZE,
    TERRAIN_COLS,
    TERRAIN_ROWS,
)


OBSERVATION_KEYS = ('terrain', 'skyline', 'falling', 'forecasts', 'state')
FOCUS_CHARGE_STATE = 10
FOCUS_AIM_STATE = 12
FOCUS_DASH_STATE = 14
FOCUS_HELD_STATE = 19


def valid_action_mask(observation):
    state = observation['state']
    mask = torch.ones((*state.shape[:-1], ACTION_COUNT), dtype=torch.bool, device=state.device)
    aiming = state[..., FOCUS_AIM_STATE] > 0
    can_press_focus = (
        (state[..., FOCUS_CHARGE_STATE] > 0) &
        (state[..., FOCUS_DASH_STATE] <= 0) &
        (state[..., FOCUS_HELD_STATE] <= 0)
    )
    mask[..., 9:] = (aiming | can_press_focus).unsqueeze(-1)
    return mask


def valid_action_mask_numpy(state):
    shape = (*state.shape[:-1], ACTION_COUNT)
    mask = np.ones(shape, dtype=bool)
    aiming = state[..., FOCUS_AIM_STATE] > 0
    can_press_focus = (
        (state[..., FOCUS_CHARGE_STATE] > 0) &
        (state[..., FOCUS_DASH_STATE] <= 0) &
        (state[..., FOCUS_HELD_STATE] <= 0)
    )
    mask[..., 9:] = np.expand_dims(aiming | can_press_focus, -1)
    return mask


def greedy_actions(quantiles, observation):
    values = quantiles.float().mean(dim=-1)
    return values.masked_fill(~valid_action_mask(observation), -torch.inf).argmax(dim=-1)


def sample_valid_actions(state, rng):
    mask = valid_action_mask_numpy(state)
    scores = rng.random(mask.shape)
    scores[~mask] = -1
    return scores.argmax(axis=-1).astype(np.uint8)


class PersistentEpsilonExplorer:
    def __init__(self, epsilons, hold, rng):
        if hold < 1:
            raise ValueError('exploration hold must be at least one frame')
        self.epsilons = np.asarray(epsilons, np.float64)
        self.hold = hold
        self.rng = rng
        self.actions = np.zeros(len(self.epsilons), np.uint8)
        self.remaining = np.zeros(len(self.epsilons), np.int32)
        self.start_probability = self.epsilons / (
            hold - self.epsilons * (hold - 1)
        )

    def select(self, greedy, state):
        greedy = np.asarray(greedy, np.uint8)
        valid = valid_action_mask_numpy(state)
        indices = np.arange(len(greedy))
        self.remaining[~valid[indices, self.actions]] = 0

        selected = greedy.copy()
        active = self.remaining > 0
        selected[active] = self.actions[active]
        self.remaining[active] -= 1

        idle = ~active
        starting = idle & (self.rng.random(len(greedy)) < self.start_probability)
        if starting.any():
            sampled = sample_valid_actions(state, self.rng)
            self.actions[starting] = sampled[starting]
            self.remaining[starting] = self.hold - 1
            selected[starting] = sampled[starting]
        return selected

    def reset(self, dones):
        self.remaining[np.asarray(dones, dtype=bool)] = 0


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(4, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(4, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class TokenEncoder(nn.Module):
    def __init__(self, features, count, output=128, queries=4):
        super().__init__()
        self.features = features
        self.count = count
        self.token = nn.Sequential(
            nn.LayerNorm(features - 1),
            nn.Linear(features - 1, output),
            nn.SiLU(),
            nn.Linear(output, output),
            nn.SiLU(),
        )
        self.queries = nn.Parameter(torch.empty(queries, output))
        nn.init.normal_(self.queries, std=output ** -0.5)
        self.project = nn.Sequential(
            nn.Linear(output * (queries + 2) + 1, output),
            nn.LayerNorm(output),
            nn.SiLU(),
        )

    def forward(self, values):
        mask = values[..., -1:] > 0
        encoded = self.token(values[..., :-1]) * mask
        token_count = mask.sum(dim=-2)
        mean = encoded.sum(dim=-2) / token_count.clamp_min(1)
        maximum = encoded.masked_fill(~mask, -torch.inf).amax(dim=-2)
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        scores = torch.einsum('...nd,qd->...qn', encoded, self.queries) / math.sqrt(encoded.shape[-1])
        scores = scores.masked_fill(~mask.squeeze(-1).unsqueeze(-2), -1e4)
        attention = torch.softmax(scores, dim=-1) * mask.squeeze(-1).unsqueeze(-2)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        pooled = torch.einsum('...qn,...nd->...qd', attention, encoded).flatten(start_dim=-2)
        density = token_count.to(encoded.dtype) / self.count
        return self.project(torch.cat((mean, maximum, pooled, density), dim=-1))


class RecurrentQNetwork(nn.Module):
    def __init__(self, hidden_size=512, quantiles=51):
        super().__init__()
        self.hidden_size = hidden_size
        self.quantiles = quantiles
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
        self.fusion = nn.Sequential(
            nn.Linear(256 + 64 + 128 + 128 + 128, 384),
            nn.LayerNorm(384),
            nn.SiLU(),
        )
        self.recurrent = nn.GRU(384, hidden_size, batch_first=True)
        self.value = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, quantiles))
        self.advantage = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, ACTION_COUNT * quantiles),
        )

    def initial_hidden(self, batch, device=None):
        return torch.zeros(1, batch, self.hidden_size, device=device)

    def encode(self, observation):
        terrain = observation['terrain']
        sequence = terrain.ndim == 4
        if not sequence:
            observation = {key: value.unsqueeze(1) for key, value in observation.items()}
            terrain = observation['terrain']
        batch, steps = terrain.shape[:2]
        flat = batch * steps

        terrain_codes = terrain.long()
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
        terrain = terrain.permute(0, 1, 4, 2, 3).reshape(flat, 15, TERRAIN_ROWS, TERRAIN_COLS)
        skyline = (observation['skyline'].float().reshape(flat, 1, SKYLINE_SIZE) - 128) / 16
        falling = observation['falling'].float().reshape(flat, FALLING_COUNT, FALLING_FEATURES)
        forecasts = observation['forecasts'].float().reshape(flat, FORECAST_COUNT, FORECAST_FEATURES)
        state = observation['state'].float().reshape(flat, STATE_SIZE)
        features = torch.cat((
            self.terrain(terrain),
            self.skyline(skyline),
            self.falling(falling),
            self.forecasts(forecasts),
            self.state(state),
        ), dim=-1)
        encoded = self.fusion(features).reshape(batch, steps, -1)
        return encoded, sequence

    def forward(self, observation, hidden=None):
        encoded, sequence = self.encode(observation)
        output, hidden = self.recurrent(encoded, hidden)
        value = self.value(output).unsqueeze(-2)
        advantage = self.advantage(output).reshape(*output.shape[:2], ACTION_COUNT, self.quantiles)
        quantile_values = value + advantage - advantage.mean(dim=-2, keepdim=True)
        if not sequence:
            quantile_values = quantile_values[:, 0]
        return quantile_values, hidden


def copy_observation(packet, index):
    return {key: packet[key][index].copy() for key in OBSERVATION_KEYS}


class SequenceAssembler:
    def __init__(self, env_count, burn_in, unroll, n_step, stride=None):
        self.env_count = env_count
        self.burn_in = burn_in
        self.unroll = unroll
        self.n_step = n_step
        self.steps = burn_in + unroll + n_step
        self.stride = stride or unroll
        self.buffers = [self._empty() for _ in range(env_count)]

    @staticmethod
    def _empty():
        return {'observations': [], 'actions': [], 'rewards': [], 'dones': [], 'world_scales': []}

    def initialize(self, packet):
        for index, buffer in enumerate(self.buffers):
            buffer['observations'].append(copy_observation(packet, index))

    @staticmethod
    def _stack_observations(observations):
        return {key: np.stack([item[key] for item in observations]) for key in OBSERVATION_KEYS}

    def _sequence(self, buffer, padded=False):
        observations = list(buffer['observations'])
        actions = list(buffer['actions'])
        rewards = list(buffer['rewards'])
        dones = list(buffer['dones'])
        world_scales = list(buffer['world_scales'])
        valid = [1] * len(actions)

        if padded and len(actions) < self.steps:
            history = max(0, len(actions) - (self.unroll + self.n_step))
            left = max(0, self.burn_in - history)
            if left:
                observations = [observations[0]] * left + observations
                actions = [0] * left + actions
                rewards = [0.0] * left + rewards
                dones = [1] * left + dones
                world_scales = [1.0] * left + world_scales
                valid = [0] * left + valid
            right = self.steps - len(actions)
            if right:
                observations += [observations[-1]] * right
                actions += [0] * right
                rewards += [0.0] * right
                dones += [1] * right
                world_scales += [1.0] * right
                valid += [0] * right

        if len(actions) > self.steps:
            start = len(actions) - self.steps
            observations = observations[start:]
            actions = actions[start:]
            rewards = rewards[start:]
            dones = dones[start:]
            world_scales = world_scales[start:]
            valid = valid[start:]

        assert len(actions) == self.steps
        assert len(observations) == self.steps + 1
        return {
            'observations': self._stack_observations(observations),
            'actions': np.asarray(actions, np.uint8),
            'rewards': np.asarray(rewards, np.float32),
            'dones': np.asarray(dones, np.uint8),
            'world_scales': np.asarray(world_scales, np.float32),
            'valid': np.asarray(valid, np.uint8),
        }

    def append(self, actions, packet):
        completed = []
        for index, buffer in enumerate(self.buffers):
            buffer['actions'].append(int(actions[index]))
            buffer['rewards'].append(float(packet['rewards'][index]))
            buffer['dones'].append(int(packet['dones'][index]))
            buffer['world_scales'].append(float(packet['world_scales'][index]))
            buffer['observations'].append(copy_observation(packet, index))

            while len(buffer['actions']) >= self.steps:
                view = {
                    'observations': buffer['observations'][:self.steps + 1],
                    'actions': buffer['actions'][:self.steps],
                    'rewards': buffer['rewards'][:self.steps],
                    'dones': buffer['dones'][:self.steps],
                    'world_scales': buffer['world_scales'][:self.steps],
                }
                completed.append(self._sequence(view))
                for key in ('actions', 'rewards', 'dones', 'world_scales'):
                    del buffer[key][:self.stride]
                del buffer['observations'][:self.stride]

            if packet['dones'][index]:
                if buffer['actions']:
                    completed.append(self._sequence(buffer, padded=True))
                self.buffers[index] = self._empty()
                self.buffers[index]['observations'].append(copy_observation(packet, index))
        return completed


class PrioritizedSequenceReplay:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.observations = None
        self.actions = None
        self.rewards = None
        self.dones = None
        self.world_scales = None
        self.valid = None
        self.priorities = np.zeros(capacity, np.float32)
        self.cursor = 0
        self.size = 0
        self.max_priority = 1.0

    def __len__(self):
        return self.size

    def _allocate(self, sequence):
        self.observations = {
            key: np.empty(
                (self.capacity, *values.shape),
                dtype=np.float16 if np.issubdtype(values.dtype, np.floating) else values.dtype,
            )
            for key, values in sequence['observations'].items()
        }
        steps = sequence['actions'].shape
        self.actions = np.empty((self.capacity, *steps), np.uint8)
        self.rewards = np.empty((self.capacity, *steps), np.float16)
        self.dones = np.empty((self.capacity, *steps), np.uint8)
        self.world_scales = np.empty((self.capacity, *steps), np.float16)
        self.valid = np.empty((self.capacity, *steps), np.uint8)

    def add(self, sequence, priority=None):
        if self.observations is None:
            self._allocate(sequence)
        priority = self.max_priority if priority is None else max(float(priority), 1e-4)
        index = self.cursor
        for key, values in sequence['observations'].items():
            self.observations[key][index] = values
        self.actions[index] = sequence['actions']
        self.rewards[index] = sequence['rewards']
        self.dones[index] = sequence['dones']
        self.world_scales[index] = sequence['world_scales']
        self.valid[index] = sequence['valid']
        self.priorities[index] = priority
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)
        self.max_priority = max(self.max_priority, priority)

    def sample(self, batch_size, beta, rng):
        size = self.size
        scaled = np.power(self.priorities[:size].clip(min=1e-6), self.alpha)
        probabilities = scaled / scaled.sum()
        indices = rng.choice(size, batch_size, replace=size < batch_size, p=probabilities)
        weights = np.power(size * probabilities[indices], -beta)
        weights /= weights.max()
        batch = {
            'observations': {key: values[indices] for key, values in self.observations.items()},
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'dones': self.dones[indices],
            'world_scales': self.world_scales[indices],
            'valid': self.valid[indices],
        }
        return indices, weights.astype(np.float32), batch

    def update_priorities(self, indices, priorities):
        for index, priority in zip(indices, priorities):
            value = max(float(priority), 1e-4)
            self.priorities[index] = value
            self.max_priority = max(self.max_priority, value)


def tensor_observation(observation, device):
    return {key: torch.as_tensor(value, device=device) for key, value in observation.items()}


def slice_observation(observation, start=None, end=None):
    return {key: value[:, start:end] for key, value in observation.items()}


def world_discounts(gamma, world_scales):
    return torch.pow(gamma, world_scales.float())


def quantile_huber_loss(prediction, target, valid, weights, kappa=1.0):
    delta = target.unsqueeze(-2) - prediction.unsqueeze(-1)
    absolute = delta.abs()
    huber = torch.where(absolute <= kappa, 0.5 * delta.square(), kappa * (absolute - 0.5 * kappa))
    count = prediction.shape[-1]
    taus = (torch.arange(count, device=prediction.device, dtype=prediction.dtype) + 0.5) / count
    quantile_weight = (taus.view(1, 1, count, 1) - (delta.detach() < 0).to(prediction.dtype)).abs()
    per_step = (quantile_weight * huber / kappa).mean(dim=(-1, -2))
    mask = valid.to(per_step.dtype)
    per_sequence = (per_step * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    return (per_sequence * weights).mean(), per_step


def learn_batch(
    online,
    target,
    optimizer,
    batch,
    importance_weights,
    burn_in,
    unroll,
    n_step,
    gamma,
    device,
    amp=True,
    grad_clip=40.0,
):
    observation = tensor_observation(batch['observations'], device)
    actions = torch.as_tensor(batch['actions'], device=device, dtype=torch.long)
    rewards = torch.as_tensor(batch['rewards'], device=device)
    dones = torch.as_tensor(batch['dones'], device=device, dtype=torch.float32)
    world_scales = torch.as_tensor(batch['world_scales'], device=device, dtype=torch.float32)
    valid = torch.as_tensor(batch['valid'], device=device, dtype=torch.float32)
    weights = torch.as_tensor(importance_weights, device=device)
    autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == 'cuda')

    with torch.no_grad(), autocast:
        _, online_hidden = online(slice_observation(observation, 0, burn_in))
        _, target_hidden = target(slice_observation(observation, 0, burn_in))

    post = slice_observation(observation, burn_in, None)
    with autocast:
        online_quantiles, _ = online(post, online_hidden.detach())
        with torch.no_grad():
            target_quantiles, _ = target(post, target_hidden.detach())

        prediction_all = online_quantiles[:, :unroll]
        learn_actions = actions[:, burn_in:burn_in + unroll]
        prediction = prediction_all.gather(
            2,
            learn_actions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, online.quantiles),
        ).squeeze(2)

        with torch.no_grad():
            returns = torch.zeros_like(prediction)
            alive = torch.ones_like(rewards[:, burn_in:burn_in + unroll])
            discount = torch.ones_like(alive)
            for offset in range(n_step):
                step_rewards = rewards[:, burn_in + offset:burn_in + offset + unroll]
                step_dones = dones[:, burn_in + offset:burn_in + offset + unroll]
                step_scales = world_scales[:, burn_in + offset:burn_in + offset + unroll]
                returns += (alive * discount * step_rewards).unsqueeze(-1)
                alive *= 1 - step_dones
                discount *= world_discounts(gamma, step_scales)

            bootstrap_online = online_quantiles[:, n_step:n_step + unroll].mean(dim=-1)
            bootstrap_mask = valid_action_mask(post)[:, n_step:n_step + unroll]
            bootstrap_online = bootstrap_online.masked_fill(~bootstrap_mask, -torch.inf)
            bootstrap_actions = bootstrap_online.argmax(dim=-1)
            bootstrap_target = target_quantiles[:, n_step:n_step + unroll].gather(
                2,
                bootstrap_actions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, online.quantiles),
            ).squeeze(2)
            returns += (alive * discount).unsqueeze(-1) * bootstrap_target

        learn_valid = valid[:, burn_in:burn_in + unroll]
        loss, _ = quantile_huber_loss(prediction.float(), returns.float(), learn_valid, weights)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(online.parameters(), grad_clip)
    optimizer.step()

    with torch.no_grad():
        td = (returns.mean(dim=-1) - prediction.mean(dim=-1)).abs() * learn_valid
        maximum = td.amax(dim=1)
        mean = td.sum(dim=1) / learn_valid.sum(dim=1).clamp_min(1)
        priorities = 0.9 * maximum + 0.1 * mean + 1e-4
        q_mean = prediction.mean().item()
    return {
        'loss': float(loss.item()),
        'grad_norm': float(grad_norm),
        'q_mean': q_mean,
        'priorities': priorities.float().cpu().numpy(),
    }


@torch.no_grad()
def ema_update(target, online, tau):
    for target_parameter, online_parameter in zip(target.parameters(), online.parameters()):
        target_parameter.lerp_(online_parameter, tau)


def exploration_rates(count):
    if count == 1:
        return np.zeros(1, np.float32)
    indices = np.arange(count, dtype=np.float32)
    return np.power(0.4, 1 + 7 * indices / (count - 1)).astype(np.float32)


def interquartile_mean(values):
    values = np.sort(np.asarray(values, np.float64))
    if not len(values):
        return math.nan
    lo = int(math.floor(len(values) * 0.25))
    hi = int(math.ceil(len(values) * 0.75))
    return float(values[lo:hi].mean())
