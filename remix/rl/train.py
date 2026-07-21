#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions.categorical import Categorical

CHANNELS, ROWS, COLS, STATE_SIZE, ACTIONS = 8, 20, 52, 16, 14
OBS_SIZE = CHANNELS * ROWS * COLS


class EnvBridge:
    def __init__(self, count, seed, hardcore=False):
        command = ['node', str(Path(__file__).with_name('env-server.mjs')), '--envs', str(count), '--seed', str(seed)]
        if hardcore:
            command.append('--hardcore')
        self.count = count
        self.packet_size = count * (OBS_SIZE + STATE_SIZE * 4 + 4 + 1 + 4 + 4 + 4)
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def _read_exact(self):
        data = bytearray(self.packet_size)
        view = memoryview(data)
        offset = 0
        while offset < len(data):
            size = self.process.stdout.readinto(view[offset:])
            if not size:
                raise RuntimeError(f'environment server exited with {self.process.poll()}')
            offset += size
        return data

    def read(self):
        data = self._read_exact()
        offset = self.count * OBS_SIZE
        obs = np.frombuffer(data, np.uint8, self.count * OBS_SIZE, 0).reshape(self.count, CHANNELS, ROWS, COLS).copy()
        states = np.frombuffer(data, '<f4', self.count * STATE_SIZE, offset).reshape(self.count, STATE_SIZE).copy()
        offset += self.count * STATE_SIZE * 4
        rewards = np.frombuffer(data, '<f4', self.count, offset).copy(); offset += self.count * 4
        dones = np.frombuffer(data, np.uint8, self.count, offset).copy(); offset += self.count
        returns = np.frombuffer(data, '<f4', self.count, offset).copy(); offset += self.count * 4
        lengths = np.frombuffer(data, '<u4', self.count, offset).copy(); offset += self.count * 4
        heights = np.frombuffer(data, '<f4', self.count, offset).copy()
        return obs, states, rewards, dones, returns, lengths, heights

    def step(self, actions):
        self.process.stdin.write(np.asarray(actions, np.uint8).tobytes())
        self.process.stdin.flush()
        return self.read()

    def close(self):
        self.process.terminate()
        self.process.wait(timeout=5)


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Sequential(
            nn.Conv2d(CHANNELS, 24, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 8)), nn.Flatten(),
            nn.Linear(48 * 4 * 8, 192), nn.SiLU(),
        )
        self.state = nn.Sequential(nn.Linear(STATE_SIZE, 64), nn.SiLU())
        self.body = nn.Sequential(nn.Linear(256, 256), nn.SiLU())
        self.actor = nn.Linear(256, ACTIONS)
        self.critic = nn.Linear(256, 1)

    def forward(self, obs, state):
        hidden = self.body(torch.cat((self.visual(obs.float().div_(255)), self.state(state)), dim=1))
        return self.actor(hidden), self.critic(hidden).squeeze(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--envs', type=int, default=128)
    parser.add_argument('--rollout', type=int, default=256)
    parser.add_argument('--total-steps', type=int, default=50_000_000)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--minibatch', type=int, default=1024)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=12)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--hardcore', action='store_true')
    parser.add_argument('--checkpoint-dir', default=str(Path.home() / 'dodgeblock-rl/checkpoints'))
    parser.add_argument('--resume')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    agent = Agent().to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4, eps=1e-5)
    step = 0
    if args.resume:
        saved = torch.load(args.resume, map_location=device)
        agent.load_state_dict(saved['agent']); optimizer.load_state_dict(saved['optimizer']); step = saved['step']

    bridge = EnvBridge(args.envs, args.seed, args.hardcore)
    obs_np, state_np, *_ = bridge.read()
    checkpoint_dir = Path(args.checkpoint_dir); checkpoint_dir.mkdir(parents=True, exist_ok=True)
    batch = args.envs * args.rollout
    updates = max(1, (args.total_steps - step) // batch)
    started = time.time()
    episode_window = []
    try:
        for update in range(1, updates + 1):
            obs_store = torch.empty((args.rollout, args.envs, CHANNELS, ROWS, COLS), dtype=torch.uint8)
            state_store = torch.empty((args.rollout, args.envs, STATE_SIZE))
            actions = torch.empty((args.rollout, args.envs), dtype=torch.long)
            logprobs = torch.empty((args.rollout, args.envs))
            rewards = torch.empty((args.rollout, args.envs))
            dones = torch.empty((args.rollout, args.envs))
            values = torch.empty((args.rollout, args.envs))
            for t in range(args.rollout):
                obs = torch.from_numpy(obs_np); state = torch.from_numpy(state_np)
                obs_store[t].copy_(obs); state_store[t].copy_(state)
                with torch.inference_mode():
                    logits, value = agent(obs.to(device), state.to(device))
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                packet = bridge.step(action.cpu().numpy())
                obs_np, state_np, reward_np, done_np, return_np, length_np, height_np = packet
                actions[t].copy_(action.cpu()); logprobs[t].copy_(dist.log_prob(action).cpu()); values[t].copy_(value.cpu())
                rewards[t].copy_(torch.from_numpy(reward_np)); dones[t].copy_(torch.from_numpy(done_np).float())
                for i in np.flatnonzero(done_np):
                    episode_window.append((float(return_np[i]), int(length_np[i]), float(height_np[i])))
                episode_window = episode_window[-200:]

            with torch.inference_mode():
                _, next_value = agent(torch.from_numpy(obs_np).to(device), torch.from_numpy(state_np).to(device))
                next_value = next_value.cpu()
            advantages = torch.zeros_like(rewards); last_gae = torch.zeros(args.envs)
            for t in reversed(range(args.rollout)):
                bootstrap = next_value if t == args.rollout - 1 else values[t + 1]
                nonterminal = 1 - dones[t]
                delta = rewards[t] + 0.99 * bootstrap * nonterminal - values[t]
                last_gae = delta + 0.99 * 0.95 * nonterminal * last_gae
                advantages[t] = last_gae
            returns = advantages + values
            flat_obs = obs_store.flatten(0, 1); flat_state = state_store.flatten(0, 1)
            flat_actions = actions.flatten(); flat_logprobs = logprobs.flatten()
            flat_adv = advantages.flatten(); flat_returns = returns.flatten(); flat_values = values.flatten()
            indices = np.arange(batch)
            for _ in range(args.epochs):
                np.random.shuffle(indices)
                for start in range(0, batch, args.minibatch):
                    mb = torch.as_tensor(indices[start:start + args.minibatch], device=device)
                    logits, new_value = agent(flat_obs[mb.cpu()].to(device), flat_state[mb.cpu()].to(device))
                    dist = Categorical(logits=logits)
                    new_logprob = dist.log_prob(flat_actions[mb.cpu()].to(device))
                    ratio = (new_logprob - flat_logprobs[mb.cpu()].to(device)).exp()
                    adv = flat_adv[mb.cpu()].to(device); adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                    policy_loss = torch.max(-adv * ratio, -adv * ratio.clamp(0.8, 1.2)).mean()
                    old_value = flat_values[mb.cpu()].to(device); target = flat_returns[mb.cpu()].to(device)
                    clipped = old_value + (new_value - old_value).clamp(-0.2, 0.2)
                    value_loss = 0.5 * torch.max((new_value - target).square(), (clipped - target).square()).mean()
                    loss = policy_loss + 0.5 * value_loss - 0.01 * dist.entropy().mean()
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), 0.5); optimizer.step()
            step += batch
            stats = {'update': update, 'step': step, 'sps': round((step) / (time.time() - started), 1)}
            if episode_window:
                stats.update({
                    'episodes': len(episode_window),
                    'mean_return': round(float(np.mean([x[0] for x in episode_window])), 3),
                    'mean_length': round(float(np.mean([x[1] for x in episode_window])), 1),
                    'mean_height': round(float(np.mean([x[2] for x in episode_window])), 1),
                    'max_height': round(max(x[2] for x in episode_window), 1),
                })
            print(json.dumps(stats), flush=True)
            if update % 10 == 0 or update == updates:
                target = checkpoint_dir / f'ppo-{step:012d}.pt'
                temporary = target.with_suffix('.tmp')
                torch.save({'agent': agent.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step, 'args': vars(args)}, temporary)
                temporary.replace(target)
                latest = checkpoint_dir / 'latest.pt'; latest.unlink(missing_ok=True); latest.symlink_to(target.name)
    finally:
        bridge.close()


if __name__ == '__main__':
    main()
