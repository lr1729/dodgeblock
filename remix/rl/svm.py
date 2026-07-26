#!/usr/bin/env python3
"""Success-visitation bonus: a bounded, self-generated dense reward.

Trains a discriminator by plain BCE between states drawn from the policy's OWN
successful episodes and its own failed ones, then adds a bounded log-odds bonus
to the sparse task reward. Following SVM (arXiv:2606.23640) for the objective
and DEMO3 (arXiv:2503.01837) for the bound.

Why this shape, given what this project measured:

- Potential-based shaping was falsified structurally: at gamma = 1 it telescopes
  to a constant per episode and cannot change how trajectories rank. A useful
  bonus must be BIASED — it must change what is optimal.
- Rescue distillation collapsed at coefficient 0.3 and was still failing at
  0.05. Tuning a weight is the wrong control surface. Here the per-step bonus is
  bounded by tanh and scaled by a per-episode budget, so its total contribution
  cannot exceed a fixed fraction of the task reward no matter what the
  discriminator outputs.
- No demonstrations are involved, so there is no survivorship or search-residue
  problem: the positives are by construction reachable by the current policy.

The discriminator reads a compact feature vector (the 38-dim state plus the
cover flag) rather than the full observation, so it costs nothing and requires
no change to the policy network or its training contract.
"""
import numpy as np
import torch
from torch import nn


class SuccessDiscriminator(nn.Module):
    def __init__(self, features, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(features),
            nn.Linear(features, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        return self.net(features).squeeze(-1)


class SuccessBonus:
    """Owns the discriminator, its optimiser, and the bounded bonus."""

    def __init__(self, features, device, budget, clip, learning_rate=3e-4,
                 epochs=2, minibatch=4096, max_samples=200_000):
        self.model = SuccessDiscriminator(features).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.device = device
        self.budget = budget
        self.clip = clip
        self.epochs = epochs
        self.minibatch = minibatch
        self.max_samples = max_samples
        self.scale = 0.0          # per-step coefficient, set from episode length
        self.ready = False
        self.metrics = {}

    def set_episode_length(self, mean_length):
        """Bound the bonus: |sum over an episode| <= budget * task reward."""
        self.scale = self.budget / max(1.0, float(mean_length))

    @torch.no_grad()
    def bonus(self, features):
        """Bounded log-odds bonus. Zero until the discriminator has seen data."""
        if not self.ready or self.scale <= 0:
            return np.zeros(len(features), np.float32)
        tensor = torch.as_tensor(features, device=self.device, dtype=torch.float32)
        logits = self.model(tensor).clamp(-self.clip, self.clip)
        return (self.scale * torch.tanh(logits)).float().cpu().numpy()

    def fit(self, features, labels):
        """One pass of BCE on this rollout's terminated episodes."""
        positive = int(labels.sum())
        negative = int(len(labels) - positive)
        if positive < 64 or negative < 64:
            self.metrics = {'skipped': True, 'positive': positive, 'negative': negative}
            return
        if len(labels) > self.max_samples:
            keep = np.random.default_rng(0).choice(
                len(labels), self.max_samples, replace=False)
            features, labels = features[keep], labels[keep]
        x = torch.as_tensor(features, device=self.device, dtype=torch.float32)
        y = torch.as_tensor(labels, device=self.device, dtype=torch.float32)
        # Successes are rare; weight them up so the discriminator does not
        # collapse to predicting "failure" everywhere.
        weight = torch.where(y > 0, negative / max(1, positive), 1.0).to(self.device)
        losses = []
        self.model.train()
        for _ in range(self.epochs):
            order = torch.randperm(len(y), device=self.device)
            for start in range(0, len(y), self.minibatch):
                index = order[start:start + self.minibatch]
                logits = self.model(x[index])
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, y[index], weight=weight[index])
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                losses.append(float(loss.item()))
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            accuracy = float(((logits > 0).float() == y).float().mean())
            separation = float(logits[y > 0].mean() - logits[y <= 0].mean())
        self.ready = True
        self.metrics = {
            'loss': round(float(np.mean(losses)), 4),
            'accuracy': round(accuracy, 4),
            'separation': round(separation, 4),
            'positive': positive,
            'negative': negative,
            'scale': round(self.scale, 8),
        }
