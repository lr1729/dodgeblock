#!/usr/bin/env python3
import torch
from torch.nn import functional as F

from r2d2 import OBSERVATION_KEYS


def action_components(actions):
    focus = torch.div(actions, 9, rounding_mode='floor')
    local = actions.remainder(9)
    vertical = torch.div(local, 3, rounding_mode='floor')
    horizontal = local.remainder(3)
    return focus, vertical, horizontal


def focus_action_mask(observation):
    state = observation['state']
    aiming = state[..., 12] > 0
    can_press = (
        (state[..., 10] > 0) &
        (state[..., 14] <= 0) &
        (state[..., 19] <= 0)
    )
    return torch.stack((torch.ones_like(aiming), aiming | can_press), dim=-1)


def tensor_demo_batch(batch, device):
    observation = {
        key: torch.as_tensor(batch[key], device=device)
        for key in OBSERVATION_KEYS
    }
    actions = torch.as_tensor(batch['actions'], dtype=torch.long, device=device)
    return observation, actions


def autoregressive_imitation_loss(
    logits,
    observation,
    actions,
    *,
    focus_positive_weight=1.0,
):
    focus, vertical, horizontal = action_components(actions)
    focus_logits, vertical_logits, horizontal_logits = logits
    focus_logits = focus_logits.masked_fill(~focus_action_mask(observation), -1e9)
    focus_weights = torch.tensor(
        [1.0, focus_positive_weight],
        dtype=focus_logits.dtype,
        device=focus_logits.device,
    )
    focus_loss = F.cross_entropy(focus_logits, focus, weight=focus_weights)
    rows = torch.arange(actions.shape[0], device=actions.device)
    selected_vertical = vertical_logits[rows, focus]
    vertical_loss = F.cross_entropy(selected_vertical, vertical)
    selected_horizontal = horizontal_logits[rows, focus, vertical]
    horizontal_loss = F.cross_entropy(selected_horizontal, horizontal)
    loss = focus_loss + vertical_loss + horizontal_loss
    with torch.no_grad():
        focus_prediction = focus_logits.argmax(dim=-1)
        vertical_prediction = selected_vertical.argmax(dim=-1)
        horizontal_prediction = selected_horizontal.argmax(dim=-1)
        joint_correct = (
            (focus_prediction == focus) &
            (vertical_prediction == vertical) &
            (horizontal_prediction == horizontal)
        )
    return loss, {
        'loss': loss.detach(),
        'focus_loss': focus_loss.detach(),
        'vertical_loss': vertical_loss.detach(),
        'horizontal_loss': horizontal_loss.detach(),
        'focus_accuracy': (focus_prediction == focus).float().mean(),
        'vertical_accuracy': (vertical_prediction == vertical).float().mean(),
        'horizontal_accuracy': (horizontal_prediction == horizontal).float().mean(),
        'joint_accuracy': joint_correct.float().mean(),
        'focus_target_fraction': focus.float().mean(),
        'focus_prediction_fraction': focus_prediction.float().mean(),
    }
