#!/usr/bin/env python3
import torch
from torch.nn import functional as F

from r2d2 import OBSERVATION_KEYS




def focus_action_mask(observation):
    state = observation['state']
    aiming = state[..., 12] > 0
    can_press = (
        (state[..., 10] > 0) &
        (state[..., 14] <= 0) &
        (state[..., 19] <= 0)
    )
    return torch.stack((torch.ones_like(aiming), aiming | can_press), dim=-1)


def previous_actions(observation):
    state = observation['state']
    horizontal = torch.where(
        state[..., 17] < 0,
        1,
        torch.where(state[..., 17] > 0, 2, 0),
    )
    vertical = torch.where(
        state[..., 18] < 0,
        1,
        torch.where(state[..., 18] > 0, 2, 0),
    )
    focus = (state[..., 19] > 0).long()
    return focus * 9 + vertical * 3 + horizontal


def sticky_joint_logprobs(logits, observation):
    repeat_logits, focus_logits, vertical_logits, horizontal_logits = logits
    focus_logprobs = torch.log_softmax(
        focus_logits.masked_fill(~focus_action_mask(observation), -1e9),
        dim=-1,
    )
    vertical_logprobs = torch.log_softmax(vertical_logits, dim=-1)
    horizontal_logprobs = torch.log_softmax(horizontal_logits, dim=-1)
    proposal_logits = (
        focus_logprobs[..., :, None, None] +
        vertical_logprobs[..., :, :, None] +
        horizontal_logprobs
    ).flatten(start_dim=-3)
    previous = previous_actions(observation)
    previous_valid = proposal_logits.gather(
        -1,
        previous.unsqueeze(-1),
    ).squeeze(-1) > -1e8
    repeat_mask = torch.stack(
        (torch.ones_like(previous_valid), previous_valid),
        dim=-1,
    )
    repeat_logprobs = torch.log_softmax(
        repeat_logits.masked_fill(~repeat_mask, -1e9),
        dim=-1,
    )
    proposal_logits = proposal_logits.scatter(
        -1,
        previous.unsqueeze(-1),
        -1e9,
    )
    proposal_logprobs = torch.log_softmax(proposal_logits, dim=-1)
    joint = (
        repeat_logprobs[..., 0, None].exp() *
        proposal_logprobs.exp()
    )
    joint.scatter_add_(
        -1,
        previous.unsqueeze(-1),
        repeat_logprobs[..., 1, None].exp(),
    )
    return joint.clamp_min(1e-30).log(), repeat_logprobs, previous


def tensor_demo_batch(batch, device):
    observation = {
        key: torch.as_tensor(batch[key], device=device)
        for key in OBSERVATION_KEYS
    }
    actions = torch.as_tensor(batch['actions'], dtype=torch.long, device=device)
    targets = torch.as_tensor(batch['targets'], device=device)
    return observation, actions, targets


def autoregressive_imitation_loss(
    logits,
    observation,
    actions,
    *,
    targets=None,
    focus_positive_weight=1.0,
    sticky_repeat_coef=None,
):
    sticky = len(logits) == 4
    if sticky:
        (
            joint_logprobs,
            repeat_logprobs,
            previous,
        ) = sticky_joint_logprobs(logits, observation)
        focus_logits, vertical_logits, horizontal_logits = logits[1:]
    else:
        focus_logits, vertical_logits, horizontal_logits = logits
    focus_logprobs = torch.log_softmax(
        focus_logits.masked_fill(~focus_action_mask(observation), -1e9),
        dim=-1,
    )
    vertical_logprobs = torch.log_softmax(vertical_logits, dim=-1)
    horizontal_logprobs = torch.log_softmax(horizontal_logits, dim=-1)
    if targets is None:
        targets = F.one_hot(actions, num_classes=18).to(focus_logprobs.dtype)
    else:
        targets = targets.to(focus_logprobs.dtype)
    joint_targets = targets.reshape(-1, 2, 3, 3)
    focus_targets = joint_targets.sum(dim=(-2, -1))
    focus_nll = -(focus_targets * focus_logprobs).sum(dim=-1)
    focus_sample_weights = (
        focus_targets[:, 0] +
        focus_positive_weight * focus_targets[:, 1]
    )
    focus_loss = (
        focus_nll * focus_sample_weights
    ).sum() / focus_sample_weights.sum().clamp_min(1)
    vertical_loss = -(
        joint_targets * vertical_logprobs.unsqueeze(-1)
    ).sum(dim=(-3, -2, -1)).mean()
    horizontal_loss = -(
        joint_targets * horizontal_logprobs
    ).sum(dim=(-3, -2, -1)).mean()
    if sticky:
        per_sample_loss = -(targets * joint_logprobs).sum(dim=-1)
        loss = (
            per_sample_loss * focus_sample_weights
        ).sum() / focus_sample_weights.sum().clamp_min(1)
        repeat_targets = targets.gather(-1, previous.unsqueeze(-1)).squeeze(-1)
        repeat_loss = -(
            (1 - repeat_targets) * repeat_logprobs[:, 0] +
            repeat_targets * repeat_logprobs[:, 1]
        ).mean()
        if sticky_repeat_coef is not None:
            loss = (
                focus_loss +
                vertical_loss +
                horizontal_loss +
                sticky_repeat_coef * repeat_loss
            )
    else:
        loss = focus_loss + vertical_loss + horizontal_loss
    with torch.no_grad():
        if not sticky:
            joint_logprobs = (
                focus_logprobs[..., :, None, None] +
                vertical_logprobs[..., :, :, None] +
                horizontal_logprobs
            ).flatten(start_dim=-3)
        prediction = joint_logprobs.argmax(dim=-1)
        target_action = targets.argmax(dim=-1)
        focus_prediction = torch.div(prediction, 9, rounding_mode='floor')
        focus_target = torch.div(target_action, 9, rounding_mode='floor')
        vertical_prediction = torch.div(
            prediction.remainder(9),
            3,
            rounding_mode='floor',
        )
        vertical_target = torch.div(
            target_action.remainder(9),
            3,
            rounding_mode='floor',
        )
        horizontal_prediction = prediction.remainder(3)
        horizontal_target = target_action.remainder(3)
    metrics = {
        'loss': loss.detach(),
        'focus_loss': focus_loss.detach(),
        'vertical_loss': vertical_loss.detach(),
        'horizontal_loss': horizontal_loss.detach(),
        'focus_accuracy': (focus_prediction == focus_target).float().mean(),
        'vertical_accuracy': (vertical_prediction == vertical_target).float().mean(),
        'horizontal_accuracy': (horizontal_prediction == horizontal_target).float().mean(),
        'joint_accuracy': (prediction == target_action).float().mean(),
        'focus_target_fraction': focus_targets[:, 1].mean(),
        'focus_prediction_fraction': focus_prediction.float().mean(),
    }
    if sticky:
        metrics['repeat_loss'] = repeat_loss.detach()
        metrics['repeat_target_fraction'] = repeat_targets.mean()
        metrics['repeat_prediction_fraction'] = (
            joint_logprobs.argmax(dim=-1) == previous
        ).float().mean()
        metrics['joint_accuracy_lift_over_repeat'] = (
            metrics['joint_accuracy'] - metrics['repeat_target_fraction']
        )
        metrics['repeat_loss_coefficient'] = (
            -1.0 if sticky_repeat_coef is None else sticky_repeat_coef
        )
    return loss, metrics
