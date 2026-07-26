#!/usr/bin/env python3
"""Value-guided option-level beam search — the search half of an ExIt loop.

Measured 2026-07-25: a greedy searcher scoring by height gained is a WORSE
teacher than the policy it would teach (9.6 h/s vs 12.6), because a myopic
objective climbs into danger. The critic already scores exactly what matters
(P(reach target from here), EV 0.94), so the leaves are evaluated with V
instead of rolled out. That is the one ingredient this project has never had:
go-explore proposed options from constant weights, and PPO had no search.

Candidates are expanded in lockstep across the batched simulator bridge: every
environment holds one candidate continuation of the same root, so a beam of B
branches x A options costs one batched rollout of `option_frames` steps plus a
single value-network call.
"""
import argparse
import json
import time

import numpy as np
import torch

from ppo_v2 import (
    ActorCriticNetwork,
    load_agent_state, AutoregressiveActionDistribution,
    STICKY_MODEL_ARCHITECTURE, StickyActorCriticNetwork,
    packet_observation, tensor_observation,
)
from v2_bridge import ParallelEnvBridge

HOLD = 254
SAVE = 253
RESTORE = 252


def option_table(option_frames, focus_press):
    """Each option is a short action program: hold a direction, optionally
    preceded by a focus press so that press->release commits a dash."""
    options = []
    for action in range(9):
        options.append(('hold', action, [action] * option_frames))
    for action in range(9):
        program = [action + 9] * focus_press + [action] * max(1, option_frames - focus_press)
        options.append(('focus', action, program[:option_frames]))
    return options


class ValueSearch:
    def __init__(self, agent, bridge, device, options, beam, discount_per_frame,
                 score_mode='value', rng=None):
        self.agent = agent
        self.bridge = bridge
        self.device = device
        self.options = options
        self.beam = beam
        self.discount = discount_per_frame
        self.score_mode = score_mode
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.envs = bridge.count
        if beam * len(options) > self.envs:
            raise ValueError(
                f'beam {beam} x {len(options)} options exceeds {self.envs} envs')

    def evaluate(self, packet):
        observation = tensor_observation(packet_observation(packet), self.device)
        with torch.inference_mode(), torch.autocast(
                device_type=self.device.type, dtype=torch.bfloat16,
                enabled=self.device.type == 'cuda'):
            logits, value, _hazard = self.agent(observation)
            distribution = AutoregressiveActionDistribution(logits, observation)
            entropy = distribution.entropies()['vertical']
        return value.float().cpu().numpy(), entropy.float().cpu().numpy()

    def expand(self, root_slot, branch_slots):
        """One search ply. Returns (scores, heights, alive, programs) per lane."""
        lanes = len(branch_slots) * len(self.options)
        restore = np.full(self.envs, -1, np.int32)
        for lane in range(lanes):
            restore[lane] = branch_slots[lane // len(self.options)]
        packet = self.bridge.restore_slots(restore)

        programs = [self.options[lane % len(self.options)][2] for lane in range(lanes)]
        depth = len(programs[0])
        died = np.zeros(self.envs, bool)
        won = np.zeros(self.envs, bool)
        reached = np.zeros(self.envs, np.float32)
        start_heights = packet['current_heights'].copy()
        for step in range(depth):
            actions = np.full(self.envs, HOLD, np.uint8)
            for lane in range(lanes):
                actions[lane] = programs[lane][step]
            packet = self.bridge.step(actions)
            # `dones` fires on SUCCESS as well as death — reaching the target
            # ends the episode. Treating them alike prunes winning branches and
            # makes success structurally unreachable.
            success = packet['successes'].astype(bool)
            finished = packet['dones'].astype(bool)
            newly_won = success & ~won & ~died
            reached = np.where(newly_won, packet['heights'], reached)
            won |= success
            died |= finished & ~success
        gained = np.where(won, reached - start_heights,
                          packet['current_heights'] - start_heights)

        if self.score_mode == 'random':
            # Control: is the critic's ranking better than no ranking at all?
            values = self.rng.random(self.envs).astype(np.float32)
        elif self.score_mode == 'height':
            values = gained.astype(np.float32)
        else:
            values, _entropy = self.evaluate(packet)
        values = np.where(died, 0.0, values)
        values = np.where(won, 1.0, values)      # terminal win
        scores = values * (self.discount ** depth)
        return scores[:lanes], gained[:lanes], ~died[:lanes], won[:lanes]

    def run_policy(self, max_frames, target):
        """Control: play the policy itself through this same harness. If this
        does not reproduce the policy's known success rate, the harness is at
        fault and any search comparison is meaningless."""
        packet = self.bridge.step(np.full(self.envs, HOLD, np.uint8))
        height, frames = 0.0, 0
        for _ in range(max_frames):
            observation = tensor_observation(packet_observation(packet), self.device)
            with torch.inference_mode(), torch.autocast(
                    device_type=self.device.type, dtype=torch.bfloat16,
                    enabled=self.device.type == 'cuda'):
                logits, _value, _hazard = self.agent(observation)
                sampled = AutoregressiveActionDistribution(logits, observation).mode()
            actions = np.full(self.envs, HOLD, np.uint8)
            actions[0] = int(sampled[0].item())
            packet = self.bridge.step(actions)
            frames += 1
            if packet['dones'][0]:
                # The env auto-resets in the same step it dies, so
                # current_heights already belongs to the NEXT episode.
                height = float(packet['heights'][0])
                break
            height = float(packet['current_heights'][0])
            if height >= target:
                break
        return {'stopped': 'policy', 'actions': [0] * frames, 'height': height}

    def run(self, max_options, target, log_every, root_slot=0):
        # Lane 0 carries the working root; slots 100+ hold the beam branches.
        # The critic scores P(reach the height it was TRAINED on), so `target`
        # must match the checkpoint's training target or V saturates and the
        # search is flying blind.
        self.bridge.save_slots(np.where(
            np.arange(self.envs) == 0, root_slot, -1).astype(np.int32))
        branch_slots = [root_slot]
        branch_actions = [[]]
        branch_height = [0.0]
        started = time.perf_counter()

        for ply in range(max_options):
            scores, gained, alive, won = self.expand(root_slot, branch_slots)
            lanes = len(branch_slots) * len(self.options)
            if won.any():
                lane = int(np.argmax(won))
                parent = lane // len(self.options)
                return {
                    'stopped': 'target',
                    'actions': branch_actions[parent] + list(
                        self.options[lane % len(self.options)][2]),
                    'height': float(branch_height[parent] + gained[lane]),
                    'seconds': round(time.perf_counter() - started, 1),
                }
            order = np.argsort(-scores[:lanes])
            keep = [lane for lane in order if alive[lane]][:self.beam]
            if not keep:
                return {'stopped': 'all branches died', 'ply': ply,
                        'actions': branch_actions[0], 'height': branch_height[0]}

            save = np.full(self.envs, -1, np.int32)
            new_slots, new_actions, new_height = [], [], []
            for index, lane in enumerate(keep):
                slot = 100 + index
                save[lane] = slot
                new_slots.append(slot)
                parent = lane // len(self.options)
                new_actions.append(branch_actions[parent] + list(
                    self.options[lane % len(self.options)][2]))
                new_height.append(float(branch_height[parent] + gained[lane]))
            self.bridge.save_slots(save)
            branch_slots, branch_actions, branch_height = new_slots, new_actions, new_height

            best = int(np.argmax(branch_height))
            if log_every and ply % log_every == 0:
                frames = len(branch_actions[best])
                print(json.dumps({
                    'ply': ply, 'frames': frames,
                    'height': round(branch_height[best], 1),
                    'climb_h_per_s': round(branch_height[best] / max(1e-9, frames / 60), 1),
                    'top_value': round(float(scores[keep[0]]), 4),
                    'seconds': round(time.perf_counter() - started, 1),
                }), flush=True)
            if branch_height[best] >= target:
                break

        best = int(np.argmax(branch_height))
        return {'stopped': 'target' if branch_height[best] >= target else 'plies',
                'actions': branch_actions[best], 'height': branch_height[best],
                'seconds': round(time.perf_counter() - started, 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    # Slot tables are per worker, so every lane of one search must live in the
    # same worker: one worker, many envs. Parallelism comes from running
    # independent searches, not from splitting one beam across workers.
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--envs-per-worker', type=int, default=256)
    parser.add_argument('--beam', type=int, default=8)
    parser.add_argument('--option-frames', type=int, default=15)
    parser.add_argument('--focus-press', type=int, default=4)
    parser.add_argument('--max-options', type=int, default=400)
    parser.add_argument('--target-height', type=float, default=10_000)
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0xD06E_B10C)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--mode', default='value', choices=('value', 'policy'))
    parser.add_argument('--score', default='value', choices=('value', 'random', 'height'))
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--baseline', type=float, default=None,
                        help='policy det success at this target, for comparison')
    parser.add_argument('--out', default='')
    args = parser.parse_args()

    device = torch.device(args.device)
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    network = (StickyActorCriticNetwork
               if saved.get('model_architecture') == STICKY_MODEL_ARCHITECTURE
               else ActorCriticNetwork)
    agent = network().to(device)
    load_agent_state(agent, saved['agent'])
    agent.eval()

    bridge = ParallelEnvBridge(
        args.workers, args.envs_per_worker, args.seed,
        target_height=args.target_height, reward_mode='target')
    results = []
    try:
        bridge.read()
        search = ValueSearch(
            agent, bridge, device,
            option_table(args.option_frames, args.focus_press),
            args.beam, args.discount, score_mode=args.score,
            rng=np.random.default_rng(args.seed))
        for episode in range(args.episodes):
            bridge.reset()
            result = (search.run_policy(12000, args.target_height)
                      if args.mode == 'policy'
                      else search.run(args.max_options, args.target_height, args.log_every))
            frames = len(result['actions'])
            result['reached'] = result['height'] >= args.target_height
            results.append(result)
            print(json.dumps({
                'episode': episode, 'reached': bool(result['reached']),
                'height': round(result['height'], 1), 'frames': frames,
                'climb_h_per_s': round(result['height'] / max(1e-9, frames / 60), 1),
                'stopped': result['stopped'], 'seconds': result.get('seconds'),
            }), flush=True)
    finally:
        bridge.close()

    heights = np.array([r['height'] for r in results], np.float32)
    reached = np.array([r['reached'] for r in results], bool)
    summary = {
        'checkpoint': args.checkpoint,
        'episodes': len(results),
        'target_height': args.target_height,
        'search_success': round(float(reached.mean()), 4),
        'policy_baseline': args.baseline,
        'median_height': float(np.median(heights)),
        'mean_height': round(float(heights.mean()), 1),
        'beam': args.beam,
        'option_frames': args.option_frames,
    }
    if args.baseline is not None:
        summary['beats_policy'] = bool(reached.mean() > args.baseline)
    print(json.dumps(summary, indent=2))
    result = results[int(np.argmax(heights))]
    frames = len(result['actions'])
    if args.out:
        import base64
        with open(args.out, 'w') as handle:
            json.dump({'version': 1, 'seed': args.seed, 'frames': frames,
                       'targetHeight': result['height'],
                       'actions': base64.b64encode(
                           bytes(bytearray(result['actions']))).decode(),
                       'search': summary}, handle)


if __name__ == '__main__':
    main()
