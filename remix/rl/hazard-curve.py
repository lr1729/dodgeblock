#!/usr/bin/env python3
"""Per-layer survival as a function of height and of elapsed time.

Every A/B in this project is at target 600 -- fifteen layers, covered in roughly
the first 40 seconds. A 10k run is 22,548 frames = 6.3 minutes, and difficulty
saturates at about 4 minutes, so ~36% of a 10k trajectory sits at saturated
difficulty and essentially none of rung 600 does. The measured per-layer survival
of ~0.92 therefore comes from the easy end of the curve and is being extrapolated
into a regime it never samples.

This measures the curve directly. Run at a target high enough that nothing
succeeds, so no episode is censored by winning, and bucket deaths by the band
they occurred in:

    per-layer survival in a band = 1 - deaths_in_band / layer_entries_in_band

Reported against height and, separately, against elapsed time -- because the
game's difficulty is a function of time, those two axes answer different
questions. The height axis says whether the pile itself gets harder; the time
axis says whether the spawn rate does.

The number that matters for the goal is the survival rate in the saturated
regime, past ~4 minutes. If it is close to the 0.92 measured at rung 600, the
distance to 10k is the one already in the log. If it is materially worse, the
distance is larger than any figure this project has quoted.
"""
import argparse
import json

import numpy as np
import torch

from ppo_v2 import (
    ActorCriticNetwork,
    AutoregressiveActionDistribution,
    STICKY_MODEL_ARCHITECTURE,
    StickyActorCriticNetwork,
    load_agent_state,
    packet_observation,
    tensor_observation,
)
from v2_bridge import ParallelEnvBridge

LAYER = 40.0
SATURATION_SECONDS = 240.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--episodes', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0x4A2A_8D10)
    parser.add_argument('--max-frames', type=int, default=30_000)
    parser.add_argument('--height-band', type=float, default=200.0)
    parser.add_argument('--time-band', type=float, default=60.0, help='seconds')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--stochastic', action='store_true')
    args = parser.parse_args()

    device = torch.device(args.device)
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    network_class = (
        StickyActorCriticNetwork
        if saved.get('model_architecture') == STICKY_MODEL_ARCHITECTURE
        else ActorCriticNetwork
    )
    agent = network_class().to(device)
    load_agent_state(agent, saved['agent'])
    agent.eval()

    # Target far above anything reachable, so no episode ends by succeeding and
    # the hazard estimate is uncensored.
    bridge = ParallelEnvBridge(
        8, max(1, args.episodes // 8), args.seed,
        target_height=1_000_000,
        reward_mode='target',
    )
    packet = bridge.read()
    count = args.episodes
    active = np.ones(count, dtype=bool)
    peak = np.zeros(count, np.float64)
    frames = np.zeros(count, np.int64)
    deaths = []          # (height, seconds) of each death

    try:
        for _ in range(args.max_frames):
            observation = tensor_observation(packet_observation(packet), device)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=device.type == 'cuda',
            ):
                logits, _value, _hazard = agent(observation)
                distribution = AutoregressiveActionDistribution(logits, observation)
                sampled = (distribution.sample() if args.stochastic
                           else distribution.mode())
            actions = sampled.cpu().numpy().astype(np.uint8)
            actions[~active] = 254
            live = packet['current_heights']
            peak[active] = np.maximum(peak[active], live[active])
            frames[active] += 1
            packet = bridge.step(actions)
            finished = active & packet['dones'].astype(bool)
            for index in np.flatnonzero(finished):
                deaths.append((float(peak[index]), frames[index] / 60.0))
            active &= ~finished
            if not active.any():
                break
    finally:
        bridge.close()

    if not deaths:
        print(json.dumps({'error': 'no deaths recorded'}))
        return
    death_height = np.array([d[0] for d in deaths])
    death_time = np.array([d[1] for d in deaths])
    # Survivors (hit max_frames) are censored: they contribute exposure but no
    # death, which is exactly what the estimator below wants.
    peak_all = peak.copy()
    time_all = frames / 60.0

    def curve(values_all, values_death, band, unit, per_layer):
        edges = np.arange(0, values_all.max() + band, band)
        rows = []
        for low in edges[:-1]:
            high = low + band
            # Exposure: layers (or seconds) each episode spent inside this band.
            entered = np.minimum(values_all, high) - low
            entered = np.clip(entered, 0, band)
            exposure = entered.sum() / (LAYER if per_layer else 1.0)
            died = int(((values_death >= low) & (values_death < high)).sum())
            if exposure < 1:
                continue
            hazard = died / exposure
            rows.append({
                unit: f'{low:.0f}-{high:.0f}',
                'exposure_layers' if per_layer else 'exposure_seconds':
                    round(float(exposure), 1),
                'deaths': died,
                'per_layer_survival' if per_layer else 'survival_per_second':
                    round(float(max(0.0, 1.0 - hazard)), 4),
            })
        return rows

    saturated = death_time >= SATURATION_SECONDS
    early_exposure = np.clip(time_all, 0, SATURATION_SECONDS)
    late_exposure = np.clip(time_all - SATURATION_SECONDS, 0, None)
    # Convert time exposure to layer exposure using each episode's own climb rate.
    rate = np.divide(peak_all, np.maximum(time_all, 1e-9))
    early_layers = (early_exposure * rate).sum() / LAYER
    late_layers = (late_exposure * rate).sum() / LAYER

    def survival(layers, died):
        return round(float(max(0.0, 1.0 - died / layers)), 4) if layers >= 1 else None

    print(json.dumps({
        'checkpoint': args.checkpoint,
        'episodes': count,
        'deaths': len(deaths),
        'mean_peak_height': round(float(peak_all.mean()), 1),
        'mean_seconds': round(float(time_all.mean()), 1),
        'by_height': curve(peak_all, death_height, args.height_band, 'height', True),
        'by_time_seconds': curve(time_all, death_time, args.time_band, 'seconds', False),
        'regime': {
            'pre_saturation_lt_240s': {
                'layer_exposure': round(float(early_layers), 1),
                'deaths': int((~saturated).sum()),
                'per_layer_survival': survival(early_layers, int((~saturated).sum())),
            },
            'saturated_ge_240s': {
                'layer_exposure': round(float(late_layers), 1),
                'deaths': int(saturated.sum()),
                'per_layer_survival': survival(late_layers, int(saturated.sum())),
            },
        },
        'reference': {
            'rung_600_measured': 0.92,
            'needed_to_reach_10k_at_all': 0.9567,
            'needed_for_consistent_10k': 0.9972,
        },
    }, indent=2))


if __name__ == '__main__':
    main()
