#!/usr/bin/env python3
from collections import defaultdict
import gzip
import json
from pathlib import Path

import numpy as np

from cell_bank import FRESH_CELL_ID


class TrajectoryStartCoordinator:
    """Samples successful trajectory snapshots uniformly by seed and time."""

    def __init__(self, paths, *, seed, probability=0.5, band_height=400):
        self.paths = [str(Path(path).resolve()) for path in paths]
        self.probability = float(probability)
        self.band_height = float(band_height)
        self.rng = np.random.default_rng(seed)
        self.variants = []
        self.by_seed = defaultdict(list)
        self.stats = defaultdict(lambda: {
            'starts': 0,
            'completions': 0,
            'successes': 0,
        })
        for filename in self.paths:
            with gzip.open(filename, 'rt') as source:
                payload = json.load(source)
            if payload.get('version') != 1:
                raise ValueError(f'unsupported trajectory bank version in {filename}')
            source_seed = int(payload['seed'])
            for entry in payload.get('entries', []):
                variant_id = len(self.variants)
                variant = {
                    'variant_id': variant_id,
                    'key': str(entry['key']),
                    'seed': source_seed,
                    'frame': int(entry.get('frame', 0)),
                    'height': float(entry['height']),
                }
                self.variants.append(variant)
                self.by_seed[source_seed].append(variant_id)
        if self.paths and not self.variants:
            raise ValueError('trajectory banks contain no snapshots')
        self.seeds = sorted(self.by_seed)
        self.keys = {variant['key'] for variant in self.variants}

    def select(self, *, heldout=False):
        if heldout or not self.variants or self.rng.random() >= self.probability:
            return FRESH_CELL_ID
        seed = self.seeds[int(self.rng.integers(len(self.seeds)))]
        variants = self.by_seed[seed]
        return int(variants[int(self.rng.integers(len(variants)))])

    def record_start(self, variant_id):
        if variant_id == FRESH_CELL_ID:
            return
        key = self.variants[int(variant_id)]['key']
        self.stats[key]['starts'] += 1

    def record_result(self, variant_id, success):
        if variant_id == FRESH_CELL_ID:
            return
        key = self.variants[int(variant_id)]['key']
        self.stats[key]['completions'] += 1
        self.stats[key]['successes'] += int(bool(success))

    def metrics(self, heldout=False):
        if heldout:
            return {
                'cells': 0,
                'attempted_cells': 0,
                'starts': 0,
                'completions': 0,
                'successes': 0,
                'success_rate': 0.0,
                'bands': [],
            }
        bands = defaultdict(lambda: {
            'cells': 0,
            'attempted': 0,
            'starts': 0,
            'completions': 0,
            'successes': 0,
        })
        for variant in self.variants:
            band = int(variant['height'] // self.band_height) * int(self.band_height)
            values = bands[band]
            values['cells'] += 1
            stat = self.stats[variant['key']]
            values['attempted'] += int(stat['starts'] > 0)
            for key in ('starts', 'completions', 'successes'):
                values[key] += stat[key]
        starts = sum(stat['starts'] for stat in self.stats.values())
        completions = sum(stat['completions'] for stat in self.stats.values())
        successes = sum(stat['successes'] for stat in self.stats.values())
        return {
            'cells': len(self.variants),
            'attempted_cells': sum(stat['starts'] > 0 for stat in self.stats.values()),
            'starts': starts,
            'completions': completions,
            'successes': successes,
            'success_rate': successes / max(1, completions),
            'bands': [{
                'height': height,
                **{
                    key: values[key]
                    for key in ('cells', 'attempted', 'starts', 'completions')
                },
                'success_rate': values['successes'] / max(1, values['completions']),
            } for height, values in sorted(bands.items())],
        }

    def state_dict(self):
        return {
            'rng_state': self.rng.bit_generator.state,
            'stats': {key: dict(value) for key, value in self.stats.items()},
        }

    def load_state_dict(self, payload, *, load_variant_starts=True):
        del load_variant_starts
        if payload.get('rng_state'):
            self.rng.bit_generator.state = payload['rng_state']
        for key, value in payload.get('stats', {}).items():
            if key in self.keys:
                self.stats[key].update({
                    field: int(value.get(field, 0))
                    for field in ('starts', 'completions', 'successes')
                })
