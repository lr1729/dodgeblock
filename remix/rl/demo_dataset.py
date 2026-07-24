#!/usr/bin/env python3
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from r2d2 import (
    FALLING_COUNT,
    FALLING_FEATURES,
    FORECAST_COUNT,
    FORECAST_FEATURES,
    SKYLINE_SIZE,
    STATE_SIZE,
    TERRAIN_COLS,
    TERRAIN_ROWS,
)


ARRAY_SPECS = {
    'terrain': (np.dtype('<u2'), (TERRAIN_ROWS, TERRAIN_COLS)),
    'skyline': (np.dtype('u1'), (SKYLINE_SIZE,)),
    'falling': (np.dtype('<f4'), (FALLING_COUNT, FALLING_FEATURES)),
    'forecasts': (np.dtype('<f4'), (FORECAST_COUNT, FORECAST_FEATURES)),
    'state': (np.dtype('<f4'), (STATE_SIZE,)),
    'actions': (np.dtype('u1'), ()),
}

DECISION_SAMPLE_WEIGHTS = {
    'uniform': 0.40,
    'opening': 0.25,
    'switch': 0.25,
    'focus': 0.05,
    'initial': 0.05,
}
OPENING_FRAMES = 1_200
INITIAL_FRAMES = 32


@dataclass
class DemoShard:
    seed: int
    frames: int
    arrays: dict
    manifest: dict
    root: Path
    sample_pools: dict


def _read_array(root, manifest, key, frames):
    dtype, trailing = ARRAY_SPECS[key]
    file_info = manifest['files'][key]
    with gzip.open(root / file_info['file'], 'rb') as source:
        data = source.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != file_info['sha256']:
        raise ValueError(f'{root}: {key} checksum mismatch')
    expected = frames * int(np.prod(trailing, dtype=np.int64) if trailing else 1)
    array = np.frombuffer(data, dtype=dtype)
    if array.size != expected:
        raise ValueError(f'{root}: {key} has {array.size} values, expected {expected}')
    return array.reshape((frames, *trailing)).copy()


class DemoDataset:
    def __init__(self, root, seeds):
        self.root = Path(root).resolve()
        self.seeds = tuple(sorted(int(seed) for seed in seeds))
        if not self.seeds:
            raise ValueError('demo dataset requires at least one seed')
        self.shards = []
        for seed in self.seeds:
            shard_root = self.root / f'seed-{seed}'
            manifest_path = shard_root / 'manifest.json'
            manifest = json.loads(manifest_path.read_text())
            if manifest.get('version') != 1 or int(manifest['seed']) != seed:
                raise ValueError(f'invalid manifest for seed {seed}')
            frames = int(manifest['frames'])
            arrays = {
                key: _read_array(shard_root, manifest, key, frames)
                for key in ARRAY_SPECS
            }
            actions = arrays['actions']
            previous = np.empty_like(actions)
            previous[0] = 0
            previous[1:] = actions[:-1]
            sample_pools = {
                'uniform': np.arange(frames, dtype=np.int64),
                'opening': np.arange(min(frames, OPENING_FRAMES), dtype=np.int64),
                'switch': np.flatnonzero(actions != previous),
                'focus': np.flatnonzero(actions >= 9),
                'initial': np.arange(min(frames, INITIAL_FRAMES), dtype=np.int64),
            }
            self.shards.append(DemoShard(
                seed,
                frames,
                arrays,
                manifest,
                shard_root,
                sample_pools,
            ))
        self.frames = sum(shard.frames for shard in self.shards)

    def contract(self):
        return [{
            'seed': shard.seed,
            'frames': shard.frames,
            'demo_sha256': shard.manifest['demo']['sha256'],
            'files': {
                key: shard.manifest['files'][key]['sha256']
                for key in ARRAY_SPECS
            },
        } for shard in self.shards]

    def sample(self, size, rng, *, decision_weighted=True):
        shard_ids = rng.integers(0, len(self.shards), size=size)
        result = {
            key: np.empty((size, *trailing), dtype=dtype)
            for key, (dtype, trailing) in ARRAY_SPECS.items()
        }
        categories = tuple(DECISION_SAMPLE_WEIGHTS)
        probabilities = np.asarray(
            [DECISION_SAMPLE_WEIGHTS[key] for key in categories],
            dtype=np.float64,
        )
        for shard_id in np.unique(shard_ids):
            output_indices = np.flatnonzero(shard_ids == shard_id)
            shard = self.shards[int(shard_id)]
            if decision_weighted:
                sampled_categories = rng.choice(
                    len(categories),
                    size=len(output_indices),
                    p=probabilities,
                )
                frame_indices = np.empty(len(output_indices), dtype=np.int64)
                for category_id in np.unique(sampled_categories):
                    selected = np.flatnonzero(sampled_categories == category_id)
                    pool = shard.sample_pools[categories[int(category_id)]]
                    if not len(pool):
                        pool = shard.sample_pools['uniform']
                    frame_indices[selected] = pool[
                        rng.integers(0, len(pool), size=len(selected))
                    ]
            else:
                frame_indices = rng.integers(
                    0,
                    shard.frames,
                    size=len(output_indices),
                )
            for key in ARRAY_SPECS:
                result[key][output_indices] = shard.arrays[key][frame_indices]
        return result

    def iter_batches(self, size):
        for shard in self.shards:
            for start in range(0, shard.frames, size):
                stop = min(shard.frames, start + size)
                yield {
                    key: values[start:stop]
                    for key, values in shard.arrays.items()
                }, shard.seed
