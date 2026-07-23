#!/usr/bin/env python3
from collections import defaultdict
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


FRESH_CELL_ID = -1
BAND_EXPLORATION_FLOOR = 0.01
BAND_COMPETENCE_EVIDENCE = 4
MAX_STALENESS_BONUS = 1.0


@dataclass(frozen=True)
class CellVariant:
    variant_id: int
    key: str
    height: float
    seed: int
    source: str


class CellBankCoordinator:
    """Owns curriculum sampling and evidence; workers only restore variants."""

    def __init__(
        self,
        paths,
        *,
        target_height,
        seed,
        probability=0.8,
        heldout_fraction=0.1,
        band_height=400,
    ):
        self.paths = [str(Path(path).resolve()) for path in paths]
        self.target_height = float(target_height)
        self.probability = float(probability)
        self.heldout_fraction = float(heldout_fraction)
        self.band_height = float(band_height)
        self.rng = np.random.default_rng(seed)
        self.variants = []
        self.groups = defaultdict(list)
        self.group_meta = {}
        self.stats = defaultdict(lambda: {
            'starts': 0,
            'completions': 0,
            'successes': 0,
            'last_sample': 0,
        })
        self.variant_starts = defaultdict(int)
        self.sample_clock = 0
        self._load()
        self.training_keys = [
            key for key in self.groups if not self.group_meta[key]['heldout']
        ]
        self.heldout_keys = [
            key for key in self.groups if self.group_meta[key]['heldout']
        ]
        self.training_bands = self._bands(self.training_keys)
        self.heldout_bands = self._bands(self.heldout_keys)

    @staticmethod
    def file_contract(paths):
        result = []
        for filename in paths:
            path = Path(filename).resolve()
            result.append({
                'path': str(path),
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            })
        return result

    def _is_heldout(self, key):
        if self.heldout_fraction <= 0:
            return False
        value = int.from_bytes(
            hashlib.sha256(f'dodgeblock-heldout|{key}'.encode()).digest()[:8],
            'little',
        ) / 2**64
        return value < self.heldout_fraction

    def _load(self):
        for filename in self.paths:
            with gzip.open(filename, 'rt') as source:
                payload = json.load(source)
            if payload.get('version') != 1:
                raise ValueError(f'unsupported cell bank version in {filename}')
            payload_target = float(payload.get('targetHeight', self.target_height))
            if payload_target != self.target_height:
                raise ValueError(
                    f'cell bank target {payload_target} does not match {self.target_height}'
                )
            seed = int(payload.get('seed', 0))
            for entry in payload.get('entries', []):
                key = str(entry['key'])
                variant = CellVariant(
                    variant_id=len(self.variants),
                    key=key,
                    height=float(
                        entry['height']
                        if 'height' in entry
                        else entry['snapshot']['height']
                    ),
                    seed=seed,
                    source=filename,
                )
                self.variants.append(variant)
                self.groups[key].append(variant.variant_id)
                if key not in self.group_meta:
                    self.group_meta[key] = {
                        'height': variant.height,
                        'heldout': self._is_heldout(key),
                    }

    def _bands(self, keys):
        bands = defaultdict(list)
        for key in keys:
            height = self.group_meta[key]['height']
            band = int(height // self.band_height)
            bands[band].append(key)
        return dict(bands)

    def _cell_weight(self, key):
        stat = self.stats[key]
        posterior = (stat['successes'] + 1) / (stat['completions'] + 2)
        learning_potential = 4 * posterior * (1 - posterior)
        staleness = min(
            MAX_STALENESS_BONUS,
            0.01 * np.sqrt(max(1, self.sample_clock - stat['last_sample'])),
        )
        uncertainty = 1 / np.sqrt(stat['completions'] + 1)
        return 0.05 + learning_potential + 0.2 * uncertainty + staleness

    def _cell_competence(self, key):
        stat = self.stats[key]
        return stat['successes'] / (
            stat['completions'] + BAND_COMPETENCE_EVIDENCE
        )

    def _band_competence(self, keys):
        if not keys:
            return 0.0
        return float(np.mean([self._cell_competence(key) for key in keys]))

    def _training_band_weights(self):
        band_ids = sorted(self.training_bands)
        weights = {}
        competences = {
            band: self._band_competence(self.training_bands[band])
            for band in band_ids
        }
        for index, band in enumerate(band_ids):
            # The target is the competent successor of the highest available
            # band. Competence then propagates downward without a hard gate.
            successor = (
                1.0
                if index == len(band_ids) - 1
                else competences[band_ids[index + 1]]
            )
            learning_weight = np.mean([
                self._cell_weight(key)
                for key in self.training_bands[band]
            ])
            weights[band] = (
                BAND_EXPLORATION_FLOOR +
                learning_weight * successor
            )
        return weights, competences

    def _weighted_choice(self, values, weights):
        weights = np.asarray(weights, np.float64)
        weights /= weights.sum()
        return values[int(self.rng.choice(len(values), p=weights))]

    def _select_key(self, heldout):
        bands = self.heldout_bands if heldout else self.training_bands
        if not bands:
            return None
        band_ids = sorted(bands)
        if heldout:
            band_weights = [1] * len(band_ids)
        else:
            weights, _competences = self._training_band_weights()
            band_weights = [weights[band] for band in band_ids]
        band = self._weighted_choice(band_ids, band_weights)
        keys = bands[band]
        seeds = sorted({
            self.variants[variant_id].seed
            for key in keys
            for variant_id in self.groups[key]
        })
        seed = seeds[int(self.rng.integers(len(seeds)))]
        seed_keys = [
            key for key in keys
            if any(self.variants[index].seed == seed for index in self.groups[key])
        ]
        key_weights = (
            [1] * len(seed_keys)
            if heldout
            else [self._cell_weight(candidate) for candidate in seed_keys]
        )
        key = self._weighted_choice(seed_keys, key_weights)
        return key, seed

    def select(self, *, heldout=False):
        if not self.variants or (not heldout and self.rng.random() >= self.probability):
            return FRESH_CELL_ID
        selected = self._select_key(heldout)
        if selected is None:
            return FRESH_CELL_ID
        key, seed = selected
        variants = [
            index for index in self.groups[key]
            if self.variants[index].seed == seed
        ]
        variant = self._weighted_choice(
            variants,
            [1 / np.sqrt(self.variant_starts[index] + 1) for index in variants],
        )
        return variant

    def record_start(self, variant_id):
        if variant_id == FRESH_CELL_ID:
            return
        variant = self.variants[int(variant_id)]
        self.sample_clock += 1
        self.variant_starts[variant.variant_id] += 1
        stat = self.stats[variant.key]
        stat['starts'] += 1
        stat['last_sample'] = self.sample_clock

    def record_result(self, variant_id, success):
        if variant_id == FRESH_CELL_ID:
            return
        variant = self.variants[int(variant_id)]
        self.stats[variant.key]['completions'] += 1
        if success:
            self.stats[variant.key]['successes'] += 1

    def metrics(self, heldout=False):
        keys = self.heldout_keys if heldout else self.training_keys
        bands = self.heldout_bands if heldout else self.training_bands
        attempted = [key for key in keys if self.stats[key]['starts']]
        starts = sum(self.stats[key]['starts'] for key in attempted)
        completions = sum(self.stats[key]['completions'] for key in attempted)
        successes = sum(self.stats[key]['successes'] for key in attempted)
        posterior = [
            (self.stats[key]['successes'] + 1) /
            (self.stats[key]['completions'] + 2)
            for key in attempted
        ]
        result = {
            'cells': len(keys),
            'attempted_cells': len(attempted),
            'starts': starts,
            'completions': completions,
            'successes': successes,
            'success_rate': round(successes / max(1, completions), 4),
            'median_cell_posterior': (
                round(float(np.median(posterior)), 4) if posterior else None
            ),
        }
        if not heldout:
            weights, competences = self._training_band_weights()
            weight_total = max(np.finfo(np.float64).tiny, sum(weights.values()))
        result['bands'] = []
        for band in sorted(bands):
            band_completions = sum(
                self.stats[key]['completions'] for key in bands[band]
            )
            band_successes = sum(
                self.stats[key]['successes'] for key in bands[band]
            )
            item = {
                    'height': round(band * self.band_height),
                    'cells': len(bands[band]),
                    'attempted': sum(
                        1 for key in bands[band] if self.stats[key]['starts']
                    ),
                    'completions': band_completions,
                    'success_rate': round(
                        band_successes / max(1, band_completions),
                        4,
                    ),
            }
            if not heldout:
                item.update({
                    'competence': round(competences[band], 4),
                    'sample_fraction': round(
                        weights[band] / weight_total,
                        4,
                    ),
                })
            result['bands'].append(item)
        return result

    def state_dict(self):
        return {
            'rng_state': self.rng.bit_generator.state,
            'sample_clock': self.sample_clock,
            'stats': dict(self.stats),
            'variant_starts': dict(self.variant_starts),
        }

    def load_state_dict(self, state, *, load_variant_starts=True):
        self.rng.bit_generator.state = state['rng_state']
        self.sample_clock = int(state.get('sample_clock', 0))
        for key, value in state.get('stats', {}).items():
            if key in self.groups:
                self.stats[key] = {
                    'starts': int(value['starts']),
                    'completions': int(value.get('completions', value['starts'])),
                    'successes': int(value['successes']),
                    'last_sample': int(value.get('last_sample', 0)),
                }
        if load_variant_starts:
            for variant_id, starts in state.get('variant_starts', {}).items():
                variant_id = int(variant_id)
                if 0 <= variant_id < len(self.variants):
                    self.variant_starts[variant_id] = int(starts)
