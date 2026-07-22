import assert from 'node:assert/strict';
import { ARENA_X } from '../src/constants.js';
import { Sim } from '../src/sim/sim.js';
import { BLOCK_TYPES } from '../src/sim/blockTypes.js';
import { ACTION_COUNT, OBS_SIZE, STATE_SIZE, actionInput, encodeObservation } from '../rl/env.mjs';
import {
  ACTION_COUNT as ACTION_COUNT_V2,
  FALLING_FEATURES,
  FORECAST_FEATURES,
  STATE_SIZE as STATE_SIZE_V2,
  TERRAIN_COLS,
  TERRAIN_EDGE_LEFT,
  TERRAIN_EDGE_RIGHT,
  TERRAIN_FAULT_SHIFT,
  TERRAIN_SIZE,
  createObservation as createObservationV2,
  encodeObservation as encodeObservationV2,
  heldActionInput,
} from '../rl/env-v2.mjs';

const sim = new Sim(42);
const first = encodeObservation(sim);
assert.equal(first.grid.length, OBS_SIZE);
assert.equal(first.state.length, STATE_SIZE);
assert.ok(first.grid.some((value) => value > 0));

for (let action = 0; action < ACTION_COUNT; action++) {
  const mapped = actionInput(action, false);
  assert.equal(typeof mapped.focusHeld, 'boolean');
  sim.step(mapped.input);
}

const focus = actionInput(8, false);
assert.equal(focus.input.focusPressed, true);
assert.equal(focus.input.focusDirY, -1);
assert.equal(actionInput(0, true).input.focusReleased, true);
assert.equal(actionInput(3, false, false).input.jumpPressed, true);
assert.equal(actionInput(3, false, true).input.jumpPressed, false);

assert.equal(ACTION_COUNT_V2, 18);
const focusUpLeft = heldActionInput(13, 0);
assert.equal(focusUpLeft.focusHeld, true);
assert.equal(focusUpLeft.focusPressed, true);
assert.equal(focusUpLeft.left, true);
assert.equal(focusUpLeft.up, true);
assert.equal(focusUpLeft.focusDirX, -1);
assert.equal(focusUpLeft.focusDirY, -1);
assert.equal(focusUpLeft.jumpPressed, true);
const heldFocus = heldActionInput(13, 13);
assert.equal(heldFocus.focusPressed, false);
assert.equal(heldFocus.jumpPressed, false);
const releaseWithoutSyntheticJump = heldActionInput(4, 13);
assert.equal(releaseWithoutSyntheticJump.focusReleased, true);
assert.equal(releaseWithoutSyntheticJump.jumpPressed, false);
assert.equal(heldActionInput(15, 0).down, true);

const timeScaleSim = new Sim(8, { director: false, rules: { autoGuard: false, checkpoints: false } });
const aimTransition = timeScaleSim.step(heldActionInput(13, 0));
assert.equal(aimTransition.worldScale, 0.1, 'Focus Aim reports slowed world time');
const dashTransition = timeScaleSim.step(heldActionInput(4, 13));
assert.equal(dashTransition.worldScale, 0.55, 'Focus dash reports its world time');

const v2sim = new Sim(7, { director: false, rules: { autoGuard: false, checkpoints: false } });
const beam = v2sim.blocks.spawnAt(100, 0, 'beam');
beam.yVel = 4;
v2sim.director.forecasts.push({
  x: 100, y: -200, w: 60, h: 40, type: 'wood', spec: BLOCK_TYPES.get('wood'), frames: 18,
});
v2sim.director.forecasts.push({
  x: 100, y: -242, w: 90, h: 40, type: 'beam', spec: BLOCK_TYPES.get('beam'), frames: 9,
});
v2sim.height = 10_000;
const v2 = encodeObservationV2(v2sim, { previousAction: 0 }, createObservationV2());
assert.equal(v2.terrain.length, TERRAIN_SIZE);
assert.equal(v2.state.length, STATE_SIZE_V2);
assert.equal(v2.falling[FALLING_FEATURES - 1], 1);
assert.equal(v2.falling[7], 1, 'falling beam identity is observable');
assert.equal(v2.forecasts[FORECAST_FEATURES - 1], 1);
assert.equal(v2.forecasts[FORECAST_FEATURES * 2 - 1], 1, 'stacked forecasts remain distinct');
assert.notEqual(v2.forecasts[4], v2.forecasts[FORECAST_FEATURES + 4]);
const encoded10k = v2.state[20];
v2sim.height = 20_000;
encodeObservationV2(v2sim, { previousAction: 0 }, v2);
assert.ok(v2.state[20] > encoded10k, 'height encoding must remain unsaturated beyond 10k');

const terrainSim = new Sim(11, { director: false, rules: { autoGuard: false, checkpoints: false } });
const lower = terrainSim.blocks.spawnAt(ARENA_X, 0, 'wood');
terrainSim.blocks.fixAt(lower, 1);
const upper = terrainSim.blocks.spawnAt(ARENA_X, 0, 'gravel');
terrainSim.blocks.fixAt(upper, 2);
lower.faultTimer = 12;
lower.faultDuration = 12;
const terrainObservation = encodeObservationV2(
  terrainSim,
  { previousAction: 0 },
  createObservationV2(),
);
const occupiedRows = [];
for (let row = 0; row < terrainObservation.terrain.length / TERRAIN_COLS; row++) {
  const cell = terrainObservation.terrain[row * TERRAIN_COLS];
  if ((cell & 3) !== 0) occupiedRows.push({ row, cell });
}
assert.equal(occupiedRows.length, 2, 'each one-layer block occupies exactly one terrain row');
assert.deepEqual(occupiedRows.map(({ cell }) => cell & 3), [2, 1]);
assert.ok(occupiedRows.every(({ cell }) => (cell & TERRAIN_EDGE_LEFT) !== 0));
assert.ok(occupiedRows.every(({ cell }) => (cell & TERRAIN_EDGE_RIGHT) === 0));
assert.ok((terrainObservation.terrain[occupiedRows[0].row * TERRAIN_COLS + 3] & TERRAIN_EDGE_RIGHT) !== 0);
const lowerCell = occupiedRows[1].cell;
assert.equal(
  (lowerCell >> TERRAIN_FAULT_SHIFT) & 255,
  12,
  'terrain exposes absolute remaining collapse warning time',
);
console.log('ok RL observations and action mapping are valid');
