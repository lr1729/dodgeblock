import assert from 'node:assert/strict';
import { Sim } from '../src/sim/sim.js';
import { ACTION_COUNT, OBS_SIZE, STATE_SIZE, actionInput, encodeObservation } from '../rl/env.mjs';

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
console.log('ok RL observations and action mapping are valid');
