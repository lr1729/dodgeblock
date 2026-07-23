import assert from 'node:assert/strict';

import { heightReward, targetReward } from '../rl/reward-v2.mjs';

function close(actual, expected, message) {
  assert.ok(Math.abs(actual - expected) < 1e-9, `${message}: ${actual} != ${expected}`);
}

const progress = targetReward({
  beforeHeight: 0,
  afterHeight: 400,
  targetHeight: 1000,
  discount: 1,
  worldScale: 1,
  dead: false,
  success: false,
});
const moreProgress = targetReward({
  beforeHeight: 400,
  afterHeight: 800,
  targetHeight: 1000,
  discount: 1,
  worldScale: 0.1,
  dead: false,
  success: false,
});
const failure = targetReward({
  beforeHeight: 800,
  afterHeight: 800,
  targetHeight: 1000,
  discount: 1,
  worldScale: 1,
  dead: true,
  success: false,
});
close(progress + moreProgress + failure, 0, 'fresh failure shaping must telescope to zero');

const success = targetReward({
  beforeHeight: 800,
  afterHeight: 1000,
  targetHeight: 1000,
  discount: 1,
  worldScale: 1,
  dead: false,
  success: true,
});
close(progress + moreProgress + success, 1, 'fresh success shaping must telescope to one');

const archivedFailure = targetReward({
  beforeHeight: 600,
  afterHeight: 600,
  targetHeight: 1000,
  discount: 1,
  worldScale: 1,
  dead: true,
  success: false,
});
close(archivedFailure, -0.6, 'archived failure must remove starting potential');

close(heightReward({
  beforeHeight: 80,
  afterHeight: 120,
  blockHeight: 40,
  dead: false,
  deathPenalty: 1,
  aliveReward: 0.001,
  worldScale: 0.1,
}), 1.0001, 'legacy height reward must remain available');

process.stdout.write('ok target reward preserves the binary success objective\n');
