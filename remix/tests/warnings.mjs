import assert from 'node:assert/strict';
import {
  estimateFallFrames,
  stackWarnings,
  warningUrgency,
} from '../src/render/warningLayout.js';

const item = (x, w, eta) => ({ x, w, eta });

const directStack = stackWarnings([
  item(100, 60, 3),
  item(100, 60, 1),
  item(100, 60, 2),
]);
assert.equal(new Set(directStack.map((warning) => warning.y)).size, 3);
const earliest = directStack.find((warning) => warning.eta === 1);
assert.equal(earliest.y, Math.max(...directStack.map((warning) => warning.y)));

const separate = stackWarnings([item(100, 60, 1), item(160, 60, 2)]);
assert.equal(separate[0].y, separate[1].y, 'edge-touching warnings should not stack');

const bridged = stackWarnings([
  item(100, 60, 1),
  item(145, 90, 2),
  item(220, 60, 3),
]);
assert.equal(new Set(bridged.map((warning) => warning.y)).size, 3);

const deep = stackWarnings(
  Array.from({ length: 12 }, (_, index) => item(300, 60, index)),
  { top: 3, maxDepth: 48 },
);
assert.equal(deep.length, 12);
assert.ok(Math.max(...deep.map((warning) => warning.y)) <= 51);
assert.equal(new Set(deep.map((warning) => warning.y)).size, 12);

assert.ok(estimateFallFrames(600, 0, 0.3, 13) > estimateFallFrames(120, 0, 0.3, 13));
const source = { yVel: 2, spec: { gravity: 0.3, maxFallSpeed: 13 } };
const farSignal = warningUrgency(source, -500, 54, 0.5);
const nearSignal = warningUrgency(source, -20, 54, 0.5);
assert.ok(farSignal.eta > nearSignal.eta);
assert.ok(farSignal.progress < nearSignal.progress);

console.log('ok warning layout preserves order and communicates arrival urgency');
